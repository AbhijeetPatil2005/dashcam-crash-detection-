import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from torchvision.models.video import r3d_18, R3D_18_Weights
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

class CrashDataset(Dataset):
    def __init__(self, data_dir, labels_dict=None, is_train=True, num_frames=16):
        self.data_dir = data_dir
        self.labels_dict = labels_dict
        self.is_train = is_train
        self.num_frames = num_frames
        
        if is_train:
            self.clip_ids = list(labels_dict.keys())
        else:
            self.clip_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
            
        self.transform = transforms.Compose([
            transforms.Resize((128, 171)),
            transforms.CenterCrop((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
        ])

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        clip_id = self.clip_ids[idx]
        clip_dir = os.path.join(self.data_dir, clip_id)
        
        # Sample frames evenly
        indices = np.linspace(0, 29, self.num_frames, dtype=int)
        
        frames = []
        for i in indices:
            path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
                frames.append(self.transform(img))
            else:
                # Fallback to zeros if missing
                frames.append(torch.zeros(3, 112, 112))
                
        # Stack to (C, T, H, W)
        video = torch.stack(frames, dim=1)
        
        if self.is_train:
            label = float(self.labels_dict[clip_id])
            return video, torch.tensor([label], dtype=torch.float32)
        else:
            return video, clip_id

def main():
    print("="*60)
    print("V13: SUPERVISED TRAINING ON CLEAN LABELS (ResNet3D)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load clean training labels
    labels_path = os.path.join(DATA_DIR, 'train_labels_clean.csv')
    train_labels = {}
    with open(labels_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_labels[row['clip_id']] = float(row['label'])
            
    print(f"Loaded {len(train_labels)} clean training labels.")
    
    # 2. Setup model
    print("Loading pretrained ResNet3D-18...")
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    
    # Freeze early layers for stability
    for name, param in model.named_parameters():
        if 'layer4' not in name and 'fc' not in name:
            param.requires_grad = False
            
    # Modify final layer for binary classification
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 1)
    )
    model = model.to(device)
    
    # 3. Training Loop
    train_dir = os.path.join(DATA_DIR, 'train')
    train_dataset = CrashDataset(train_dir, train_labels, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-3)
    
    num_epochs = 3
    print("\nStarting Training...")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in pbar:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1)})
            
    # 4. Inference on Test Set
    print("\nGenerating Predictions on Test Set...")
    test_dir = os.path.join(DATA_DIR, 'test')
    test_dataset = CrashDataset(test_dir, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    model.eval()
    predictions = {}
    with torch.no_grad():
        for inputs, clip_ids in tqdm(test_loader, desc="Testing"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs).squeeze(-1).cpu().numpy()
            
            # Handle single item batch edge case
            if probs.ndim == 0:
                probs = [probs.item()]
                clip_ids = [clip_ids[0]]
                
            for cid, prob in zip(clip_ids, probs):
                predictions[cid] = prob
                
    # 5. Save final submission
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v13_clean_r3d.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        
        # Ensure we write in the order of sample_submission.csv
        sample_path = os.path.join(DATA_DIR, 'sample_submission.csv')
        with open(sample_path, 'r') as sf:
            reader = csv.DictReader(sf)
            for row in reader:
                cid = row['clip_id']
                score = predictions.get(cid, 0.5)
                writer.writerow([cid, f'{score:.6f}'])
                
    print(f"\nSaved to {out_path}")
    print("Ready to submit!")

if __name__ == '__main__':
    main()
