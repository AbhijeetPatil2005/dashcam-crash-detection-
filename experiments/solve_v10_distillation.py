import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR
TEST_DIR = os.path.join(DATA_DIR, 'test')

class PseudoDataset(Dataset):
    def __init__(self, clip_ids, labels, transform):
        self.samples = []
        for cid, lbl in zip(clip_ids, labels):
            # Extract 5 frames per clip to prevent overfitting and speed up training
            for idx in [5, 10, 15, 20, 25]:
                path = os.path.join(TEST_DIR, cid, f'frame_{idx:03d}.jpg')
                if os.path.exists(path):
                    self.samples.append((path, lbl))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        return self.transform(img), torch.tensor(label, dtype=torch.float32)

class TestDataset(Dataset):
    def __init__(self, clip_ids, transform):
        self.clip_ids = clip_ids
        self.transform = transform

    def __len__(self):
        return len(self.clip_ids)

    def __getitem__(self, idx):
        cid = self.clip_ids[idx]
        # Average prediction over 5 frames
        frames = []
        for f_idx in [5, 10, 15, 20, 25]:
            path = os.path.join(TEST_DIR, cid, f'frame_{f_idx:03d}.jpg')
            if os.path.exists(path):
                img = Image.open(path).convert('RGB')
                frames.append(self.transform(img))
        if len(frames) == 0:
            frames.append(torch.zeros(3, 224, 224))
        return cid, torch.stack(frames)

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / len(scores)

def main():
    print("="*60)
    print("V10: IN-DOMAIN KNOWLEDGE DISTILLATION (Self-Training CNN)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load our best V7 (YOLO+CLIP) predictions
    v7_path = os.path.join(SUBMISSIONS_DIR, 'submission_v7_yolo.csv')
    test_ids = []
    v7_scores = []
    with open(v7_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            v7_scores.append(float(row['label']))
            
    test_ids = np.array(test_ids)
    v7_scores = np.array(v7_scores)
    
    # 2. Select ultra-high confidence clips for Pseudo-Labeling
    # Top 5% crashes, Bottom 5% normal
    n_pseudo = int(len(test_ids) * 0.05)
    sorted_idx = np.argsort(v7_scores)
    
    normal_idx = sorted_idx[:n_pseudo]
    crash_idx = sorted_idx[-n_pseudo:]
    
    train_ids = np.concatenate([test_ids[normal_idx], test_ids[crash_idx]])
    train_labels = np.concatenate([np.zeros(n_pseudo), np.ones(n_pseudo)])
    
    print(f"Distilling knowledge into CNN using {n_pseudo} normal and {n_pseudo} crash clips from Test Set.")
    
    # 3. Augmentations to prevent overfitting to these 174 clips
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = PseudoDataset(train_ids, train_labels, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    
    test_dataset = TestDataset(test_ids, test_transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # 4. Train EfficientNet-B0 (Small, fast, highly accurate)
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
    model = model.to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    
    epochs = 3
    print("Training self-distilled CNN...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                outs = model(imgs).squeeze(-1)
                loss = criterion(outs, lbls)
                
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            preds = (torch.sigmoid(outs) > 0.5).float()
            correct += (preds == lbls).sum().item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f} Acc: {correct/len(train_dataset):.4f}")
        
    # 5. Predict on full test set
    model.eval()
    cnn_scores = []
    print("Extracting CNN predictions...")
    with torch.no_grad():
        for cid, frames in tqdm(test_loader):
            # frames: [batch_size, 5, 3, 224, 224]
            batch_size, n_frames, c, h, w = frames.shape
            frames = frames.view(-1, c, h, w).to(device)
            
            with torch.amp.autocast('cuda'):
                outs = model(frames).squeeze(-1)
                outs = torch.sigmoid(outs).view(batch_size, n_frames).mean(dim=1)
                
            cnn_scores.extend(outs.cpu().numpy())
            
    cnn_scores = np.array(cnn_scores)
    
    # 6. Final Fusion
    # We combine our new CNN (which learned the test domain visually) with our YOLO/CLIP baseline
    print("Fusing CNN predictions with V7 Physics/Semantics baseline...")
    final_score = rank_normalize(v7_scores) * 2.0 + rank_normalize(cnn_scores) * 1.5
    final_score = rank_normalize(final_score)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v10_distillation.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_score):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved to {out_path}")
    print("Ready to submit!")

if __name__ == '__main__':
    main()
