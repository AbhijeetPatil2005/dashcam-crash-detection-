"""
STEP 1: Score every training clip with CLIP to find mislabeled examples.
This identifies clips where the AI strongly disagrees with the provided label.
These are the ones most likely to be mislabeled (the "poison" in the dataset).

Run this first, then run train_labeler.py to review the disagreements.
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import open_clip

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

CRASH_PROMPTS = [
    "a dashcam video of a car crash",
    "a dashcam video of a violent vehicle collision",
    "a dashcam video of cars smashing into each other",
    "a dashcam video capturing the moment of a traffic accident",
    "a dashcam video of a car hitting another car",
    "a dashcam video showing a car accident with debris flying",
    "a dashcam video of a head on collision",
    "a dashcam video of a car spinning out and crashing",
    "a video of a car crash from a dashboard camera",
    "a video showing the moment of impact in a car accident",
]

NORMAL_PROMPTS = [
    "a dashcam video of a car driving normally on a road",
    "a dashcam video of peaceful highway driving",
    "a dashcam video of smooth traffic flow",
    "a dashcam video of a calm and uneventful drive",
    "a dashcam video of a car following traffic safely",
    "a dashcam video of everyday commute driving",
    "a dashcam video of a car on an empty road",
    "a video of a boring normal car ride",
    "a video of a car driving without any incidents",
    "a dashcam recording of a routine drive",
]

def main():
    print("="*60)
    print("TRAIN SET SCORING: Finding Mislabeled Examples")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load CLIP
    print("Loading CLIP ViT-L-14 (DataComp)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='datacomp_xl_s13b_b90k'
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    
    # Encode text
    text_crash = tokenizer(CRASH_PROMPTS).to(device)
    text_normal = tokenizer(NORMAL_PROMPTS).to(device)
    
    with torch.no_grad():
        feat_crash = F.normalize(model.encode_text(text_crash), dim=-1).mean(0, keepdim=True)
        feat_normal = F.normalize(model.encode_text(text_normal), dim=-1).mean(0, keepdim=True)
    
    # Load training labels
    train_data = []
    with open(os.path.join(DATA_DIR, 'train_labels.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_data.append({
                'clip_id': row['clip_id'],
                'label': int(row['label']),
                'group_id': row['group_id']
            })
    
    train_dir = os.path.join(DATA_DIR, 'train')
    frame_indices = [0, 5, 10, 15, 20, 25, 29]
    
    # Score each training clip
    results = []
    for item in tqdm(train_data, desc="Scoring Train Set"):
        clip_dir = os.path.join(train_dir, item['clip_id'])
        crash_sims = []
        normal_sims = []
        
        for idx in frame_indices:
            path = os.path.join(clip_dir, f'frame_{idx:03d}.jpg')
            try:
                img = preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
            except Exception:
                continue
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    img_feat = F.normalize(model.encode_image(img), dim=-1)
                    crash_sims.append((img_feat @ feat_crash.T).item())
                    normal_sims.append((img_feat @ feat_normal.T).item())
        
        if crash_sims:
            ai_score = max(crash_sims) - min(normal_sims)
        else:
            ai_score = 0.0
        
        results.append({
            'clip_id': item['clip_id'],
            'label': item['label'],
            'group_id': item['group_id'],
            'ai_score': ai_score
        })
    
    # Save results
    out_path = os.path.join(CACHE_DIR, 'train_clip_scores.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label', 'group_id', 'ai_score'])
        for r in results:
            writer.writerow([r['clip_id'], r['label'], r['group_id'], f"{r['ai_score']:.6f}"])
    
    # Print disagreement stats
    ai_scores = np.array([r['ai_score'] for r in results])
    labels = np.array([r['label'] for r in results])
    
    median_score = np.median(ai_scores)
    ai_pred = (ai_scores > median_score).astype(int)
    
    disagree = (ai_pred != labels)
    print(f"\nTotal clips: {len(results)}")
    print(f"AI disagrees with label: {disagree.sum()} ({100*disagree.mean():.1f}%)")
    
    # Find the worst offenders
    # Clips labeled CRASH but AI says NORMAL (low score)
    crash_but_low = [(r['ai_score'], r['clip_id']) for r in results if r['label'] == 1]
    crash_but_low.sort()
    print(f"\nTop 10 'Crash' clips that AI thinks are Normal (possible mislabels):")
    for score, cid in crash_but_low[:10]:
        print(f"  {cid}: AI={score:.4f}")
    
    # Clips labeled NORMAL but AI says CRASH (high score)
    normal_but_high = [(r['ai_score'], r['clip_id']) for r in results if r['label'] == 0]
    normal_but_high.sort(reverse=True)
    print(f"\nTop 10 'Normal' clips that AI thinks are Crashes (possible mislabels):")
    for score, cid in normal_but_high[:10]:
        print(f"  {cid}: AI={score:.4f}")
    
    print(f"\nSaved all scores to {out_path}")
    print("Now run: python train_labeler.py")

if __name__ == '__main__':
    main()
