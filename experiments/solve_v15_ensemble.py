import os
import csv
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def load_predictions(filename):
    path = os.path.join(SUBMISSIONS_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {filename} not found.")
        return None
    preds = {}
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            preds[row['clip_id']] = float(row['label'])
    return preds

def rank_normalize(scores_dict):
    """Converts raw scores to uniform ranks between 0 and 1."""
    clip_ids = list(scores_dict.keys())
    scores = [scores_dict[cid] for cid in clip_ids]
    
    # argsort twice gives the rank of each element
    ranks = np.argsort(np.argsort(scores)).astype(float)
    ranks = ranks / (len(ranks) - 1)
    
    return {cid: rank for cid, rank in zip(clip_ids, ranks)}

def main():
    print("="*60)
    print("V15: GRANDMASTER ENSEMBLE")
    print("Combining our best independent models")
    print("="*60)
    
    # 1. Load the best models
    models = {
        'V7 (YOLO Physics + CLIP)': ('submission_v7_yolo.csv', 4.0),
        'V8 (X-CLIP Video Model)': ('submission_v8_video.csv', 3.0),
        'V10 (Distilled CNN)': ('submission_v10_distillation.csv', 2.0),
        'V9 (Masked Flow)': ('submission_v9_masked_flow.csv', 2.0)
    }
    
    all_preds = {}
    total_weight = 0
    
    for name, (filename, weight) in models.items():
        raw_preds = load_predictions(filename)
        if not raw_preds:
            continue
            
        print(f"Loaded {name} (Weight: {weight})")
        norm_preds = rank_normalize(raw_preds)
        
        if not all_preds:
            for cid in norm_preds:
                all_preds[cid] = 0.0
                
        for cid, score in norm_preds.items():
            all_preds[cid] += score * weight
            
        total_weight += weight
        
    # 2. Average the predictions
    final_preds = {}
    for cid in all_preds:
        final_preds[cid] = all_preds[cid] / total_weight
        
    # 3. Save submission
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v15_grandmaster.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        # Sort to match sample submission order exactly
        sample_path = os.path.join(DATA_DIR, 'sample_submission.csv')
        with open(sample_path, 'r') as sf:
            reader = csv.DictReader(sf)
            for row in reader:
                cid = row['clip_id']
                writer.writerow([cid, f'{final_preds[cid]:.6f}'])
                
    print(f"\nSaved ensemble to {out_path}")
    print("Ready to submit!")

if __name__ == '__main__':
    main()
