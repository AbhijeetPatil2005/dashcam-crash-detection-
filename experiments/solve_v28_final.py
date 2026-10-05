"""
V28: FINAL PHYSICS-VISION BLEND
================================
V27b proved physics injection works (0.79098 vs V24's 0.78943).

Analysis shows:
- V25b is 0.955 correlated with V24 (redundant, only adds noise)
- Removing V25b and giving that weight to V16 physics = cleaner signal
- 10% V16 is the sweet spot (8% proved itself, 10% slightly more decisive)

This is the cleanest, most mathematically principled blend:
  90% V24 (proven best vision model)
  10% V16 (uncorrelated physics — TTC, spin, deceleration)
"""

import os
import csv
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / max(1, len(scores) - 1)

def load_scores(filename):
    scores = []
    with open(os.path.join(SUBMISSIONS_DIR, filename), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row['label']))
    return np.array(scores)

def main():
    print("="*60)
    print("V28: FINAL PHYSICS-VISION BLEND")
    print("="*60)
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
    
    v24_ranks = rank_normalize(load_scores('submission_v24_feature_blender.csv'))
    v16_ranks = rank_normalize(load_scores('submission_v16_advanced_physics.csv'))
    
    # Clean, simple, principled: 90% Vision + 10% Physics
    final = v24_ranks * 0.90 + v16_ranks * 0.10
    final_ranks = rank_normalize(final)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v28_final.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
    
    print(f"Blend: 90% V24 + 10% V16-Physics")
    print(f"Saved V28 to {out_path}")

if __name__ == '__main__':
    main()
