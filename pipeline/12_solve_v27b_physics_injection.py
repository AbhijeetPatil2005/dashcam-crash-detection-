"""
V27b: SURGICAL PHYSICS INJECTION
=================================
The LightGBM approach (V27) failed to use V16 because the tree model 
was trained on pseudo-labels from vision models, so it just learned to copy
the vision meta-scores and ignored physics entirely.

Instead, we do a DIRECT rank-blend. V16 has only 0.21 correlation with V24,
meaning it sees fundamentally different crash signals (Time-to-Collision, 
spin detection, deceleration from multi-object tracking).

We inject just 5-8% V16 physics signal into V24. This is a tiny surgical 
correction that nudges ambiguous clips in the right direction without 
overwhelming V24's proven 0.789 score.
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
    path = os.path.join(SUBMISSIONS_DIR, filename)
    scores = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row['label']))
    return np.array(scores)

def main():
    print("="*60)
    print("V27b: SURGICAL PHYSICS INJECTION")
    print("="*60)
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
    
    # Load our components
    v24_ranks = rank_normalize(load_scores('submission_v24_feature_blender.csv'))
    v16_ranks = rank_normalize(load_scores('submission_v16_advanced_physics.csv'))
    v25b_ranks = rank_normalize(load_scores('submission_v25b_final_blend.csv'))
    
    # V24 (0.789) is our absolute best. V16 is our most diverse physics model.
    # V25b (0.788) provides overnight model diversity (SigLIP2 + ViT-H).
    # 
    # Optimal blend: 
    #   85% V24 (proven champion)
    #   8%  V16 (uncorrelated physics — this is the key injection)
    #   7%  V25b (overnight model diversity)
    
    blend = v24_ranks * 0.85 + v16_ranks * 0.08 + v25b_ranks * 0.07
    final_ranks = rank_normalize(blend)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v27b_physics_injection.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
    
    print(f"Saved V27b to {out_path}")
    print(f"Blend: 85% V24 + 8% V16-Physics + 7% V25b-Overnight")

if __name__ == '__main__':
    main()
