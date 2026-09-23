import os
import csv
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / len(scores)

def main():
    motion_cache = os.path.join(CACHE_DIR, 'cache_motion_features.npz')
    data = np.load(motion_cache, allow_pickle=True)
    X_test_motion = data['X_test']
    motion_feats = data['feature_names'].tolist()
    
    # Load test IDs
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    # Unsupervised Physics Heuristics
    # 1. diff_spike_ratio: max frame difference relative to baseline
    score_diff = X_test_motion[:, motion_feats.index('diff_spike_ratio')]
    
    # 2. blur_drop_max: max sudden loss of camera sharpness (impact)
    score_blur = X_test_motion[:, motion_feats.index('blur_drop_max')]
    
    # 3. flow_spike_ratio: max optical flow magnitude relative to baseline
    score_flow = X_test_motion[:, motion_feats.index('flow_spike_ratio')]
    
    # Average the ranks
    final_scores = (
        rank_normalize(score_diff) + 
        rank_normalize(score_blur) + 
        rank_normalize(score_flow)
    ) / 3.0
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_zero_shot_ensemble.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_scores):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
