import os
import csv
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def main():
    motion_cache = os.path.join(CACHE_DIR, 'cache_motion_features.npz')
    data = np.load(motion_cache, allow_pickle=True)
    
    X_test = data['X_test']
    feature_names = data['feature_names'].tolist()
    
    # Load test IDs
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    # Which feature is best? 
    # 'diff_max' is the maximum absolute pixel difference between consecutive frames.
    # 'diff_spike_ratio' is diff_max / diff_mean
    
    idx_max = feature_names.index('diff_max')
    idx_spike = feature_names.index('diff_spike_ratio')
    idx_flow = feature_names.index('flow_max')
    
    # Let's try diff_spike_ratio as it is normalized against the camera's baseline vibration
    scores = X_test[:, idx_spike]
    
    # Rank normalize just in case
    ranks = np.argsort(np.argsort(scores)).astype(float) / len(scores)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_zero_shot.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, ranks):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
