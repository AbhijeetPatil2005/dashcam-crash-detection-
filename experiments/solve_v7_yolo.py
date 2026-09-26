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
    print("="*60)
    print("V7: YOLO Kinematics + V6 Foundation Models")
    print("="*60)
    
    # 1. Load V6 ultimate zero-shot predictions
    v6_path = os.path.join(SUBMISSIONS_DIR, 'submission_v6_ultimate.csv')
    test_ids = []
    v6_scores = []
    with open(v6_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            v6_scores.append(float(row['label']))
    
    v6_scores = np.array(v6_scores)
    
    # 2. Load YOLO kinematics features
    yolo_path = os.path.join(CACHE_DIR, 'cache_yolo_features.npz')
    if not os.path.exists(yolo_path):
        print(f"Error: {yolo_path} not found. Run extract_yolo_kinematics.py first.")
        return
        
    data = np.load(yolo_path)
    X_yolo = data['X_test']
    
    # feature_names = ['max_decel', 'max_area_change', 'max_iou', 'obj_count']
    max_decel = X_yolo[:, 0]
    max_area_change = X_yolo[:, 1]
    max_iou = X_yolo[:, 2]
    
    # Create physics scores based on mathematical thresholds of collisions
    # A crash is characterized by massive deceleration, rapid area expansion, and high bounding box overlap
    physics_score = (rank_normalize(max_decel) * 1.5 + 
                     rank_normalize(max_area_change) * 0.5 + 
                     rank_normalize(max_iou) * 1.0)
    
    # Combine semantic understanding (V6) and physical reality (YOLO)
    print("Combining Semantic Foundation Models with YOLO Kinematics...")
    final_score = rank_normalize(v6_scores) * 2.0 + rank_normalize(physics_score) * 1.0
    final_score = rank_normalize(final_score)
    
    # Write submission
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v7_yolo.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_score):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved to {out_path}")
    print("Ready to submit!")

if __name__ == '__main__':
    main()
