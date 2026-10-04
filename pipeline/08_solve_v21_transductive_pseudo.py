"""
V21: TRANSDUCTIVE PHYSICS PSEUDO-LABELING
==========================================
We know Physics models (YOLO/Flow) fail because of domain shift 
(trained on Sunny, tested on Rain/Night).
BUT, our V20 High-Res SigLIP model scored 0.785 on the TEST set directly!
This means V20 is highly accurate at identifying true test-set crashes.

We will use V20's most confident predictions to pseudo-label the test set.
Then we train a LightGBM Physics classifier ON THE TEST SET (Transductive Learning).
This physics model will learn what a crash looks like in the rain/night!
Finally, we ensemble the new domain-adapted physics model with V20.
"""

import os
import csv
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / max(1, len(scores) - 1)

def main():
    print("="*60)
    print("V21: TRANSDUCTIVE PHYSICS PSEUDO-LABELING")
    print("="*60)
    
    # 1. Load test IDs
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    # 2. Load V20 predictions
    v20_scores = []
    v20_path = os.path.join(SUBMISSIONS_DIR, 'submission_v20_highres.csv')
    if not os.path.exists(v20_path):
        print("Error: submission_v20_highres.csv not found!")
        return
        
    with open(v20_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            v20_scores.append(float(row['label']))
    v20_scores = np.array(v20_scores)
    
    # 3. Load Cached Physics Features for the Test Set
    print("Loading Cached Physics Features (Optical Flow + YOLO Kinematics)...")
    flow_features = np.load(os.path.join(CACHE_DIR, 'cache_masked_flow.npz'))['X']  # Shape: (1750, 3)
    
    yolo_cache = np.load(os.path.join(CACHE_DIR, 'cache_yolo_features.npz'))
    # yolo_cache['X_test'] was saved in V7/V9. Let's verify shape.
    if 'X_test' in yolo_cache:
        yolo_features = yolo_cache['X_test']
    else:
        # Fallback if the key was named differently
        keys = list(yolo_cache.keys())
        yolo_features = yolo_cache[keys[0]]
        
    # Combine features (if shapes match)
    if len(flow_features) == len(test_ids) and len(yolo_features) == len(test_ids):
        X_test_physics = np.hstack([flow_features, yolo_features])
        print(f"Physics Feature Shape: {X_test_physics.shape}")
    else:
        print("Error: Feature shapes do not match test_ids length!")
        return
        
    # 4. Generate Pseudo-Labels from V20
    print("Generating Pseudo-Labels...")
    # Rank clips by V20 score
    ranked_indices = np.argsort(v20_scores)
    
    # Bottom 1000 are very safely Normal (0)
    # Top 200 are very safely Crash (1)
    normal_idx = ranked_indices[:1000]
    crash_idx = ranked_indices[-200:]
    
    train_idx = np.concatenate([normal_idx, crash_idx])
    
    X_train_pseudo = X_test_physics[train_idx]
    y_train_pseudo = np.concatenate([np.zeros(len(normal_idx)), np.ones(len(crash_idx))])
    
    print(f"Created Pseudo-Training Set: {len(X_train_pseudo)} samples (1000 Normal, 200 Crash)")
    
    # 5. Train Domain-Adapted LightGBM
    print("Training Transductive LightGBM...")
    # Use class_weight='balanced' because of 1000:200 ratio
    clf = lgb.LGBMClassifier(n_estimators=150, learning_rate=0.03, max_depth=5, class_weight='balanced', random_state=42)
    clf.fit(X_train_pseudo, y_train_pseudo)
    
    train_preds = clf.predict_proba(X_train_pseudo)[:, 1]
    print(f"Pseudo-Train AUC: {roc_auc_score(y_train_pseudo, train_preds):.4f}")
    
    # 6. Predict on the Entire Test Set
    print("Predicting on Test Set...")
    physics_preds = clf.predict_proba(X_test_physics)[:, 1]
    
    # 7. Ultimate Ensemble
    print("Ensembling V21...")
    # V20 is incredibly strong (0.785), so we give it 80% weight.
    # The domain-adapted physics model gets 20% weight to provide the final boost.
    v20_ranks = rank_normalize(v20_scores)
    physics_ranks = rank_normalize(physics_preds)
    
    final_ensemble = v20_ranks * 0.80 + physics_ranks * 0.20
    final_ranks = rank_normalize(final_ensemble)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v21_transductive.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"Saved V21 to {out_path}")

if __name__ == '__main__':
    main()
