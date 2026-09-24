import os
import csv
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / len(scores)

def main():
    print("="*60)
    print("V5: PSEUDO-LABELING (SEMI-SUPERVISED SELF-TRAINING)")
    print("="*60)
    
    # 1. Load the best predictions so far (V4 scored 0.684)
    v4_path = os.path.join(SUBMISSIONS_DIR, 'submission_v4.csv')
    if not os.path.exists(v4_path):
        print("Error: Could not find V4 submission to use for pseudo-labeling.")
        return
        
    test_ids = []
    v4_scores = []
    with open(v4_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            v4_scores.append(float(row['label']))
            
    v4_scores = np.array(v4_scores)
    
    # 2. Select the most confident predictions to become pseudo-labels
    # Top 10% as Crash (1), Bottom 10% as Normal (0)
    n_pseudo = int(len(test_ids) * 0.15) # 15% from each side
    
    sorted_indices = np.argsort(v4_scores)
    normal_idx = sorted_indices[:n_pseudo]  # lowest scores
    crash_idx = sorted_indices[-n_pseudo:]  # highest scores
    
    pseudo_idx = np.concatenate([normal_idx, crash_idx])
    pseudo_labels = np.concatenate([np.zeros(n_pseudo), np.ones(n_pseudo)])
    
    print(f"Extracted {n_pseudo} confident Normal and {n_pseudo} confident Crash clips from Test Set.")
    
    # 3. Load all cached features for the Test Set
    # We will train our model ONLY on the pseudo-labeled test clips!
    # This completely ignores the poisoned Kaggle training set.
    print("Loading test set features...")
    cnn_data = np.load(os.path.join(CACHE_DIR, 'cache_cnn_features.npz'))
    X_test_cnn = cnn_data['X_test']
    
    clip_data = np.load(os.path.join(CACHE_DIR, 'cache_clip_features.npz'))
    X_test_clip = clip_data['X_test']
    
    motion_data = np.load(os.path.join(CACHE_DIR, 'cache_motion_features.npz'), allow_pickle=True)
    X_test_motion = motion_data['X_test']
    
    # Combine all features for the test set
    X_test_all = np.hstack([X_test_cnn, X_test_clip, X_test_motion])
    
    # Extract the training subset (the pseudo-labeled clips)
    X_pseudo_train = X_test_all[pseudo_idx]
    y_pseudo_train = pseudo_labels
    
    print(f"Training LightGBM on {len(y_pseudo_train)} pseudo-labeled test clips (Feature dim: {X_pseudo_train.shape[1]})...")
    
    # 4. Train a LightGBM model with Cross-Validation
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    test_preds = np.zeros(len(test_ids))
    oof_preds = np.zeros(len(X_pseudo_train))
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': 4,
        'feature_fraction': 0.3,
        'verbosity': -1,
        'random_state': 42
    }
    
    for fold, (trn_idx, val_idx) in enumerate(folds.split(X_pseudo_train, y_pseudo_train)):
        X_trn, y_trn = X_pseudo_train[trn_idx], y_pseudo_train[trn_idx]
        X_val, y_val = X_pseudo_train[val_idx], y_pseudo_train[val_idx]
        
        dtrain = lgb.Dataset(X_trn, label=y_trn)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
        
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=1000,
            valid_sets=[dtrain, dval],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        # Predict on validation set
        oof_preds[val_idx] = model.predict(X_val)
        
        # Predict on entire test set
        test_preds += model.predict(X_test_all) / folds.n_splits
        
    cv_auc = roc_auc_score(y_pseudo_train, oof_preds)
    print(f"Pseudo-label CV AUC: {cv_auc:.4f}")
    
    # 5. Final Ensemble
    # We mix the new Self-Trained predictions with our V4 baseline
    final_combined = (
        rank_normalize(test_preds) * 1.5 + 
        rank_normalize(v4_scores) * 1.0
    )
    
    final_combined = rank_normalize(final_combined)
    
    # Write submission
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v5_pseudo.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_combined):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved to {out_path}")

if __name__ == '__main__':
    main()
