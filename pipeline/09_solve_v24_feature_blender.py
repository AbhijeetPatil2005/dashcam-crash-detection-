"""
V24: SUPERVISED FEATURE BLENDER
================================
The KEY gap in V20-V23: We have 16 features per clip per backbone 
(text similarity + velocity) across 3 backbones = 48 rich features total.
But we only used HAND-TUNED linear combinations.

In V24, we let LightGBM find the OPTIMAL non-linear combination 
of all 48 features. The critical insight: we use V21's most confident 
predictions (which scored 0.788!) as pseudo-labels, then train a 
tree-based model to learn complex interactions between features 
(e.g., "if SigLIP spike is high AND EVA velocity is high, DEFINITELY crash").

Trees can capture non-linear interactions that our weighted-sum approach 
completely misses!
"""

import os
import csv
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / max(1, len(scores) - 1)

def load_features(cache_dir, cache_key, test_ids):
    """Load cached features and build a feature matrix."""
    path = os.path.join(cache_dir, f'{cache_key}_features.json')
    with open(path, 'r') as f:
        feats = json.load(f)
    
    feature_names = list(feats[test_ids[0]].keys())
    X = np.zeros((len(test_ids), len(feature_names)))
    for i, cid in enumerate(test_ids):
        for j, fname in enumerate(feature_names):
            X[i, j] = feats[cid].get(fname, 0.0)
    return X, [f"{cache_key}_{fn}" for fn in feature_names]

def main():
    print("="*60)
    print("V24: SUPERVISED FEATURE BLENDER")
    print("="*60)
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
    
    # ============================================================
    # 1. Load ALL cached features from V19 (3 backbones × 16 features = 48 features)
    # ============================================================
    print("Loading cached features...")
    
    all_X = []
    all_names = []
    
    # V19 cache: SigLIP-378, EVA02, CLIP (each with 16 features)
    for cache_key in ['siglip_378', 'eva02_velocity', 'clip_l14_velocity']:
        X, names = load_features(os.path.join(CACHE_DIR, 'v19_cache'), cache_key, test_ids)
        all_X.append(X)
        all_names.extend(names)
        print(f"  {cache_key}: {X.shape[1]} features")
    
    # Also load V18 cache: EVA02, CLIP (with 8 text-only features each)
    for cache_key in ['eva02', 'clip_l14']:
        path = os.path.join(CACHE_DIR, 'v18_cache', f'{cache_key}_features.json')
        if os.path.exists(path):
            X, names = load_features(os.path.join(CACHE_DIR, 'v18_cache'), cache_key, test_ids)
            all_X.append(X)
            all_names.extend(names)
            print(f"  v18_{cache_key}: {X.shape[1]} features")
    
    # Add cached physics features
    flow_path = os.path.join(CACHE_DIR, 'cache_masked_flow.npz')
    if os.path.exists(flow_path):
        flow_data = np.load(flow_path)['X']
        if len(flow_data) == len(test_ids):
            all_X.append(flow_data)
            all_names.extend(['flow_max_vel', 'flow_vel_var', 'flow_ang_var'])
            print(f"  optical_flow: {flow_data.shape[1]} features")
    
    yolo_path = os.path.join(CACHE_DIR, 'cache_yolo_features.npz')
    if os.path.exists(yolo_path):
        yolo_data = np.load(yolo_path)
        if 'X_test' in yolo_data:
            yolo_feats = yolo_data['X_test']
        else:
            yolo_feats = yolo_data[list(yolo_data.keys())[0]]
        if len(yolo_feats) == len(test_ids):
            all_X.append(yolo_feats)
            all_names.extend([f'yolo_feat_{i}' for i in range(yolo_feats.shape[1])])
            print(f"  yolo_physics: {yolo_feats.shape[1]} features")
    
    # Add previous submission scores as meta-features
    meta_submissions = [
        'submission_v7_yolo.csv', 'submission_v8_video.csv',
        'submission_v10_distillation.csv', 'submission_v15_grandmaster.csv',
        'submission_v18_ultimate.csv', 'submission_v20_highres.csv',
    ]
    for sub_file in meta_submissions:
        sub_path = os.path.join(SUBMISSIONS_DIR, sub_file)
        if os.path.exists(sub_path):
            scores = []
            with open(sub_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scores.append(float(row['label']))
            if len(scores) == len(test_ids):
                all_X.append(np.array(scores).reshape(-1, 1))
                all_names.append(f'meta_{sub_file}')
                print(f"  meta_{sub_file}")
    
    X_all = np.hstack(all_X)
    print(f"\nTotal Feature Matrix: {X_all.shape} ({X_all.shape[1]} features)")
    
    # ============================================================
    # 2. Create Pseudo-Labels from V21 (our best legal model: 0.788)
    # ============================================================
    v21_scores = []
    with open(os.path.join(SUBMISSIONS_DIR, 'submission_v21_transductive.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            v21_scores.append(float(row['label']))
    v21_scores = np.array(v21_scores)
    
    # Also load V20 for consensus
    v20_scores = []
    with open(os.path.join(SUBMISSIONS_DIR, 'submission_v20_highres.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            v20_scores.append(float(row['label']))
    v20_scores = np.array(v20_scores)
    
    # Use CONSENSUS: only label clips where V20 and V21 AGREE strongly
    v21_ranks = rank_normalize(v21_scores)
    v20_ranks = rank_normalize(v20_scores)
    avg_ranks = (v21_ranks + v20_ranks) / 2.0
    
    sorted_idx = np.argsort(avg_ranks)
    
    # Very conservative: Bottom 700 = Normal, Top 150 = Crash
    # These are clips where BOTH models are extremely confident
    normal_idx = sorted_idx[:700]
    crash_idx = sorted_idx[-150:]
    
    train_idx = np.concatenate([normal_idx, crash_idx])
    X_train = X_all[train_idx]
    y_train = np.concatenate([np.zeros(len(normal_idx)), np.ones(len(crash_idx))])
    
    print(f"\nPseudo-Label Set: {len(normal_idx)} Normal + {len(crash_idx)} Crash = {len(train_idx)} total")
    
    # ============================================================
    # 3. Train LightGBM with Cross-Validation
    # ============================================================
    print("\nTraining LightGBM Ensemble (5-fold CV)...")
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 300,
        'learning_rate': 0.02,
        'max_depth': 4,
        'num_leaves': 15,
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'reg_alpha': 1.0,
        'reg_lambda': 2.0,
        'class_weight': 'balanced',
        'random_state': 42,
        'verbose': -1,
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_idx))
    test_preds = np.zeros(len(test_ids))
    
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)])
        
        oof_preds[val_idx] = clf.predict_proba(X_val)[:, 1]
        test_preds += clf.predict_proba(X_all)[:, 1] / 5
        
        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        print(f"  Fold {fold+1} AUC: {fold_auc:.4f}")
    
    overall_auc = roc_auc_score(y_train, oof_preds)
    print(f"  Overall OOF AUC: {overall_auc:.4f}")
    
    # ============================================================
    # 4. Final Ensemble: LightGBM + V21 + V20
    # ============================================================
    print("\nBuilding Final Ensemble...")
    
    lgb_ranks = rank_normalize(test_preds)
    
    # The LightGBM model captures non-linear feature interactions.
    # But V21 is our proven champion. So we give V21 50%, LGB 35%, V20 15%.
    final = lgb_ranks * 0.35 + v21_ranks * 0.50 + v20_ranks * 0.15
    final_ranks = rank_normalize(final)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v24_feature_blender.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
    
    print(f"\nSaved V24 to {out_path}")
    
    # Print top important features
    print("\nTop 15 Most Important Features:")
    importances = clf.feature_importances_
    sorted_imp = np.argsort(importances)[::-1]
    for i in sorted_imp[:15]:
        if i < len(all_names):
            print(f"  {all_names[i]}: {importances[i]}")

if __name__ == '__main__':
    main()
