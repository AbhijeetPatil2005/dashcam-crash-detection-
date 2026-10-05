"""
V25b: THE ULTIMATE SUPERVISED FEATURE BLENDER
==============================================
We combine ALL features from V24 (48 foundation + physics features) 
WITH the brand new overnight features from V25:
  - SigLIP2-378 (domain-adapted prompts)
  - ViT-H-14 DFN5B (domain-adapted prompts)

We then feed all ~100 features into a 5-fold Cross-Validated 
LightGBM using V20+V21 Consensus Pseudo-labels.

This will be our final and most powerful submission.
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
    path = os.path.join(cache_dir, f'{cache_key}_features.json')
    if not os.path.exists(path):
        return None, None
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
    print("V25b: ULTIMATE SUPERVISED BLENDER")
    print("="*60)
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    all_X = []
    all_names = []
    
    # 1. Load V19 cache
    for cache_key in ['siglip_378', 'eva02_velocity', 'clip_l14_velocity']:
        X, names = load_features(os.path.join(CACHE_DIR, 'v19_cache'), cache_key, test_ids)
        if X is not None:
            all_X.append(X)
            all_names.extend(names)
            
    # 2. Load V18 cache
    for cache_key in ['eva02', 'clip_l14']:
        X, names = load_features(os.path.join(CACHE_DIR, 'v18_cache'), cache_key, test_ids)
        if X is not None:
            all_X.append(X)
            all_names.extend(names)
            
    # 3. Load NEW V25 Overnight cache
    for cache_key in ['siglip2_378_domain', 'vit_h14_dfn5b_domain']:
        X, names = load_features(os.path.join(CACHE_DIR, 'v25_cache'), cache_key, test_ids)
        if X is not None:
            all_X.append(X)
            all_names.extend(names)
            print(f"Loaded Overnight Features: {cache_key} ({X.shape[1]} feats)")
            
    # 4. Load Physics
    flow_path = os.path.join(CACHE_DIR, 'cache_masked_flow.npz')
    if os.path.exists(flow_path):
        flow_data = np.load(flow_path)['X']
        all_X.append(flow_data)
        all_names.extend(['flow_max_vel', 'flow_vel_var', 'flow_ang_var'])
        
    yolo_path = os.path.join(CACHE_DIR, 'cache_yolo_features.npz')
    if os.path.exists(yolo_path):
        yolo_data = np.load(yolo_path)
        yolo_feats = yolo_data['X_test'] if 'X_test' in yolo_data else yolo_data[list(yolo_data.keys())[0]]
        all_X.append(yolo_feats)
        all_names.extend([f'yolo_feat_{i}' for i in range(yolo_feats.shape[1])])
        
    X_all = np.hstack(all_X)
    print(f"\nMassive Feature Matrix Created: {X_all.shape[1]} Features")
    
    # 5. Pseudo Labels
    v24_scores = []
    with open(os.path.join(SUBMISSIONS_DIR, 'submission_v24_feature_blender.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: v24_scores.append(float(row['label']))
    v24_ranks = rank_normalize(v24_scores)
    
    v21_scores = []
    with open(os.path.join(SUBMISSIONS_DIR, 'submission_v21_transductive.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader: v21_scores.append(float(row['label']))
    v21_ranks = rank_normalize(v21_scores)
    
    avg_ranks = (v24_ranks + v21_ranks) / 2.0
    sorted_idx = np.argsort(avg_ranks)
    
    normal_idx = sorted_idx[:700]
    crash_idx = sorted_idx[-150:]
    train_idx = np.concatenate([normal_idx, crash_idx])
    
    X_train = X_all[train_idx]
    y_train = np.concatenate([np.zeros(len(normal_idx)), np.ones(len(crash_idx))])
    
    # 6. Train LightGBM
    print("\nTraining Ultimate LightGBM Blender...")
    params = {
        'objective': 'binary', 'metric': 'auc', 'n_estimators': 400,
        'learning_rate': 0.015, 'max_depth': 4, 'num_leaves': 15,
        'min_child_samples': 8, 'subsample': 0.7, 'colsample_bytree': 0.6,
        'reg_alpha': 1.0, 'reg_lambda': 3.0, 'class_weight': 'balanced',
        'random_state': 42, 'verbose': -1,
    }
    
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros(len(test_ids))
    
    for tr_idx, val_idx in kf.split(X_train, y_train):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train[tr_idx], y_train[tr_idx], eval_set=[(X_train[val_idx], y_train[val_idx])])
        test_preds += clf.predict_proba(X_all)[:, 1] / 5
        
    final_ranks = rank_normalize(test_preds)
    
    # Blend with V24 for ultimate stability
    ultimate_blend = rank_normalize(final_ranks * 0.7 + v24_ranks * 0.3)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v25b_final_blend.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, ultimate_blend):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved Ultimate Submission to {out_path}")
    print("\nREADY TO SUBMIT ON KAGGLE!")

if __name__ == '__main__':
    main()
