"""
V27: DEEP PHYSICS-VISION FUSION
=================================
The KEY insight: V16's advanced multi-object tracking physics 
(TTC, spin, deceleration) has only 0.21 correlation with V24!
This means it sees COMPLETELY DIFFERENT crash signals.

All our V20-V25b models are 0.95+ correlated with each other — they 
are basically copies. But V16 is genuinely orthogonal.

By fusing V16's physics with our best vision model (V24), the errors 
that V24 makes on certain clips will be CORRECTED by V16's physics, 
and vice versa. This is the mathematical principle behind why 
diverse ensembles always beat single models.
"""

import os
import csv
import json
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
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

def load_submission_scores(filename, test_ids):
    path = os.path.join(SUBMISSIONS_DIR, filename)
    if not os.path.exists(path):
        return None
    scores = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row['label']))
    if len(scores) != len(test_ids):
        return None
    return np.array(scores)

def main():
    print("="*60)
    print("V27: DEEP PHYSICS-VISION FUSION")
    print("="*60)
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
    
    # ============================================================
    # FEATURE BLOCK 1: All Vision Foundation Model Features
    # ============================================================
    all_X = []
    all_names = []
    
    # V19 cache (SigLIP-378, EVA02, CLIP with velocity features)
    for cache_key in ['siglip_378', 'eva02_velocity', 'clip_l14_velocity']:
        X, names = load_features(os.path.join(CACHE_DIR, 'v19_cache'), cache_key, test_ids)
        if X is not None:
            all_X.append(X); all_names.extend(names)
    
    # V18 cache (EVA02, CLIP text-similarity features)
    for cache_key in ['eva02', 'clip_l14']:
        X, names = load_features(os.path.join(CACHE_DIR, 'v18_cache'), cache_key, test_ids)
        if X is not None:
            all_X.append(X); all_names.extend(names)
    
    # V25 Overnight cache (SigLIP2-378 domain, ViT-H-14 domain)
    for cache_key in ['siglip2_378_domain', 'vit_h14_dfn5b_domain']:
        X, names = load_features(os.path.join(CACHE_DIR, 'v25_cache'), cache_key, test_ids)
        if X is not None:
            all_X.append(X); all_names.extend(names)
            print(f"  Loaded {cache_key}: {X.shape[1]} features")
    
    # ============================================================
    # FEATURE BLOCK 2: Physics Features
    # ============================================================
    
    # Basic optical flow
    flow_path = os.path.join(CACHE_DIR, 'cache_masked_flow.npz')
    if os.path.exists(flow_path):
        flow_data = np.load(flow_path)['X']
        if len(flow_data) == len(test_ids):
            all_X.append(flow_data)
            all_names.extend(['flow_max_vel', 'flow_vel_var', 'flow_ang_var'])
    
    # Basic YOLO
    yolo_path = os.path.join(CACHE_DIR, 'cache_yolo_features.npz')
    if os.path.exists(yolo_path):
        yolo_data = np.load(yolo_path)
        yolo_feats = yolo_data['X_test'] if 'X_test' in yolo_data else yolo_data[list(yolo_data.keys())[0]]
        if len(yolo_feats) == len(test_ids):
            all_X.append(yolo_feats)
            all_names.extend([f'yolo_feat_{i}' for i in range(yolo_feats.shape[1])])
    
    # ============================================================
    # FEATURE BLOCK 3: Meta-Scores (Including V16 Advanced Physics!)
    # ============================================================
    meta_models = {
        # Advanced Physics (THE KEY NEW SIGNAL — 0.21 correlation with V24!)
        'meta_v16_advanced_physics': 'submission_v16_advanced_physics.csv',
        'meta_v17_physics_grandmaster': 'submission_v17_ultimate_physics.csv',
        # Foundation Models (proven performers)
        'meta_v7_yolo_physics': 'submission_v7_yolo.csv',
        'meta_v8_video': 'submission_v8_video.csv',
        'meta_v10_distillation': 'submission_v10_distillation.csv',
        'meta_v15_grandmaster': 'submission_v15_grandmaster.csv',
        'meta_v18_ultimate': 'submission_v18_ultimate.csv',
        'meta_v20_highres': 'submission_v20_highres.csv',
        'meta_v25_standalone': 'submission_v25_standalone.csv',
    }
    
    for name, filename in meta_models.items():
        scores = load_submission_scores(filename, test_ids)
        if scores is not None:
            all_X.append(rank_normalize(scores).reshape(-1, 1))
            all_names.append(name)
            print(f"  Added meta: {name}")
    
    X_all = np.hstack(all_X)
    print(f"\nTotal Feature Matrix: {X_all.shape[1]} features")
    
    # ============================================================
    # PSEUDO-LABELS: Consensus from V24 + V21 (our two best)
    # ============================================================
    v24_ranks = rank_normalize(load_submission_scores('submission_v24_feature_blender.csv', test_ids))
    v21_ranks = rank_normalize(load_submission_scores('submission_v21_transductive.csv', test_ids))
    
    avg_ranks = (v24_ranks + v21_ranks) / 2.0
    sorted_idx = np.argsort(avg_ranks)
    
    normal_idx = sorted_idx[:700]
    crash_idx = sorted_idx[-150:]
    train_idx = np.concatenate([normal_idx, crash_idx])
    
    X_train = X_all[train_idx]
    y_train = np.concatenate([np.zeros(len(normal_idx)), np.ones(len(crash_idx))])
    
    print(f"Pseudo-Labels: {len(normal_idx)} Normal + {len(crash_idx)} Crash")
    
    # ============================================================
    # TRAINING: Repeated Stratified K-Fold for Maximum Robustness
    # ============================================================
    print("\nTraining Physics-Vision Fusion LightGBM (3×5-fold)...")
    
    params = {
        'objective': 'binary', 'metric': 'auc', 'n_estimators': 400,
        'learning_rate': 0.015, 'max_depth': 4, 'num_leaves': 15,
        'min_child_samples': 8, 'subsample': 0.7, 'colsample_bytree': 0.5,
        'reg_alpha': 1.5, 'reg_lambda': 3.0, 'class_weight': 'balanced',
        'random_state': 42, 'verbose': -1,
    }
    
    # Use 3 repeats × 5 folds = 15 models for ultra-stable predictions
    rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
    test_preds = np.zeros(len(test_ids))
    n_folds = 0
    
    for tr_idx, val_idx in rkf.split(X_train, y_train):
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train[tr_idx], y_train[tr_idx])
        test_preds += clf.predict_proba(X_all)[:, 1]
        n_folds += 1
    
    test_preds /= n_folds
    print(f"  Averaged predictions across {n_folds} fold models")
    
    # ============================================================
    # FINAL BLEND: LGB fusion + V24 safety net
    # ============================================================
    lgb_ranks = rank_normalize(test_preds)
    
    # 60% new fusion model, 40% proven V24
    final = lgb_ranks * 0.60 + v24_ranks * 0.40
    final_ranks = rank_normalize(final)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v27_physics_vision_fusion.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
    
    print(f"\nSaved V27 to {out_path}")
    
    # Print top features
    importances = clf.feature_importances_
    sorted_imp = np.argsort(importances)[::-1]
    print("\nTop 15 Most Important Features:")
    for i in sorted_imp[:15]:
        if i < len(all_names):
            print(f"  {all_names[i]}: {importances[i]}")

if __name__ == '__main__':
    main()
