"""
V2 Solution for Cooked or Not — Focused on DOMAIN GENERALIZATION.

WHY V1 FAILED (0.53 AUC on LB):
- CV AUC was 0.995 but LB AUC was 0.53 — catastrophic domain shift
- Train no-crash = seconds BEFORE a crash (same video as crash clips)
- Test no-crash = ordinary driving (completely different videos)
- Model learned camera/scene-specific features, not crash detection

V2 STRATEGY — Three domain-invariant feature extractors + ensemble:

1. TEMPORAL MOTION FEATURES + LightGBM
   - Frame differences, optical flow stats, motion spike detection
   - These measure MOTION DYNAMICS, not appearance → domain-robust
   - Crashes universally produce sudden large motion changes

2. CLIP FEATURES + LightGBM  
   - CLIP encodes high-level semantic content ("car crash" vs "normal driving")
   - Pretrained on 400M image-text pairs → extremely domain-robust
   - Extract features from key frames, aggregate temporally

3. IMPROVED CNN (EfficientNet) with anti-overfitting
   - Freeze backbone completely, only train temporal head
   - Extreme augmentation + high dropout
   - Focus on temporal dynamics, not appearance
"""

import os
import sys
import csv
import json
import time
import gc
import numpy as np
import cv2
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast

import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR


def load_train_labels():
    csv_path = os.path.join(DATA_DIR, 'train_labels.csv')
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'clip_id': row['clip_id'],
                'label': int(row['label']),
                'group_id': row['group_id'],
            })
    return samples


def load_test_ids():
    csv_path = os.path.join(DATA_DIR, 'sample_submission.csv')
    clip_ids = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_ids.append(row['clip_id'])
    return clip_ids


def load_frames(clip_dir, indices=None, grayscale=False):
    """Load frames from a clip directory."""
    if indices is None:
        indices = range(30)
    frames = []
    for i in indices:
        path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
        if grayscale:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        else:
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if img is None:
            if grayscale:
                img = np.zeros((224, 384), dtype=np.uint8)
            else:
                img = np.zeros((224, 384, 3), dtype=np.uint8)
        frames.append(img)
    return frames


# ============================================================
# FEATURE EXTRACTOR 1: TEMPORAL MOTION FEATURES
# ============================================================

def extract_motion_features(clip_dir):
    """
    Extract rich temporal motion features from a clip.
    
    Crashes produce distinctive motion signatures:
    - Sudden spike in frame differences (impact)
    - Camera shake / blur after impact
    - Rapid deceleration (optical flow changes)
    
    These features are inherently domain-invariant because they 
    measure RELATIVE motion, not absolute appearance.
    """
    frames_gray = load_frames(clip_dir, grayscale=True)
    frames_color = load_frames(clip_dir, grayscale=False)
    
    features = {}
    
    # --- Frame Difference Features ---
    diffs = []
    diffs_center = []  # Center region (more important for crash detection)
    for i in range(1, 30):
        diff = cv2.absdiff(frames_gray[i], frames_gray[i-1]).astype(np.float32)
        diffs.append(diff.mean())
        
        # Center crop (middle 50% of frame — where crashes usually happen)
        h, w = diff.shape
        center = diff[h//4:3*h//4, w//4:3*w//4]
        diffs_center.append(center.mean())
    
    diffs = np.array(diffs)
    diffs_center = np.array(diffs_center)
    
    # Basic statistics
    features['diff_mean'] = diffs.mean()
    features['diff_std'] = diffs.std()
    features['diff_max'] = diffs.max()
    features['diff_min'] = diffs.min()
    features['diff_median'] = np.median(diffs)
    features['diff_range'] = diffs.max() - diffs.min()
    
    # Center region statistics
    features['diff_center_mean'] = diffs_center.mean()
    features['diff_center_std'] = diffs_center.std()
    features['diff_center_max'] = diffs_center.max()
    
    # Where does maximum change occur? (normalized position)
    features['diff_argmax'] = np.argmax(diffs) / len(diffs)
    features['diff_center_argmax'] = np.argmax(diffs_center) / len(diffs_center)
    
    # Spike detection — ratio of max to mean
    features['diff_spike_ratio'] = diffs.max() / (diffs.mean() + 1e-8)
    features['diff_center_spike_ratio'] = diffs_center.max() / (diffs_center.mean() + 1e-8)
    
    # Second half vs first half (crashes usually progress)
    half = len(diffs) // 2
    first_half = diffs[:half].mean()
    second_half = diffs[half:].mean()
    features['diff_half_ratio'] = second_half / (first_half + 1e-8)
    
    # Last quarter vs first quarter
    q1 = diffs[:len(diffs)//4].mean()
    q4 = diffs[3*len(diffs)//4:].mean()
    features['diff_quarter_ratio'] = q4 / (q1 + 1e-8)
    
    # Percentiles
    for p in [75, 90, 95, 99]:
        features[f'diff_p{p}'] = np.percentile(diffs, p)
    
    # Rate of change of differences (acceleration of motion)
    diff_accel = np.diff(diffs)
    features['diff_accel_mean'] = np.abs(diff_accel).mean()
    features['diff_accel_max'] = diff_accel.max()  # Max sudden increase
    features['diff_accel_min'] = diff_accel.min()  # Max sudden decrease
    features['diff_accel_std'] = diff_accel.std()
    features['diff_accel_argmax'] = np.argmax(diff_accel) / len(diff_accel)
    
    # Number of "spike" frames (diff > 2x mean)
    features['diff_num_spikes_2x'] = (diffs > 2 * diffs.mean()).sum()
    features['diff_num_spikes_3x'] = (diffs > 3 * diffs.mean()).sum()
    
    # Consecutive high-diff frames (sustained motion = crash aftermath)
    high_diff = diffs > np.percentile(diffs, 75)
    max_consecutive = 0
    current = 0
    for hd in high_diff:
        if hd:
            current += 1
            max_consecutive = max(max_consecutive, current)
        else:
            current = 0
    features['diff_max_consecutive_high'] = max_consecutive
    
    # --- Blur Detection (crashes cause motion blur) ---
    blur_scores = []
    for frame in frames_gray:
        laplacian_var = cv2.Laplacian(frame, cv2.CV_64F).var()
        blur_scores.append(laplacian_var)
    blur_scores = np.array(blur_scores)
    
    features['blur_mean'] = blur_scores.mean()
    features['blur_std'] = blur_scores.std()
    features['blur_min'] = blur_scores.min()
    features['blur_max'] = blur_scores.max()
    features['blur_range'] = blur_scores.max() - blur_scores.min()
    features['blur_argmin'] = np.argmin(blur_scores) / len(blur_scores)
    
    # Blur drop (sudden loss of sharpness = impact/camera shake)
    blur_changes = np.diff(blur_scores)
    features['blur_drop_max'] = (-blur_changes).max()  # Biggest drop in sharpness
    features['blur_drop_argmax'] = np.argmax(-blur_changes) / len(blur_changes)
    
    # --- Color Statistics (crash may change scene dramatically) ---
    color_means = []
    for frame in frames_color:
        color_means.append(frame.mean(axis=(0, 1)))
    color_means = np.array(color_means)  # (30, 3)
    
    # Color change over time
    color_diffs = np.abs(np.diff(color_means, axis=0)).mean(axis=1)  # (29,)
    features['color_diff_mean'] = color_diffs.mean()
    features['color_diff_max'] = color_diffs.max()
    features['color_diff_std'] = color_diffs.std()
    features['color_diff_spike'] = color_diffs.max() / (color_diffs.mean() + 1e-8)
    
    # --- Structural Similarity Changes ---
    # Use histogram comparison as a proxy for SSIM (much faster)
    hist_diffs = []
    for i in range(1, 30):
        hist1 = cv2.calcHist([frames_gray[i-1]], [0], None, [64], [0, 256])
        hist2 = cv2.calcHist([frames_gray[i]], [0], None, [64], [0, 256])
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        hist_diffs.append(1 - corr)  # Dissimilarity
    
    hist_diffs = np.array(hist_diffs)
    features['hist_diff_mean'] = hist_diffs.mean()
    features['hist_diff_max'] = hist_diffs.max()
    features['hist_diff_std'] = hist_diffs.std()
    features['hist_diff_spike'] = hist_diffs.max() / (hist_diffs.mean() + 1e-8)
    features['hist_diff_argmax'] = np.argmax(hist_diffs) / len(hist_diffs)
    
    # --- Edge Density Changes (structural damage detection) ---
    edge_densities = []
    for frame in frames_gray:
        edges = cv2.Canny(frame, 50, 150)
        edge_densities.append(edges.mean())
    edge_densities = np.array(edge_densities)
    
    edge_changes = np.abs(np.diff(edge_densities))
    features['edge_density_mean'] = edge_densities.mean()
    features['edge_density_std'] = edge_densities.std()
    features['edge_change_max'] = edge_changes.max()
    features['edge_change_mean'] = edge_changes.mean()
    
    # --- Optical Flow (dense flow magnitude) ---
    flow_magnitudes = []
    for i in range(1, min(30, len(frames_gray))):
        flow = cv2.calcOpticalFlowFarneback(
            frames_gray[i-1], frames_gray[i], 
            None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        flow_magnitudes.append(mag.mean())
    
    flow_mags = np.array(flow_magnitudes)
    features['flow_mean'] = flow_mags.mean()
    features['flow_std'] = flow_mags.std()
    features['flow_max'] = flow_mags.max()
    features['flow_min'] = flow_mags.min()
    features['flow_spike_ratio'] = flow_mags.max() / (flow_mags.mean() + 1e-8)
    features['flow_argmax'] = np.argmax(flow_mags) / len(flow_mags)
    
    flow_accel = np.diff(flow_mags)
    features['flow_accel_max'] = flow_accel.max()
    features['flow_accel_mean'] = np.abs(flow_accel).mean()
    
    # Second half vs first half
    features['flow_half_ratio'] = flow_mags[half:].mean() / (flow_mags[:half].mean() + 1e-8)
    
    # Percentiles
    for p in [75, 90, 95]:
        features[f'flow_p{p}'] = np.percentile(flow_mags, p)
    
    return features


def extract_all_motion_features(data_dir, clip_ids, desc="Extracting motion features"):
    """Extract motion features for all clips."""
    all_features = []
    for clip_id in tqdm(clip_ids, desc=desc):
        clip_dir = os.path.join(data_dir, clip_id)
        features = extract_motion_features(clip_dir)
        features['clip_id'] = clip_id
        all_features.append(features)
    return all_features


# ============================================================
# FEATURE EXTRACTOR 2: CLIP FEATURES
# ============================================================

def extract_clip_features(data_dir, clip_ids, desc="Extracting CLIP features"):
    """
    Extract CLIP visual features from key frames.
    
    CLIP is trained on 400M image-text pairs and encodes high-level 
    semantic content. It can distinguish "car crash scene" from 
    "normal driving" without being tied to specific cameras/datasets.
    """
    import open_clip
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Use ViT-B/32 — good balance of quality and speed
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-B-32', pretrained='laion2b_s34b_b79k'
    )
    model = model.to(device)
    model.eval()
    
    # Sample key frames: first, 1/4, middle, 3/4, last
    key_frame_indices = [0, 7, 14, 21, 29]
    
    all_features = {}
    
    for clip_id in tqdm(clip_ids, desc=desc):
        clip_dir = os.path.join(data_dir, clip_id)
        
        frame_features = []
        for idx in key_frame_indices:
            frame_path = os.path.join(clip_dir, f'frame_{idx:03d}.jpg')
            from PIL import Image
            try:
                img = Image.open(frame_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)
            except Exception:
                img_tensor = torch.zeros(1, 3, 224, 224).to(device)
            
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    feat = model.encode_image(img_tensor)
                    feat = F.normalize(feat, dim=-1)
            
            frame_features.append(feat.cpu().float().numpy().flatten())
        
        # Aggregate frame features
        frame_features = np.array(frame_features)  # (5, 512)
        
        # Statistics across frames
        clip_feat = np.concatenate([
            frame_features.mean(axis=0),   # Average semantic content
            frame_features.std(axis=0),    # Variability (crashes = high variability)
            frame_features[-1] - frame_features[0],  # Temporal change start→end
            frame_features.max(axis=0) - frame_features.min(axis=0),  # Range
        ])
        
        all_features[clip_id] = clip_feat
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return all_features


# ============================================================
# FEATURE EXTRACTOR 3: IMPROVED CNN FEATURES (frozen backbone)
# ============================================================

def extract_cnn_temporal_features(data_dir, clip_ids, desc="Extracting CNN features"):
    """
    Use frozen EfficientNet to extract frame features, then compute
    temporal statistics. By freezing the backbone, we prevent overfitting
    to training domain appearance.
    """
    import timm
    from torchvision import transforms
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool='avg')
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Sample 8 evenly spaced frames
    frame_indices = np.linspace(0, 29, 8, dtype=int).tolist()
    
    all_features = {}
    
    for clip_id in tqdm(clip_ids, desc=desc):
        clip_dir = os.path.join(data_dir, clip_id)
        frames = load_frames(clip_dir, indices=frame_indices, grayscale=False)
        
        frame_feats = []
        for frame in frames:
            img_tensor = transform(frame).unsqueeze(0).to(device)
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    feat = model(img_tensor)
            frame_feats.append(feat.cpu().float().numpy().flatten())
        
        frame_feats = np.array(frame_feats)  # (8, feat_dim)
        
        # Temporal statistics
        feat = np.concatenate([
            frame_feats.mean(axis=0),
            frame_feats.std(axis=0),
            frame_feats[-1] - frame_feats[0],  # Overall change
            np.abs(np.diff(frame_feats, axis=0)).mean(axis=0),  # Avg frame-to-frame change
            np.abs(np.diff(frame_feats, axis=0)).max(axis=0),   # Max frame-to-frame change
        ])
        
        all_features[clip_id] = feat
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return all_features


# ============================================================
# TRAINING WITH LIGHTGBM
# ============================================================

def train_lgbm_model(X_train, y_train, groups_train, X_test, feature_name="features"):
    """
    Train LightGBM with GroupKFold CV.
    LightGBM is much less prone to overfitting than deep learning
    when combined with proper cross-validation.
    """
    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    
    oof_preds = np.zeros(len(y_train))
    test_preds = np.zeros(len(X_test))
    fold_aucs = []
    
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 20,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'min_gain_to_split': 0.02,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups_train)):
        print(f"\n  Fold {fold}: train={len(train_idx)}, val={len(val_idx)}")
        
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        train_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            num_boost_round=2000,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(50),
                lgb.log_evaluation(100),
            ],
        )
        
        val_pred = model.predict(X_val)
        oof_preds[val_idx] = val_pred
        
        test_pred = model.predict(X_test)
        test_preds += test_pred / n_folds
        
        fold_auc = roc_auc_score(y_val, val_pred)
        fold_aucs.append(fold_auc)
        print(f"  Fold {fold} AUC: {fold_auc:.4f}")
    
    mean_auc = np.mean(fold_aucs)
    std_auc = np.std(fold_aucs)
    print(f"\n  {feature_name} CV AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    
    return oof_preds, test_preds, fold_aucs


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("COOKED OR NOT v2 — Domain-Robust Crash Detection")
    print("=" * 60)
    
    train_dir = os.path.join(DATA_DIR, 'train')
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # Load labels
    samples = load_train_labels()
    train_clip_ids = [s['clip_id'] for s in samples]
    train_labels = np.array([s['label'] for s in samples])
    train_groups = np.array([s['group_id'] for s in samples])
    
    test_clip_ids = load_test_ids()
    
    print(f"Train: {len(train_clip_ids)} clips, Test: {len(test_clip_ids)} clips")
    
    all_test_preds = []
    all_cv_aucs = []
    
    # =============================================
    # APPROACH 1: Motion Features + LightGBM
    # =============================================
    print("\n" + "=" * 60)
    print("APPROACH 1: Temporal Motion Features + LightGBM")
    print("=" * 60)
    
    motion_cache = os.path.join(CACHE_DIR, 'cache_motion_features.npz')
    
    if os.path.exists(motion_cache):
        print("Loading cached motion features...")
        data = np.load(motion_cache, allow_pickle=True)
        X_train_motion = data['X_train']
        X_test_motion = data['X_test']
        motion_feature_names = data['feature_names'].tolist()
    else:
        print("Extracting motion features (this takes ~15 min)...")
        
        train_features = extract_all_motion_features(train_dir, train_clip_ids, "Train motion features")
        test_features = extract_all_motion_features(test_dir, test_clip_ids, "Test motion features")
        
        # Convert to arrays
        motion_feature_names = [k for k in train_features[0].keys() if k != 'clip_id']
        X_train_motion = np.array([[f[k] for k in motion_feature_names] for f in train_features])
        X_test_motion = np.array([[f[k] for k in motion_feature_names] for f in test_features])
        
        # Handle NaN/inf
        X_train_motion = np.nan_to_num(X_train_motion, nan=0.0, posinf=100, neginf=-100)
        X_test_motion = np.nan_to_num(X_test_motion, nan=0.0, posinf=100, neginf=-100)
        
        np.savez(motion_cache, X_train=X_train_motion, X_test=X_test_motion,
                 feature_names=np.array(motion_feature_names))
        print(f"Cached motion features: {X_train_motion.shape}")
    
    print(f"Motion features shape: {X_train_motion.shape}")
    
    oof_motion, test_motion, aucs_motion = train_lgbm_model(
        X_train_motion, train_labels, train_groups, X_test_motion, "Motion"
    )
    all_test_preds.append(test_motion)
    all_cv_aucs.append(np.mean(aucs_motion))
    
    # =============================================
    # APPROACH 2: CLIP Features + LightGBM
    # =============================================
    print("\n" + "=" * 60)
    print("APPROACH 2: CLIP Features + LightGBM")
    print("=" * 60)
    
    clip_cache = os.path.join(CACHE_DIR, 'cache_clip_features.npz')
    
    if os.path.exists(clip_cache):
        print("Loading cached CLIP features...")
        data = np.load(clip_cache)
        X_train_clip = data['X_train']
        X_test_clip = data['X_test']
    else:
        print("Extracting CLIP features (this takes ~20 min)...")
        
        train_clip_feats = extract_clip_features(train_dir, train_clip_ids, "Train CLIP features")
        test_clip_feats = extract_clip_features(test_dir, test_clip_ids, "Test CLIP features")
        
        X_train_clip = np.array([train_clip_feats[cid] for cid in train_clip_ids])
        X_test_clip = np.array([test_clip_feats[cid] for cid in test_clip_ids])
        
        X_train_clip = np.nan_to_num(X_train_clip, nan=0.0, posinf=100, neginf=-100)
        X_test_clip = np.nan_to_num(X_test_clip, nan=0.0, posinf=100, neginf=-100)
        
        np.savez(clip_cache, X_train=X_train_clip, X_test=X_test_clip)
        print(f"Cached CLIP features: {X_train_clip.shape}")
    
    print(f"CLIP features shape: {X_train_clip.shape}")
    
    oof_clip, test_clip, aucs_clip = train_lgbm_model(
        X_train_clip, train_labels, train_groups, X_test_clip, "CLIP"
    )
    all_test_preds.append(test_clip)
    all_cv_aucs.append(np.mean(aucs_clip))
    
    # =============================================
    # APPROACH 3: CNN Features + LightGBM
    # =============================================
    print("\n" + "=" * 60)
    print("APPROACH 3: Frozen CNN Features + LightGBM")
    print("=" * 60)
    
    cnn_cache = os.path.join(CACHE_DIR, 'cache_cnn_features.npz')
    
    if os.path.exists(cnn_cache):
        print("Loading cached CNN features...")
        data = np.load(cnn_cache)
        X_train_cnn = data['X_train']
        X_test_cnn = data['X_test']
    else:
        print("Extracting CNN features (this takes ~15 min)...")
        
        train_cnn_feats = extract_cnn_temporal_features(train_dir, train_clip_ids, "Train CNN features")
        test_cnn_feats = extract_cnn_temporal_features(test_dir, test_clip_ids, "Test CNN features")
        
        X_train_cnn = np.array([train_cnn_feats[cid] for cid in train_clip_ids])
        X_test_cnn = np.array([test_cnn_feats[cid] for cid in test_clip_ids])
        
        X_train_cnn = np.nan_to_num(X_train_cnn, nan=0.0, posinf=100, neginf=-100)
        X_test_cnn = np.nan_to_num(X_test_cnn, nan=0.0, posinf=100, neginf=-100)
        
        np.savez(cnn_cache, X_train=X_train_cnn, X_test=X_test_cnn)
        print(f"Cached CNN features: {X_train_cnn.shape}")
    
    print(f"CNN features shape: {X_train_cnn.shape}")
    
    oof_cnn, test_cnn, aucs_cnn = train_lgbm_model(
        X_train_cnn, train_labels, train_groups, X_test_cnn, "CNN"
    )
    all_test_preds.append(test_cnn)
    all_cv_aucs.append(np.mean(aucs_cnn))
    
    # =============================================
    # APPROACH 4: Combined Features + LightGBM
    # =============================================
    print("\n" + "=" * 60)
    print("APPROACH 4: ALL Features Combined + LightGBM")
    print("=" * 60)
    
    X_train_all = np.hstack([X_train_motion, X_train_clip, X_train_cnn])
    X_test_all = np.hstack([X_test_motion, X_test_clip, X_test_cnn])
    
    print(f"Combined features shape: {X_train_all.shape}")
    
    oof_combined, test_combined, aucs_combined = train_lgbm_model(
        X_train_all, train_labels, train_groups, X_test_all, "Combined"
    )
    all_test_preds.append(test_combined)
    all_cv_aucs.append(np.mean(aucs_combined))
    
    # =============================================
    # FINAL ENSEMBLE (Rank Average)
    # =============================================
    print("\n" + "=" * 60)
    print("FINAL ENSEMBLE")
    print("=" * 60)
    
    # Weight by CV AUC performance
    weights = np.array(all_cv_aucs)
    weights = weights / weights.sum()
    
    print(f"Model weights: {dict(zip(['Motion', 'CLIP', 'CNN', 'Combined'], weights))}")
    
    # Rank-average ensemble (better for AUC than simple average)
    def rank_normalize(preds):
        return np.argsort(np.argsort(preds)).astype(float) / len(preds)
    
    final_preds = np.zeros(len(test_clip_ids))
    for pred, w in zip(all_test_preds, weights):
        final_preds += w * rank_normalize(pred)
    
    # Clip to [0, 1]
    final_preds = np.clip(final_preds, 0, 1)
    
    # Write submission
    submission_path = os.path.join(SUBMISSIONS_DIR, 'submission_v2.csv')
    with open(submission_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for clip_id, pred in zip(test_clip_ids, final_preds):
            writer.writerow([clip_id, f'{pred:.6f}'])
    
    print(f"\nSubmission saved to {submission_path}")
    print(f"Prediction stats: mean={final_preds.mean():.4f}, std={final_preds.std():.4f}")
    print(f"Min={final_preds.min():.4f}, Max={final_preds.max():.4f}")
    
    # Also save individual model submissions for comparison
    for name, preds in zip(['motion', 'clip_model', 'cnn', 'combined'], all_test_preds):
        sub_path = os.path.join(SUBMISSIONS_DIR, f'submission_v2_{name}.csv')
        with open(sub_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['clip_id', 'label'])
            for clip_id, pred in zip(test_clip_ids, preds):
                writer.writerow([clip_id, f'{pred:.6f}'])
    
    print("\nIndividual model submissions also saved!")
    print(f"\nTo submit: kaggle competitions submit -c cooked-or-not -f \"{submission_path}\" -m \"V2 ensemble\"")


if __name__ == '__main__':
    main()
