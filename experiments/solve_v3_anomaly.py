import os
import csv
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / len(scores)

def main():
    print("="*60)
    print("V3: UNSEEN DOMAIN ANOMALY DETECTION")
    print("="*60)
    
    # 1. Load Labels to find "Normal Driving" (Label 0)
    train_labels = []
    with open(os.path.join(DATA_DIR, 'train_labels.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_labels.append(int(row['label']))
    train_labels = np.array(train_labels)
    normal_idx = np.where(train_labels == 0)[0]
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    print(f"Found {len(normal_idx)} 'Normal Driving' clips in training set.")

    # 2. Load Features (CNN and CLIP)
    cnn_data = np.load(os.path.join(CACHE_DIR, 'cache_cnn_features.npz'))
    X_train_cnn = cnn_data['X_train']
    X_test_cnn = cnn_data['X_test']
    
    clip_data = np.load(os.path.join(CACHE_DIR, 'cache_clip_features.npz'))
    X_train_clip = clip_data['X_train']
    X_test_clip = clip_data['X_test']
    
    # Combine features
    X_train_all = np.hstack([X_train_cnn, X_train_clip])
    X_test_all = np.hstack([X_test_cnn, X_test_clip])
    
    # Isolate normal driving
    X_train_normal = X_train_all[normal_idx]
    
    # Scale and reduce dimensionality (Anomaly detection struggles in huge dimensions)
    scaler = StandardScaler()
    X_train_normal_scaled = scaler.fit_transform(X_train_normal)
    X_test_scaled = scaler.transform(X_test_all)
    
    pca = PCA(n_components=128)
    X_train_normal_pca = pca.fit_transform(X_train_normal_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    print("Training Anomaly Detectors on Normal Driving only...")
    # 3. Train Anomaly Detectors
    # Isolation Forest: Trees isolate anomalies faster (shorter paths)
    iso_forest = IsolationForest(n_estimators=300, contamination=0.1, random_state=42)
    iso_forest.fit(X_train_normal_pca)
    
    # Note: decision_function returns >0 for normal, <0 for anomalies. We want anomalies to be higher score (crash)
    score_iso = -iso_forest.decision_function(X_test_pca)
    
    # One-Class SVM: Finds a tight hypersphere around normal data
    ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    ocsvm.fit(X_train_normal_pca)
    score_svm = -ocsvm.decision_function(X_test_pca)
    
    anomaly_score = rank_normalize(score_iso) + rank_normalize(score_svm)
    
    # 4. Load Physics Zero-Shot Heuristics (we know these scored 0.64 alone!)
    print("Loading Zero-Shot Physics & CLIP scores...")
    motion_data = np.load(os.path.join(CACHE_DIR, 'cache_motion_features.npz'), allow_pickle=True)
    X_test_motion = motion_data['X_test']
    motion_feats = motion_data['feature_names'].tolist()
    
    score_diff = X_test_motion[:, motion_feats.index('diff_spike_ratio')]
    score_blur = X_test_motion[:, motion_feats.index('blur_drop_max')]
    score_flow = X_test_motion[:, motion_feats.index('flow_spike_ratio')]
    
    # We will also pull the final CLIP text scores we generated in our 0.677 submission
    # By reading the output file of solve_zero_shot_final.csv directly since we don't have the cache
    prev_final_scores = []
    try:
        with open(os.path.join(SUBMISSIONS_DIR, 'submission_zero_shot_final.csv'), 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prev_final_scores.append(float(row['label']))
    except FileNotFoundError:
        # Fallback if the file was deleted
        prev_final_scores = np.zeros(len(test_ids))
    prev_final_scores = np.array(prev_final_scores)

    # 5. Final Ultimate Ensemble
    # We mix the Anomaly Detection score with our 0.677 baseline
    final_combined = (
        rank_normalize(anomaly_score) * 1.5 +  # The new unsupervised anomaly detection
        rank_normalize(prev_final_scores) * 2.0  # The previous physics+clip zero-shot
    )
    
    final_combined = rank_normalize(final_combined)
    
    # Write submission
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v3_anomaly.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_combined):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved to {out_path}")
    print("Ready to submit tomorrow! Command:")
    print(f'kaggle competitions submit -c cooked-or-not -f "{out_path}" -m "V3 Anomaly+ZeroShot"')

if __name__ == '__main__':
    main()
