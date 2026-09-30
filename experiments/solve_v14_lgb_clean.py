import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def extract_masked_flow_features(clip_dir, model):
    frames = []
    for i in range(30):
        path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
        if not os.path.exists(path): return [0, 0, 0]
        frames.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        
    rgb_path = os.path.join(clip_dir, f'frame_25.jpg')
    rgb_frame = cv2.imread(rgb_path)
    if rgb_frame is None: return [0, 0, 0]
        
    results = model(rgb_frame, verbose=False, classes=[2, 3, 5, 7])
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    if len(boxes) == 0: return [0, 0, 0]
        
    car_flows = []
    for i in range(20, 29):
        flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        mask = np.zeros_like(mag)
        for x1, y1, x2, y2 in boxes:
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
            mask[y1:y2, x1:x2] = 1
            
        car_mag = mag[mask == 1]
        car_ang = ang[mask == 1]
        
        if len(car_mag) > 0:
            car_flows.append({
                'max_vel': np.max(car_mag),
                'vel_var': np.var(car_mag),
                'ang_var': np.var(car_ang)
            })
            
    if not car_flows: return [0, 0, 0]
    
    max_v = max(f['max_vel'] for f in car_flows)
    max_v_var = max(f['vel_var'] for f in car_flows)
    max_a_var = max(f['ang_var'] for f in car_flows)
    return [max_v, max_v_var, max_a_var]

def main():
    print("Loading YOLOv8...")
    model = YOLO("yolov8n.pt")
    
    train_labels = {}
    with open(os.path.join(DATA_DIR, 'train_labels_clean.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_labels[row['clip_id']] = int(row['label'])
            
    train_ids = list(train_labels.keys())
    train_dir = os.path.join(DATA_DIR, 'train')
    
    cache_path = os.path.join(CACHE_DIR, 'cache_masked_flow_train.npz')
    if os.path.exists(cache_path):
        print("Loading cached train features...")
        X_train = np.load(cache_path)['X']
    else:
        print("Extracting train features...")
        X_train = []
        for cid in tqdm(train_ids):
            X_train.append(extract_masked_flow_features(os.path.join(train_dir, cid), model))
        X_train = np.array(X_train)
        np.savez(cache_path, X=X_train)
        
    y_train = np.array([train_labels[cid] for cid in train_ids])
    
    # Load test features (already extracted in V9)
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    X_test = np.load(os.path.join(CACHE_DIR, 'cache_masked_flow.npz'))['X']
    
    print("\nTraining LightGBM on Masked Flow Physics...")
    clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    clf.fit(X_train, y_train)
    
    train_preds = clf.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, train_preds)
    print(f"Train AUC: {auc:.4f}")
    
    test_preds = clf.predict_proba(X_test)[:, 1]
    
    # Fuse with V7
    v7_scores = []
    with open(os.path.join(SUBMISSIONS_DIR, 'submission_v7_yolo.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            v7_scores.append(float(row['label']))
            
    def rank_normalize(scores):
        return np.argsort(np.argsort(scores)).astype(float) / len(scores)
        
    final_score = rank_normalize(v7_scores) * 2.0 + rank_normalize(test_preds) * 1.5
    final_score = rank_normalize(final_score)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v14_lgb_clean.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_score):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
