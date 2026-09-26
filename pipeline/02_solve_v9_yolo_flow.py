"""
Module: 02_solve_v9_yolo_flow.py
Description: Extracts dense optical flow variance features to detect ego-vehicle camera shake.
"""
import os
import cv2
import csv
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.ensemble import IsolationForest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def extract_masked_flow_features(clip_dir, model):
    """
    Key insight from Kaggle Winners: Isolate the optical flow strictly to the YOLO 
    bounding boxes. This removes all background ego-motion (trees, road moving) 
    and perfectly tracks the physical chaos of the vehicles themselves.
    """
    frames = []
    for i in range(30):
        path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
        frames.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        
    # Run YOLO on the middle and late frames (where crashes happen)
    # We only need the boxes, not tracking.
    rgb_path = os.path.join(clip_dir, f'frame_25.jpg')
    rgb_frame = cv2.imread(rgb_path)
    if rgb_frame is None:
        return [0, 0, 0]
        
    results = model(rgb_frame, verbose=False, classes=[2, 3, 5, 7]) # vehicles
    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
    
    if len(boxes) == 0:
        return [0, 0, 0]
        
    # Calculate optical flow over the last 10 frames (impact zone)
    # Average the flow, but ONLY inside the bounding boxes
    car_flows = []
    
    for i in range(20, 29):
        flow = cv2.calcOpticalFlowFarneback(frames[i], frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Create a mask for the vehicles
        mask = np.zeros_like(mag)
        for x1, y1, x2, y2 in boxes:
            # Ensure coordinates are within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
            mask[y1:y2, x1:x2] = 1
            
        # Extract flow only where cars are
        car_mag = mag[mask == 1]
        car_ang = ang[mask == 1]
        
        if len(car_mag) > 0:
            car_flows.append({
                'max_vel': np.max(car_mag),
                'mean_vel': np.mean(car_mag),
                'vel_var': np.var(car_mag),
                'ang_var': np.var(car_ang) # Chaotic direction = spinning/crash
            })
            
    if not car_flows:
        return [0, 0, 0]
        
    # Max velocity of any car
    max_v = max(f['max_vel'] for f in car_flows)
    
    # Velocity variance (sudden stop causes high spatial variance across the car body)
    max_v_var = max(f['vel_var'] for f in car_flows)
    
    # Angular variance (different parts of car moving in different directions = collision/crumpling)
    max_a_var = max(f['ang_var'] for f in car_flows)
    
    return [max_v, max_v_var, max_a_var]

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / len(scores)

def main():
    print("="*60)
    print("V9: MASKED OPTICAL FLOW KINEMATICS (Inspired by Top 3 Kaggle Solution)")
    print("="*60)
    
    print("Loading YOLOv8...")
    model = YOLO("yolov8n.pt")
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # Check if cached
    cache_path = os.path.join(CACHE_DIR, 'cache_masked_flow.npz')
    if os.path.exists(cache_path):
        print("Loading cached features...")
        X_feats = np.load(cache_path)['X']
    else:
        print("Extracting features (this will take a few minutes)...")
        X_feats = []
        for clip_id in tqdm(test_ids):
            feats = extract_masked_flow_features(os.path.join(test_dir, clip_id), model)
            X_feats.append(feats)
        X_feats = np.array(X_feats)
        np.savez(cache_path, X=X_feats)
        
    # Feature 1: The sheer magnitude of velocity inside bounding boxes
    score_vel = rank_normalize(X_feats[:, 0])
    
    # Feature 2: The physical chaos (variance of flow angles indicating spinning/crumpling)
    score_chaos = rank_normalize(X_feats[:, 2])
    
    # Pure Physics Score
    physics_score = score_vel * 0.5 + score_chaos * 0.5
    physics_score = rank_normalize(physics_score)
    
    # Blend with V7 baseline to create V9
    # (Since V7 contains our best CLIP + YOLO Tracking)
    v7_path = os.path.join(SUBMISSIONS_DIR, 'submission_v7_yolo.csv')
    if os.path.exists(v7_path):
        v7_scores = []
        with open(v7_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                v7_scores.append(float(row['label']))
        v7_scores = np.array(v7_scores)
        
        print("Fusing V9 Physics with V7 Semantics...")
        final_score = rank_normalize(v7_scores) * 2.0 + physics_score * 1.0
        final_score = rank_normalize(final_score)
    else:
        final_score = physics_score
        
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v9_masked_flow.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_score):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved to {out_path}")
    print("Ready to submit!")

if __name__ == '__main__':
    main()
