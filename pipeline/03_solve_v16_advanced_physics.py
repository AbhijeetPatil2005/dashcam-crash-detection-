"""
Module: 03_solve_v16_advanced_physics.py
Description: Pure physics engine that scores clips based on tracking and kinematics without semantic vision.
"""
import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def process_clip_physics(clip_dir, model):
    """
    Advanced Physics Engine using Multi-Object Tracking & Time-to-Collision.
    """
    frames = []
    for i in range(30):
        path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
        if not os.path.exists(path): return 0.0
        frames.append(cv2.imread(path))
        
    if len(frames) < 30: return 0.0
    
    # 1. Multi-Object Tracking across the 30 frames
    tracks = {}
    
    # We only care about vehicles
    for i, frame in enumerate(frames):
        # use ByteTrack (fast, robust)
        results = model.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False, tracker="bytetrack.yaml")
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, obj_id in zip(boxes, ids):
                if obj_id not in tracks:
                    tracks[obj_id] = []
                x, y, w, h = box
                tracks[obj_id].append({'frame': i, 'x': x, 'y': y, 'area': w*h, 'aspect': w/max(h, 1)})
                
    max_danger_score = 0.0
    
    # 2. Physics Analysis per Vehicle
    for obj_id, track in tracks.items():
        if len(track) < 5: continue # Ignore fleeting detections
            
        areas = [t['area'] for t in track]
        aspects = [t['aspect'] for t in track]
        xs = [t['x'] for t in track]
        ys = [t['y'] for t in track]
        
        # Calculate rates of change
        area_diffs = np.diff(areas)
        aspect_diffs = np.diff(aspects)
        
        # A. Time-to-Collision (TTC) Danger
        # If area is expanding extremely rapidly, it's a collision course
        # TTC ~ Area / (dArea/dt). Smaller TTC = Higher Danger.
        max_expansion = np.max(area_diffs) if len(area_diffs) > 0 else 0
        mean_area = np.mean(areas)
        ttc_danger = (max_expansion / max(mean_area, 1)) * 100 
        
        # B. Spin-out / T-Bone Danger
        # If the aspect ratio changes violently, the car is spinning or flipping
        max_spin = np.max(np.abs(aspect_diffs)) if len(aspect_diffs) > 0 else 0
        
        # C. Sudden Stop (Extreme Deceleration)
        # Calculate velocity in 2D space
        vx = np.diff(xs)
        vy = np.diff(ys)
        velocities = np.sqrt(vx**2 + vy**2)
        accelerations = np.diff(velocities)
        max_decel = abs(np.min(accelerations)) if len(accelerations) > 0 and np.min(accelerations) < 0 else 0
        
        # Combine danger signals for this vehicle
        vehicle_danger = (ttc_danger * 2.0) + (max_spin * 50.0) + (max_decel * 0.5)
        
        if vehicle_danger > max_danger_score:
            max_danger_score = vehicle_danger
            
    # 3. Global Camera Shake (Ego-Vehicle Crash)
    # If the dashcam vehicle hits something (e.g. a wall), the whole camera shakes violently.
    # We measure this using global optical flow variance in the last 10 frames
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames[-10:]]
    shake_scores = []
    for i in range(len(gray_frames)-1):
        flow = cv2.calcOpticalFlowFarneback(gray_frames[i], gray_frames[i+1], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        shake_scores.append(np.var(mag))
        
    ego_danger = np.max(shake_scores) if shake_scores else 0
    
    # Final combined physics score
    final_score = max_danger_score + (ego_danger * 0.1)
    return float(final_score)

def main():
    print("="*60)
    print("V16: ADVANCED MULTI-OBJECT TRACKING PHYSICS")
    print("="*60)
    
    print("Loading YOLOv8 Tracker...")
    model = YOLO("yolov8n.pt")
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    test_dir = os.path.join(DATA_DIR, 'test')
    
    raw_scores = {}
    print("Calculating advanced physics for Test Set...")
    for cid in tqdm(test_ids):
        raw_scores[cid] = process_clip_physics(os.path.join(test_dir, cid), model)
        
    # Rank normalize
    scores_list = [raw_scores[cid] for cid in test_ids]
    ranks = np.argsort(np.argsort(scores_list)).astype(float)
    ranks = ranks / (len(ranks) - 1)
    
    # Save standalone V16
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v16_advanced_physics.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, rank in zip(test_ids, ranks):
            writer.writerow([cid, f'{rank:.6f}'])
            
    print(f"\nSaved V16 to {out_path}")
    
    # Now create V17 (Grandmaster + V16)
    print("Fusing with V15 Grandmaster Ensemble...")
    v15_scores = []
    with open(os.path.join(SUBMISSIONS_DIR, 'submission_v15_grandmaster.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            v15_scores.append(float(row['label']))
            
    final_fusion = (np.array(v15_scores) * 2.0 + ranks * 1.0)
    final_ranks = np.argsort(np.argsort(final_fusion)).astype(float)
    final_ranks = final_ranks / (len(final_ranks) - 1)
    
    fusion_path = os.path.join(SUBMISSIONS_DIR, 'submission_v17_ultimate_physics.csv')
    with open(fusion_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, rank in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{rank:.6f}'])
            
    print(f"Saved V17 to {fusion_path}")

if __name__ == '__main__':
    main()
