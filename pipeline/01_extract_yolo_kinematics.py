"""
Module: 01_extract_yolo_kinematics.py
Description: Extracts Time-to-Collision (TTC) and kinematic features using YOLOv8 tracking.
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm
import csv
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def compute_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def extract_kinematics(clip_dir, model):
    frames = []
    for i in range(30):
        frame_path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
        frames.append(frame_path)
        
    # Run YOLOv8 tracking
    results = model.track(source=frames, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[2, 3, 5, 7]) # cars, motorcycles, buses, trucks
    
    # Track histories: {id: [(frame_idx, cx, cy, w, h, box)]}
    tracks = {}
    
    for frame_idx, r in enumerate(results):
        if r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.int().cpu().numpy()
            
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                
                if track_id not in tracks:
                    tracks[track_id] = []
                tracks[track_id].append((frame_idx, cx, cy, w, h, box))
                
    if len(tracks) == 0:
        return [0, 0, 0, 0] # max_decel, max_area_change, max_iou, obj_count
        
    # Compute kinematics
    max_decel = 0
    max_area_change = 0
    
    for track_id, history in tracks.items():
        if len(history) < 3:
            continue
            
        velocities = []
        areas = []
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            # Distance moved
            dx = curr[1] - prev[1]
            dy = curr[2] - prev[2]
            v = np.sqrt(dx**2 + dy**2)
            
            # Normalize by width to account for perspective (objects far away move less pixels)
            norm_v = v / max(curr[3], 1)
            velocities.append(norm_v)
            
            areas.append(curr[3] * curr[4])
            
        if len(velocities) > 1:
            # Acceleration = change in velocity
            accels = np.diff(velocities)
            # Deceleration is negative acceleration. We want the max absolute deceleration
            decel = -np.min(accels) if np.min(accels) < 0 else 0
            max_decel = max(max_decel, decel)
            
        if len(areas) > 1:
            # Sudden increase in area = rapid approach
            area_changes = np.diff(areas) / np.array(areas[:-1])
            max_area_change = max(max_area_change, np.max(area_changes))
            
    # Compute max bounding box IoU (overlapping cars)
    max_iou = 0
    for frame_idx, r in enumerate(results):
        boxes = r.boxes.xyxy.cpu().numpy()
        if len(boxes) > 1:
            for i in range(len(boxes)):
                for j in range(i+1, len(boxes)):
                    iou = compute_iou(boxes[i], boxes[j])
                    max_iou = max(max_iou, iou)
                    
    return [max_decel, max_area_change, max_iou, len(tracks)]

def main():
    print("Loading YOLOv8 model for tracking...")
    model = YOLO("yolov8n.pt")
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    test_dir = os.path.join(DATA_DIR, 'test')
    
    X_yolo = []
    print("Extracting Kinematics using YOLOv8+ByteTrack...")
    for clip_id in tqdm(test_ids):
        clip_dir = os.path.join(test_dir, clip_id)
        feats = extract_kinematics(clip_dir, model)
        X_yolo.append(feats)
        
    X_yolo = np.array(X_yolo)
    
    feature_names = ['max_decel', 'max_area_change', 'max_iou', 'obj_count']
    
    out_path = os.path.join(CACHE_DIR, 'cache_yolo_features.npz')
    np.savez(out_path, X_test=X_yolo, feature_names=feature_names)
    print(f"\nSaved YOLO kinematics to {out_path}")
    
    print("\nStats:")
    for i, name in enumerate(feature_names):
        print(f"  {name}: Mean={X_yolo[:, i].mean():.4f}, Max={X_yolo[:, i].max():.4f}")

if __name__ == '__main__':
    main()
