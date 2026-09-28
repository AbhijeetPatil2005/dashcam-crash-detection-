"""
STEP 2: Visual Review Tool for Training Set Disagreements
==========================================================
Shows you the clips where the AI strongly disagrees with the label.
These are the most likely mislabeled examples in the training data.

Controls:
  [Y] or [1]: This IS a crash (label = 1)
  [N] or [0]: This is NOT a crash (label = 0)  
  [K]: Keep the original label (skip without changing)
  [Q]: Quit and save

Run: python train_labeler.py
"""

import os
import csv
import cv2
import numpy as np
import time

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
SCORES_PATH = os.path.join(CACHE_DIR, 'train_clip_scores.csv')
CLEAN_LABELS_PATH = os.path.join(DATA_DIR, 'train_labels_clean.csv')

def load_scores():
    data = []
    with open(SCORES_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({
                'clip_id': row['clip_id'],
                'label': int(row['label']),
                'group_id': row['group_id'],
                'ai_score': float(row['ai_score'])
            })
    return data

def load_existing_clean_labels():
    """Load already-reviewed labels so we can resume."""
    reviewed = {}
    if os.path.exists(CLEAN_LABELS_PATH):
        with open(CLEAN_LABELS_PATH, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                reviewed[row['clip_id']] = {
                    'label': int(row['label']),
                    'group_id': row['group_id'],
                    'reviewed': row.get('reviewed', 'no')
                }
    return reviewed

def save_clean_labels(all_data, reviewed_labels):
    """Save the full label file (original + corrections)."""
    with open(CLEAN_LABELS_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label', 'group_id', 'reviewed'])
        for item in all_data:
            cid = item['clip_id']
            if cid in reviewed_labels:
                writer.writerow([cid, reviewed_labels[cid]['label'], 
                               item['group_id'], reviewed_labels[cid]['reviewed']])
            else:
                writer.writerow([cid, item['label'], item['group_id'], 'no'])

def play_clip(clip_id, original_label, ai_score, idx, total):
    """Play the clip in a loop until the user makes a choice."""
    clip_dir = os.path.join(TRAIN_DIR, clip_id)
    frames = []
    
    for i in range(30):
        path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
        if os.path.exists(path):
            img = cv2.imread(path)
            img = cv2.resize(img, (768, 448))
            frames.append(img)
    
    if not frames:
        return None
    
    label_text = "CRASH" if original_label == 1 else "NORMAL"
    label_color = (0, 0, 255) if original_label == 1 else (0, 255, 0)
    
    frame_idx = 0
    while True:
        display = frames[frame_idx].copy()
        
        # Header
        cv2.putText(display, f"[{idx}/{total}] {clip_id}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        # Original label
        cv2.putText(display, f"Original Label: {label_text}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2)
        
        # AI opinion
        ai_text = "AI thinks: CRASH" if ai_score > 0 else "AI thinks: NORMAL"
        ai_color = (0, 0, 255) if ai_score > 0 else (0, 255, 0)
        cv2.putText(display, f"{ai_text} ({ai_score:.3f})", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, ai_color, 2)
        
        # Disagreement warning
        ai_pred = 1 if ai_score > 0 else 0
        if ai_pred != original_label:
            cv2.putText(display, ">>> DISAGREEMENT - POSSIBLE MISLABEL <<<", (10, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Controls
        cv2.putText(display, "[Y/1]=CRASH  [N/0]=NORMAL  [K]=Keep  [Q]=Quit", (10, 430),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Frame counter
        cv2.putText(display, f"Frame {frame_idx+1}/30", (680, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Train Set Review", display)
        key = cv2.waitKey(50) & 0xFF
        
        if key in [ord('y'), ord('Y'), ord('1')]:
            return 1
        elif key in [ord('n'), ord('N'), ord('0')]:
            return 0
        elif key in [ord('k'), ord('K')]:
            return original_label  # keep original
        elif key in [ord('q'), ord('Q')]:
            return -1  # quit
        
        frame_idx = (frame_idx + 1) % len(frames)

def main():
    print("="*60)
    print("   TRAINING DATA CLEANER")
    print("   Fix mislabeled clips to train a better model")
    print("="*60)
    
    if not os.path.exists(SCORES_PATH):
        print(f"Error: {SCORES_PATH} not found!")
        print("Run 'python score_train_set.py' first to generate AI scores.")
        return
    
    # Load data
    all_data = load_scores()
    reviewed_labels = load_existing_clean_labels()
    
    already_reviewed = sum(1 for v in reviewed_labels.values() if v['reviewed'] == 'yes')
    print(f"Already reviewed: {already_reviewed} clips")
    
    # Find disagreements (sorted by severity)
    # Strongest disagreements first: 
    #   - Clips labeled CRASH but AI gives very low score
    #   - Clips labeled NORMAL but AI gives very high score
    disagreements = []
    for item in all_data:
        if item['clip_id'] in reviewed_labels and reviewed_labels[item['clip_id']]['reviewed'] == 'yes':
            continue
        
        ai_pred = 1 if item['ai_score'] > 0 else 0
        if ai_pred != item['label']:
            severity = abs(item['ai_score'])
            disagreements.append((severity, item))
    
    # Sort by severity (strongest disagreements first)
    disagreements.sort(key=lambda x: -x[0])
    
    print(f"Remaining disagreements to review: {len(disagreements)}")
    print(f"\nStarting review... (most suspicious clips first)")
    print("Click on the video window to give it focus!")
    time.sleep(2)
    
    reviewed_count = 0
    corrected_count = 0
    
    for i, (severity, item) in enumerate(disagreements):
        result = play_clip(item['clip_id'], item['label'], item['ai_score'], 
                          i+1, len(disagreements))
        
        if result == -1:  # quit
            print("Quitting and saving...")
            break
        
        reviewed_labels[item['clip_id']] = {
            'label': result,
            'group_id': item['group_id'],
            'reviewed': 'yes'
        }
        reviewed_count += 1
        
        if result != item['label']:
            corrected_count += 1
            action = "FLIPPED"
        else:
            action = "confirmed"
        
        old_text = "CRASH" if item['label'] == 1 else "NORMAL"
        new_text = "CRASH" if result == 1 else "NORMAL"
        print(f"  {item['clip_id']}: {old_text} -> {new_text} ({action})")
        
        # Auto-save every 10 reviews
        if reviewed_count % 10 == 0:
            save_clean_labels(all_data, reviewed_labels)
            print(f"  [Auto-saved] Reviewed: {reviewed_count}, Corrected: {corrected_count}")
    
    cv2.destroyAllWindows()
    
    # Final save
    save_clean_labels(all_data, reviewed_labels)
    
    print("\n" + "="*60)
    print(f"Session complete!")
    print(f"  Reviewed: {reviewed_count} clips")
    print(f"  Corrected: {corrected_count} labels")
    print(f"  Clean labels saved to: {CLEAN_LABELS_PATH}")
    print(f"\nWhen done reviewing, run: python solve_v13_clean_train.py")
    print("="*60)

if __name__ == '__main__':
    main()
