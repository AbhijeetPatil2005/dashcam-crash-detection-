"""
V22: THE APEX ENSEMBLE
======================
We have pushed every single dimension of this problem to the absolute limit:
1. V15: Spatial-Temporal Deep Learning (ResNet3D, Distillation)
2. V18: Multi-Model Zero-Shot Foundation Models (SigLIP, EVA02, CLIP)
3. V20: High-Resolution Vision (SigLIP-378px)
4. V21: Transductive Domain-Adapted Physics (LightGBM on Test Set)

By fusing all of these completely independent, fully legal approaches 
together, any remaining noise in individual models will cancel out, 
giving us the highest possible probability of breaking the 0.80 barrier.
"""

import os
import csv
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / max(1, len(scores) - 1)

def load_scores(filename):
    path = os.path.join(SUBMISSIONS_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: {filename} not found!")
        return None
    scores = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row['label']))
    return np.array(scores)

def main():
    print("="*60)
    print("V22: THE APEX ENSEMBLE")
    print("="*60)
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    # Load all our best legal submissions
    models = {
        'V21_Transductive': (load_scores('submission_v21_transductive.csv'), 3.0),
        'V20_HighRes': (load_scores('submission_v20_highres.csv'), 2.5),
        'V18_MultiFoundation': (load_scores('submission_v18_ultimate.csv'), 1.5),
        'V15_Grandmaster': (load_scores('submission_v15_grandmaster.csv'), 1.0),
        'V7_Physics': (load_scores('submission_v7_yolo.csv'), 0.5)
    }
    
    # Filter out missing models
    models = {k: v for k, v in models.items() if v[0] is not None}
    
    if not models:
        print("Error: No models found!")
        return
        
    final_ensemble = np.zeros(len(test_ids))
    total_weight = 0
    
    for name, (scores, weight) in models.items():
        print(f"Fusing {name} (Weight: {weight})")
        final_ensemble += rank_normalize(scores) * weight
        total_weight += weight
        
    final_ranks = rank_normalize(final_ensemble / total_weight)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v22_apex.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved V22 APEX ENSEMBLE to {out_path}")

if __name__ == '__main__':
    main()
