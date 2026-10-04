"""
V23: OPTIMAL AUTO-BLENDER (MEGA ENSEMBLE)
=========================================
We have generated 20 different submissions over the course of this competition.
Instead of manually guessing the weights (like we did in V22), we will use 
mathematics to find the perfect combination.

We will treat our strongest legal model (V21) as "Pseudo Ground Truth" for 
the top 500 and bottom 500 clips. 
Then, we train a Ridge Regression model to learn the mathematically optimal 
weights for blending ALL 20 past models together to perfectly reconstruct 
the high-confidence predictions.

This guarantees we squeeze every last decimal point out of our existing work!
"""

import os
import csv
import glob
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / max(1, len(scores) - 1)

def load_scores(filepath):
    scores = []
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores.append(float(row['label']))
    return np.array(scores)

def main():
    print("="*60)
    print("V23: OPTIMAL AUTO-BLENDER")
    print("="*60)
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    # Load ALL submission CSVs (except V12 which was human-in-loop, and anything that is a blend of blends)
    all_csvs = glob.glob(os.path.join(SUBMISSIONS_DIR, 'submission_*.csv'))
    
    # We exclude V12 (banned), V22 (manual blend), and V23 (this file)
    banned = ['v12', 'v22', 'v23']
    
    models = {}
    for csv_file in all_csvs:
        filename = os.path.basename(csv_file)
        if any(b in filename.lower() for b in banned):
            continue
        scores = load_scores(csv_file)
        if len(scores) == len(test_ids):
            models[filename] = rank_normalize(scores)
            
    print(f"Loaded {len(models)} independent models for blending.")
    
    # Convert to feature matrix X (Shape: 1750, num_models)
    model_names = list(models.keys())
    X_all = np.column_stack([models[name] for name in model_names])
    
    # We need a Ground Truth to train the blender.
    # We will use V21 (our best legal model) to select the most confident targets.
    v21_scores = models.get('submission_v21_transductive.csv')
    if v21_scores is None:
        print("Error: V21 not found!")
        return
        
    ranked_indices = np.argsort(v21_scores)
    # Top 300 crashes (1), Bottom 800 normals (0)
    normal_idx = ranked_indices[:800]
    crash_idx = ranked_indices[-300:]
    
    train_idx = np.concatenate([normal_idx, crash_idx])
    
    X_train = X_all[train_idx]
    y_train = np.concatenate([np.zeros(len(normal_idx)), np.ones(len(crash_idx))])
    
    # Train Ridge Regression to find optimal weights (alpha=10 for strong regularization to avoid overfitting)
    print("\nOptimizing weights using Ridge Regression...")
    blender = Ridge(alpha=10.0, positive=True) # positive=True forces all models to have non-negative weights
    blender.fit(X_train, y_train)
    
    weights = blender.coef_
    weights = weights / weights.sum() # Normalize to sum to 1
    
    print("\nOptimal Weights Found:")
    for name, w in zip(model_names, weights):
        if w > 0.01: # Only print models that contributed more than 1%
            print(f"  {name}: {w*100:.1f}%")
            
    # Generate final ensemble
    final_ensemble = np.zeros(len(test_ids))
    for i, w in enumerate(weights):
        if w > 0:
            final_ensemble += X_all[:, i] * w
            
    final_ranks = rank_normalize(final_ensemble)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v23_autoblend.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved V23 Auto-Blender to {out_path}")

if __name__ == '__main__':
    main()
