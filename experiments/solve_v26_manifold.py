"""
V26: EMBEDDING MANIFOLD SMOOTHING (K-NN)
=========================================
Even our best model can be uncertain about specific clips (ranking them around 0.5).
However, if an uncertain clip looks VERY similar (in CLIP embedding space) to 
5 other clips that are almost certainly crashes, it's highly likely that 
the uncertain clip is also a crash.

This script applies K-Nearest Neighbors (K-NN) smoothing over the Test Set.
It diffuses the predictions across the semantic manifold, directly improving 
ranking consistency and AUC.
"""

import os
import csv
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / max(1, len(scores) - 1)

def main():
    print("="*60)
    print("V26: EMBEDDING MANIFOLD SMOOTHING")
    print("="*60)
    
    # 1. Load test IDs and Best Scores
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    best_scores = []
    # Using V24 since it had our highest public leaderboard score (0.78943)
    with open(os.path.join(SUBMISSIONS_DIR, 'submission_v24_feature_blender.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            best_scores.append(float(row['label']))
    best_scores = np.array(best_scores)
    
    # 2. Load Raw Embeddings
    print("Loading CLIP embeddings...")
    clip_data = np.load(os.path.join(CACHE_DIR, 'cache_clip_features.npz'))
    X_test = clip_data['X_test']  # Shape: (1750, 2048)
    
    # 3. Calculate Similarity Graph
    print("Calculating Semantic Similarity Graph...")
    similarity_matrix = cosine_similarity(X_test)
    
    # 4. K-NN Smoothing
    print("Applying Manifold Diffusion...")
    K = 15  # Number of neighbors to look at
    smoothed_scores = np.zeros_like(best_scores)
    
    for i in range(len(test_ids)):
        # Get indices of top K most similar clips (including self)
        # We sort ascending, so the last K elements are the most similar
        similar_idx = np.argsort(similarity_matrix[i])[-K:]
        
        # Get their similarities (weights) and scores
        weights = similarity_matrix[i, similar_idx]
        neighbor_scores = best_scores[similar_idx]
        
        # Softmax-like temperature scaling on weights to prioritize very close neighbors
        weights = weights ** 4
        weights = weights / np.sum(weights)
        
        # Calculate weighted average score
        smoothed_scores[i] = np.sum(neighbor_scores * weights)
        
    # 5. Blend Original and Smoothed
    # We don't want to completely overwrite the original scores, just nudge them
    print("Blending original and smoothed scores...")
    orig_ranks = rank_normalize(best_scores)
    smooth_ranks = rank_normalize(smoothed_scores)
    
    # 70% Original, 30% Smoothed
    final_blend = rank_normalize(orig_ranks * 0.70 + smooth_ranks * 0.30)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v26_manifold_smoothed.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_blend):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved V26 to {out_path}")

if __name__ == '__main__':
    main()
