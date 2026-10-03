"""
V20: HIGH-RES V18 ENSEMBLE
=========================
V19 (Semantic Velocity) dropped the score because raw embedding shifts 
are too sensitive to normal camera bumps and turns. 

However, we spent hours processing the High-Res SigLIP-378 model!
This script throws away the noisy velocity features and goes back to the 
proven V18 logic (pure text-similarity temporal spikes), but uses the 
brand new high-res SigLIP-378 features we cached in V19.
"""

import os
import csv
import json
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR
V19_CACHE = os.path.join(CACHE_DIR, 'v19_cache')

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / max(1, len(scores) - 1)

def features_to_v18_score(features_dict, test_ids):
    """Exactly the same logic that got 0.763 in V18, ignoring V19's velocity."""
    crash_scores = []
    for cid in test_ids:
        f = features_dict.get(cid, {})
        # If cache is missing something, return 0
        if not f:
            crash_scores.append(0.0)
            continue
            
        score = (
            f.get('max_score', 0) * 2.0 +
            f.get('final_score', 0) * 3.0 +
            f.get('spike', 0) * 4.0 +
            f.get('temporal_gradient', 0) * 3.0 +
            f.get('late_mean', 0) * 2.0 +
            f.get('dynamic_range', 0) * 1.5 +
            f.get('variance', 0) * 1.0 -
            f.get('early_mean', 0) * 1.0
        )
        crash_scores.append(score)
    return np.array(crash_scores)

def main():
    print("="*60)
    print("V20: HIGH-RES V18 ENSEMBLE")
    print("="*60)
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    # 1. Load High-Res SigLIP (378px) from V19 cache
    siglip378_path = os.path.join(V19_CACHE, 'siglip_378_features.json')
    if os.path.exists(siglip378_path):
        with open(siglip378_path, 'r') as f:
            siglip_feats = json.load(f)
        siglip_scores = features_to_v18_score(siglip_feats, test_ids)
        print("Loaded High-Res SigLIP-378")
    else:
        print("Error: High-Res SigLIP cache not found!")
        return
        
    # 2. Load EVA02 from V19 cache
    eva_path = os.path.join(V19_CACHE, 'eva02_velocity_features.json')
    with open(eva_path, 'r') as f:
        eva_feats = json.load(f)
    eva_scores = features_to_v18_score(eva_feats, test_ids)
    print("Loaded EVA02-L-14")
    
    # 3. Load CLIP from V19 cache
    clip_path = os.path.join(V19_CACHE, 'clip_l14_velocity_features.json')
    with open(clip_path, 'r') as f:
        clip_feats = json.load(f)
    clip_scores = features_to_v18_score(clip_feats, test_ids)
    print("Loaded CLIP ViT-L-14")
    
    # 4. Load V7 Physics
    v7_scores = []
    with open(os.path.join(SUBMISSIONS_DIR, 'submission_v7_yolo.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            v7_scores.append(float(row['label']))
    v7_scores = np.array(v7_scores)
    print("Loaded V7 Physics")
    
    # 5. Build the New Ultimate Ensemble
    print("\nBuilding V20 Ensemble...")
    components = {
        'SigLIP_378_temporal': (rank_normalize(siglip_scores), 5.5),  # Boosted weight for high-res
        'EVA_temporal': (rank_normalize(eva_scores), 3.0),
        'CLIP_temporal': (rank_normalize(clip_scores), 3.0),
        'V7_physics': (rank_normalize(v7_scores), 3.0),
    }
    
    total_weight = sum(w for _, w in components.values())
    final_ensemble = np.zeros(len(test_ids))
    for name, (scores, weight) in components.items():
        final_ensemble += scores * weight
        print(f"  {name}: weight={weight}")
        
    final_ranks = rank_normalize(final_ensemble / total_weight)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v20_highres.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved V20 to {out_path}")

if __name__ == '__main__':
    main()
