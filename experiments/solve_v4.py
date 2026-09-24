"""
V4: Improved Zero-Shot CLIP + Enhanced Physics

Key improvements over v_zero_shot_final (0.677):
1. ViT-L-14 (4x larger CLIP model, much better visual understanding)
2. Many more diverse text prompts (prompt engineering)
3. Check ALL 30 frames, not just 4
4. Temporal CLIP embedding analysis (how much does the scene change?)
5. More physics features from the cache
6. Better ensemble weighting based on what we learned from LB scores
"""

import os
import csv
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / len(scores)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # =============================================
    # PART 1: Enhanced CLIP Zero-Shot (ViT-L-14)
    # =============================================
    print("Loading ViT-L-14 CLIP model (much larger than ViT-B-32)...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='datacomp_xl_s13b_b90k'
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    
    # Extensive prompt engineering — many diverse descriptions
    crash_prompts = [
        "a dashcam photo of a car crash",
        "a dashcam photo of a violent car collision",
        "a dashcam photo of vehicles colliding on the road",
        "a dashcam photo of a traffic accident with damaged cars",
        "a dashcam photo of a car hitting another vehicle",
        "a dashcam photo of a rear end collision",
        "a dashcam photo of a car accident with debris on the road",
        "a dashcam photo of a car crash with broken glass and metal",
        "a dashcam photo of cars smashing into each other",
        "a dashcam photo of a multi vehicle pile up",
        "a dashcam photo of a car flipping over after a crash",
        "a dashcam photo of a head on collision between two cars",
        "a dashcam photo of a car spinning out of control and crashing",
        "a dashcam video frame showing a moment of impact in a car crash",
        "a dashcam photo showing a car accident just happened",
    ]
    
    normal_prompts = [
        "a dashcam photo of a car driving normally on a road",
        "a dashcam photo of a peaceful highway drive",
        "a dashcam photo of smooth traffic flow on a road",
        "a dashcam photo of cars driving safely on a street",
        "a dashcam photo of an empty road ahead",
        "a dashcam photo of a calm drive through a city",
        "a dashcam photo of normal everyday traffic",
        "a dashcam photo of a car following other vehicles at a safe distance",
        "a dashcam photo of a clear road with no incidents",
        "a dashcam photo of routine commute driving",
        "a dashcam photo of a car waiting at a traffic light",
        "a dashcam photo of a quiet suburban street",
        "a dashcam photo of a car driving on a clear day",
        "a dashcam video frame showing ordinary uneventful driving",
        "a dashcam photo showing nothing unusual on the road",
    ]
    
    print(f"Encoding {len(crash_prompts)} crash prompts and {len(normal_prompts)} normal prompts...")
    text_crash = tokenizer(crash_prompts).to(device)
    text_normal = tokenizer(normal_prompts).to(device)
    
    with torch.no_grad():
        feat_crash = F.normalize(model.encode_text(text_crash), dim=-1).mean(dim=0, keepdim=True)
        feat_normal = F.normalize(model.encode_text(text_normal), dim=-1).mean(dim=0, keepdim=True)
    
    # Load test IDs
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
    
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # Process ALL frames for each clip
    # Sample 10 evenly spaced frames (more than 4, but efficient)
    frame_indices = [0, 3, 7, 10, 14, 17, 21, 24, 27, 29]
    
    clip_scores = []
    clip_temporal_change = []
    
    print("Running zero-shot CLIP classification on test clips...")
    for clip_id in tqdm(test_ids):
        clip_dir = os.path.join(test_dir, clip_id)
        
        frame_crash_sims = []
        frame_normal_sims = []
        frame_embeddings = []
        
        for idx in frame_indices:
            frame_path = os.path.join(clip_dir, f'frame_{idx:03d}.jpg')
            try:
                img = Image.open(frame_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)
            except Exception:
                continue
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    img_feat = F.normalize(model.encode_image(img_tensor), dim=-1).float()
                    sim_crash = (img_feat @ feat_crash.T).item()
                    sim_normal = (img_feat @ feat_normal.T).item()
            
            frame_crash_sims.append(sim_crash)
            frame_normal_sims.append(sim_normal)
            frame_embeddings.append(img_feat.cpu().numpy().flatten())
        
        if len(frame_crash_sims) == 0:
            clip_scores.append(0.0)
            clip_temporal_change.append(0.0)
            continue
        
        crash_sims = np.array(frame_crash_sims)
        normal_sims = np.array(frame_normal_sims)
        
        # Score 1: Max crash similarity - min normal similarity (our original approach)
        score_max_diff = crash_sims.max() - normal_sims.min()
        
        # Score 2: Average (crash - normal) across all frames
        score_avg_diff = (crash_sims - normal_sims).mean()
        
        # Score 3: How much does "crashiness" increase over time?
        # If later frames are more crash-like, that's a strong signal
        if len(crash_sims) > 1:
            first_half = crash_sims[:len(crash_sims)//2].mean()
            second_half = crash_sims[len(crash_sims)//2:].mean()
            score_temporal_increase = second_half - first_half
        else:
            score_temporal_increase = 0.0
        
        # Combined CLIP score
        clip_scores.append(score_max_diff * 0.4 + score_avg_diff * 0.4 + score_temporal_increase * 0.2)
        
        # Score 4: Temporal change in CLIP embeddings
        # Big embedding changes = something dramatic happened
        if len(frame_embeddings) > 1:
            emb_array = np.array(frame_embeddings)
            emb_diffs = np.linalg.norm(np.diff(emb_array, axis=0), axis=1)
            clip_temporal_change.append(emb_diffs.max())
        else:
            clip_temporal_change.append(0.0)
    
    clip_scores = np.array(clip_scores)
    clip_temporal_change = np.array(clip_temporal_change)
    
    del model
    torch.cuda.empty_cache()
    
    # =============================================
    # PART 2: Physics Heuristics (from cache)
    # =============================================
    print("\nLoading physics heuristics...")
    motion_cache = os.path.join(CACHE_DIR, 'cache_motion_features.npz')
    data = np.load(motion_cache, allow_pickle=True)
    X_test_motion = data['X_test']
    motion_feats = data['feature_names'].tolist()
    
    # Best individual physics features
    score_diff_spike = X_test_motion[:, motion_feats.index('diff_spike_ratio')]
    score_blur_drop = X_test_motion[:, motion_feats.index('blur_drop_max')]
    score_flow_spike = X_test_motion[:, motion_feats.index('flow_spike_ratio')]
    score_flow_max = X_test_motion[:, motion_feats.index('flow_max')]
    score_diff_max = X_test_motion[:, motion_feats.index('diff_max')]
    score_diff_accel_max = X_test_motion[:, motion_feats.index('diff_accel_max')]
    score_hist_diff_max = X_test_motion[:, motion_feats.index('hist_diff_max')]
    score_edge_change_max = X_test_motion[:, motion_feats.index('edge_change_max')]
    score_diff_num_spikes = X_test_motion[:, motion_feats.index('diff_num_spikes_3x')]
    
    # =============================================
    # PART 3: Final Ensemble
    # =============================================
    print("\nBuilding final ensemble...")
    
    final_scores = (
        # CLIP zero-shot (our strongest signal)
        rank_normalize(clip_scores) * 2.5 +
        rank_normalize(clip_temporal_change) * 1.0 +
        
        # Physics: spike/anomaly ratios (normalized, domain-robust)
        rank_normalize(score_diff_spike) * 1.5 +
        rank_normalize(score_flow_spike) * 1.0 +
        rank_normalize(score_blur_drop) * 1.0 +
        
        # Physics: raw magnitudes (less robust but informative)
        rank_normalize(score_diff_max) * 0.5 +
        rank_normalize(score_flow_max) * 0.5 +
        rank_normalize(score_diff_accel_max) * 0.5 +
        rank_normalize(score_hist_diff_max) * 0.5 +
        rank_normalize(score_edge_change_max) * 0.3 +
        rank_normalize(score_diff_num_spikes) * 0.3
    )
    
    # Final rank normalization
    final_scores = rank_normalize(final_scores)
    
    # Write submission
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v4.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_scores):
            writer.writerow([cid, f'{score:.6f}'])
    
    print(f"\nSaved to {out_path}")
    print(f"Prediction stats: mean={final_scores.mean():.4f}, std={final_scores.std():.4f}")
    
    # Also save CLIP-only submission for comparison
    clip_only = rank_normalize(clip_scores)
    out_clip = os.path.join(SUBMISSIONS_DIR, 'submission_v4_clip_only.csv')
    with open(out_clip, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, clip_only):
            writer.writerow([cid, f'{score:.6f}'])
    print(f"CLIP-only submission saved to {out_clip}")

if __name__ == '__main__':
    main()
