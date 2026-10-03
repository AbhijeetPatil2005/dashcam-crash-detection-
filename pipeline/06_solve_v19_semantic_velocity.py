"""
V19: SEMANTIC VELOCITY + HIGH-RES SIGLIP
==========================================
Two brand-new signals never tried before:

1. SEMANTIC VELOCITY: Instead of comparing frames to text prompts,
   we measure how fast the MEANING of the scene is changing.
   - Normal driving: smooth, slow embedding changes
   - Crash: EXPLOSIVE embedding change (cars crumple, debris flies, camera shakes)
   This is like "optical flow" but in MEANING space, not pixel space.

2. HIGH-RES SIGLIP (378px): 378px resolution catches brake lights,
   small debris, and distant vehicles that 224px completely misses.

3. MULTI-SCALE TEMPORAL: Measures changes over 1, 3, 5, and 10 frame windows
   to capture both sudden impacts and gradual build-ups.
"""

import os
import csv
import json
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm
import open_clip

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR
CACHE_DIR = os.path.join(CACHE_DIR, 'v19_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

CRASH_PROMPTS = [
    "a dashcam video of a car crash",
    "a dashcam video of a violent vehicle collision",
    "a dashcam video of a traffic accident",
    "a dashcam video of cars smashing into each other",
    "a dashcam video capturing the moment of impact in a car accident",
    "a dashcam recording of a car hitting another vehicle",
    "a video of a car crash from a dashboard camera",
    "a dashcam video showing vehicles colliding on the road",
    "a dashcam video of a rear-end collision",
    "a dashcam video of a head-on collision between two cars",
    "a dashcam video of a T-bone crash at an intersection",
    "a dashcam video of a car running a red light and crashing",
    "a dashcam video of a car spinning out of control",
    "a dashcam video of a car rolling over after a crash",
    "a dashcam video of a car hitting a guardrail",
    "a dashcam video of a car swerving and hitting another vehicle",
    "a dashcam video of a multi-car pileup",
    "a dashcam video of a car rear-ending a stopped vehicle",
    "a dashcam video of a car losing control on a wet road",
    "a dashcam video of a side-swipe collision on a highway",
    "a dashcam video showing broken glass and car debris on the road",
    "a dashcam video showing airbags deploying in a crash",
    "a dashcam video showing a car with crumpled front end",
    "a dashcam video of a car accident with smoke and debris",
    "a dashcam video showing sudden impact and windshield cracking",
    "a dashcam video of a car about to crash",
    "a dashcam video of an unavoidable collision",
    "a dashcam video showing a car sliding towards another vehicle",
    "a dashcam video of emergency braking before impact",
    "a dashcam video of a near miss that turns into a crash",
]

NORMAL_PROMPTS = [
    "a dashcam video of a car driving normally on a road",
    "a dashcam video of peaceful highway driving",
    "a dashcam video of smooth traffic flow on a highway",
    "a dashcam video of a calm and uneventful drive",
    "a dashcam video of everyday commute driving",
    "a dashcam video of a car cruising on an open road",
    "a video of a boring normal car ride from a dashcam",
    "a dashcam video of safe driving in light traffic",
    "a dashcam video of a car waiting at a traffic light",
    "a dashcam video of cars merging smoothly onto a highway",
    "a dashcam video of driving through a residential neighborhood",
    "a dashcam video of a car following traffic at a safe distance",
    "a dashcam video of a car changing lanes smoothly",
    "a dashcam video of driving on a country road with no traffic",
    "a dashcam video of a car parked or moving very slowly",
    "a dashcam video of normal city driving with pedestrians",
    "a dashcam video of a car driving in the rain without incident",
    "a dashcam video of driving on a clear sunny day",
    "a dashcam video of slow traffic moving normally",
    "a dashcam video of a car at a stop sign",
]

def extract_clip_features(clip_dir, model, preprocess, feat_crash, feat_normal, device):
    """
    Extract BOTH text-similarity scores AND raw embeddings for all 30 frames.
    Returns temporal text scores + semantic velocity features.
    """
    frame_scores = []
    embeddings = []
    
    for i in range(30):
        path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            frame_scores.append(0.0)
            embeddings.append(None)
            continue
        
        tta_scores = []
        tta_embeds = []
        for aug_img in [img, ImageOps.mirror(img)]:
            inp = preprocess(aug_img).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    img_feat = F.normalize(model.encode_image(inp), dim=-1)
                    crash_sim = (img_feat @ feat_crash.T).mean().item()
                    normal_sim = (img_feat @ feat_normal.T).mean().item()
                    tta_scores.append(crash_sim - normal_sim)
                    tta_embeds.append(img_feat.cpu().float().numpy().flatten())
        
        frame_scores.append(np.mean(tta_scores))
        embeddings.append(np.mean(tta_embeds, axis=0))
    
    scores = np.array(frame_scores)
    
    # ============================================================
    # A. TEXT SIMILARITY TEMPORAL FEATURES (same as V18)
    # ============================================================
    if len(scores) < 5:
        text_features = {
            'max_score': 0, 'final_score': 0, 'spike': 0,
            'temporal_gradient': 0, 'variance': 0, 'late_mean': 0,
            'early_mean': 0, 'dynamic_range': 0
        }
    else:
        diffs = np.diff(scores)
        text_features = {
            'max_score': float(np.max(scores)),
            'final_score': float(np.mean(scores[-5:])),
            'spike': float(np.max(diffs)) if len(diffs) > 0 else 0,
            'temporal_gradient': float(np.polyfit(np.arange(len(scores)), scores, 1)[0]),
            'variance': float(np.var(scores)),
            'late_mean': float(np.mean(scores[-10:])),
            'early_mean': float(np.mean(scores[:10])),
            'dynamic_range': float(np.max(scores) - np.min(scores))
        }
    
    # ============================================================
    # B. SEMANTIC VELOCITY FEATURES (brand new!)
    # ============================================================
    # Compute embedding distances between frames at multiple scales
    valid_embeddings = [(i, e) for i, e in enumerate(embeddings) if e is not None]
    
    if len(valid_embeddings) < 5:
        velocity_features = {
            'max_velocity_1': 0, 'max_velocity_3': 0, 'max_velocity_5': 0,
            'max_velocity_10': 0, 'late_velocity': 0, 'velocity_spike': 0,
            'velocity_gradient': 0, 'embedding_anomaly': 0
        }
    else:
        # Multi-scale semantic velocity
        def compute_velocities(embeds_list, gap):
            velocities = []
            for i in range(gap, len(embeds_list)):
                idx_a, emb_a = embeds_list[i - gap]
                idx_b, emb_b = embeds_list[i]
                # Cosine distance = 1 - cosine_similarity
                cos_sim = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b) + 1e-8)
                velocities.append(1.0 - cos_sim)
            return np.array(velocities) if velocities else np.array([0.0])
        
        vel_1 = compute_velocities(valid_embeddings, 1)   # Frame-to-frame
        vel_3 = compute_velocities(valid_embeddings, 3)   # 3-frame window
        vel_5 = compute_velocities(valid_embeddings, 5)   # 5-frame window
        vel_10 = compute_velocities(valid_embeddings, min(10, len(valid_embeddings) - 1))
        
        # Compute the mean embedding to find outlier frames
        all_embeds = np.stack([e for _, e in valid_embeddings])
        mean_embed = np.mean(all_embeds, axis=0)
        anomaly_scores = [1.0 - np.dot(e, mean_embed) / (np.linalg.norm(e) * np.linalg.norm(mean_embed) + 1e-8)
                         for _, e in valid_embeddings]
        
        # Late-clip velocity (last 1/3 of the clip)
        n_late = max(1, len(vel_1) // 3)
        
        velocity_features = {
            'max_velocity_1': float(np.max(vel_1)),
            'max_velocity_3': float(np.max(vel_3)),
            'max_velocity_5': float(np.max(vel_5)),
            'max_velocity_10': float(np.max(vel_10)),
            'late_velocity': float(np.mean(vel_1[-n_late:])),
            'velocity_spike': float(np.max(np.diff(vel_1))) if len(vel_1) > 1 else 0,
            'velocity_gradient': float(np.polyfit(np.arange(len(vel_1)), vel_1, 1)[0]) if len(vel_1) >= 2 else 0,
            'embedding_anomaly': float(np.max(anomaly_scores))
        }
    
    # Merge all features
    all_features = {}
    all_features.update(text_features)
    all_features.update(velocity_features)
    return all_features

def run_backbone_with_checkpoint(model_name, pretrained, cache_key, test_ids, test_dir, device):
    cache_path = os.path.join(CACHE_DIR, f'{cache_key}_features.json')
    
    if os.path.exists(cache_path):
        print(f"\n[CHECKPOINT] Loading cached {model_name} from {cache_path}")
        with open(cache_path, 'r') as f:
            return json.load(f)
    
    print(f"\nLoading {model_name} ({pretrained})...")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    
    text_crash = tokenizer(CRASH_PROMPTS).to(device)
    text_normal = tokenizer(NORMAL_PROMPTS).to(device)
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            feat_crash = F.normalize(model.encode_text(text_crash), dim=-1)
            feat_normal = F.normalize(model.encode_text(text_normal), dim=-1)
    
    all_features = {}
    for cid in tqdm(test_ids, desc=f"Scoring with {model_name}"):
        clip_dir = os.path.join(test_dir, cid)
        all_features[cid] = extract_clip_features(clip_dir, model, preprocess, feat_crash, feat_normal, device)
    
    print(f"[CHECKPOINT] Saving to {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(all_features, f)
    
    del model
    torch.cuda.empty_cache()
    return all_features

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / max(1, len(scores) - 1)

def features_to_crash_score(features_dict, test_ids):
    """Convert the rich feature set into a single crash probability."""
    crash_scores = []
    for cid in test_ids:
        f = features_dict[cid]
        
        # Text similarity component
        text_score = (
            f['max_score'] * 2.0 +
            f['final_score'] * 3.0 +
            f['spike'] * 4.0 +
            f['temporal_gradient'] * 3.0 +
            f['late_mean'] * 2.0 +
            f['dynamic_range'] * 1.5 +
            f['variance'] * 1.0 -
            f['early_mean'] * 1.0
        )
        
        # Semantic velocity component (brand new!)
        velocity_score = (
            f['max_velocity_1'] * 200.0 +     # Sudden frame-to-frame change
            f['max_velocity_3'] * 150.0 +      # Medium-window change
            f['max_velocity_5'] * 100.0 +      # Longer-window change
            f['max_velocity_10'] * 50.0 +      # Full-clip change
            f['late_velocity'] * 300.0 +        # Late-clip velocity (most important!)
            f['velocity_spike'] * 200.0 +       # Sudden acceleration in velocity
            f['velocity_gradient'] * 150.0 +    # Increasing velocity trend
            f['embedding_anomaly'] * 100.0      # Outlier frames
        )
        
        crash_scores.append(text_score + velocity_score)
    return np.array(crash_scores)

def main():
    print("="*60)
    print("V19: SEMANTIC VELOCITY + HIGH-RES SIGLIP")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # === Backbone 1: SigLIP-378 (HIGH RESOLUTION - brand new!) ===
    siglip_hr_features = run_backbone_with_checkpoint(
        'ViT-SO400M-14-SigLIP-378', 'webli', 'siglip_378', test_ids, test_dir, device)
    siglip_hr_scores = features_to_crash_score(siglip_hr_features, test_ids)
    
    # === Backbone 2: EVA02-L-14 (with semantic velocity - upgraded!) ===
    eva_features = run_backbone_with_checkpoint(
        'EVA02-L-14', 'merged2b_s4b_b131k', 'eva02_velocity', test_ids, test_dir, device)
    eva_scores = features_to_crash_score(eva_features, test_ids)
    
    # === Backbone 3: CLIP ViT-L-14 (with semantic velocity - upgraded!) ===
    clip_features = run_backbone_with_checkpoint(
        'ViT-L-14', 'datacomp_xl_s13b_b90k', 'clip_l14_velocity', test_ids, test_dir, device)
    clip_scores = features_to_crash_score(clip_features, test_ids)
    
    # === Load V18 results ===
    v18_scores = []
    v18_path = os.path.join(SUBMISSIONS_DIR, 'submission_v18_ultimate.csv')
    if os.path.exists(v18_path):
        with open(v18_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                v18_scores.append(float(row['label']))
        v18_scores = np.array(v18_scores)
        print("Loaded V18 results")
    
    # === Load V7 physics ===
    v7_scores = []
    v7_path = os.path.join(SUBMISSIONS_DIR, 'submission_v7_yolo.csv')
    if os.path.exists(v7_path):
        with open(v7_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                v7_scores.append(float(row['label']))
        v7_scores = np.array(v7_scores)
        print("Loaded V7 physics")
    
    # === MEGA ENSEMBLE ===
    print("\n" + "="*60)
    print("Building V19 Ultimate Ensemble...")
    print("="*60)
    
    components = {
        'SigLIP_378_velocity': (rank_normalize(siglip_hr_scores), 6.0),   # Best new model
        'EVA_velocity': (rank_normalize(eva_scores), 3.0),
        'CLIP_velocity': (rank_normalize(clip_scores), 3.0),
        'V18_ensemble': (rank_normalize(v18_scores), 4.0),                # Previous best
        'V7_physics': (rank_normalize(v7_scores), 2.0),                   # Pure physics
    }
    
    total_weight = sum(w for _, w in components.values())
    final_ensemble = np.zeros(len(test_ids))
    for name, (scores, weight) in components.items():
        final_ensemble += scores * weight
        print(f"  {name}: weight={weight}")
    
    final_ranks = rank_normalize(final_ensemble / total_weight)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v19_semantic_velocity.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
    
    print(f"\nSaved V19 to {out_path}")
    print("DONE!")

if __name__ == '__main__':
    main()
