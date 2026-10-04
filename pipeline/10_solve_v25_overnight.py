"""
V25: OVERNIGHT FINAL PUSH
==========================
Two brand new models + domain-adapted prompts for the final day.

1. SigLIP2-378 (upgraded SigLIP, better training, same speed)
2. ViT-H-14 DFN5B (much larger model, trained on 5B filtered images)
3. Domain-specific rain/night prompts (directly fix known domain shift)

All with checkpointing so no work is lost if the PC crashes.

Estimated time: ~8 hours total (start 7:30 PM, done ~3:30 AM)
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
CACHE_DIR = os.path.join(CACHE_DIR, 'v25_cache')
os.makedirs(CACHE_DIR, exist_ok=True)

# ============================================================
# DOMAIN-SPECIFIC PROMPTS (rain/night matched to test set)
# ============================================================
CRASH_PROMPTS = [
    # Generic crash prompts (proven)
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
    # NEW: Domain-specific rain/night prompts
    "a dashcam video of a car crash in heavy rain",
    "a dashcam video of a car accident at night",
    "a dashcam video of a car collision on a wet road at night",
    "a dashcam video of a car hydroplaning and crashing in the rain",
    "a dashcam video of a nighttime car crash with headlights",
    "a dashcam video of a car skidding on a rainy road and hitting another car",
    "a dashcam video of a car accident during a rainstorm",
    "a dashcam video of cars crashing in poor visibility",
    "a dashcam video of a car accident on a dark wet highway",
    "a dashcam video of a car crash in foggy rainy conditions",
    "a dashcam video of a car losing traction and colliding in the rain",
    "a dashcam video of a multi-vehicle accident on a slippery wet road",
    "a dashcam video of a car crashing at a rainy intersection at night",
    "a dashcam video of headlights illuminating a car crash scene",
    "a dashcam video of a car spinning on a wet surface and crashing",
]

NORMAL_PROMPTS = [
    # Generic normal prompts (proven)
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
    # NEW: Domain-specific rain/night normal prompts
    "a dashcam video of safe driving in the rain at night",
    "a dashcam video of careful driving on a wet road",
    "a dashcam video of normal nighttime driving with headlights on",
    "a dashcam video of a car driving safely through a rainstorm",
    "a dashcam video of a car following traffic on a rainy highway",
    "a dashcam video of steady driving on a dark wet road",
    "a dashcam video of a car slowly driving through rain puddles",
    "a dashcam video of uneventful driving on a dark highway",
    "a dashcam video of windshield wipers running during normal driving",
    "a dashcam video of a car driving cautiously on a slippery road",
]

def extract_features(clip_dir, model, preprocess, feat_crash, feat_normal, device):
    """Extract temporal text-similarity features for all 30 frames."""
    frame_scores = []
    
    for i in range(30):
        path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            frame_scores.append(0.0)
            continue
        
        tta_scores = []
        for aug_img in [img, ImageOps.mirror(img)]:
            inp = preprocess(aug_img).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    img_feat = F.normalize(model.encode_image(inp), dim=-1)
                    crash_sim = (img_feat @ feat_crash.T).mean().item()
                    normal_sim = (img_feat @ feat_normal.T).mean().item()
                    tta_scores.append(crash_sim - normal_sim)
        
        frame_scores.append(np.mean(tta_scores))
    
    scores = np.array(frame_scores)
    
    if len(scores) < 5:
        return {k: 0.0 for k in ['max_score', 'final_score', 'spike', 'temporal_gradient', 
                                   'variance', 'late_mean', 'early_mean', 'dynamic_range',
                                   'top3_mean', 'late_spike', 'acceleration', 'range_late']}
    
    diffs = np.diff(scores)
    sorted_scores = np.sort(scores)[::-1]
    
    return {
        'max_score': float(np.max(scores)),
        'final_score': float(np.mean(scores[-5:])),
        'spike': float(np.max(diffs)) if len(diffs) > 0 else 0.0,
        'temporal_gradient': float(np.polyfit(np.arange(len(scores)), scores, 1)[0]),
        'variance': float(np.var(scores)),
        'late_mean': float(np.mean(scores[-10:])),
        'early_mean': float(np.mean(scores[:10])),
        'dynamic_range': float(np.max(scores) - np.min(scores)),
        # NEW features to help with ambiguous clips
        'top3_mean': float(np.mean(sorted_scores[:3])),  # Average of top 3 frames
        'late_spike': float(np.max(diffs[-10:])) if len(diffs) >= 10 else float(np.max(diffs)) if len(diffs) > 0 else 0.0,  # Spike in last 10 frames
        'acceleration': float(np.max(np.diff(diffs))) if len(diffs) > 1 else 0.0,  # Rate of change of rate of change
        'range_late': float(np.max(scores[-10:]) - np.min(scores[-10:])),  # Dynamic range in last 10 frames
    }

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
        all_features[cid] = extract_features(clip_dir, model, preprocess, feat_crash, feat_normal, device)
    
    print(f"[CHECKPOINT] Saving to {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(all_features, f)
    
    del model
    torch.cuda.empty_cache()
    return all_features

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / max(1, len(scores) - 1)

def features_to_score(features_dict, test_ids):
    crash_scores = []
    for cid in test_ids:
        f = features_dict[cid]
        score = (
            f['max_score'] * 2.0 + f['final_score'] * 3.0 + f['spike'] * 4.0 +
            f['temporal_gradient'] * 3.0 + f['late_mean'] * 2.0 + f['dynamic_range'] * 1.5 +
            f['variance'] * 1.0 - f['early_mean'] * 1.0 +
            f.get('top3_mean', 0) * 2.0 + f.get('late_spike', 0) * 3.0 +
            f.get('acceleration', 0) * 1.5 + f.get('range_late', 0) * 1.0
        )
        crash_scores.append(score)
    return np.array(crash_scores)

def main():
    print("="*60)
    print("V25: FINAL OVERNIGHT PUSH")
    print("  - SigLIP2-378 (upgraded model)")
    print("  - ViT-H-14 DFN5B (much larger model)")
    print("  - Domain-adapted rain/night prompts")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # === Model 1: SigLIP2-378 (UPGRADE from SigLIP v1) ===
    siglip2_features = run_backbone_with_checkpoint(
        'ViT-SO400M-14-SigLIP2-378', 'webli', 'siglip2_378_domain', test_ids, test_dir, device)
    siglip2_scores = features_to_score(siglip2_features, test_ids)
    
    # === Model 2: ViT-H-14 DFN5B (MUCH larger model) ===
    vith_features = run_backbone_with_checkpoint(
        'ViT-H-14', 'dfn5b', 'vit_h14_dfn5b_domain', test_ids, test_dir, device)
    vith_scores = features_to_score(vith_features, test_ids)
    
    # === Quick standalone submission (just these 2 new models + V7 physics) ===
    print("\nBuilding V25 standalone ensemble...")
    
    v7_scores = []
    with open(os.path.join(SUBMISSIONS_DIR, 'submission_v7_yolo.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            v7_scores.append(float(row['label']))
    v7_scores = np.array(v7_scores)
    
    components = {
        'SigLIP2_378': (rank_normalize(siglip2_scores), 5.0),
        'ViT_H14_DFN5B': (rank_normalize(vith_scores), 4.0),
        'V7_physics': (rank_normalize(v7_scores), 1.5),
    }
    
    total_weight = sum(w for _, w in components.values())
    final = np.zeros(len(test_ids))
    for name, (scores, weight) in components.items():
        final += scores * weight
        print(f"  {name}: weight={weight}")
    
    final_ranks = rank_normalize(final / total_weight)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v25_standalone.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
    
    print(f"\nSaved V25 standalone to {out_path}")
    print("="*60)
    print("V25 OVERNIGHT PROCESSING COMPLETE!")
    print("Run solve_v25b_final_blend.py in the morning for the ultimate submission.")
    print("="*60)

if __name__ == '__main__':
    main()
