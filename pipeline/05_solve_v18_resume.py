"""
V18 RESUME with CHECKPOINTING
- Model 1 (SigLIP): Loads from saved file (already done)
- Model 2 (EVA02): Runs fresh, saves result to disk when done
- Model 3 (CLIP): Runs fresh, saves result to disk when done
- If the script is run again after a crash, it will skip any model
  whose results already exist on disk.
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
CACHE_DIR = os.path.join(CACHE_DIR, 'v18_cache')
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

def score_clip_temporal(clip_dir, model, preprocess, feat_crash, feat_normal, device):
    frame_scores = []
    for i in range(30):
        path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            frame_scores.append(0.0)
            continue
        scores_for_frame = []
        for aug_img in [img, ImageOps.mirror(img)]:
            inp = preprocess(aug_img).unsqueeze(0).to(device)
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    img_feat = F.normalize(model.encode_image(inp), dim=-1)
                    crash_sim = (img_feat @ feat_crash.T).mean().item()
                    normal_sim = (img_feat @ feat_normal.T).mean().item()
                    scores_for_frame.append(crash_sim - normal_sim)
        frame_scores.append(np.mean(scores_for_frame))
    
    scores = np.array(frame_scores)
    if len(scores) < 5:
        return {'max_score': 0, 'final_score': 0, 'spike': 0, 'temporal_gradient': 0, 'variance': 0, 'late_mean': 0, 'early_mean': 0, 'dynamic_range': 0}
    
    return {
        'max_score': float(np.max(scores)),
        'final_score': float(np.mean(scores[-5:])),
        'spike': float(np.max(np.diff(scores))) if len(scores) > 1 else 0,
        'temporal_gradient': float(np.polyfit(np.arange(len(scores)), scores, 1)[0]) if len(scores) >= 2 else 0,
        'variance': float(np.var(scores)),
        'late_mean': float(np.mean(scores[-10:])),
        'early_mean': float(np.mean(scores[:10])),
        'dynamic_range': float(np.max(scores) - np.min(scores))
    }

def run_backbone_with_checkpoint(model_name, pretrained, cache_key, test_ids, test_dir, device):
    """Run a backbone and save results to disk. If results already exist, load them."""
    cache_path = os.path.join(CACHE_DIR, f'{cache_key}_features.json')
    
    if os.path.exists(cache_path):
        print(f"\n[CHECKPOINT] Loading cached results for {model_name} from {cache_path}")
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
        all_features[cid] = score_clip_temporal(clip_dir, model, preprocess, feat_crash, feat_normal, device)
    
    # SAVE CHECKPOINT
    print(f"[CHECKPOINT] Saving {model_name} results to {cache_path}")
    with open(cache_path, 'w') as f:
        json.dump(all_features, f)
    
    del model
    torch.cuda.empty_cache()
    return all_features

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / max(1, len(scores) - 1)

def features_to_crash_score(features_dict, test_ids):
    crash_scores = []
    for cid in test_ids:
        f = features_dict[cid]
        score = (f['max_score']*2.0 + f['final_score']*3.0 + f['spike']*4.0 + 
                 f['temporal_gradient']*3.0 + f['late_mean']*2.0 + f['dynamic_range']*1.5 + 
                 f['variance']*1.0 - f['early_mean']*1.0)
        crash_scores.append(score)
    return np.array(crash_scores)

def main():
    print("="*60)
    print("V18: ULTIMATE ZERO-SHOT (WITH CHECKPOINTING)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # === Model 1: SigLIP (load from pre-saved CSV) ===
    siglip_cache = os.path.join(CACHE_DIR, 'siglip_features.json')
    if os.path.exists(siglip_cache):
        print("\n[CHECKPOINT] Loading SigLIP from cache...")
        with open(siglip_cache, 'r') as f:
            siglip_features = json.load(f)
        siglip_scores = features_to_crash_score(siglip_features, test_ids)
    else:
        # Fall back to the standalone CSV if features aren't cached
        siglip_csv = os.path.join(SUBMISSIONS_DIR, 'submission_v18a_siglip.csv')
        if os.path.exists(siglip_csv):
            print("\n[CHECKPOINT] Loading SigLIP from standalone CSV...")
            siglip_scores = []
            with open(siglip_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    siglip_scores.append(float(row['label']))
            siglip_scores = np.array(siglip_scores)
        else:
            # Must run from scratch
            siglip_features = run_backbone_with_checkpoint(
                'ViT-SO400M-14-SigLIP', 'webli', 'siglip', test_ids, test_dir, device)
            siglip_scores = features_to_crash_score(siglip_features, test_ids)
    
    # === Model 2: EVA02 (with checkpoint) ===
    eva_features = run_backbone_with_checkpoint(
        'EVA02-L-14', 'merged2b_s4b_b131k', 'eva02', test_ids, test_dir, device)
    eva_scores = features_to_crash_score(eva_features, test_ids)
    
    # === Model 3: CLIP (with checkpoint) ===
    clip_features = run_backbone_with_checkpoint(
        'ViT-L-14', 'datacomp_xl_s13b_b90k', 'clip_l14', test_ids, test_dir, device)
    clip_scores = features_to_crash_score(clip_features, test_ids)
    
    # === Mega Ensemble ===
    print("\n" + "="*60)
    print("Building Ultimate Ensemble...")
    print("="*60)
    
    prev_models = {}
    for name, filename in [('V7', 'submission_v7_yolo.csv'), ('V8', 'submission_v8_video.csv'), ('V6', 'submission_v6_ultimate.csv')]:
        path = os.path.join(SUBMISSIONS_DIR, filename)
        if os.path.exists(path):
            scores = []
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scores.append(float(row['label']))
            prev_models[name] = np.array(scores)
            print(f"  Loaded {name}: {filename}")
    
    components = {
        'SigLIP_temporal': (rank_normalize(siglip_scores), 5.0),
        'EVA_temporal': (rank_normalize(eva_scores), 3.0),
        'CLIP_temporal': (rank_normalize(clip_scores), 3.0),
        'V7_physics': (rank_normalize(prev_models['V7']), 3.0),
        'V8_video': (rank_normalize(prev_models['V8']), 2.0),
        'V6_foundation': (rank_normalize(prev_models['V6']), 1.5),
    }
    
    total_weight = sum(w for _, w in components.values())
    final_ensemble = np.zeros(len(test_ids))
    for name, (scores, weight) in components.items():
        final_ensemble += scores * weight
        print(f"  {name}: weight={weight}")
    
    final_ranks = rank_normalize(final_ensemble / total_weight)
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v18_ultimate.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_ranks):
            writer.writerow([cid, f'{score:.6f}'])
    
    print(f"\nSaved Ultimate Ensemble to {out_path}")
    print("="*60)
    print("DONE! Ready to submit.")

if __name__ == '__main__':
    main()
