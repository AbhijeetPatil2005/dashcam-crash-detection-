"""
V8: VIDEO FOUNDATION MODEL - X-CLIP + Multi-Scale Temporal Analysis
===================================================================
This is the GAME CHANGER. Instead of analyzing individual frames like photos,
X-CLIP watches the entire video clip with temporal transformers that understand
motion, causality, and events unfolding over time.

Think of it this way:
- CLIP (V6): Shows a doctor 7 separate X-ray photos → "hmm, looks like damage"
- X-CLIP (V8): Shows a doctor the full video → "I can SEE the car colliding"

We also run at MULTIPLE temporal scales:
- Full clip (all 30 frames): catches the overall event
- Last 8 frames only: catches the critical moment of impact
- First vs Last comparison: catches the before/after transition
"""

import os
import csv
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import XCLIPModel, XCLIPProcessor
import open_clip
import gc

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def to_tensor(output):
    """Extract tensor from HuggingFace model output (handles API changes)."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, 'pooler_output') and output.pooler_output is not None:
        return output.pooler_output
    if hasattr(output, 'last_hidden_state'):
        return output.last_hidden_state[:, 0]
    return output[1] if isinstance(output, (tuple, list)) else output

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / len(scores)

# ============================================================
# CRASH TEXT PROMPTS (extensive prompt ensembling)
# ============================================================
CRASH_VIDEO_PROMPTS = [
    "a dashcam video of a car crash",
    "a dashcam video of a violent vehicle collision",
    "a dashcam video of cars smashing into each other",
    "a dashcam video capturing the moment of a traffic accident",
    "a dashcam video of a car hitting another car",
    "a dashcam video of a rear end collision",
    "a dashcam video showing a car accident with debris flying",
    "a dashcam video of a head on collision",
    "a dashcam video of a car spinning out and crashing",
    "a dashcam video of a multi vehicle pile up on the highway",
    "a video of a car crash from a dashboard camera",
    "a video showing the moment of impact in a car accident",
    "a dramatic dashcam video of a vehicular collision",
    "a dashcam recording of a sudden car crash",
    "a video of two cars colliding on the road",
]

NORMAL_VIDEO_PROMPTS = [
    "a dashcam video of a car driving normally on a road",
    "a dashcam video of peaceful highway driving",
    "a dashcam video of smooth traffic flow",
    "a dashcam video of a calm and uneventful drive",
    "a dashcam video of a car following traffic safely",
    "a dashcam video of everyday commute driving",
    "a dashcam video of a car on an empty road",
    "a dashcam video of a car waiting at a traffic light",
    "a dashcam video of ordinary suburban driving",
    "a dashcam video showing nothing unusual happening on the road",
    "a video of a boring normal car ride",
    "a video of a car driving without any incidents",
    "a dashcam recording of a routine drive",
    "a dashcam video of safe driving in clear weather",
    "a video of regular uneventful traffic from a dashboard camera",
]

def load_frames(clip_dir, indices):
    """Load specific frame indices from a clip directory."""
    frames = []
    for idx in indices:
        path = os.path.join(clip_dir, f'frame_{idx:03d}.jpg')
        try:
            img = Image.open(path).convert('RGB')
            frames.append(img)
        except Exception:
            pass
    return frames

def run_xclip(test_ids, test_dir, device):
    """
    Run X-CLIP Video-Language Model at multiple temporal scales.
    X-CLIP processes 8 frames at a time with temporal cross-attention.
    """
    print("\n" + "="*60)
    print("Loading X-CLIP Large (ViT-L/14) Video-Language Model...")
    print("="*60)
    
    model_name = "microsoft/xclip-large-patch14"
    processor = XCLIPProcessor.from_pretrained(model_name)
    model = XCLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # Encode text prompts
    crash_inputs = processor(text=CRASH_VIDEO_PROMPTS, return_tensors="pt", padding=True).to(device)
    normal_inputs = processor(text=NORMAL_VIDEO_PROMPTS, return_tensors="pt", padding=True).to(device)
    
    with torch.no_grad():
        crash_text_feats = to_tensor(model.get_text_features(**crash_inputs))
        crash_text_feats = F.normalize(crash_text_feats, dim=-1).mean(dim=0, keepdim=True)
        
        normal_text_feats = to_tensor(model.get_text_features(**normal_inputs))
        normal_text_feats = F.normalize(normal_text_feats, dim=-1).mean(dim=0, keepdim=True)
    
    # Define multiple temporal windows
    # Scale 1: Full clip (evenly spaced 8 frames across all 30)
    full_indices = [0, 4, 8, 12, 16, 20, 24, 29]
    # Scale 2: Last 8 frames (the critical impact zone)
    late_indices = [22, 23, 24, 25, 26, 27, 28, 29]
    # Scale 3: Middle section (the approach phase)
    mid_indices = [10, 12, 14, 16, 18, 20, 22, 24]
    
    all_scales = [
        ("full_clip", full_indices),
        ("impact_zone", late_indices),
        ("approach", mid_indices),
    ]
    
    scale_scores = {name: [] for name, _ in all_scales}
    
    for clip_id in tqdm(test_ids, desc="X-CLIP Video Scoring"):
        clip_dir = os.path.join(test_dir, clip_id)
        
        for scale_name, indices in all_scales:
            frames = load_frames(clip_dir, indices)
            
            if len(frames) < 8:
                # Pad with last frame if needed
                while len(frames) < 8:
                    frames.append(frames[-1] if frames else Image.new('RGB', (224, 224)))
            
            # Process video (X-CLIP expects exactly 8 frames)
            video_inputs = processor(videos=[frames[:8]], return_tensors="pt", padding=True).to(device)
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    video_feats = to_tensor(model.get_video_features(**video_inputs))
                    video_feats = F.normalize(video_feats, dim=-1)
                    
                    sim_crash = (video_feats @ crash_text_feats.T).item()
                    sim_normal = (video_feats @ normal_text_feats.T).item()
            
            scale_scores[scale_name].append(sim_crash - sim_normal)
    
    # Combine temporal scales
    combined = (
        rank_normalize(np.array(scale_scores["full_clip"])) * 1.0 +
        rank_normalize(np.array(scale_scores["impact_zone"])) * 1.5 +  # Impact zone matters most
        rank_normalize(np.array(scale_scores["approach"])) * 0.5
    )
    
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    
    return rank_normalize(combined)

def run_clip_l14(test_ids, test_dir, device):
    """Our proven best: ViT-L-14 CLIP on individual frames (from V6)."""
    print("\n" + "="*60)
    print("Loading CLIP ViT-L-14 (DataComp) for frame-level semantics...")
    print("="*60)
    
    CRASH_PROMPTS = CRASH_VIDEO_PROMPTS  # Reuse same prompts
    NORMAL_PROMPTS = NORMAL_VIDEO_PROMPTS
    
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='datacomp_xl_s13b_b90k'
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    
    text_crash = tokenizer(CRASH_PROMPTS).to(device)
    text_normal = tokenizer(NORMAL_PROMPTS).to(device)
    
    with torch.no_grad():
        feat_crash = F.normalize(model.encode_text(text_crash), dim=-1).mean(dim=0, keepdim=True)
        feat_normal = F.normalize(model.encode_text(text_normal), dim=-1).mean(dim=0, keepdim=True)
    
    frame_indices = [0, 5, 10, 15, 20, 25, 29]
    scores = []
    
    for clip_id in tqdm(test_ids, desc="CLIP Frame Scoring"):
        clip_dir = os.path.join(test_dir, clip_id)
        crash_sims = []
        normal_sims = []
        
        for idx in frame_indices:
            path = os.path.join(clip_dir, f'frame_{idx:03d}.jpg')
            try:
                img = preprocess(Image.open(path).convert('RGB')).unsqueeze(0).to(device)
            except Exception:
                continue
            
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    img_feat = F.normalize(model.encode_image(img), dim=-1)
                    crash_sims.append((img_feat @ feat_crash.T).item())
                    normal_sims.append((img_feat @ feat_normal.T).item())
        
        if not crash_sims:
            scores.append(0.0)
            continue
        
        c = np.array(crash_sims)
        n = np.array(normal_sims)
        scores.append(c.max() - n.min())
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return rank_normalize(np.array(scores))

def main():
    print("="*60)
    print("V8: VIDEO FOUNDATION MODEL MEGA-ENSEMBLE")
    print("  X-CLIP (Video AI) + CLIP (Frame AI) + YOLO (Physics)")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load test IDs
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
    
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # ============================================================
    # LAYER 1: X-CLIP Video Understanding (THE BIG NEW THING)
    # ============================================================
    xclip_scores = run_xclip(test_ids, test_dir, device)
    
    # ============================================================
    # LAYER 2: CLIP Frame-Level Semantics (proven performer)
    # ============================================================
    clip_scores = run_clip_l14(test_ids, test_dir, device)
    
    # ============================================================
    # LAYER 3: YOLO Physics (if available)
    # ============================================================
    yolo_path = os.path.join(CACHE_DIR, 'cache_yolo_features.npz')
    if os.path.exists(yolo_path):
        print("\nLoading YOLO kinematics...")
        data = np.load(yolo_path)
        X_yolo = data['X_test']
        yolo_physics = (
            rank_normalize(X_yolo[:, 0]) * 1.5 +  # deceleration
            rank_normalize(X_yolo[:, 1]) * 0.5 +  # area change
            rank_normalize(X_yolo[:, 2]) * 1.0     # IoU overlap
        )
        yolo_scores = rank_normalize(yolo_physics)
        has_yolo = True
    else:
        has_yolo = False
        print("\nNo YOLO features found, skipping physics layer.")
    
    # ============================================================
    # MEGA FUSION
    # ============================================================
    print("\n" + "="*60)
    print("MEGA FUSION: Video AI + Frame AI + Physics")
    print("="*60)
    
    if has_yolo:
        final = (
            xclip_scores * 2.5 +   # Video understanding (biggest weight - this is the game changer)
            clip_scores * 1.5 +     # Frame semantics (proven strong)
            yolo_scores * 0.5       # Physics (supplementary)
        )
    else:
        final = (
            xclip_scores * 2.0 +
            clip_scores * 1.0
        )
    
    final = rank_normalize(final)
    
    # Write submission
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v8_video.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final):
            writer.writerow([cid, f'{score:.6f}'])
    
    print(f"\nSaved to {out_path}")
    print("Ready to submit!")

if __name__ == '__main__':
    main()
