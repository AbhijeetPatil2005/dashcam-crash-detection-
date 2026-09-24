"""
V6: ULTIMATE ZERO-SHOT FOUNDATION ENSEMBLE
This uses multiple massive Vision-Language models and extensive Prompt Ensembling
to maximize semantic understanding of the scene without touching the poisoned training data.
"""

import os
import csv
import numpy as np
import torch
import open_clip
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import gc

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, CACHE_DIR, SUBMISSIONS_DIR

def rank_normalize(scores):
    return np.argsort(np.argsort(scores)).astype(float) / len(scores)

# Extensive prompt ensembling (similar to original CLIP paper technique)
CRASH_PROMPTS = [
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
    "a photo of a severe car crash",
    "a photo of a vehicular accident",
    "a dramatic photo of a car crash",
    "a photo from a dashboard camera capturing an accident",
    "a tragic car collision on the highway"
]

NORMAL_PROMPTS = [
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
    "a photo of a boring car ride",
    "a photo of regular driving",
    "a mundane photo of a road from a car",
    "a photo from a dashboard camera capturing normal driving",
    "a safe and quiet drive on the highway"
]

def run_clip_model(model_name, pretrained_name, test_ids, test_dir, device):
    print(f"\n--- Loading {model_name} ({pretrained_name}) ---")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained_name
    )
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)
    
    text_crash = tokenizer(CRASH_PROMPTS).to(device)
    text_normal = tokenizer(NORMAL_PROMPTS).to(device)
    
    with torch.no_grad():
        # Average the embeddings of all prompts for a robust single representation
        feat_crash = F.normalize(model.encode_text(text_crash), dim=-1).mean(dim=0, keepdim=True)
        feat_normal = F.normalize(model.encode_text(text_normal), dim=-1).mean(dim=0, keepdim=True)
        
    frame_indices = [0, 5, 10, 15, 20, 25, 29] # 7 evenly spaced frames
    clip_scores = []
    
    for clip_id in tqdm(test_ids, desc=f"Scoring {model_name}"):
        clip_dir = os.path.join(test_dir, clip_id)
        frame_crash_sims = []
        frame_normal_sims = []
        
        for idx in frame_indices:
            frame_path = os.path.join(clip_dir, f'frame_{idx:03d}.jpg')
            try:
                img = Image.open(frame_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)
            except Exception:
                continue
                
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    img_feat = F.normalize(model.encode_image(img_tensor), dim=-1)
                    sim_crash = (img_feat @ feat_crash.T).item()
                    sim_normal = (img_feat @ feat_normal.T).item()
                    
            frame_crash_sims.append(sim_crash)
            frame_normal_sims.append(sim_normal)
            
        if len(frame_crash_sims) == 0:
            clip_scores.append(0.0)
            continue
            
        crash_sims = np.array(frame_crash_sims)
        normal_sims = np.array(frame_normal_sims)
        
        score_max_diff = crash_sims.max() - normal_sims.min()
        score_avg_diff = (crash_sims - normal_sims).mean()
        clip_scores.append(score_max_diff * 0.5 + score_avg_diff * 0.5)
        
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    return rank_normalize(np.array(clip_scores))

def main():
    print("="*60)
    print("V6: ULTIMATE ZERO-SHOT FOUNDATION ENSEMBLE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    test_dir = os.path.join(DATA_DIR, 'test')
    
    # We will ensemble 3 massive foundation models
    # 1. ViT-L-14 with LAION/DataComp weights (best generalizer)
    scores_datacomp = run_clip_model('ViT-L-14', 'datacomp_xl_s13b_b90k', test_ids, test_dir, device)
    
    # 2. ViT-L-14 with OpenAI weights (different training distribution)
    scores_openai = run_clip_model('ViT-L-14', 'openai', test_ids, test_dir, device)
    
    # 3. CoCa ViT-L-14 (Contrastive Captioners - state of the art for zero-shot)
    scores_coca = run_clip_model('coca_ViT-L-14', 'laion2b_s13b_b90k', test_ids, test_dir, device)
    
    # Final mega-ensemble
    print("\nFusing foundation models...")
    final_scores = (
        scores_datacomp * 1.0 + 
        scores_openai * 1.0 + 
        scores_coca * 1.0
    )
    
    final_scores = rank_normalize(final_scores)
    
    # Write submission
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_v6_ultimate.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_scores):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"\nSaved to {out_path}")
    print("Ready to submit!")

if __name__ == '__main__':
    main()
