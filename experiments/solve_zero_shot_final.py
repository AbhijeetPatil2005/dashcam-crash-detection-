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
    print("Loading physics heuristics...")
    motion_cache = os.path.join(CACHE_DIR, 'cache_motion_features.npz')
    data = np.load(motion_cache, allow_pickle=True)
    X_test_motion = data['X_test']
    motion_feats = data['feature_names'].tolist()
    
    # Load test IDs
    test_ids = []
    with open(os.path.join(DATA_DIR, 'sample_submission.csv'), 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_ids.append(row['clip_id'])
            
    # Physics scores (we know these work well from LB 0.638)
    score_diff = X_test_motion[:, motion_feats.index('diff_spike_ratio')]
    score_blur = X_test_motion[:, motion_feats.index('blur_drop_max')]
    score_flow = X_test_motion[:, motion_feats.index('flow_spike_ratio')]
    
    # Now let's do Zero-Shot Text-to-Image CLIP Classification!
    print("Running Zero-Shot CLIP Text Classification...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    
    # Text prompts
    text_normal = tokenizer(["a photo from a dashboard camera of a car driving normally on a road", "boring driving on a highway"]).to(device)
    text_crash = tokenizer(["a dashboard camera photo of a violent car crash", "car accident collision on the road"]).to(device)
    
    with torch.no_grad():
        feat_normal = F.normalize(model.encode_text(text_normal), dim=-1).mean(dim=0, keepdim=True)
        feat_crash = F.normalize(model.encode_text(text_crash), dim=-1).mean(dim=0, keepdim=True)
        
    score_clip = []
    
    test_dir = os.path.join(DATA_DIR, 'test')
    
    for clip_id in tqdm(test_ids):
        clip_dir = os.path.join(test_dir, clip_id)
        # Check frames 15, 20, 25, 29 (crashes usually happen later in the clip)
        max_crash_sim = -100
        min_normal_sim = 100
        
        for idx in [15, 20, 25, 29]:
            frame_path = os.path.join(clip_dir, f'frame_{idx:03d}.jpg')
            try:
                img = Image.open(frame_path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)
            except:
                continue
                
            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    img_feat = F.normalize(model.encode_image(img_tensor), dim=-1)
                    sim_crash = (img_feat @ feat_crash.T).item()
                    sim_normal = (img_feat @ feat_normal.T).item()
                    
            max_crash_sim = max(max_crash_sim, sim_crash)
            min_normal_sim = min(min_normal_sim, sim_normal)
            
        # CLIP score is the difference between how much it looks like a crash vs normal
        score_clip.append(max_crash_sim - min_normal_sim)
        
    score_clip = np.array(score_clip)
    
    # Combine EVERYTHING
    final_scores = (
        rank_normalize(score_diff) * 1.0 + 
        rank_normalize(score_blur) * 1.0 + 
        rank_normalize(score_flow) * 1.0 +
        rank_normalize(score_clip) * 1.5  # Give CLIP text zero-shot slightly more weight
    ) / 4.5
    
    out_path = os.path.join(SUBMISSIONS_DIR, 'submission_zero_shot_final.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['clip_id', 'label'])
        for cid, score in zip(test_ids, final_scores):
            writer.writerow([cid, f'{score:.6f}'])
            
    print(f"Saved to {out_path}")

if __name__ == '__main__':
    main()
