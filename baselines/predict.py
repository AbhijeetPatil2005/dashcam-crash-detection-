"""
Inference and submission generation for crash detection.

Supports:
- Multi-fold ensemble
- Test-Time Augmentation (TTA)
- Multi-model ensemble
- Submission file generation
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.amp import autocast

from dataset import (
    CrashClipDataset, load_test_ids,
    get_val_augmentation, get_tta_augmentation
)
from models import get_model


@torch.no_grad()
def predict_single_model(model, dataloader, device, use_diff=False):
    """Generate predictions for a single model."""
    model.eval()
    all_clip_ids = []
    all_preds = []
    
    for batch in tqdm(dataloader, desc="Predicting"):
        if use_diff:
            clip_ids, frames, diff_feat = batch
            frames = frames.to(device, non_blocking=True)
            diff_feat = diff_feat.to(device, non_blocking=True)
        else:
            clip_ids, frames = batch
            frames = frames.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            if use_diff:
                logits = model(frames, diff_feat)
            else:
                logits = model(frames)
        
        probs = torch.sigmoid(logits.squeeze(-1))
        all_clip_ids.extend(clip_ids)
        all_preds.extend(probs.cpu().numpy().tolist())
    
    return all_clip_ids, np.array(all_preds)


def predict_with_tta(model, test_dir, test_samples, config, device, use_diff=False):
    """Predict with Test-Time Augmentation (horizontal flip)."""
    num_frames = config.get('num_frames', 8)
    batch_size = config.get('batch_size', 4)
    num_workers = config.get('num_workers', 2)
    
    # Normal predictions
    normal_dataset = CrashClipDataset(
        data_dir=test_dir,
        samples=test_samples,
        num_frames=num_frames,
        transform=get_val_augmentation(),
        is_test=True,
        use_diff=use_diff,
    )
    normal_loader = DataLoader(
        normal_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    clip_ids, normal_preds = predict_single_model(model, normal_loader, device, use_diff)
    
    # TTA: horizontal flip
    tta_dataset = CrashClipDataset(
        data_dir=test_dir,
        samples=test_samples,
        num_frames=num_frames,
        transform=get_tta_augmentation(),
        is_test=True,
        use_diff=use_diff,
    )
    tta_loader = DataLoader(
        tta_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    _, tta_preds = predict_single_model(model, tta_loader, device, use_diff)
    
    # Average normal and TTA predictions
    avg_preds = (normal_preds + tta_preds) / 2.0
    
    return clip_ids, avg_preds


def ensemble_predict(model_configs, test_dir, test_samples, device):
    """
    Ensemble predictions from multiple models/folds.
    
    Args:
        model_configs: list of dicts with 'checkpoint', 'model_type', 'config'
        test_dir: path to test data
        test_samples: list of test clip dicts
        device: torch device
    
    Returns:
        clip_ids, averaged predictions
    """
    all_predictions = []
    clip_ids = None
    
    for mc in model_configs:
        print(f"\nLoading model: {mc['checkpoint']}")
        
        # Load config
        config = mc['config']
        model_type = mc['model_type']
        use_diff = model_type == 'gru_diff'
        
        # Create model
        model_kwargs = {
            'backbone_name': config.get('backbone', 'efficientnet_b3'),
            'num_frames': config.get('num_frames', 8),
            'dropout': config.get('dropout', 0.4),
            'pretrained': False,  # We're loading weights
        }
        if model_type in ['gru', 'gru_diff']:
            model_kwargs['hidden_dim'] = config.get('hidden_dim', 256)
        
        model = get_model(model_type, **model_kwargs)
        
        # Load checkpoint
        ckpt = torch.load(mc['checkpoint'], map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)
        
        print(f"  Loaded checkpoint (val AUC: {ckpt.get('val_auc', 'N/A')})")
        
        # Predict with TTA
        ids, preds = predict_with_tta(
            model, test_dir, test_samples, config, device, use_diff
        )
        
        if clip_ids is None:
            clip_ids = ids
        
        all_predictions.append(preds)
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
    
    # Average all predictions (rank-averaging would be better for AUC but simple avg works well)
    ensemble_preds = np.mean(all_predictions, axis=0)
    
    return clip_ids, ensemble_preds


def rank_average(predictions_list):
    """
    Rank-average multiple prediction arrays.
    Better than simple averaging for ROC-AUC since it preserves ranking.
    """
    ranked = []
    for preds in predictions_list:
        # Convert to ranks (0 to 1)
        ranks = np.argsort(np.argsort(preds)).astype(float) / len(preds)
        ranked.append(ranks)
    
    return np.mean(ranked, axis=0)


def write_submission(clip_ids, predictions, output_path):
    """Write submission CSV file."""
    with open(output_path, 'w') as f:
        f.write("clip_id,label\n")
        for clip_id, pred in zip(clip_ids, predictions):
            # Clip to [0, 1] range
            pred = float(np.clip(pred, 0.0, 1.0))
            f.write(f"{clip_id},{pred:.6f}\n")
    print(f"Submission saved to {output_path} ({len(clip_ids)} clips)")


def main():
    parser = argparse.ArgumentParser(description='Generate crash detection predictions')
    parser.add_argument('--test-dir', type=str, default='data/crash_competition_data/test',
                        help='Path to test data directory')
    parser.add_argument('--sample-csv', type=str, default='data/crash_competition_data/sample_submission.csv',
                        help='Path to sample_submission.csv')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--model-type', type=str, default='gru',
                        help='Model type(s), comma-separated for ensemble')
    parser.add_argument('--output', type=str, default='submissions/submission_v1_effnet_gru.csv',
                        help='Output submission file path')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--use-rank-avg', action='store_true',
                        help='Use rank averaging instead of simple averaging')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load test clip IDs
    test_clip_ids = load_test_ids(args.sample_csv)
    test_samples = [{'clip_id': cid} for cid in test_clip_ids]
    print(f"Test clips: {len(test_samples)}")
    
    # Find all checkpoints
    model_types = args.model_type.split(',')
    model_configs = []
    
    for mt in model_types:
        mt = mt.strip()
        config_path = os.path.join(args.checkpoint_dir, f'{mt}_config.json')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            config = {
                'backbone': 'efficientnet_b3',
                'num_frames': 8,
                'batch_size': args.batch_size,
                'num_workers': args.num_workers,
                'dropout': 0.4,
                'hidden_dim': 256,
            }
        
        config['batch_size'] = args.batch_size
        config['num_workers'] = args.num_workers
        
        # Find fold checkpoints
        for fold in range(10):  # Check up to 10 folds
            ckpt_path = os.path.join(args.checkpoint_dir, f'{mt}_fold{fold}_best.pth')
            if os.path.exists(ckpt_path):
                model_configs.append({
                    'checkpoint': ckpt_path,
                    'model_type': mt,
                    'config': config,
                })
    
    print(f"Found {len(model_configs)} checkpoints to ensemble")
    
    if not model_configs:
        print("ERROR: No checkpoints found!")
        return
    
    # Generate predictions
    if args.use_rank_avg:
        # Predict each model separately, then rank-average
        all_predictions = []
        clip_ids = None
        
        for mc in model_configs:
            config = mc['config']
            model_type = mc['model_type']
            use_diff = model_type == 'gru_diff'
            
            model_kwargs = {
                'backbone_name': config.get('backbone', 'efficientnet_b3'),
                'num_frames': config.get('num_frames', 8),
                'dropout': config.get('dropout', 0.4),
                'pretrained': False,
            }
            if model_type in ['gru', 'gru_diff']:
                model_kwargs['hidden_dim'] = config.get('hidden_dim', 256)
            
            model = get_model(model_type, **model_kwargs)
            ckpt = torch.load(mc['checkpoint'], map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            model = model.to(device)
            
            ids, preds = predict_with_tta(
                model, args.test_dir, test_samples, config, device, use_diff
            )
            
            if clip_ids is None:
                clip_ids = ids
            all_predictions.append(preds)
            
            del model
            torch.cuda.empty_cache()
        
        final_preds = rank_average(all_predictions)
    else:
        clip_ids, final_preds = ensemble_predict(
            model_configs, args.test_dir, test_samples, device
        )
    
    # Write submission
    write_submission(clip_ids, final_preds, args.output)
    
    # Print statistics
    print(f"\nPrediction statistics:")
    print(f"  Mean: {final_preds.mean():.4f}")
    print(f"  Std:  {final_preds.std():.4f}")
    print(f"  Min:  {final_preds.min():.4f}")
    print(f"  Max:  {final_preds.max():.4f}")
    print(f"  Median: {np.median(final_preds):.4f}")
    
    # Distribution of predictions
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        count = (final_preds > threshold).sum()
        print(f"  P > {threshold}: {count} ({count/len(final_preds)*100:.1f}%)")


if __name__ == '__main__':
    main()
