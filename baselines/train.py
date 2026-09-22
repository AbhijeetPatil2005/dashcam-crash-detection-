"""
Training script for crash detection models.

Supports:
- GroupKFold cross-validation
- Mixed precision training (AMP)
- Early stopping on validation AUC
- Gradient accumulation for effective larger batch sizes
- Label smoothing
"""

import os
import sys
import time
import argparse
import json
import numpy as np
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from dataset import (
    CrashClipDataset, MixupDataset,
    load_train_labels, get_train_augmentation, get_val_augmentation
)
from models import get_model


def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


class LabelSmoothingBCE(nn.Module):
    """Binary cross-entropy with label smoothing."""
    def __init__(self, smoothing=0.05):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(logits.squeeze(-1), targets)


def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, 
                    accumulation_steps=1, use_diff=False):
    """Train for one epoch with gradient accumulation and mixed precision."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(dataloader):
        if use_diff:
            frames, diff_feat, labels = batch
            frames = frames.to(device, non_blocking=True)
            diff_feat = diff_feat.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            frames, labels = batch
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            if use_diff:
                logits = model(frames, diff_feat)
            else:
                logits = model(frames)
            loss = criterion(logits, labels)
            loss = loss / accumulation_steps
        
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accumulation_steps
        num_batches += 1
        
        if (batch_idx + 1) % 20 == 0:
            print(f"  Batch {batch_idx+1}/{len(dataloader)}, Loss: {total_loss/num_batches:.4f}")
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, criterion, device, use_diff=False):
    """Validate and compute AUC."""
    model.eval()
    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        if use_diff:
            frames, diff_feat, labels = batch
            frames = frames.to(device, non_blocking=True)
            diff_feat = diff_feat.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            frames, labels = batch
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            if use_diff:
                logits = model(frames, diff_feat)
            else:
                logits = model(frames)
            loss = criterion(logits, labels)
        
        probs = torch.sigmoid(logits.squeeze(-1))
        all_preds.extend(probs.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # Compute AUC
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.5
    
    return avg_loss, auc, np.array(all_preds)


def train_fold(fold, train_samples, val_samples, data_dir, model_type, 
               save_dir, config):
    """Train a single fold."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Training Fold {fold} on {device}")
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}")
    print(f"{'='*60}")
    
    use_diff = model_type == 'gru_diff'
    
    # Create datasets
    train_transform = get_train_augmentation()
    val_transform = get_val_augmentation()
    
    train_dataset = CrashClipDataset(
        data_dir=data_dir,
        samples=train_samples,
        num_frames=config['num_frames'],
        transform=train_transform,
        use_diff=use_diff,
    )
    
    val_dataset = CrashClipDataset(
        data_dir=data_dir,
        samples=val_samples,
        num_frames=config['num_frames'],
        transform=val_transform,
        use_diff=use_diff,
    )
    
    # Apply mixup to training
    if config.get('use_mixup', True):
        train_dataset = MixupDataset(train_dataset, alpha=0.4, p=0.3)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if config.get('num_workers', 2) > 0 else False,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=True,
        persistent_workers=True if config.get('num_workers', 2) > 0 else False,
    )
    
    # Create model
    model_kwargs = {
        'backbone_name': config.get('backbone', 'efficientnet_b3'),
        'num_frames': config['num_frames'],
        'dropout': config.get('dropout', 0.4),
        'pretrained': True,
    }
    if model_type in ['gru', 'gru_diff']:
        model_kwargs['hidden_dim'] = config.get('hidden_dim', 256)
    
    model = get_model(model_type, **model_kwargs)
    model = model.to(device)
    
    # Optimizer 
    # Use different learning rates for backbone vs head
    backbone_params = list(model.backbone.parameters()) if hasattr(model, 'backbone') else []
    if hasattr(model, 'frame_model'):
        backbone_params = list(model.frame_model.backbone.parameters())
    
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['lr'] * 0.1},  # Lower LR for pretrained backbone
        {'params': head_params, 'lr': config['lr']},
    ], weight_decay=config.get('weight_decay', 0.01))
    
    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'], eta_min=1e-7
    )
    
    criterion = LabelSmoothingBCE(smoothing=config.get('label_smoothing', 0.05))
    scaler = GradScaler('cuda')
    
    # Training loop
    best_auc = 0
    patience_counter = 0
    accumulation_steps = config.get('accumulation_steps', 2)
    
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        start_time = time.time()
        
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            accumulation_steps=accumulation_steps, use_diff=use_diff
        )
        
        val_loss, val_auc, val_preds = validate(
            model, val_loader, criterion, device, use_diff=use_diff
        )
        
        scheduler.step()
        
        elapsed = time.time() - start_time
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Val AUC: {val_auc:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s")
        
        # Save best model
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            save_path = os.path.join(save_dir, f'{model_type}_fold{fold}_best.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_auc': val_auc,
                'epoch': epoch,
                'config': config,
            }, save_path)
            print(f"  >> New best AUC: {val_auc:.4f}, saved to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= config.get('patience', 5):
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    print(f"\nFold {fold} best AUC: {best_auc:.4f}")
    
    # Cleanup
    del model, optimizer, scaler
    torch.cuda.empty_cache()
    
    return best_auc


def main():
    parser = argparse.ArgumentParser(description='Train crash detection model')
    parser.add_argument('--data-dir', type=str, default='data/crash_competition_data/train',
                        help='Path to training data directory')
    parser.add_argument('--labels-csv', type=str, default='data/crash_competition_data/train_labels.csv',
                        help='Path to train_labels.csv')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--model-type', type=str, default='gru',
                        choices=['gru', 'attention', 'gru_diff'],
                        help='Model architecture')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3',
                        help='Backbone model name from timm')
    parser.add_argument('--num-frames', type=int, default=8,
                        help='Number of frames to sample per clip')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=15,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate for head (backbone gets 0.1x)')
    parser.add_argument('--num-folds', type=int, default=5,
                        help='Number of CV folds')
    parser.add_argument('--fold', type=int, default=-1,
                        help='Specific fold to train (-1 for all)')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='DataLoader workers')
    parser.add_argument('--accumulation-steps', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    set_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load labels
    samples = load_train_labels(args.labels_csv)
    print(f"Loaded {len(samples)} training samples")
    
    # GroupKFold split
    clip_ids = [s['clip_id'] for s in samples]
    labels = np.array([s['label'] for s in samples])
    groups = np.array([s['group_id'] for s in samples])
    
    print(f"Unique groups: {len(np.unique(groups))}")
    print(f"Label distribution: {np.bincount(labels)}")
    
    config = {
        'backbone': args.backbone,
        'num_frames': args.num_frames,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'lr': args.lr,
        'dropout': 0.4,
        'hidden_dim': 256,
        'label_smoothing': 0.05,
        'weight_decay': 0.01,
        'patience': 5,
        'use_mixup': True,
        'accumulation_steps': args.accumulation_steps,
        'num_workers': args.num_workers,
    }
    
    # Save config
    config_path = os.path.join(args.save_dir, f'{args.model_type}_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    gkf = GroupKFold(n_splits=args.num_folds)
    fold_aucs = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(clip_ids, labels, groups)):
        if args.fold >= 0 and fold != args.fold:
            continue
        
        train_samples = [samples[i] for i in train_idx]
        val_samples = [samples[i] for i in val_idx]
        
        fold_auc = train_fold(
            fold=fold,
            train_samples=train_samples,
            val_samples=val_samples,
            data_dir=args.data_dir,
            model_type=args.model_type,
            save_dir=args.save_dir,
            config=config,
        )
        fold_aucs.append(fold_auc)
    
    if fold_aucs:
        mean_auc = np.mean(fold_aucs)
        std_auc = np.std(fold_aucs)
        print(f"\n{'='*60}")
        print(f"Cross-validation Results ({args.model_type})")
        print(f"{'='*60}")
        for i, auc in enumerate(fold_aucs):
            print(f"  Fold {i}: AUC = {auc:.4f}")
        print(f"  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"{'='*60}")
        
        # Save results
        results_path = os.path.join(args.save_dir, f'{args.model_type}_cv_results.json')
        with open(results_path, 'w') as f:
            json.dump({
                'fold_aucs': fold_aucs,
                'mean_auc': float(mean_auc),
                'std_auc': float(std_auc),
            }, f, indent=2)


if __name__ == '__main__':
    main()
