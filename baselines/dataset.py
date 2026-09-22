"""
Dataset classes for crash detection from dashcam clips.
Handles frame loading, sampling, augmentation, and GroupKFold splitting.
"""

import os
import csv
import random
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_train_labels(csv_path):
    """Load train_labels.csv -> list of dicts with clip_id, label, group_id."""
    samples = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append({
                'clip_id': row['clip_id'],
                'label': int(row['label']),
                'group_id': row['group_id'],
            })
    return samples


def load_test_ids(csv_path):
    """Load sample_submission.csv -> list of clip_ids."""
    clip_ids = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            clip_ids.append(row['clip_id'])
    return clip_ids


def get_train_augmentation(img_size=(224, 384)):
    """Strong augmentation for training — domain-robust."""
    return A.Compose([
        A.RandomResizedCrop(size=(img_size[0], img_size[1]), scale=(0.8, 1.0), ratio=(1.5, 1.9), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        ], p=0.8),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
        ], p=0.3),
        A.GaussNoise(p=0.2),
        A.ToGray(p=0.1),
        A.ImageCompression(quality_range=(50, 95), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 3), hole_height_range=(20, 60), hole_width_range=(20, 60), p=0.2),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_augmentation(img_size=(224, 384)):
    """Minimal augmentation for validation."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_tta_augmentation(img_size=(224, 384)):
    """Test-time augmentation — just horizontal flip."""
    return A.Compose([
        A.HorizontalFlip(p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class CrashClipDataset(Dataset):
    """
    Dataset for loading dashcam clips for crash detection.
    
    Samples N frames from each 30-frame clip for temporal analysis.
    Supports both regular frames and frame differences.
    """
    
    def __init__(self, data_dir, samples, num_frames=8, transform=None, 
                 is_test=False, use_diff=False):
        """
        Args:
            data_dir: path to train/ or test/ directory
            samples: list of dicts with 'clip_id' (and 'label', 'group_id' for train)
            num_frames: number of frames to sample from the 30
            transform: albumentations transform
            is_test: if True, samples have no label
            use_diff: if True, also compute frame differences
        """
        self.data_dir = data_dir
        self.samples = samples
        self.num_frames = num_frames
        self.transform = transform
        self.is_test = is_test
        self.use_diff = use_diff
        
        # Precompute frame indices to sample
        self.total_frames = 30
        
    def _get_frame_indices(self):
        """Get evenly spaced frame indices."""
        indices = np.linspace(0, self.total_frames - 1, self.num_frames, dtype=int)
        return indices.tolist()
    
    def _load_frame(self, clip_dir, frame_idx):
        """Load a single frame as numpy array."""
        frame_path = os.path.join(clip_dir, f'frame_{frame_idx:03d}.jpg')
        img = cv2.imread(frame_path)
        if img is None:
            # Fallback: create black frame
            img = np.zeros((224, 384, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        clip_id = sample['clip_id'] if isinstance(sample, dict) else sample
        clip_dir = os.path.join(self.data_dir, clip_id)
        
        frame_indices = self._get_frame_indices()
        
        frames = []
        for fi in frame_indices:
            img = self._load_frame(clip_dir, fi)
            if self.transform:
                augmented = self.transform(image=img)
                img_tensor = augmented['image']
            else:
                img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            frames.append(img_tensor)
        
        # Stack: (num_frames, C, H, W)
        frames_tensor = torch.stack(frames, dim=0)
        
        diff_tensor = None
        if self.use_diff:
            # Compute frame differences for motion detection
            # Load all 30 frames at lower res for efficiency
            diffs = self._compute_differences(clip_dir)
            diff_tensor = diffs
        
        if self.is_test:
            if self.use_diff and diff_tensor is not None:
                return clip_id, frames_tensor, diff_tensor
            return clip_id, frames_tensor
        
        label = sample['label']
        if self.use_diff and diff_tensor is not None:
            return frames_tensor, diff_tensor, torch.tensor(label, dtype=torch.float32)
        return frames_tensor, torch.tensor(label, dtype=torch.float32)
    
    def _compute_differences(self, clip_dir):
        """
        Compute temporal difference features from all 30 frames.
        Returns a tensor of aggregated difference statistics.
        """
        # Load frames at reduced resolution for speed
        prev_gray = None
        diff_magnitudes = []
        
        for i in range(self.total_frames):
            frame_path = os.path.join(clip_dir, f'frame_{i:03d}.jpg')
            img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((224, 384), dtype=np.uint8)
            
            # Resize for speed
            img = cv2.resize(img, (192, 112))
            
            if prev_gray is not None:
                diff = cv2.absdiff(img, prev_gray)
                diff_magnitudes.append(diff.mean())
            
            prev_gray = img
        
        # Create feature vector from temporal differences
        diff_array = np.array(diff_magnitudes, dtype=np.float32)
        
        # Statistics that capture crash dynamics
        features = np.array([
            diff_array.mean(),
            diff_array.std(),
            diff_array.max(),
            np.argmax(diff_array) / len(diff_array),  # When does max diff occur (normalized)
            np.percentile(diff_array, 90),
            np.percentile(diff_array, 95),
            # Ratio of max to mean — spike detection
            diff_array.max() / (diff_array.mean() + 1e-8),
            # Second half vs first half — crash usually in second half
            diff_array[len(diff_array)//2:].mean() / (diff_array[:len(diff_array)//2].mean() + 1e-8),
            # Rate of change of differences
            np.abs(np.diff(diff_array)).mean(),
            np.abs(np.diff(diff_array)).max(),
        ], dtype=np.float32)
        
        return torch.from_numpy(features)


class MixupDataset(Dataset):
    """Wrapper that applies Mixup augmentation."""
    
    def __init__(self, dataset, alpha=0.4, p=0.5):
        self.dataset = dataset
        self.alpha = alpha
        self.p = p
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        result = self.dataset[idx]
        
        if random.random() > self.p:
            return result
        
        # Get another random sample
        idx2 = random.randint(0, len(self.dataset) - 1)
        result2 = self.dataset[idx2]
        
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Mixup frames and labels
        if len(result) == 2:  # frames, label
            frames1, label1 = result
            frames2, label2 = result2
            mixed_frames = lam * frames1 + (1 - lam) * frames2
            mixed_label = lam * label1 + (1 - lam) * label2
            return mixed_frames, mixed_label
        else:  # frames, diff, label
            frames1, diff1, label1 = result
            frames2, diff2, label2 = result2
            mixed_frames = lam * frames1 + (1 - lam) * frames2
            mixed_diff = lam * diff1 + (1 - lam) * diff2
            mixed_label = lam * label1 + (1 - lam) * label2
            return mixed_frames, mixed_diff, mixed_label
