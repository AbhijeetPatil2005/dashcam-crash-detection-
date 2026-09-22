"""
Model architectures for crash detection.

Model A: EfficientNet-B3 + GRU temporal model
Model B: EfficientNet-B3 + Temporal Attention (alternative head)
Both designed to fit in 6GB VRAM with mixed precision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class TemporalAttention(nn.Module):
    """Self-attention over temporal dimension to find crash-relevant frames."""
    
    def __init__(self, feat_dim):
        super().__init__()
        self.query = nn.Linear(feat_dim, feat_dim // 4)
        self.key = nn.Linear(feat_dim, feat_dim // 4)
        self.value = nn.Linear(feat_dim, feat_dim)
        self.scale = (feat_dim // 4) ** -0.5
    
    def forward(self, x):
        """x: (B, T, D) -> (B, D)"""
        q = self.query(x)  # (B, T, D//4)
        k = self.key(x)    # (B, T, D//4)
        v = self.value(x)  # (B, T, D)
        
        attn = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, T, T)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v)  # (B, T, D)
        # Pool over time
        out = out.mean(dim=1)  # (B, D)
        return out


class CrashDetectorGRU(nn.Module):
    """
    Primary model: EfficientNet-B3 frame encoder + GRU temporal head.
    
    Processes N frames through shared CNN, then GRU captures temporal patterns.
    Crashes produce distinctive temporal signatures (sudden visual changes).
    """
    
    def __init__(self, backbone_name='efficientnet_b3', num_frames=8, 
                 hidden_dim=256, dropout=0.4, pretrained=True):
        super().__init__()
        
        self.num_frames = num_frames
        
        # Backbone: EfficientNet-B3 (pretrained)
        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head, get features
            global_pool='avg'
        )
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 384)
            feat_dim = self.backbone(dummy).shape[-1]
        
        self.feat_dim = feat_dim
        
        # Temporal model: bidirectional GRU
        self.gru = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Temporal attention for weighted frame pooling
        self.temporal_attn = TemporalAttention(hidden_dim * 2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, T, C, H, W) - batch of frame sequences
        Returns:
            logits: (B, 1)
        """
        B, T, C, H, W = x.shape
        
        # Reshape to process all frames through backbone at once
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)  # (B*T, feat_dim)
        features = features.view(B, T, -1)  # (B, T, feat_dim)
        
        # Temporal modeling
        gru_out, _ = self.gru(features)  # (B, T, hidden*2)
        
        # Attention-weighted pooling
        temporal_feat = self.temporal_attn(gru_out)  # (B, hidden*2)
        
        # Classify
        logits = self.classifier(temporal_feat)  # (B, 1)
        return logits


class CrashDetectorAttention(nn.Module):
    """
    Alternative model: EfficientNet + pure attention-based temporal head.
    
    Uses self-attention instead of GRU for temporal modeling.
    Sometimes better at finding the exact crash frame.
    """
    
    def __init__(self, backbone_name='efficientnet_b3', num_frames=8,
                 dropout=0.4, pretrained=True):
        super().__init__()
        
        self.num_frames = num_frames
        
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained, 
            num_classes=0,
            global_pool='avg'
        )
        
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 384)
            feat_dim = self.backbone(dummy).shape[-1]
        
        self.feat_dim = feat_dim
        
        # Positional encoding for frame order
        self.pos_embed = nn.Parameter(torch.randn(1, num_frames, feat_dim) * 0.02)
        
        # Transformer encoder for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feat_dim,
            nhead=8,
            dim_feedforward=feat_dim * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.temporal_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        B, T, C, H, W = x.shape
        
        x = x.view(B * T, C, H, W)
        features = self.backbone(x)
        features = features.view(B, T, -1)
        
        # Add positional encoding
        features = features + self.pos_embed[:, :T, :]
        
        # Temporal self-attention
        temporal_out = self.temporal_encoder(features)  # (B, T, D)
        
        # Global average pool over time
        pooled = temporal_out.mean(dim=1)  # (B, D)
        
        logits = self.classifier(pooled)
        return logits


class CrashDetectorWithDiff(nn.Module):
    """
    Enhanced model that combines frame features with motion/difference statistics.
    
    The diff features capture temporal dynamics (sudden changes = crash)
    and are inherently domain-robust since they measure relative motion.
    """
    
    def __init__(self, backbone_name='efficientnet_b3', num_frames=8,
                 diff_feat_dim=10, hidden_dim=256, dropout=0.4, pretrained=True):
        super().__init__()
        
        # Frame-level model
        self.frame_model = CrashDetectorGRU(
            backbone_name=backbone_name,
            num_frames=num_frames,
            hidden_dim=hidden_dim,
            dropout=dropout,
            pretrained=pretrained
        )
        
        # Replace the last linear layer to concatenate with diff features
        # Remove the classifier from frame_model
        old_classifier = self.frame_model.classifier
        self.frame_model.classifier = nn.Identity()
        
        # Combined classifier  
        gru_out_dim = hidden_dim * 2
        
        # Diff feature processor
        self.diff_processor = nn.Sequential(
            nn.Linear(diff_feat_dim, 32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.GELU(),
        )
        
        # Combined head
        combined_dim = gru_out_dim + 32
        self.classifier = nn.Sequential(
            nn.LayerNorm(combined_dim),
            nn.Dropout(dropout),
            nn.Linear(combined_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, 1),
        )
    
    def forward(self, frames, diff_features):
        """
        Args:
            frames: (B, T, C, H, W)
            diff_features: (B, diff_feat_dim) 
        """
        # Get temporal features from frames
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        features = self.frame_model.backbone(x)
        features = features.view(B, T, -1)
        
        gru_out, _ = self.frame_model.gru(features)
        temporal_feat = self.frame_model.temporal_attn(gru_out)
        
        # Process diff features
        diff_feat = self.diff_processor(diff_features)
        
        # Combine
        combined = torch.cat([temporal_feat, diff_feat], dim=-1)
        logits = self.classifier(combined)
        return logits


def get_model(model_type='gru', **kwargs):
    """Model factory function."""
    if model_type == 'gru':
        return CrashDetectorGRU(**kwargs)
    elif model_type == 'attention':
        return CrashDetectorAttention(**kwargs)
    elif model_type == 'gru_diff':
        return CrashDetectorWithDiff(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
