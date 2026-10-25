#!/usr/bin/env python3
"""
Segmentation Model with ViT Encoder

Uses pretrained ViT-Small encoder with a lightweight decoder for LV segmentation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add IJEPA path
IJEPA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)
from src.models.vision_transformer import vit_small


class SegmentationDecoder(nn.Module):
    """
    Lightweight decoder for segmentation.
    Takes patch embeddings from ViT and upsamples to full resolution mask.
    """
    def __init__(self, embed_dim=384, patch_size=16, img_size=224, num_classes=1):
        super().__init__()
        
        self.patch_size = patch_size
        self.img_size = img_size
        self.num_patches_side = img_size // patch_size  # 14 for 224/16
        
        # Project embeddings
        self.proj = nn.Linear(embed_dim, 256)
        
        # Upsample blocks (14 -> 28 -> 56 -> 112 -> 224)
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.GELU(),
        )
        
        # Final conv
        self.head = nn.Conv2d(16, num_classes, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: [B, N, D] patch embeddings from ViT (N=196 for 14x14 grid)
        Returns:
            [B, num_classes, H, W] segmentation logits
        """
        B, N, D = x.shape
        
        # Project
        x = self.proj(x)  # [B, N, 256]
        
        # Reshape to spatial grid
        x = x.transpose(1, 2)  # [B, 256, N]
        x = x.reshape(B, 256, self.num_patches_side, self.num_patches_side)  # [B, 256, 14, 14]
        
        # Upsample
        x = self.up1(x)  # [B, 128, 28, 28]
        x = self.up2(x)  # [B, 64, 56, 56]
        x = self.up3(x)  # [B, 32, 112, 112]
        x = self.up4(x)  # [B, 16, 224, 224]
        
        x = self.head(x)  # [B, 1, 224, 224]
        
        return x


class ViTSegmentation(nn.Module):
    """
    ViT-based segmentation model.
    
    Uses pretrained ViT-Small encoder with segmentation decoder.
    """
    def __init__(self, encoder_checkpoint=None, freeze_encoder=False, img_size=224):
        super().__init__()
        
        # Encoder
        self.encoder = vit_small()
        self.embed_dim = self.encoder.embed_dim  # 384
        
        # Load pretrained weights if provided
        if encoder_checkpoint is not None:
            self._load_encoder(encoder_checkpoint)
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Decoder
        self.decoder = SegmentationDecoder(
            embed_dim=self.embed_dim,
            patch_size=16,
            img_size=img_size,
            num_classes=1
        )
    
    def _load_encoder(self, checkpoint_path):
        """Load encoder weights from checkpoint."""
        state = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'encoder' in state:
            encoder_state = state['encoder']
        elif 'target_encoder' in state:
            encoder_state = state['target_encoder']
        else:
            encoder_state = state
        
        # Load weights
        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"Loaded encoder: {len(encoder_state)} keys, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
    
    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] input images
        Returns:
            [B, 1, H, W] segmentation logits
        """
        # Encode
        features = self.encoder(x)  # [B, 196, 384]
        
        # Decode
        logits = self.decoder(features)  # [B, 1, 224, 224]
        
        return logits
    
    def get_encoder_state(self):
        """Return encoder state dict for saving."""
        return self.encoder.state_dict()


def dice_loss(pred, target, smooth=1e-6):
    """
    Compute Dice loss.
    
    Args:
        pred: [B, 1, H, W] predicted logits
        target: [B, 1, H, W] ground truth masks (0 or 1)
    """
    pred = torch.sigmoid(pred)
    
    # Flatten
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    
    intersection = (pred_flat * target_flat).sum()
    dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
    
    return 1 - dice


def dice_score(pred, target, threshold=0.5, smooth=1e-6):
    """
    Compute Dice score (for evaluation).
    
    Args:
        pred: [B, 1, H, W] predicted logits
        target: [B, 1, H, W] ground truth masks
    Returns:
        float: mean Dice score
    """
    pred = (torch.sigmoid(pred) > threshold).float()
    
    # Per-sample dice
    batch_size = pred.shape[0]
    dice_scores = []
    
    for i in range(batch_size):
        p = pred[i].view(-1)
        t = target[i].view(-1)
        intersection = (p * t).sum()
        dice = (2. * intersection + smooth) / (p.sum() + t.sum() + smooth)
        dice_scores.append(dice.item())
    
    return sum(dice_scores) / len(dice_scores)


class CombinedLoss(nn.Module):
    """Combined BCE + Dice loss for segmentation."""
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        d_loss = dice_loss(pred, target)
        return self.bce_weight * bce_loss + self.dice_weight * d_loss
