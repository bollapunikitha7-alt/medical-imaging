#!/usr/bin/env python3
"""
MAE (Masked Autoencoder) Pretraining on TB CXR Dataset - v2

Uses vit_small() from JEPA codebase as encoder for EXACT architecture match.

Architecture:
- Encoder: vit_small (embed_dim=384, depth=12, num_heads=6, mlp_ratio=4.0)
- Decoder: Lightweight 4-layer transformer

This ensures identical encoder weights can be compared with JEPA.
"""

import os
import sys
import argparse
import json
import logging
import random
import math
from datetime import datetime
import time
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Add IJEPA_Meta path for JEPA modules
IJEPA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)
from src.models.vision_transformer import vit_small

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class TBCXRDataset(Dataset):
    """TB Chest Radiography Database for pretraining."""
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.image_paths = []
        
        for folder_name in ['Normal', 'Tuberculosis']:
            folder = os.path.join(root_path, folder_name)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith(self.SUPPORTED_EXTENSIONS):
                        self.image_paths.append(os.path.join(folder, f))
        
        logger.info(f"Loaded {len(self.image_paths)} images for MAE pretraining")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img


# ============================================================================
# Position Embeddings (for decoder)
# ============================================================================

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """Generate 2D sincos position embeddings."""
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size, grid_size])
    
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega
    
    pos = pos.reshape(-1)
    out = np.einsum('m,d->md', pos, omega)
    
    emb_sin = np.sin(out)
    emb_cos = np.cos(out)
    emb = np.concatenate([emb_sin, emb_cos], axis=1)
    return emb


# ============================================================================
# Decoder Components
# ============================================================================

class DecoderBlock(nn.Module):
    """Transformer block for decoder."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop)
        )
    
    def forward(self, x):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class MAEDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction."""
    def __init__(self, num_patches, patch_size=16, in_chans=3,
                 encoder_embed_dim=384, decoder_embed_dim=192, 
                 decoder_depth=4, decoder_num_heads=6):
        super().__init__()
        
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.decoder_embed_dim = decoder_embed_dim
        
        # Project encoder dim to decoder dim
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Position embeddings for decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio=4., qkv_bias=True)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Predict pixels
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans)
        
        self._init_weights()
    
    def _init_weights(self):
        # Position embeddings
        pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(self.num_patches**0.5), cls_token=False
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, ids_restore):
        """
        Args:
            x: (B, N_visible, encoder_embed_dim) - visible patch embeddings
            ids_restore: (B, N) - indices to restore original order
        Returns:
            pred: (B, N, patch_size**2 * 3) - predicted pixel values
        """
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        B, N_visible, D = x.shape
        N = self.num_patches
        
        mask_tokens = self.mask_token.repeat(B, N - N_visible, 1)
        x_ = torch.cat([x, mask_tokens], dim=1)
        
        # Unshuffle to original order
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # Add position embeddings
        x = x + self.decoder_pos_embed
        
        # Decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        
        x = self.decoder_norm(x)
        
        # Predict pixels
        x = self.decoder_pred(x)
        return x


# ============================================================================
# MAE Model (using vit_small encoder)
# ============================================================================

class MAE(nn.Module):
    """
    Masked Autoencoder with vit_small encoder from JEPA codebase.
    
    This ensures EXACT architecture match with JEPA for fair comparison.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=6,
                 mask_ratio=0.75, norm_pix_loss=True):
        super().__init__()
        
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.norm_pix_loss = norm_pix_loss
        
        # Encoder: USE EXACT vit_small from JEPA
        self.encoder = vit_small(patch_size=patch_size, drop_path_rate=0.1)
        self.encoder_embed_dim = self.encoder.embed_dim  # 384
        self.num_patches = self.encoder.patch_embed.num_patches  # 196
        
        logger.info(f"MAE Encoder: vit_small (embed_dim={self.encoder_embed_dim}, "
                   f"num_patches={self.num_patches})")
        
        # Decoder
        self.decoder = MAEDecoder(
            num_patches=self.num_patches, patch_size=patch_size, in_chans=in_chans,
            encoder_embed_dim=self.encoder_embed_dim, decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth, decoder_num_heads=decoder_num_heads
        )
    
    def patchify(self, imgs):
        """Convert images to patches."""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, h * w, p * p * C)
        return x
    
    def unpatchify(self, x):
        """Convert patches back to images."""
        p = self.patch_size
        B, N, _ = x.shape
        h = w = int(N ** 0.5)
        C = 3
        x = x.reshape(B, h, w, p, p, C)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, C, h * p, w * p)
        return x
    
    def random_masking(self, x):
        """
        Random masking: keep subset of patches, mask the rest.
        
        Args:
            x: (B, N, D) patch embeddings with position
            
        Returns:
            x_masked: (B, N_visible, D)
            mask: (B, N) binary mask, 0=keep, 1=masked
            ids_restore: (B, N) indices to restore original order
        """
        B, N, D = x.shape
        keep_len = int(N * (1 - self.mask_ratio))
        
        # Random noise for shuffling
        noise = torch.rand(B, N, device=x.device)
        
        # Sort and get indices
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep first keep_len patches
        ids_keep = ids_shuffle[:, :keep_len]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0=keep, 1=masked
        mask = torch.ones(B, N, device=x.device)
        mask[:, :keep_len] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x):
        """Encode visible patches only."""
        # Patch embed
        x = self.encoder.patch_embed(x)  # (B, N, D)
        
        # Add position embeddings
        x = x + self.encoder.pos_embed
        
        # Random masking
        x, mask, ids_restore = self.random_masking(x)
        
        # Apply transformer blocks to visible patches only
        for blk in self.encoder.blocks:
            x = blk(x)
        
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """Decode and predict pixels."""
        return self.decoder(x, ids_restore)
    
    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss on masked patches only.
        
        Args:
            imgs: (B, 3, H, W) original images
            pred: (B, N, patch_size**2 * 3) predicted pixels
            mask: (B, N) binary mask, 1=masked
        """
        target = self.patchify(imgs)
        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # Mean over patch pixels
        
        # Only compute loss on masked patches
        loss = (loss * mask).sum() / mask.sum()
        return loss
    
    def forward(self, imgs):
        """
        Forward pass: encode, decode, compute loss.
        """
        latent, mask, ids_restore = self.forward_encoder(imgs)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
    def get_encoder_state_dict(self):
        """Get only the encoder weights for downstream use."""
        return self.encoder.state_dict()


# ============================================================================
# Training
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch_idx, images in enumerate(dataloader):
        images = images.to(device)
        
        optimizer.zero_grad()
        loss, _, _ = model(images)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 20 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}")
    
    scheduler.step()
    avg_loss = total_loss / num_batches
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='MAE Pretraining with vit_small')
    parser.add_argument('--data-root', type=str, required=True, help='Path to TB CXR dataset')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.05, help='Weight decay')
    parser.add_argument('--mask-ratio', type=float, default=0.75, help='Mask ratio')
    parser.add_argument('--warmup-epochs', type=int, default=10, help='Warmup epochs')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Setup
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data augmentation for pretraining
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = TBCXRDataset(args.data_root, transform=transform)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True
    )
    
    logger.info(f"Dataset: {len(dataset)} images")
    logger.info(f"Batches per epoch: {len(dataloader)}")
    
    # Model
    model = MAE(
        img_size=224, patch_size=16,
        decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=6,
        mask_ratio=args.mask_ratio, norm_pix_loss=True
    ).to(device)
    
    # Count parameters
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Encoder params: {encoder_params:,}")
    logger.info(f"Decoder params: {decoder_params:,}")
    logger.info(f"Total params: {total_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay
    )
    
    # Scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        else:
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training history
    history = {'epoch': [], 'loss': [], 'lr': []}
    
    logger.info("=" * 60)
    logger.info("Starting MAE Pretraining with vit_small encoder")
    logger.info("=" * 60)
    
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        
        avg_loss = train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch)
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        history['epoch'].append(epoch)
        history['loss'].append(avg_loss)
        history['lr'].append(current_lr)
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
        
        logger.info(f"Epoch {epoch:3d}/{args.epochs} | Loss: {avg_loss:.4f} | "
                   f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s" +
                   (" *BEST*" if is_best else ""))
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'encoder': model.get_encoder_state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_ep{epoch}.pth'))
        
        # Save best
        if is_best:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'encoder': model.get_encoder_state_dict(),
                'loss': avg_loss,
                'args': vars(args)
            }
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
    
    total_time = time.time() - start_time
    
    # Save final encoder
    final_encoder = {
        'encoder': model.get_encoder_state_dict(),
        'embed_dim': model.encoder_embed_dim,
        'num_patches': model.num_patches,
        'epoch': args.epochs,
        'final_loss': avg_loss
    }
    torch.save(final_encoder, os.path.join(args.output_dir, 'encoder_final.pth'))
    
    # Save history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("=" * 60)
    logger.info(f"Training complete!")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Final encoder saved to: {args.output_dir}/encoder_final.pth")
    logger.info("=" * 60)
    
    # Verify encoder can be loaded
    logger.info("\nVerifying encoder checkpoint...")
    test_ckpt = torch.load(os.path.join(args.output_dir, 'encoder_final.pth'), 
                          map_location='cpu', weights_only=False)
    test_encoder = vit_small(patch_size=16)
    loaded = test_encoder.load_state_dict(test_ckpt['encoder'], strict=True)
    logger.info(f"Encoder verification: SUCCESS - all {len(test_ckpt['encoder'])} params loaded")


if __name__ == '__main__':
    main()
