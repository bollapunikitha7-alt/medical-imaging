#!/usr/bin/env python3
"""
Fine-tune pretrained encoders for LV Segmentation

Loads JEPA, MAE, or Supervised encoder and adds segmentation decoder.
Trains on EchoNet-Dynamic frames with ground truth LV masks.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.echo_dataset import (
    load_dynamic_data, EchoFrameDataset
)
from utils.common import DYNAMIC_PATH
from segmentation.seg_model import ViTSegmentation, CombinedLoss, dice_score


# Force flush logging
class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def setup_logging(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'finetune_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            FlushFileHandler(log_file)
        ]
    )
    return log_file

logger = logging.getLogger(__name__)


def get_gpu_memory():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        return f"GPU: {allocated:.2f}GB"
    return ""


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    for batch_idx, batch in enumerate(dataloader):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(images)
        loss = criterion(logits, masks)
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        with torch.no_grad():
            dice = dice_score(logits, masks)
        
        total_loss += loss.item()
        total_dice += dice
        num_batches += 1
        
        if batch_idx % 25 == 0:
            logger.info(f"  [{batch_idx}/{len(dataloader)}] Loss: {loss.item():.4f}, Dice: {dice:.4f}")
        
        # Clear cache
        if batch_idx % 50 == 0:
            torch.cuda.empty_cache()
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    return avg_loss, avg_dice


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    total_dice = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            logits = model(images)
            loss = criterion(logits, masks)
            dice = dice_score(logits, masks)
            
            total_loss += loss.item()
            total_dice += dice
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_dice = total_dice / num_batches
    return avg_loss, avg_dice


def main():
    parser = argparse.ArgumentParser(description='Fine-tune for LV Segmentation')
    
    # Model
    parser.add_argument('--encoder_checkpoint', type=str, required=True,
                        help='Path to pretrained encoder checkpoint')
    parser.add_argument('--model_name', type=str, default='model',
                        help='Name for this model (jepa/mae/supervised)')
    parser.add_argument('--freeze_encoder', action='store_true', default=False,
                        help='Freeze encoder weights during training')
    
    # Data
    parser.add_argument('--data_root', type=str, default=DYNAMIC_PATH)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=0)
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = setup_logging(args.output_dir)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Log file: {log_file}")
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Encoder: {args.encoder_checkpoint}")
    logger.info(f"Freeze encoder: {args.freeze_encoder}")
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Save config (convert Path objects to strings)
    config = {k: str(v) if hasattr(v, '__fspath__') else v for k, v in vars(args).items()}
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load data
    logger.info("Loading EchoNet-Dynamic data...")
    filelist, frames, trace = load_dynamic_data(args.data_root)
    
    # Create datasets
    train_dataset = EchoFrameDataset(
        root_path=args.data_root,
        filelist_df=filelist,
        frames_dict=frames,
        trace_dict=trace,
        split="TRAIN",
        dataset_type="dynamic",
        frame_type="both",
        transform=train_transform,
        img_size=args.img_size,
        return_mask=True,
    )
    
    val_dataset = EchoFrameDataset(
        root_path=args.data_root,
        filelist_df=filelist,
        frames_dict=frames,
        trace_dict=trace,
        split="VAL",
        dataset_type="dynamic",
        frame_type="both",
        transform=val_transform,
        img_size=args.img_size,
        return_mask=True,
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False
    )
    
    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    logger.info(f"Val: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Model
    logger.info("Creating segmentation model...")
    model = ViTSegmentation(
        encoder_checkpoint=args.encoder_checkpoint,
        freeze_encoder=args.freeze_encoder,
        img_size=args.img_size
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total params: {num_params:,}, Trainable: {num_trainable:,}")
    
    # Loss and optimizer
    criterion = CombinedLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training for {args.epochs} epochs")
    logger.info(f"{'='*60}")
    
    best_dice = 0
    train_losses, train_dices = [], []
    val_losses, val_dices = [], []
    
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch+1}/{args.epochs}")
        logger.info(f"{get_gpu_memory()}")
        
        # Train
        train_loss, train_dice = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        train_losses.append(train_loss)
        train_dices.append(train_dice)
        
        # Validate
        val_loss, val_dice = evaluate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        val_dices.append(val_dice)
        
        scheduler.step()
        
        logger.info(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")
        
        # Save best
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'encoder': model.get_encoder_state(),
                'optimizer': optimizer.state_dict(),
                'val_dice': val_dice,
            }, os.path.join(args.output_dir, 'best_model.pth'))
            logger.info(f"New best! Val Dice: {val_dice:.4f}")
        
        # Clear cache
        torch.cuda.empty_cache()
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model': model.state_dict(),
        'encoder': model.get_encoder_state(),
        'val_dice': val_dice,
    }, os.path.join(args.output_dir, 'final_model.pth'))
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(train_losses, label='Train')
    axes[0].plot(val_losses, label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(train_dices, label='Train')
    axes[1].plot(val_dices, label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_title('Dice Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    # Save results
    results = {
        'model_name': args.model_name,
        'encoder_checkpoint': args.encoder_checkpoint,
        'best_val_dice': best_dice,
        'final_val_dice': val_dice,
        'epochs': args.epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_dices': train_dices,
        'val_dices': val_dices,
    }
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete!")
    logger.info(f"Best Val Dice: {best_dice:.4f}")
    logger.info(f"Results saved to: {args.output_dir}")
    logger.info(f"{'='*60}")


if __name__ == '__main__':
    main()
