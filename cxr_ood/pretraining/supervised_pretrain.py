#!/usr/bin/env python3
"""
Supervised Pretraining on TB CXR Dataset

Pretrains a ViT-Small encoder using standard supervised classification
on the same TB CXR dataset used for JEPA pretraining.

This ensures fair comparison: same data, same architecture, different method.

Note: Supervised uses labels (Normal vs Tuberculosis), while JEPA/MAE are self-supervised.
"""

import os
import sys
import argparse
import json
import logging
import random
import math
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

# Add IJEPA_Meta to path
IJEPA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "IJEPA_Meta")
sys.path.insert(0, IJEPA_PATH)
from src.models.vision_transformer import vit_small

logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset
# ============================================================================

class TBCXRDataset(Dataset):
    """TB Chest Radiography Database with labels for supervised pretraining."""
    SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    
    def __init__(self, root_path, transform=None):
        self.root_path = root_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Load from both Normal and Tuberculosis folders
        for label, folder_name in [(0, 'Normal'), (1, 'Tuberculosis')]:
            folder = os.path.join(root_path, folder_name)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.endswith(self.SUPPORTED_EXTENSIONS):
                        self.image_paths.append(os.path.join(folder, f))
                        self.labels.append(label)
        
        logger.info(f"Loaded {len(self.image_paths)} images for supervised pretraining")
        logger.info(f"  Normal: {self.labels.count(0)}, Tuberculosis: {self.labels.count(1)}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]


# ============================================================================
# Supervised Model
# ============================================================================

class SupervisedViTSmall(nn.Module):
    """
    ViT-Small with classification head for supervised pretraining.
    
    Architecture matches JEPA's vit_small for fair comparison.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Encoder (same as JEPA)
        self.encoder = vit_small()
        self.embed_dim = self.encoder.embed_dim  # 384
        
        # Classification head
        self.head = nn.Linear(self.embed_dim, num_classes)
    
    def forward(self, x, return_features=False):
        # Get features from encoder
        features = self.encoder(x)  # [B, 196, 384] - all patch embeddings
        global_repr = features.mean(dim=1)  # [B, 384] - mean pool over patches
        
        if return_features:
            return global_repr
        
        logits = self.head(global_repr)
        return logits


# ============================================================================
# Training
# ============================================================================

def train_supervised(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save config
    config = vars(args)
    config['start_time'] = datetime.now().isoformat()
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset
    full_dataset = TBCXRDataset(args.data_root, transform=train_transform)
    
    # Split into train/val (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    # Update val dataset transform
    val_dataset.dataset.transform = val_transform
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # Model
    model = SupervisedViTSmall(num_classes=2).to(device)
    
    logger.info(f"Supervised model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    logger.info(f"Encoder: {sum(p.numel() for p in model.encoder.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return epoch / args.warmup_epochs
        return 0.5 * (1 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    logger.info(f"\nStarting supervised pretraining for {args.epochs} epochs")
    print(f"\n{'='*100}")
    print(f"Supervised Pretraining on TB CXR Dataset")
    print(f"{'='*100}")
    print(f"{'Epoch':^8} | {'Train Loss':^12} | {'Train Acc':^10} | {'Val Loss':^12} | {'Val Acc':^10} | {'Val AUROC':^10} | {'LR':^12}")
    print(f"{'-'*100}")
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_auroc': [],
        'lr': []
    }
    
    best_auroc = 0.0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            optimizer.step()
            
            train_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += imgs.size(0)
        
        scheduler.step()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                
                logits = model(imgs)
                loss = criterion(logits, labels)
                
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                
                probs = F.softmax(logits, dim=1)[:, 1]
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auroc = roc_auc_score(all_labels, all_probs)
        
        current_lr = scheduler.get_last_lr()[0]
        
        # Log history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auroc'].append(val_auroc)
        history['lr'].append(current_lr)
        
        # Check if best
        is_best = ""
        if val_auroc > best_auroc:
            best_auroc = val_auroc
            best_epoch = epoch + 1
            is_best = " *BEST*"
            
            # Save best model
            best_checkpoint = {
                'epoch': epoch + 1,
                'encoder': model.encoder.state_dict(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_auroc': val_auroc,
                'config': config
            }
            torch.save(best_checkpoint, os.path.join(args.output_dir, 'best_model.pth'))
        
        print(f"{epoch+1:^8} | {train_loss:^12.4f} | {train_acc:^10.4f} | {val_loss:^12.4f} | {val_acc:^10.4f} | {val_auroc:^10.4f} | {current_lr:^12.2e}{is_best}")
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0 or (epoch + 1) == args.epochs:
            checkpoint = {
                'epoch': epoch + 1,
                'encoder': model.encoder.state_dict(),
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_auroc': val_auroc,
                'config': config
            }
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_ep{epoch+1}.pth')
            torch.save(checkpoint, ckpt_path)
    
    print(f"{'-'*100}")
    print(f"Best validation AUROC: {best_auroc:.4f} at epoch {best_epoch}")
    print(f"{'='*100}")
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final encoder weights (for easy loading in comparison script)
    encoder_path = os.path.join(args.output_dir, 'encoder_final.pth')
    torch.save({
        'encoder': model.encoder.state_dict(),
        'embed_dim': model.embed_dim,
        'best_auroc': best_auroc,
        'config': config
    }, encoder_path)
    logger.info(f"Saved encoder weights to {encoder_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser(description='Supervised Pretraining on TB CXR')
    
    # Data
    parser.add_argument('--data-root', type=str, required=True,
                        help='Path to TB_Chest_Radiography_Database')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for checkpoints')
    
    # Model
    parser.add_argument('--img-size', type=int, default=224)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    train_supervised(args)


if __name__ == '__main__':
    main()
