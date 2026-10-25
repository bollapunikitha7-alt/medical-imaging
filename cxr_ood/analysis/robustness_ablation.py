#!/usr/bin/env python3
"""
Subtask 6: Robustness Stress Test + Ablation Study

1. Apply input corruptions to ID test data:
   - Gaussian noise
   - Blur  
   - Contrast shift

2. Evaluate JEPA vs MAE vs Supervised with frozen encoders:
   - AUC drop
   - ECE increase

3. Ablation: JEPA Target Encoder (with EMA) vs Context Encoder (without EMA)
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score
from PIL import Image, ImageFilter

# Add IJEPA_Meta to path
IJEPA_PATH = Path(__file__).parent.parent.parent / "IJEPA_Meta"
sys.path.insert(0, str(IJEPA_PATH))
from src.models.vision_transformer import vit_small


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Corruption Functions
# ============================================================================

class GaussianNoise:
    """Add Gaussian noise to image."""
    def __init__(self, severity=1):
        # Severity 1-5 maps to std 0.04, 0.08, 0.12, 0.16, 0.20
        self.std = 0.04 * severity
    
    def __call__(self, img):
        # img is a tensor [C, H, W]
        noise = torch.randn_like(img) * self.std
        return torch.clamp(img + noise, 0, 1)


class GaussianBlur:
    """Apply Gaussian blur to image."""
    def __init__(self, severity=1):
        # Severity 1-5 maps to kernel size 3, 5, 7, 9, 11
        self.kernel_size = 2 * severity + 1
    
    def __call__(self, img):
        # img is a tensor [C, H, W], need to add batch dim
        img = img.unsqueeze(0)
        # Create Gaussian kernel
        sigma = self.kernel_size / 6.0
        x = torch.arange(self.kernel_size, dtype=torch.float32) - self.kernel_size // 2
        kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
        kernel_2d = kernel_2d.expand(3, 1, self.kernel_size, self.kernel_size)
        
        # Apply blur
        padding = self.kernel_size // 2
        img_blurred = F.conv2d(img, kernel_2d.to(img.device), padding=padding, groups=3)
        return img_blurred.squeeze(0)


class ContrastShift:
    """Reduce contrast of image."""
    def __init__(self, severity=1):
        # Severity 1-5 maps to contrast factor 0.8, 0.6, 0.4, 0.3, 0.2
        factors = [0.8, 0.6, 0.4, 0.3, 0.2]
        self.factor = factors[severity - 1]
    
    def __call__(self, img):
        # img is a tensor [C, H, W]
        mean = img.mean()
        return torch.clamp((img - mean) * self.factor + mean, 0, 1)


# ============================================================================
# Encoder Classes
# ============================================================================

class JEPAEncoder(nn.Module):
    """JEPA encoder - uses target encoder (with EMA)."""
    def __init__(self, checkpoint_path: str, use_target=True):
        super().__init__()
        self.encoder = vit_small(patch_size=16, drop_path_rate=0.1)
        self.embed_dim = self.encoder.embed_dim
        self.use_target = use_target
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Choose target_encoder (EMA) or encoder (context)
        if use_target:
            encoder_state = ckpt.get('target_encoder', ckpt.get('encoder', ckpt))
            encoder_name = "Target (EMA)"
        else:
            encoder_state = ckpt.get('encoder', ckpt)
            encoder_name = "Context (no EMA)"
        
        model_state = self.encoder.state_dict()
        filtered_state = {}
        for k, v in encoder_state.items():
            clean_key = k.replace('module.', '')
            if clean_key in model_state and v.shape == model_state[clean_key].shape:
                filtered_state[clean_key] = v
        
        self.encoder.load_state_dict(filtered_state, strict=False)
        logger.info(f"Loaded JEPA {encoder_name} encoder ({len(filtered_state)} params)")
    
    def forward(self, x):
        features = self.encoder(x)
        return features.mean(dim=1)


class MAEEncoder(nn.Module):
    """MAE encoder."""
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = vit_small(patch_size=16, drop_path_rate=0.1)
        self.embed_dim = self.encoder.embed_dim
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        encoder_state = ckpt.get('encoder', ckpt)
        self.encoder.load_state_dict(encoder_state, strict=True)
        logger.info(f"Loaded MAE encoder ({len(encoder_state)} keys)")
    
    def forward(self, x):
        features = self.encoder(x)
        return features.mean(dim=1)


class SupervisedEncoder(nn.Module):
    """Supervised encoder."""
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = vit_small(patch_size=16, drop_path_rate=0.1)
        self.embed_dim = self.encoder.embed_dim
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        encoder_state = ckpt.get('encoder', ckpt)
        
        model_state = self.encoder.state_dict()
        filtered_state = {}
        for k, v in encoder_state.items():
            clean_key = k.replace('module.', '')
            if clean_key in model_state and v.shape == model_state[clean_key].shape:
                filtered_state[clean_key] = v
        
        self.encoder.load_state_dict(filtered_state, strict=False)
        logger.info(f"Loaded Supervised encoder ({len(filtered_state)} params)")
    
    def forward(self, x):
        features = self.encoder(x)
        return features.mean(dim=1)


# ============================================================================
# Linear Classifier
# ============================================================================

class LinearClassifier(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        return self.fc(x)


def train_linear_probe(embeddings: np.ndarray, labels: np.ndarray, 
                       embed_dim: int, device: torch.device,
                       epochs: int = 100, lr: float = 0.01) -> LinearClassifier:
    """Train a linear probe on embeddings."""
    classifier = LinearClassifier(embed_dim, num_classes=2).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.long).to(device)
    
    classifier.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = classifier(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
    
    classifier.eval()
    return classifier


# ============================================================================
# ECE Computation
# ============================================================================

def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return float(ece)


# ============================================================================
# Evaluation Functions
# ============================================================================

@torch.no_grad()
def extract_embeddings_with_corruption(encoder, dataloader, device, corruption_fn=None):
    """Extract embeddings with optional corruption applied."""
    encoder.eval()
    embeddings = []
    labels = []
    
    for images, targets in dataloader:
        # Apply corruption if specified
        if corruption_fn is not None:
            # Denormalize first
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            images_denorm = images * std + mean
            
            # Apply corruption to each image
            corrupted = []
            for img in images_denorm:
                corrupted.append(corruption_fn(img))
            images_denorm = torch.stack(corrupted)
            
            # Re-normalize
            images = (images_denorm - mean) / std
        
        images = images.to(device)
        emb = encoder(images)
        embeddings.append(emb.cpu().numpy())
        labels.append(targets.numpy())
    
    return np.vstack(embeddings), np.concatenate(labels)


def evaluate_with_corruption(encoder, classifier, dataloader, device, corruption_fn=None):
    """Evaluate encoder+classifier with corruption."""
    embeddings, labels = extract_embeddings_with_corruption(encoder, dataloader, device, corruption_fn)
    
    # Get predictions
    with torch.no_grad():
        X = torch.tensor(embeddings, dtype=torch.float32).to(device)
        logits = classifier(X).cpu().numpy()
    
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    
    # Compute metrics
    auc = roc_auc_score(labels, probs[:, 1])
    ece = compute_ece(probs, labels)
    
    return auc, ece


# ============================================================================
# Visualization
# ============================================================================

def create_robustness_plot(results: dict, output_path: str):
    """Create robustness plot showing AUC vs corruption severity."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    corruptions = ['gaussian_noise', 'blur', 'contrast']
    corruption_names = ['Gaussian Noise', 'Blur', 'Contrast Shift']
    methods = ['JEPA', 'MAE', 'Supervised']
    colors = {'JEPA': '#3498db', 'MAE': '#e74c3c', 'Supervised': '#2ecc71'}
    
    severities = [0, 1, 2, 3, 4, 5]  # 0 = clean
    
    for col, (corruption, corruption_name) in enumerate(zip(corruptions, corruption_names)):
        # AUC plot
        ax_auc = axes[0, col]
        for method in methods:
            aucs = [results[method][corruption][s]['auc'] for s in severities]
            ax_auc.plot(severities, aucs, 'o-', label=method, color=colors[method], linewidth=2, markersize=6)
        
        ax_auc.set_xlabel('Corruption Severity', fontsize=10)
        ax_auc.set_ylabel('AUC', fontsize=10)
        ax_auc.set_title(f'{corruption_name}', fontsize=12, fontweight='bold')
        ax_auc.legend(fontsize=9)
        ax_auc.grid(True, alpha=0.3)
        ax_auc.set_ylim(0.4, 1.05)
        ax_auc.set_xticks(severities)
        
        # ECE plot
        ax_ece = axes[1, col]
        for method in methods:
            eces = [results[method][corruption][s]['ece'] for s in severities]
            ax_ece.plot(severities, eces, 'o-', label=method, color=colors[method], linewidth=2, markersize=6)
        
        ax_ece.set_xlabel('Corruption Severity', fontsize=10)
        ax_ece.set_ylabel('ECE ↓', fontsize=10)
        ax_ece.set_title(f'{corruption_name}', fontsize=12, fontweight='bold')
        ax_ece.legend(fontsize=9)
        ax_ece.grid(True, alpha=0.3)
        ax_ece.set_xticks(severities)
    
    axes[0, 0].set_ylabel('AUC ↑', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('ECE ↓', fontsize=11, fontweight='bold')
    
    plt.suptitle('Robustness to Input Corruptions', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Robustness plot saved to {output_path}")


def create_robustness_table(results: dict, output_path: str):
    """Create robustness summary table."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')
    
    methods = ['JEPA', 'MAE', 'Supervised']
    
    headers = [
        'Method',
        'Clean\nAUC',
        'Noise-5\nAUC',
        'Blur-5\nAUC', 
        'Contrast-5\nAUC',
        'Avg Drop\n(Severity 5)',
        'Clean\nECE',
        'Noise-5\nECE',
        'Blur-5\nECE',
        'Contrast-5\nECE'
    ]
    
    rows = []
    for method in methods:
        clean_auc = results[method]['gaussian_noise'][0]['auc']
        noise5_auc = results[method]['gaussian_noise'][5]['auc']
        blur5_auc = results[method]['blur'][5]['auc']
        contrast5_auc = results[method]['contrast'][5]['auc']
        
        avg_drop = clean_auc - (noise5_auc + blur5_auc + contrast5_auc) / 3
        
        clean_ece = results[method]['gaussian_noise'][0]['ece']
        noise5_ece = results[method]['gaussian_noise'][5]['ece']
        blur5_ece = results[method]['blur'][5]['ece']
        contrast5_ece = results[method]['contrast'][5]['ece']
        
        row = [
            method,
            f"{clean_auc:.3f}",
            f"{noise5_auc:.3f}",
            f"{blur5_auc:.3f}",
            f"{contrast5_auc:.3f}",
            f"{avg_drop:.3f}",
            f"{clean_ece:.3f}",
            f"{noise5_ece:.3f}",
            f"{blur5_ece:.3f}",
            f"{contrast5_ece:.3f}"
        ]
        rows.append(row)
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.1] + [0.1] * 9
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.8)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#3498db')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(rows) + 1):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Robustness Summary: Performance Under Corruption (Severity 0=Clean, 5=Severe)', 
              fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Robustness table saved to {output_path}")


def create_ablation_table(ablation_results: dict, output_path: str):
    """Create ablation study table."""
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis('off')
    
    headers = [
        'JEPA Variant',
        'Clean AUC ↑',
        'Noise-5 AUC ↑',
        'Blur-5 AUC ↑',
        'Contrast-5 AUC ↑',
        'Avg Robust AUC ↑',
        'Clean ECE ↓'
    ]
    
    rows = []
    for variant_name, res in ablation_results.items():
        clean_auc = res['gaussian_noise'][0]['auc']
        noise5_auc = res['gaussian_noise'][5]['auc']
        blur5_auc = res['blur'][5]['auc']
        contrast5_auc = res['contrast'][5]['auc']
        avg_robust = (noise5_auc + blur5_auc + contrast5_auc) / 3
        clean_ece = res['gaussian_noise'][0]['ece']
        
        row = [
            variant_name,
            f"{clean_auc:.4f}",
            f"{noise5_auc:.4f}",
            f"{blur5_auc:.4f}",
            f"{contrast5_auc:.4f}",
            f"{avg_robust:.4f}",
            f"{clean_ece:.4f}"
        ]
        rows.append(row)
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
        colWidths=[0.2, 0.13, 0.13, 0.13, 0.13, 0.14, 0.13]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.0)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#9b59b6')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    for i in range(1, len(rows) + 1):
        color = '#ecf0f1' if i % 2 == 0 else 'white'
        for j in range(len(headers)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Ablation Study: JEPA Target Encoder (with EMA) vs Context Encoder (without EMA)', 
              fontsize=12, fontweight='bold', pad=20)
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Ablation table saved to {output_path}")


def print_results(results: dict, ablation_results: dict):
    """Print comprehensive results to console."""
    
    methods = ['JEPA', 'MAE', 'Supervised']
    corruptions = ['gaussian_noise', 'blur', 'contrast']
    corruption_names = {'gaussian_noise': 'Noise', 'blur': 'Blur', 'contrast': 'Contrast'}
    
    print("\n" + "=" * 100)
    print("ROBUSTNESS STRESS TEST RESULTS")
    print("=" * 100)
    
    # Header
    print(f"\n{'Method':<12} | {'Corruption':<12} | {'Sev 0 (Clean)':<15} | {'Sev 1':<10} | {'Sev 2':<10} | {'Sev 3':<10} | {'Sev 4':<10} | {'Sev 5':<10}")
    print("-" * 100)
    
    for method in methods:
        for corruption in corruptions:
            aucs = [results[method][corruption][s]['auc'] for s in range(6)]
            print(f"{method:<12} | {corruption_names[corruption]:<12} | {aucs[0]:<15.4f} | {aucs[1]:<10.4f} | {aucs[2]:<10.4f} | {aucs[3]:<10.4f} | {aucs[4]:<10.4f} | {aucs[5]:<10.4f}")
        print("-" * 100)
    
    # Summary
    print("\n" + "=" * 100)
    print("ROBUSTNESS SUMMARY (AUC Drop from Clean to Severity 5)")
    print("=" * 100)
    print(f"\n{'Method':<12} | {'Noise Drop':>12} | {'Blur Drop':>12} | {'Contrast Drop':>14} | {'Avg Drop':>12}")
    print("-" * 70)
    
    for method in methods:
        clean = results[method]['gaussian_noise'][0]['auc']
        noise_drop = clean - results[method]['gaussian_noise'][5]['auc']
        blur_drop = clean - results[method]['blur'][5]['auc']
        contrast_drop = clean - results[method]['contrast'][5]['auc']
        avg_drop = (noise_drop + blur_drop + contrast_drop) / 3
        
        print(f"{method:<12} | {noise_drop:>12.4f} | {blur_drop:>12.4f} | {contrast_drop:>14.4f} | {avg_drop:>12.4f}")
    
    # Ablation
    print("\n" + "=" * 100)
    print("ABLATION STUDY: JEPA with EMA vs without EMA")
    print("=" * 100)
    
    print(f"\n{'Variant':<25} | {'Clean AUC':>12} | {'Noise-5 AUC':>12} | {'Blur-5 AUC':>12} | {'Contrast-5 AUC':>14} | {'Clean ECE':>12}")
    print("-" * 95)
    
    for variant_name, res in ablation_results.items():
        clean_auc = res['gaussian_noise'][0]['auc']
        noise5_auc = res['gaussian_noise'][5]['auc']
        blur5_auc = res['blur'][5]['auc']
        contrast5_auc = res['contrast'][5]['auc']
        clean_ece = res['gaussian_noise'][0]['ece']
        
        print(f"{variant_name:<25} | {clean_auc:>12.4f} | {noise5_auc:>12.4f} | {blur5_auc:>12.4f} | {contrast5_auc:>14.4f} | {clean_ece:>12.4f}")
    
    print("=" * 100)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Robustness Stress Test & Ablation')
    parser.add_argument('--jepa-checkpoint', type=str, required=True)
    parser.add_argument('--mae-checkpoint', type=str, required=True)
    parser.add_argument('--supervised-checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Data transforms (without normalization for corruption)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load ID dataset
    id_dataset = ImageFolder(os.path.join(args.data_root, 'TB_Chest_Radiography_Database'), transform=transform)
    
    # Load validation split
    baseline_dir = os.path.dirname(args.output_dir)
    split_path = os.path.join(baseline_dir, 'baseline_comparison', 'dataset_split.json')
    
    if os.path.exists(split_path):
        with open(split_path, 'r') as f:
            split = json.load(f)
        val_indices = split['val_indices']
        train_indices = split['train_indices']
    else:
        # Create split
        n_total = len(id_dataset)
        indices = list(range(n_total))
        np.random.shuffle(indices)
        train_indices = indices[:int(0.8 * n_total)]
        val_indices = indices[int(0.8 * n_total):]
    
    train_subset = Subset(id_dataset, train_indices)
    val_subset = Subset(id_dataset, val_indices)
    
    logger.info(f"Train: {len(train_subset)}, Val: {len(val_subset)}")
    
    train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Load encoders
    encoders = {
        'JEPA': JEPAEncoder(args.jepa_checkpoint, use_target=True).to(device),
        'MAE': MAEEncoder(args.mae_checkpoint).to(device),
        'Supervised': SupervisedEncoder(args.supervised_checkpoint).to(device)
    }
    
    # Ablation encoders
    ablation_encoders = {
        'JEPA Target (with EMA)': JEPAEncoder(args.jepa_checkpoint, use_target=True).to(device),
        'JEPA Context (no EMA)': JEPAEncoder(args.jepa_checkpoint, use_target=False).to(device)
    }
    
    # Corruption functions
    corruption_fns = {
        'gaussian_noise': {s: GaussianNoise(s) if s > 0 else None for s in range(6)},
        'blur': {s: GaussianBlur(s) if s > 0 else None for s in range(6)},
        'contrast': {s: ContrastShift(s) if s > 0 else None for s in range(6)}
    }
    
    # Results storage
    results = {method: {c: {} for c in corruption_fns.keys()} for method in encoders.keys()}
    ablation_results = {name: {c: {} for c in corruption_fns.keys()} for name in ablation_encoders.keys()}
    
    # ===== Robustness Test =====
    logger.info("\n" + "=" * 60)
    logger.info("ROBUSTNESS STRESS TEST")
    logger.info("=" * 60)
    
    for method, encoder in encoders.items():
        logger.info(f"\nProcessing {method}...")
        
        # Extract clean embeddings for training linear probe
        train_emb, train_labels = extract_embeddings_with_corruption(encoder, train_loader, device, None)
        
        # Train linear probe
        classifier = train_linear_probe(train_emb, train_labels, encoder.embed_dim, device)
        
        # Evaluate under each corruption and severity
        for corruption_name, severity_fns in corruption_fns.items():
            for severity, corruption_fn in severity_fns.items():
                auc, ece = evaluate_with_corruption(encoder, classifier, val_loader, device, corruption_fn)
                results[method][corruption_name][severity] = {'auc': auc, 'ece': ece}
                
                if severity == 0 or severity == 5:
                    logger.info(f"  {corruption_name} sev={severity}: AUC={auc:.4f}, ECE={ece:.4f}")
    
    # ===== Ablation Study =====
    logger.info("\n" + "=" * 60)
    logger.info("ABLATION STUDY: EMA vs No EMA")
    logger.info("=" * 60)
    
    for name, encoder in ablation_encoders.items():
        logger.info(f"\nProcessing {name}...")
        
        # Extract embeddings and train probe
        train_emb, train_labels = extract_embeddings_with_corruption(encoder, train_loader, device, None)
        classifier = train_linear_probe(train_emb, train_labels, encoder.embed_dim, device)
        
        # Evaluate
        for corruption_name, severity_fns in corruption_fns.items():
            for severity, corruption_fn in severity_fns.items():
                auc, ece = evaluate_with_corruption(encoder, classifier, val_loader, device, corruption_fn)
                ablation_results[name][corruption_name][severity] = {'auc': auc, 'ece': ece}
                
                if severity == 0 or severity == 5:
                    logger.info(f"  {corruption_name} sev={severity}: AUC={auc:.4f}, ECE={ece:.4f}")
    
    # Print results
    print_results(results, ablation_results)
    
    # Create visualizations
    logger.info("\nGenerating visualizations...")
    
    plot_path = os.path.join(args.output_dir, 'robustness_plot.png')
    create_robustness_plot(results, plot_path)
    
    table_path = os.path.join(args.output_dir, 'robustness_table.png')
    create_robustness_table(results, table_path)
    
    ablation_path = os.path.join(args.output_dir, 'ablation_table.png')
    create_ablation_table(ablation_results, ablation_path)
    
    # Save results to JSON
    all_results = {
        'robustness': results,
        'ablation': ablation_results
    }
    
    results_path = os.path.join(args.output_dir, 'robustness_ablation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Output files
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"  - {plot_path}")
    print(f"  - {table_path}")
    print(f"  - {ablation_path}")
    print(f"  - {results_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
