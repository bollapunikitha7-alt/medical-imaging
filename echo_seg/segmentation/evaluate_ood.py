#!/usr/bin/env python3
"""
Evaluate segmentation models on OOD datasets

Tests fine-tuned models on:
- EchoNet-Dynamic TEST (in-distribution)
- EchoNet-Pediatric A4C (OOD: age shift)
- EchoNet-Pediatric PSAX (OOD: view shift)
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
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.echo_dataset import (
    load_dynamic_data, load_pediatric_a4c_data, load_pediatric_psax_data, EchoFrameDataset
)
from utils.common import DYNAMIC_PATH, PEDIATRIC_A4C_PATH, PEDIATRIC_PSAX_PATH
from segmentation.seg_model import ViTSegmentation, dice_score


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_dataset(model, dataloader, device, dataset_name):
    """Evaluate model on a dataset, return per-sample dice scores."""
    model.eval()
    all_dice = []
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            logits = model(images)
            
            # Per-sample dice
            pred = (torch.sigmoid(logits) > 0.5).float()
            for i in range(pred.shape[0]):
                p = pred[i].view(-1)
                t = masks[i].view(-1)
                intersection = (p * t).sum()
                dice = (2. * intersection + 1e-6) / (p.sum() + t.sum() + 1e-6)
                all_dice.append(dice.item())
    
    mean_dice = np.mean(all_dice)
    std_dice = np.std(all_dice)
    
    logger.info(f"{dataset_name}: Dice = {mean_dice:.4f} ± {std_dice:.4f} (n={len(all_dice)})")
    
    return {
        'mean_dice': mean_dice,
        'std_dice': std_dice,
        'n_samples': len(all_dice),
        'all_dice': all_dice,
    }


def load_model(model_path, device):
    """Load fine-tuned segmentation model."""
    model = ViTSegmentation(encoder_checkpoint=None, freeze_encoder=False)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='OOD Evaluation')
    
    # Model paths
    parser.add_argument('--jepa_model', type=str, default=None,
                        help='Path to JEPA fine-tuned model')
    parser.add_argument('--mae_model', type=str, default=None,
                        help='Path to MAE fine-tuned model')
    parser.add_argument('--supervised_model', type=str, default=None,
                        help='Path to Supervised fine-tuned model')
    
    # Data
    parser.add_argument('--dynamic_root', type=str, default=DYNAMIC_PATH)
    parser.add_argument('--pediatric_a4c_root', type=str, default=PEDIATRIC_A4C_PATH)
    parser.add_argument('--pediatric_psax_root', type=str, default=PEDIATRIC_PSAX_PATH)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=8)
    
    # Output
    parser.add_argument('--output_dir', type=str, required=True)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    datasets = {}
    
    # 1. EchoNet-Dynamic TEST
    logger.info("Loading EchoNet-Dynamic TEST...")
    dyn_filelist, dyn_frames, dyn_trace = load_dynamic_data(args.dynamic_root)
    dynamic_test = EchoFrameDataset(
        root_path=args.dynamic_root,
        filelist_df=dyn_filelist,
        frames_dict=dyn_frames,
        trace_dict=dyn_trace,
        split="TEST",
        dataset_type="dynamic",
        frame_type="both",
        transform=transform,
        img_size=args.img_size,
        return_mask=True,
    )
    datasets['Dynamic_TEST'] = DataLoader(
        dynamic_test, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    logger.info(f"  Dynamic TEST: {len(dynamic_test)} samples")
    
    # 2. EchoNet-Pediatric A4C (if available)
    if os.path.exists(args.pediatric_a4c_root):
        logger.info("Loading EchoNet-Pediatric A4C...")
        try:
            ped_a4c_filelist, ped_a4c_frames, ped_a4c_trace = load_pediatric_a4c_data(
                args.pediatric_a4c_root
            )
            pediatric_a4c = EchoFrameDataset(
                root_path=args.pediatric_a4c_root,
                filelist_df=ped_a4c_filelist,
                frames_dict=ped_a4c_frames,
                trace_dict=ped_a4c_trace,
                split=None,  # Use all
                dataset_type="pediatric",
                frame_type="both",
                transform=transform,
                img_size=args.img_size,
                return_mask=True,
            )
            datasets['Pediatric_A4C'] = DataLoader(
                pediatric_a4c, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            logger.info(f"  Pediatric A4C: {len(pediatric_a4c)} samples")
        except Exception as e:
            logger.warning(f"Could not load Pediatric A4C: {e}")
    
    # 3. EchoNet-Pediatric PSAX (if available)
    if os.path.exists(args.pediatric_psax_root):
        logger.info("Loading EchoNet-Pediatric PSAX...")
        try:
            ped_psax_filelist, ped_psax_frames, ped_psax_trace = load_pediatric_psax_data(
                args.pediatric_psax_root
            )
            pediatric_psax = EchoFrameDataset(
                root_path=args.pediatric_psax_root,
                filelist_df=ped_psax_filelist,
                frames_dict=ped_psax_frames,
                trace_dict=ped_psax_trace,
                split=None,
                dataset_type="pediatric",
                frame_type="both",
                transform=transform,
                img_size=args.img_size,
                return_mask=True,
            )
            datasets['Pediatric_PSAX'] = DataLoader(
                pediatric_psax, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            logger.info(f"  Pediatric PSAX: {len(pediatric_psax)} samples")
        except Exception as e:
            logger.warning(f"Could not load Pediatric PSAX: {e}")
    
    # Models to evaluate
    models = {}
    if args.jepa_model and os.path.exists(args.jepa_model):
        models['JEPA'] = load_model(args.jepa_model, device)
        logger.info(f"Loaded JEPA model from {args.jepa_model}")
    if args.mae_model and os.path.exists(args.mae_model):
        models['MAE'] = load_model(args.mae_model, device)
        logger.info(f"Loaded MAE model from {args.mae_model}")
    if args.supervised_model and os.path.exists(args.supervised_model):
        models['Supervised'] = load_model(args.supervised_model, device)
        logger.info(f"Loaded Supervised model from {args.supervised_model}")
    
    if not models:
        logger.error("No models to evaluate!")
        return
    
    # Evaluate all combinations
    results = {}
    
    logger.info(f"\n{'='*60}")
    logger.info("EVALUATION RESULTS")
    logger.info(f"{'='*60}")
    
    for model_name, model in models.items():
        results[model_name] = {}
        logger.info(f"\n{model_name}:")
        
        for dataset_name, dataloader in datasets.items():
            eval_result = evaluate_dataset(model, dataloader, device, dataset_name)
            results[model_name][dataset_name] = {
                'mean_dice': eval_result['mean_dice'],
                'std_dice': eval_result['std_dice'],
                'n_samples': eval_result['n_samples'],
            }
    
    # Summary table
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY TABLE")
    logger.info(f"{'='*60}")
    
    # Print header
    dataset_names = list(datasets.keys())
    header = f"{'Model':<15}"
    for ds in dataset_names:
        header += f" | {ds:<20}"
    logger.info(header)
    logger.info("-" * len(header))
    
    # Print rows
    for model_name in models.keys():
        row = f"{model_name:<15}"
        for ds in dataset_names:
            if ds in results[model_name]:
                dice = results[model_name][ds]['mean_dice']
                std = results[model_name][ds]['std_dice']
                row += f" | {dice:.4f} ± {std:.4f}    "
            else:
                row += f" | {'N/A':<20}"
        logger.info(row)
    
    # Save results
    with open(os.path.join(args.output_dir, 'ood_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(dataset_names))
    width = 0.25
    
    for i, (model_name, model_results) in enumerate(results.items()):
        means = [model_results.get(ds, {}).get('mean_dice', 0) for ds in dataset_names]
        stds = [model_results.get(ds, {}).get('std_dice', 0) for ds in dataset_names]
        
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, yerr=stds, label=model_name, capsize=3)
    
    ax.set_xlabel('Dataset')
    ax.set_ylabel('Dice Score')
    ax.set_title('OOD Evaluation: LV Segmentation')
    ax.set_xticks(x)
    ax.set_xticklabels(dataset_names, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'ood_comparison.png'), dpi=150)
    plt.close()
    
    logger.info(f"\nResults saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
