#!/usr/bin/env python3
"""
Master script to run all echo segmentation experiments

1. Fine-tune JEPA, MAE, Supervised encoders for LV segmentation
2. Evaluate on OOD datasets
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime
import json


def run_command(cmd, description):
    """Run a command and log output."""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"WARNING: {description} returned code {result.returncode}")
    
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run Echo Segmentation Experiments')
    
    # Pretrained encoder paths
    parser.add_argument('--jepa_encoder', type=str, 
                        default='experiments/echo_seg_pilot/jepa/checkpoint_ep30.pth',
                        help='Path to JEPA pretrained encoder')
    parser.add_argument('--mae_encoder', type=str,
                        default='experiments/echo_seg_pilot/mae/encoder_final.pth',
                        help='Path to MAE pretrained encoder')
    parser.add_argument('--supervised_encoder', type=str,
                        default='experiments/echo_seg_pilot/supervised/encoder_final.pth',
                        help='Path to Supervised pretrained encoder')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Control what to run
    parser.add_argument('--skip_finetune', action='store_true',
                        help='Skip fine-tuning, only run evaluation')
    parser.add_argument('--models', nargs='+', default=['jepa', 'mae', 'supervised'],
                        choices=['jepa', 'mae', 'supervised'],
                        help='Which models to run')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='experiments/echo_seg_results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save run config
    config = vars(args)
    config['timestamp'] = timestamp
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n{'#'*60}")
    print("ECHO SEGMENTATION EXPERIMENTS")
    print(f"Output: {output_dir}")
    print(f"Models: {args.models}")
    print(f"{'#'*60}")
    
    # Define encoder paths
    encoder_paths = {
        'jepa': args.jepa_encoder,
        'mae': args.mae_encoder,
        'supervised': args.supervised_encoder,
    }
    
    # Fine-tuned model paths (output)
    finetuned_paths = {}
    
    # Step 1: Fine-tune each model
    if not args.skip_finetune:
        print("\n" + "="*60)
        print("STEP 1: FINE-TUNING")
        print("="*60)
        
        for model_name in args.models:
            encoder_path = encoder_paths[model_name]
            model_output = os.path.join(output_dir, model_name)
            
            if not os.path.exists(encoder_path):
                print(f"WARNING: {model_name} encoder not found at {encoder_path}, skipping...")
                continue
            
            cmd = [
                sys.executable,
                os.path.join(script_dir, 'segmentation', 'finetune.py'),
                '--encoder_checkpoint', encoder_path,
                '--model_name', model_name,
                '--output_dir', model_output,
                '--epochs', str(args.epochs),
                '--batch_size', str(args.batch_size),
                '--lr', str(args.lr),
                '--freeze_encoder',  # Start with frozen encoder
            ]
            
            ret = run_command(cmd, f"Fine-tune {model_name.upper()}")
            
            if ret == 0:
                finetuned_paths[model_name] = os.path.join(model_output, 'best_model.pth')
            else:
                print(f"ERROR: Fine-tuning {model_name} failed")
    else:
        # Look for existing fine-tuned models
        print("\nSkipping fine-tuning, looking for existing models...")
        for model_name in args.models:
            # Check common locations
            candidates = [
                os.path.join(args.output_dir, model_name, 'best_model.pth'),
                os.path.join(output_dir, model_name, 'best_model.pth'),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    finetuned_paths[model_name] = cand
                    print(f"Found {model_name}: {cand}")
                    break
    
    # Step 2: OOD Evaluation
    print("\n" + "="*60)
    print("STEP 2: OOD EVALUATION")
    print("="*60)
    
    eval_output = os.path.join(output_dir, 'evaluation')
    
    cmd = [
        sys.executable,
        os.path.join(script_dir, 'segmentation', 'evaluate_ood.py'),
        '--output_dir', eval_output,
    ]
    
    # Add model paths
    if 'jepa' in finetuned_paths:
        cmd.extend(['--jepa_model', finetuned_paths['jepa']])
    if 'mae' in finetuned_paths:
        cmd.extend(['--mae_model', finetuned_paths['mae']])
    if 'supervised' in finetuned_paths:
        cmd.extend(['--supervised_model', finetuned_paths['supervised']])
    
    if len(finetuned_paths) > 0:
        run_command(cmd, "OOD Evaluation")
    else:
        print("No fine-tuned models available for evaluation!")
    
    # Summary
    print("\n" + "#"*60)
    print("EXPERIMENTS COMPLETE")
    print("#"*60)
    print(f"\nOutput directory: {output_dir}")
    print("\nFine-tuned models:")
    for name, path in finetuned_paths.items():
        print(f"  {name}: {path}")
    print(f"\nEvaluation results: {eval_output}/ood_results.json")
    print(f"Comparison plot: {eval_output}/ood_comparison.png")


if __name__ == '__main__':
    main()
