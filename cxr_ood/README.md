# CXR Out-of-Distribution Detection

Chest X-Ray Out-of-Distribution detection using JEPA, MAE, and Supervised Learning.

## Project Structure

```
cxr_ood/
├── pretraining/           # Pretraining scripts
│   ├── jepa_pretrain.py   # JEPA pretraining with target EMA
│   ├── mae_pretrain.py    # MAE reconstruction pretraining
│   └── supervised_pretrain.py  # Supervised classification
│
├── evaluation/            # Evaluation scripts
│   ├── linear_probe.py    # Linear evaluation protocol
│   ├── ood_detection.py   # Energy & Mahalanobis OOD scoring
│   └── embedding_analysis.py  # t-SNE visualization & statistics
│
├── analysis/              # Analysis scripts
│   ├── baseline_comparison.py  # Compare JEPA vs MAE vs Supervised
│   ├── robustness_ablation.py  # Corruption robustness testing
│   ├── calibration_analysis.py # ECE & reliability diagrams
│   └── embedding_3d_visualization.py  # 3D point cloud visualization
│
├── utils/                 # Utilities
│   ├── common.py          # Common paths & utilities
│   ├── cxr_dataset.py     # CXR dataset loader
│   └── sanity_check.py    # Checkpoint verification
│
└── configs/               # Configuration files
    └── cxr_vit_small.yaml # ViT-Small config for CXR
```

## Usage

All scripts should be run from the project root directory (`Med_JEPA_ODD/`):

```bash
# JEPA Pretraining
python cxr_ood/pretraining/jepa_pretrain.py --config cxr_ood/configs/cxr_vit_small.yaml

# MAE Pretraining
python cxr_ood/pretraining/mae_pretrain.py --data-root Datasets/CXR/TB_Chest_Radiography_Database --output-dir experiments/cxr_jepa_pilot/baseline_comparison/mae_pretrain

# Supervised Pretraining
python cxr_ood/pretraining/supervised_pretrain.py --data-root Datasets/CXR/TB_Chest_Radiography_Database --output-dir experiments/cxr_jepa_pilot/baseline_comparison/supervised_pretrain

# Linear Probe Evaluation
python cxr_ood/evaluation/linear_probe.py --checkpoint experiments/cxr_jepa_pilot/checkpoint_ep50.pth

# OOD Detection
python cxr_ood/evaluation/ood_detection.py --checkpoint experiments/cxr_jepa_pilot/checkpoint_ep50.pth

# Baseline Comparison
python cxr_ood/analysis/baseline_comparison.py

# Robustness Testing
python cxr_ood/analysis/robustness_ablation.py

# 3D Visualization
python cxr_ood/analysis/embedding_3d_visualization.py --jepa-checkpoint experiments/cxr_jepa_pilot/checkpoint_ep50.pth --mae-checkpoint experiments/cxr_jepa_pilot/baseline_comparison/mae_pretrain_v2/encoder_final.pth --supervised-checkpoint experiments/cxr_jepa_pilot/baseline_comparison/supervised_pretrain/best_model.pth --data-root Datasets/CXR --output-dir experiments/cxr_jepa_pilot/3d_visualization
```

## Dependencies

This project uses the core JEPA implementation from `IJEPA_Meta/src/`:
- `src/models/vision_transformer.py` - ViT architecture
- `src/masks/` - Masking strategies
- `src/utils/` - Logging, tensors, schedulers

## Results

All outputs are saved to `experiments/cxr_jepa_pilot/`:
- Checkpoints: `checkpoint_ep*.pth`
- Baselines: `baseline_comparison/`
- Visualizations: `3d_visualization/`
- Results: `*_results.json`

## Key Findings

1. **OOD Detection**: Energy-based scoring achieves 0.77 AUROC on Montgomery, 0.72 on Shenzhen
2. **Robustness**: MAE shows superior contrast robustness (0.98 AUC) vs JEPA (0.67)
3. **EMA Ablation**: Target encoder (with EMA) outperforms context encoder in OOD detection
