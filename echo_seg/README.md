# EchoNet Left Ventricular Segmentation

Left Ventricular (LV) Segmentation on Echocardiogram using JEPA-pretrained encoders.

## Project Structure

```
echo_seg/
├── pretraining/           # JEPA pretraining on echo frames
│   └── (to be implemented)
│
├── segmentation/          # LV segmentation models
│   └── (to be implemented)
│
├── analysis/              # Analysis and visualization
│   └── (to be implemented)
│
├── utils/                 # Utilities and dataset loaders
│   └── (to be implemented)
│
└── configs/               # Configuration files
    └── (to be implemented)
```

## Datasets

- **EchoNet-Dynamic**: Adults echocardiogram dataset with LV segmentation masks
- **EchoNet-Pediatric**: Pediatric echocardiogram dataset

## Planned Approach

1. Extract frames from echocardiogram videos
2. Pretrain JEPA encoder on echo frames (self-supervised)
3. Fine-tune for LV segmentation with frozen/unfrozen encoder
4. Compare JEPA vs MAE vs Supervised baselines
5. Evaluate on pediatric data (OOD transfer)
