"""
Echo Segmentation Pretraining Scripts

- jepa_pretrain.py: JEPA (Joint Embedding Predictive Architecture)
- mae_pretrain.py: MAE (Masked Autoencoder)
- supervised_pretrain.py: ImageNet-pretrained baseline
"""

from .jepa_pretrain import train_jepa
from .mae_pretrain import train_mae
from .supervised_pretrain import create_supervised_baseline
