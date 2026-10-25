"""
Pretraining modules for CXR OOD detection.

- jepa_pretrain: JEPA pretraining with target EMA
- mae_pretrain: Masked Autoencoder reconstruction
- supervised_pretrain: Supervised classification baseline
"""

from pathlib import Path
import sys

# Ensure IJEPA_Meta is in path
IJEPA_PATH = Path(__file__).parent.parent.parent / "IJEPA_Meta"
if str(IJEPA_PATH) not in sys.path:
    sys.path.insert(0, str(IJEPA_PATH))
