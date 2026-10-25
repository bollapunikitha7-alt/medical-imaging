"""
CXR Out-of-Distribution Detection using JEPA, MAE, and Supervised Learning

This package contains code for:
- Pretraining: JEPA, MAE, and Supervised models on TB Chest X-rays
- Evaluation: Linear probe, OOD detection, embedding analysis
- Analysis: Robustness testing, calibration, 3D visualization

The core JEPA implementation is in IJEPA_Meta/src/ (kept static).
"""

import sys
from pathlib import Path

# Add IJEPA_Meta to path for importing src modules
IJEPA_PATH = Path(__file__).parent.parent / "IJEPA_Meta"
if str(IJEPA_PATH) not in sys.path:
    sys.path.insert(0, str(IJEPA_PATH))

__version__ = "0.1.0"
