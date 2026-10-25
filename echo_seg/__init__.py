"""
EchoNet Left Ventricular Segmentation using JEPA Representations

This package contains code for:
- Pretraining: JEPA on echocardiogram frames
- Segmentation: LV segmentation with JEPA encoder as backbone
- Analysis: Performance comparison and visualization

Datasets:
- EchoNet-Dynamic: Adults echocardiogram videos
- EchoNet-Pediatric: Pediatric echocardiogram videos

The core JEPA implementation is in IJEPA_Meta/src/ (kept static).
"""

import sys
from pathlib import Path

# Add IJEPA_Meta to path for importing src modules
IJEPA_PATH = Path(__file__).parent.parent / "IJEPA_Meta"
if str(IJEPA_PATH) not in sys.path:
    sys.path.insert(0, str(IJEPA_PATH))

__version__ = "0.1.0"
