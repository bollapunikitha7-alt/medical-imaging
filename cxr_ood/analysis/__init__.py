"""
Analysis modules for CXR OOD detection.

- baseline_comparison: Compare JEPA vs MAE vs Supervised
- robustness_ablation: Corruption robustness testing
- calibration_analysis: ECE and reliability diagrams
- embedding_3d_visualization: 3D point cloud visualization
"""

from pathlib import Path
import sys

# Ensure IJEPA_Meta is in path
IJEPA_PATH = Path(__file__).parent.parent.parent / "IJEPA_Meta"
if str(IJEPA_PATH) not in sys.path:
    sys.path.insert(0, str(IJEPA_PATH))
