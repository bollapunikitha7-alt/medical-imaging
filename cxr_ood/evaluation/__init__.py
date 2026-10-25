"""
Evaluation modules for CXR OOD detection.

- linear_probe: Linear evaluation protocol
- ood_detection: Energy and Mahalanobis OOD scoring
- embedding_analysis: t-SNE visualization and statistics
"""

from pathlib import Path
import sys

# Ensure IJEPA_Meta is in path
IJEPA_PATH = Path(__file__).parent.parent.parent / "IJEPA_Meta"
if str(IJEPA_PATH) not in sys.path:
    sys.path.insert(0, str(IJEPA_PATH))
