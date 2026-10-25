"""
Utility modules for CXR OOD detection.

- datasets: CXR, Montgomery, Shenzhen dataset loaders
- common: Shared utilities
"""

from pathlib import Path
import sys

# Ensure IJEPA_Meta is in path
IJEPA_PATH = Path(__file__).parent.parent.parent / "IJEPA_Meta"
if str(IJEPA_PATH) not in sys.path:
    sys.path.insert(0, str(IJEPA_PATH))
