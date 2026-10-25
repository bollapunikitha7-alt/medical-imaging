"""
Common utilities and path setup for CXR OOD detection.

This module handles path configuration to ensure IJEPA_Meta modules are accessible.
"""

import sys
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# IJEPA directory (original code - kept static)
IJEPA_ROOT = PROJECT_ROOT / "IJEPA_Meta"

# Add IJEPA to path if not already there
if str(IJEPA_ROOT) not in sys.path:
    sys.path.insert(0, str(IJEPA_ROOT))

# Common paths
DATASETS_ROOT = PROJECT_ROOT / "Datasets" / "CXR"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

# Default checkpoint paths
DEFAULT_CHECKPOINTS = {
    'jepa': EXPERIMENTS_ROOT / "cxr_jepa_pilot" / "checkpoint_ep50.pth",
    'mae': EXPERIMENTS_ROOT / "cxr_jepa_pilot" / "baseline_comparison" / "mae_pretrain_v2" / "encoder_final.pth",
    'supervised': EXPERIMENTS_ROOT / "cxr_jepa_pilot" / "baseline_comparison" / "supervised_pretrain" / "best_model.pth",
}

def get_device(device_str: str = 'auto'):
    """Get the best available device."""
    import torch
    if device_str == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_str

def setup_logging(log_file=None, level='INFO'):
    """Setup logging configuration."""
    import logging
    
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)
