"""
Common utilities and path setup for Echo Segmentation OOD experiments.

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
DATASETS_ROOT = PROJECT_ROOT / "Datasets" / "Echo"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

# Echo dataset paths
DYNAMIC_PATH = DATASETS_ROOT / "EchoNet_Dynamic"
PEDIATRIC_A4C_PATH = DATASETS_ROOT / "Echonet_Pediatric" / "A4C"
PEDIATRIC_PSAX_PATH = DATASETS_ROOT / "Echonet_Pediatric" / "PSAX"

# Default checkpoint paths
DEFAULT_CHECKPOINTS = {
    'jepa': EXPERIMENTS_ROOT / "echo_seg_pilot" / "jepa" / "checkpoint_ep50.pth",
    'mae': EXPERIMENTS_ROOT / "echo_seg_pilot" / "mae" / "encoder_final.pth",
    'supervised': EXPERIMENTS_ROOT / "echo_seg_pilot" / "supervised" / "best_model.pth",
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
