"""
EchoNet Dataset utilities for LV Segmentation.

Supports:
- EchoNet-Dynamic (Adult A4C) - Training data
- EchoNet-Pediatric A4C - OOD testing (same view, different population)
- EchoNet-Pediatric PSAX - OOD testing (different view)
"""

import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
import logging

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import skimage.draw
from PIL import Image

logger = logging.getLogger(__name__)


# ============================================================================
# Video Loading
# ============================================================================

def loadvideo(filename: str) -> np.ndarray:
    """
    Load a video from a file.
    
    Args:
        filename: Path to video file
        
    Returns:
        np.ndarray with shape (C, F, H, W) - channels=3, frames, height, width
        Values are uint8 ranging from 0 to 255
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    
    capture = cv2.VideoCapture(filename)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video array (F, H, W, C)
    v = np.zeros((frame_count, frame_height, frame_width, 3), np.uint8)
    
    for count in range(frame_count):
        ret, frame = capture.read()
        if not ret:
            raise ValueError(f"Failed to load frame #{count} of {filename}")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        v[count, :, :] = frame
    
    capture.release()
    
    # Transpose to (C, F, H, W)
    v = v.transpose((3, 0, 1, 2))
    return v


# ============================================================================
# Tracing Parsing
# ============================================================================

def parse_dynamic_tracings(csv_path):
    """
    Parse EchoNet-Dynamic VolumeTracings.csv
    
    Format: FileName, X1, Y1, X2, Y2, Frame
    Each row contains a paired coordinate (left and right side of heart)
    
    Returns:
        frames: dict mapping filename -> list of frame indices with tracings
        trace: dict mapping filename -> frame -> np.array of (X1, Y1, X2, Y2) coordinates
    """
    frames = defaultdict(list)
    trace = defaultdict(lambda: defaultdict(list))
    
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        if header != ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]:
            raise ValueError(f"Unexpected header: {header}")
        
        for line in f:
            filename, x1, y1, x2, y2, frame = line.strip().split(',')
            x1 = float(x1)
            y1 = float(y1)
            x2 = float(x2)
            y2 = float(y2)
            frame = int(frame)
            
            if frame not in trace[filename]:
                frames[filename].append(frame)
            trace[filename][frame].append((x1, y1, x2, y2))
    
    # Convert to numpy arrays
    for filename in frames:
        for frame in frames[filename]:
            trace[filename][frame] = np.array(trace[filename][frame])
    
    return dict(frames), trace


def parse_pediatric_tracings(csv_path):
    """
    Parse EchoNet-Pediatric VolumeTracings.csv
    
    Format: FileName, X, Y, Frame
    Each row contains a single coordinate (polygon point)
    
    Returns:
        frames: dict mapping filename -> list of frame indices with tracings
        trace: dict mapping filename -> frame -> np.array of (X, Y) coordinates
    """
    frames = defaultdict(list)
    trace = defaultdict(lambda: defaultdict(list))
    
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
        if header != ["FileName", "X", "Y", "Frame"]:
            raise ValueError(f"Unexpected header: {header}")
        
        for line in f:
            parts = line.strip().split(',')
            if len(parts) != 4:
                continue
            filename, x, y, frame = parts
            
            # Skip empty values
            if not x or not y or not frame:
                continue
            
            try:
                x = float(x)
                y = float(y)
                frame = int(frame) - 1  # Convert to 0-indexed
            except ValueError:
                continue
            
            if frame < 0:
                continue  # Skip invalid frames
            
            if frame not in trace[filename]:
                frames[filename].append(frame)
            trace[filename][frame].append((x, y))
    
    # Convert to numpy arrays
    for filename in frames:
        for frame in frames[filename]:
            trace[filename][frame] = np.array(trace[filename][frame])
    
    return dict(frames), trace


# ============================================================================
# Mask Creation
# ============================================================================

def create_mask_from_dynamic_trace(trace, height, width):
    """
    Create binary mask from EchoNet-Dynamic trace format.
    
    The trace has shape (N, 4) with columns [X1, Y1, X2, Y2].
    """
    x1, y1, x2, y2 = trace[:, 0], trace[:, 1], trace[:, 2], trace[:, 3]
    
    # Concatenate to form closed polygon
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))
    
    # Create binary mask using skimage.draw.polygon
    r, c = skimage.draw.polygon(
        np.rint(y).astype(np.int32), 
        np.rint(x).astype(np.int32), 
        (height, width)
    )
    
    mask = np.zeros((height, width), np.float32)
    mask[r, c] = 1
    return mask


def create_mask_from_pediatric_trace(trace, height, width):
    """
    Create binary mask from EchoNet-Pediatric trace format.
    
    The trace has shape (N, 2) with columns [X, Y].
    """
    x = trace[:, 0]
    y = trace[:, 1]
    
    # Create binary mask using skimage.draw.polygon
    r, c = skimage.draw.polygon(
        np.rint(y).astype(np.int32), 
        np.rint(x).astype(np.int32), 
        (height, width)
    )
    
    mask = np.zeros((height, width), np.float32)
    mask[r, c] = 1
    return mask


# ============================================================================
# Datasets
# ============================================================================

class EchoFrameDataset(Dataset):
    """
    PyTorch Dataset for EchoNet frames with LV segmentation masks.
    
    Each sample is a single frame (ED or ES) with its corresponding LV mask.
    This is for frame-level pretraining and segmentation, not video-level.
    """
    
    def __init__(
        self, 
        root_path,
        filelist_df,
        frames_dict,
        trace_dict,
        split="TRAIN",
        dataset_type="dynamic",  # "dynamic" or "pediatric"
        frame_type="both",  # "ED", "ES", or "both"
        transform=None,
        img_size=224,
        return_mask=True,
    ):
        self.root = Path(root_path)
        self.filelist = filelist_df.copy()
        self.frames_dict = frames_dict
        self.trace_dict = trace_dict
        self.dataset_type = dataset_type
        self.transform = transform
        self.img_size = img_size
        self.return_mask = return_mask
        
        # Filter by split if applicable
        if split is not None and 'Split' in self.filelist.columns:
            self.filelist = self.filelist[self.filelist['Split'] == split].reset_index(drop=True)
        
        # Build list of (filename, frame_idx, frame_type) samples
        self.samples = []
        for _, row in self.filelist.iterrows():
            filename = row['FileName']
            
            # Try both with and without .avi extension for lookup
            if filename in self.frames_dict:
                lookup_key = filename
            elif filename + '.avi' in self.frames_dict:
                lookup_key = filename + '.avi'
            else:
                continue
            
            traced_frames = self.frames_dict[lookup_key]
            if len(traced_frames) < 2:
                continue
            
            if frame_type in ["ES", "both"]:
                self.samples.append((lookup_key, traced_frames[0], "ES"))
            if frame_type in ["ED", "both"]:
                self.samples.append((lookup_key, traced_frames[-1], "ED"))
        
        logger.info(f"EchoFrameDataset: {len(self.samples)} samples ({frame_type} frames, split={split})")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, frame_idx, frame_type = self.samples[idx]
        
        # Load video
        video_filename = filename if filename.endswith('.avi') else filename + '.avi'
        video_path = self.root / "Videos" / video_filename
        video = loadvideo(str(video_path))  # (C, F, H, W)
        
        # Get frame
        frame = video[:, frame_idx, :, :]  # (C, H, W)
        frame = frame.transpose(1, 2, 0)  # (H, W, C)
        
        # Get original dimensions
        h, w = frame.shape[:2]
        
        # Convert to PIL for transforms
        frame_pil = Image.fromarray(frame)
        
        # Resize to target size
        frame_pil = frame_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Apply transforms
        if self.transform:
            frame_tensor = self.transform(frame_pil)
        else:
            frame_tensor = transforms.ToTensor()(frame_pil)
        
        if self.return_mask:
            # Get trace and create mask
            trace = self.trace_dict[filename][frame_idx]
            
            if self.dataset_type == "dynamic":
                mask = create_mask_from_dynamic_trace(trace, h, w)
            else:
                mask = create_mask_from_pediatric_trace(trace, h, w)
            
            # Resize mask
            mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
            mask_pil = mask_pil.resize((self.img_size, self.img_size), Image.NEAREST)
            mask_tensor = torch.from_numpy(np.array(mask_pil) / 255.0).float().unsqueeze(0)
            
            return {
                'image': frame_tensor,
                'mask': mask_tensor,
                'filename': filename,
                'frame_idx': frame_idx,
                'frame_type': frame_type
            }
        else:
            # For pretraining, just return the image
            return frame_tensor


class EchoPretrainDataset(Dataset):
    """
    Simplified dataset for JEPA/MAE pretraining (no masks needed).
    Returns only images for self-supervised learning.
    """
    
    def __init__(
        self, 
        root_path,
        filelist_df,
        frames_dict,
        split="TRAIN",
        frame_type="both",
        transform=None,
        img_size=224,
    ):
        self.root = Path(root_path)
        self.filelist = filelist_df.copy()
        self.frames_dict = frames_dict
        self.transform = transform
        self.img_size = img_size
        
        # Filter by split if applicable
        if split is not None and 'Split' in self.filelist.columns:
            self.filelist = self.filelist[self.filelist['Split'] == split].reset_index(drop=True)
        
        # Build list of (filename, frame_idx) samples
        self.samples = []
        for _, row in self.filelist.iterrows():
            filename = row['FileName']
            
            # Try both with and without .avi extension
            if filename in self.frames_dict:
                lookup_key = filename
            elif filename + '.avi' in self.frames_dict:
                lookup_key = filename + '.avi'
            else:
                continue
            
            traced_frames = self.frames_dict[lookup_key]
            if len(traced_frames) < 2:
                continue
            
            if frame_type in ["ES", "both"]:
                self.samples.append((lookup_key, traced_frames[0]))
            if frame_type in ["ED", "both"]:
                self.samples.append((lookup_key, traced_frames[-1]))
        
        logger.info(f"EchoPretrainDataset: {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        filename, frame_idx = self.samples[idx]
        
        # Load video
        video_filename = filename if filename.endswith('.avi') else filename + '.avi'
        video_path = self.root / "Videos" / video_filename
        video = loadvideo(str(video_path))  # (C, F, H, W)
        
        # Get frame
        frame = video[:, frame_idx, :, :]  # (C, H, W)
        frame = frame.transpose(1, 2, 0)  # (H, W, C)
        
        # Convert to PIL for transforms
        frame_pil = Image.fromarray(frame)
        
        # Resize to target size
        frame_pil = frame_pil.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Apply transforms
        if self.transform:
            frame_tensor = self.transform(frame_pil)
        else:
            frame_tensor = transforms.ToTensor()(frame_pil)
        
        return frame_tensor


# ============================================================================
# Dataset Factory Functions
# ============================================================================

def load_dynamic_data(root_path=None):
    """Load EchoNet-Dynamic metadata and tracings."""
    from .common import DYNAMIC_PATH
    
    if root_path is None:
        root_path = DYNAMIC_PATH
    else:
        root_path = Path(root_path)
    
    filelist = pd.read_csv(root_path / "FileList.csv")
    frames, trace = parse_dynamic_tracings(root_path / "VolumeTracings.csv")
    
    logger.info(f"Loaded Dynamic: {len(filelist)} videos, {len(frames)} with tracings")
    return filelist, frames, trace


def load_pediatric_a4c_data(root_path=None):
    """Load EchoNet-Pediatric A4C metadata and tracings."""
    from .common import PEDIATRIC_A4C_PATH
    
    if root_path is None:
        root_path = PEDIATRIC_A4C_PATH
    else:
        root_path = Path(root_path)
    
    filelist = pd.read_csv(root_path / "FileList.csv")
    frames, trace = parse_pediatric_tracings(root_path / "VolumeTracings.csv")
    
    logger.info(f"Loaded Pediatric A4C: {len(filelist)} videos, {len(frames)} with tracings")
    return filelist, frames, trace


def load_pediatric_psax_data(root_path=None):
    """Load EchoNet-Pediatric PSAX metadata and tracings."""
    from .common import PEDIATRIC_PSAX_PATH
    
    if root_path is None:
        root_path = PEDIATRIC_PSAX_PATH
    else:
        root_path = Path(root_path)
    
    filelist = pd.read_csv(root_path / "FileList.csv")
    frames, trace = parse_pediatric_tracings(root_path / "VolumeTracings.csv")
    
    logger.info(f"Loaded Pediatric PSAX: {len(filelist)} videos, {len(frames)} with tracings")
    return filelist, frames, trace


def make_echo_pretrain_dataset(
    transform=None,
    batch_size=32,
    collator=None,
    pin_mem=True,
    num_workers=4,
    root_path=None,
    split="TRAIN",
    frame_type="both",
    img_size=224,
    drop_last=True,
    shuffle=True,
):
    """
    Create DataLoader for Echo pretraining (JEPA/MAE).
    
    Uses EchoNet-Dynamic TRAIN split by default.
    """
    from .common import DYNAMIC_PATH
    
    if root_path is None:
        root_path = DYNAMIC_PATH
    
    # Load data
    filelist, frames, trace = load_dynamic_data(root_path)
    
    # Create dataset
    dataset = EchoPretrainDataset(
        root_path=root_path,
        filelist_df=filelist,
        frames_dict=frames,
        split=split,
        frame_type=frame_type,
        transform=transform,
        img_size=img_size,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        collate_fn=collator,
    )
    
    return dataset, dataloader, (filelist, frames, trace)
