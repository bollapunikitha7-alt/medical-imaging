#!/usr/bin/env python3
"""
3D Embedding Space Visualization for JEPA, MAE, and Supervised

Creates interactive 3D visualizations using:
1. PyVista - 3D point clouds saved as VTK/PLY files
2. Plotly - Interactive HTML visualizations
3. UMAP/t-SNE for dimensionality reduction to 3D

Output files:
- embedding_3d_{method}.vtk - PyVista point cloud
- embedding_3d_{method}.ply - Standard 3D format
- embedding_3d_interactive.html - Interactive Plotly visualization
- embedding_3d_combined.html - All methods in one view
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Try importing visualization libraries
try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
except ImportError:
    PYVISTA_AVAILABLE = False
    print("Warning: PyVista not installed. Install with: pip install pyvista")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not installed. Install with: pip install plotly")

# Using t-SNE only (UMAP removed due to TensorFlow/protobuf conflicts)

# Add IJEPA_Meta to path
IJEPA_PATH = Path(__file__).parent.parent.parent / "IJEPA_Meta"
sys.path.insert(0, str(IJEPA_PATH))
from src.models.vision_transformer import vit_small

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Dataset Classes
# ============================================================================

class MontgomeryDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str, transform=None):
        self.root = Path(root_path) / "Montgomery TB CXR"
        self.transform = transform
        self.samples = []
        
        import pandas as pd
        metadata_path = self.root / "montgomery_metadata.csv"
        df = pd.read_csv(metadata_path)
        
        images_dir = self.root / "images"
        for _, row in df.iterrows():
            filename = row['study_id']
            finding = str(row['findings']).lower().strip()
            label = 0 if finding == 'normal' else 1
            
            img_path = images_dir / filename
            if img_path.exists():
                self.samples.append((str(img_path), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


class ShenzhenDataset(torch.utils.data.Dataset):
    def __init__(self, root_path: str, transform=None):
        self.root = Path(root_path) / "Shenzhen TB CXR"
        self.transform = transform
        self.samples = []
        
        import pandas as pd
        metadata_path = self.root / "shenzhen_metadata.csv"
        df = pd.read_csv(metadata_path)
        
        images_dir = self.root / "images" / "images"
        for _, row in df.iterrows():
            filename = row['study_id']
            finding = str(row['findings']).lower().strip()
            label = 0 if finding == 'normal' else 1
            
            img_path = images_dir / filename
            if img_path.exists():
                self.samples.append((str(img_path), label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================================
# Encoder Classes
# ============================================================================

class JEPAEncoder(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = vit_small(patch_size=16, drop_path_rate=0.1)
        self.embed_dim = self.encoder.embed_dim
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        encoder_state = ckpt.get('target_encoder', ckpt.get('encoder', ckpt))
        
        model_state = self.encoder.state_dict()
        filtered_state = {k.replace('module.', ''): v for k, v in encoder_state.items() 
                         if k.replace('module.', '') in model_state}
        self.encoder.load_state_dict(filtered_state, strict=False)
        logger.info(f"Loaded JEPA encoder ({len(filtered_state)} params)")
    
    def forward(self, x):
        return self.encoder(x).mean(dim=1)


class MAEEncoder(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = vit_small(patch_size=16, drop_path_rate=0.1)
        self.embed_dim = self.encoder.embed_dim
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        encoder_state = ckpt.get('encoder', ckpt)
        self.encoder.load_state_dict(encoder_state, strict=True)
        logger.info(f"Loaded MAE encoder ({len(encoder_state)} keys)")
    
    def forward(self, x):
        return self.encoder(x).mean(dim=1)


class SupervisedEncoder(nn.Module):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.encoder = vit_small(patch_size=16, drop_path_rate=0.1)
        self.embed_dim = self.encoder.embed_dim
        
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        encoder_state = ckpt.get('encoder', ckpt)
        
        model_state = self.encoder.state_dict()
        filtered_state = {k.replace('module.', ''): v for k, v in encoder_state.items() 
                         if k.replace('module.', '') in model_state}
        self.encoder.load_state_dict(filtered_state, strict=False)
        logger.info(f"Loaded Supervised encoder ({len(filtered_state)} params)")
    
    def forward(self, x):
        return self.encoder(x).mean(dim=1)


# ============================================================================
# Embedding Extraction
# ============================================================================

@torch.no_grad()
def extract_embeddings(encoder, dataloader, device):
    encoder.eval()
    embeddings, labels = [], []
    for images, targets in dataloader:
        images = images.to(device)
        emb = encoder(images)
        embeddings.append(emb.cpu().numpy())
        labels.append(targets.numpy())
    return np.vstack(embeddings), np.concatenate(labels)


def reduce_to_3d(embeddings: np.ndarray, method='tsne', random_state=42):
    """Reduce embeddings to 3D using t-SNE."""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Use t-SNE for 3D reduction (stable and reliable)
    logger.info(f"Running t-SNE 3D reduction on {embeddings.shape[0]} samples...")
    reducer = TSNE(n_components=3, random_state=random_state, perplexity=min(30, len(embeddings)-1), n_iter=1000)
    return reducer.fit_transform(embeddings_scaled)


# ============================================================================
# PyVista Visualization
# ============================================================================

def create_pyvista_pointcloud(points_3d: np.ndarray, labels: np.ndarray, 
                               method_name: str, output_dir: str):
    """Create PyVista point cloud and save as VTK/PLY."""
    if not PYVISTA_AVAILABLE:
        logger.warning("PyVista not available, skipping point cloud creation")
        return
    
    # Create point cloud
    cloud = pv.PolyData(points_3d)
    cloud['labels'] = labels
    cloud['colors'] = labels  # Will be used for coloring
    
    # Save as VTK
    vtk_path = os.path.join(output_dir, f'embedding_3d_{method_name.lower()}.vtk')
    cloud.save(vtk_path)
    logger.info(f"Saved VTK: {vtk_path}")
    
    # Save as PLY
    ply_path = os.path.join(output_dir, f'embedding_3d_{method_name.lower()}.ply')
    cloud.save(ply_path)
    logger.info(f"Saved PLY: {ply_path}")
    
    return cloud


def create_pyvista_screenshot(all_data: dict, output_dir: str):
    """Create static screenshot of 3D visualization."""
    if not PYVISTA_AVAILABLE:
        return
    
    pv.set_plot_theme('document')
    
    for method_name, data in all_data.items():
        plotter = pv.Plotter(off_screen=True)
        
        points = data['points']
        labels = data['labels']
        datasets = data['datasets']
        
        # Color map for datasets
        colors = {'ID-Normal': '#2ecc71', 'ID-TB': '#e74c3c', 
                  'Montgomery-Normal': '#3498db', 'Montgomery-TB': '#9b59b6',
                  'Shenzhen-Normal': '#f39c12', 'Shenzhen-TB': '#1abc9c'}
        
        unique_datasets = np.unique(datasets)
        for ds in unique_datasets:
            mask = datasets == ds
            color = colors.get(ds, '#95a5a6')
            plotter.add_points(points[mask], color=color, point_size=8, 
                             render_points_as_spheres=True, label=ds)
        
        plotter.add_legend()
        plotter.add_title(f'{method_name} Embedding Space (3D)', font_size=14)
        
        # Save screenshot
        img_path = os.path.join(output_dir, f'embedding_3d_{method_name.lower()}.png')
        plotter.screenshot(img_path, window_size=[1920, 1080])
        plotter.close()
        logger.info(f"Saved screenshot: {img_path}")


# ============================================================================
# Plotly Interactive Visualization
# ============================================================================

def create_plotly_visualization(all_data: dict, output_dir: str):
    """Create interactive Plotly 3D scatter plots."""
    if not PLOTLY_AVAILABLE:
        logger.warning("Plotly not available, skipping interactive visualization")
        return
    
    # Color scheme
    color_map = {
        'ID-Normal': '#2ecc71',
        'ID-TB': '#e74c3c', 
        'Montgomery-Normal': '#3498db',
        'Montgomery-TB': '#9b59b6',
        'Shenzhen-Normal': '#f39c12',
        'Shenzhen-TB': '#1abc9c'
    }
    
    # Create individual visualizations for each method
    for method_name, data in all_data.items():
        fig = go.Figure()
        
        points = data['points']
        datasets = data['datasets']
        
        for ds_name in np.unique(datasets):
            mask = datasets == ds_name
            fig.add_trace(go.Scatter3d(
                x=points[mask, 0],
                y=points[mask, 1],
                z=points[mask, 2],
                mode='markers',
                name=ds_name,
                marker=dict(
                    size=4,
                    color=color_map.get(ds_name, '#95a5a6'),
                    opacity=0.8,
                    line=dict(width=0.5, color='white')
                ),
                hovertemplate=f'{ds_name}<br>x: %{{x:.2f}}<br>y: %{{y:.2f}}<br>z: %{{z:.2f}}<extra></extra>'
            ))
        
        fig.update_layout(
            title=dict(
                text=f'{method_name} Embedding Space (3D t-SNE)',
                font=dict(size=20)
            ),
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3',
                bgcolor='rgb(240, 240, 240)'
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            margin=dict(l=0, r=0, b=0, t=50),
            width=1000,
            height=800
        )
        
        # Save individual HTML
        html_path = os.path.join(output_dir, f'embedding_3d_{method_name.lower()}.html')
        fig.write_html(html_path)
        logger.info(f"Saved interactive HTML: {html_path}")
    
    # Create combined visualization with subplots
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=['JEPA', 'MAE', 'Supervised'],
        horizontal_spacing=0.02
    )
    
    for col, method_name in enumerate(['JEPA', 'MAE', 'Supervised'], 1):
        if method_name not in all_data:
            continue
            
        data = all_data[method_name]
        points = data['points']
        datasets = data['datasets']
        
        for ds_name in np.unique(datasets):
            mask = datasets == ds_name
            fig.add_trace(go.Scatter3d(
                x=points[mask, 0],
                y=points[mask, 1],
                z=points[mask, 2],
                mode='markers',
                name=ds_name,
                marker=dict(
                    size=3,
                    color=color_map.get(ds_name, '#95a5a6'),
                    opacity=0.7
                ),
                showlegend=(col == 1),  # Only show legend for first subplot
                legendgroup=ds_name
            ), row=1, col=col)
    
    fig.update_layout(
        title=dict(
            text='3D Embedding Space Comparison: JEPA vs MAE vs Supervised',
            font=dict(size=18)
        ),
        height=700,
        width=1800,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    # Save combined HTML
    combined_path = os.path.join(output_dir, 'embedding_3d_combined.html')
    fig.write_html(combined_path)
    logger.info(f"Saved combined interactive HTML: {combined_path}")


def create_animated_rotation(all_data: dict, output_dir: str):
    """Create animated GIF of rotating 3D view (requires imageio)."""
    if not PYVISTA_AVAILABLE:
        return
    
    try:
        import imageio
    except ImportError:
        logger.warning("imageio not available for animation")
        return
    
    for method_name, data in all_data.items():
        points = data['points']
        datasets = data['datasets']
        
        plotter = pv.Plotter(off_screen=True, window_size=[800, 600])
        
        colors = {'ID-Normal': '#2ecc71', 'ID-TB': '#e74c3c',
                  'Montgomery-Normal': '#3498db', 'Montgomery-TB': '#9b59b6',
                  'Shenzhen-Normal': '#f39c12', 'Shenzhen-TB': '#1abc9c'}
        
        for ds in np.unique(datasets):
            mask = datasets == ds
            cloud = pv.PolyData(points[mask])
            plotter.add_mesh(cloud, color=colors.get(ds, '#95a5a6'), 
                           point_size=8, render_points_as_spheres=True)
        
        plotter.add_title(f'{method_name}', font_size=16)
        
        # Create rotation animation
        gif_path = os.path.join(output_dir, f'embedding_3d_{method_name.lower()}_rotation.gif')
        plotter.open_gif(gif_path)
        
        for angle in range(0, 360, 5):
            plotter.camera.azimuth = angle
            plotter.write_frame()
        
        plotter.close()
        logger.info(f"Saved animated GIF: {gif_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='3D Embedding Visualization')
    parser.add_argument('--jepa-checkpoint', type=str, required=True)
    parser.add_argument('--mae-checkpoint', type=str, required=True)
    parser.add_argument('--supervised-checkpoint', type=str, required=True)
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--reduction', type=str, default='tsne', choices=['tsne', 'umap'])
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-samples', type=int, default=500, 
                        help='Max samples per dataset for faster visualization')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load datasets
    logger.info("Loading datasets...")
    id_dataset = ImageFolder(os.path.join(args.data_root, 'TB_Chest_Radiography_Database'), transform=transform)
    montgomery_dataset = MontgomeryDataset(args.data_root, transform=transform)
    shenzhen_dataset = ShenzhenDataset(args.data_root, transform=transform)
    
    # Subsample for visualization
    def subsample(dataset, max_n):
        if len(dataset) <= max_n:
            return dataset
        indices = np.random.choice(len(dataset), max_n, replace=False)
        return Subset(dataset, indices)
    
    id_subset = subsample(id_dataset, args.max_samples)
    mont_subset = subsample(montgomery_dataset, min(args.max_samples, len(montgomery_dataset)))
    shen_subset = subsample(shenzhen_dataset, args.max_samples)
    
    logger.info(f"Samples - ID: {len(id_subset)}, Montgomery: {len(mont_subset)}, Shenzhen: {len(shen_subset)}")
    
    id_loader = DataLoader(id_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    mont_loader = DataLoader(mont_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    shen_loader = DataLoader(shen_subset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Load encoders
    encoders = {
        'JEPA': JEPAEncoder(args.jepa_checkpoint).to(device),
        'MAE': MAEEncoder(args.mae_checkpoint).to(device),
        'Supervised': SupervisedEncoder(args.supervised_checkpoint).to(device)
    }
    
    # Extract embeddings and create visualizations
    all_data = {}
    
    for method_name, encoder in encoders.items():
        logger.info(f"\nProcessing {method_name}...")
        
        # Extract embeddings
        id_emb, id_labels = extract_embeddings(encoder, id_loader, device)
        mont_emb, mont_labels = extract_embeddings(encoder, mont_loader, device)
        shen_emb, shen_labels = extract_embeddings(encoder, shen_loader, device)
        
        # Combine all embeddings
        all_emb = np.vstack([id_emb, mont_emb, shen_emb])
        all_labels = np.concatenate([id_labels, mont_labels, shen_labels])
        
        # Create dataset labels
        datasets = []
        for i, label in enumerate(id_labels):
            datasets.append('ID-Normal' if label == 0 else 'ID-TB')
        for i, label in enumerate(mont_labels):
            datasets.append('Montgomery-Normal' if label == 0 else 'Montgomery-TB')
        for i, label in enumerate(shen_labels):
            datasets.append('Shenzhen-Normal' if label == 0 else 'Shenzhen-TB')
        datasets = np.array(datasets)
        
        # Reduce to 3D
        logger.info(f"  Reducing to 3D using {args.reduction.upper()}...")
        points_3d = reduce_to_3d(all_emb, method=args.reduction)
        
        # Store data
        all_data[method_name] = {
            'points': points_3d,
            'labels': all_labels,
            'datasets': datasets
        }
        
        # Create PyVista point cloud
        if PYVISTA_AVAILABLE:
            create_pyvista_pointcloud(points_3d, all_labels, method_name, args.output_dir)
    
    # Create visualizations
    logger.info("\nCreating visualizations...")
    
    if PLOTLY_AVAILABLE:
        create_plotly_visualization(all_data, args.output_dir)
    
    if PYVISTA_AVAILABLE:
        create_pyvista_screenshot(all_data, args.output_dir)
        create_animated_rotation(all_data, args.output_dir)
    
    # Save embedding data as JSON for later use
    json_data = {}
    for method, data in all_data.items():
        json_data[method] = {
            'points': data['points'].tolist(),
            'labels': data['labels'].tolist(),
            'datasets': data['datasets'].tolist()
        }
    
    json_path = os.path.join(args.output_dir, 'embedding_3d_data.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f)
    logger.info(f"Saved embedding data: {json_path}")
    
    # Print output files
    print("\n" + "=" * 60)
    print("OUTPUT FILES")
    print("=" * 60)
    print(f"\nInteractive HTML (open in browser):")
    if PLOTLY_AVAILABLE:
        print(f"  - {os.path.join(args.output_dir, 'embedding_3d_combined.html')}")
        for method in ['jepa', 'mae', 'supervised']:
            print(f"  - {os.path.join(args.output_dir, f'embedding_3d_{method}.html')}")
    
    if PYVISTA_AVAILABLE:
        print(f"\n3D Point Cloud Files:")
        for method in ['jepa', 'mae', 'supervised']:
            print(f"  - {os.path.join(args.output_dir, f'embedding_3d_{method}.vtk')}")
            print(f"  - {os.path.join(args.output_dir, f'embedding_3d_{method}.ply')}")
        
        print(f"\nScreenshots and Animations:")
        for method in ['jepa', 'mae', 'supervised']:
            print(f"  - {os.path.join(args.output_dir, f'embedding_3d_{method}.png')}")
            print(f"  - {os.path.join(args.output_dir, f'embedding_3d_{method}_rotation.gif')}")
    
    print(f"\nRaw Data:")
    print(f"  - {json_path}")
    print("=" * 60)


if __name__ == '__main__':
    main()
