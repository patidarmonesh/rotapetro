import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import math

def plot_alignment_qc(ppl_frames, site_id, save_path):
    """Side-by-side PPL at 0/45/90/135 degrees."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    angles = [0, 45, 90, 135]
    indices = [int(a / 360 * len(ppl_frames)) for a in angles]
    for ax, idx, angle in zip(axes, indices, angles):
        if idx < len(ppl_frames):
            ax.imshow(ppl_frames[idx])
            ax.set_title(f'PPL {angle}°')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_maps_qc(maps_dict, ppl_0, xpl_0, site_id, save_path):
    """6-panel figure (PPL 0°, XPL 0°, XPL max proj, extinction colour map, anisotropy, pleochroism)"""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()
    
    axes[0].imshow(ppl_0)
    axes[0].set_title('PPL 0°')
    
    axes[1].imshow(xpl_0)
    axes[1].set_title('XPL 0°')
    
    axes[2].imshow(maps_dict.get('xpl_max', np.zeros_like(xpl_0)))
    axes[2].set_title('XPL Max Proj')
    
    axes[3].imshow(maps_dict.get('extinction', np.zeros_like(xpl_0)))
    axes[3].set_title('Extinction Colour')
    
    axes[4].imshow(maps_dict.get('anisotropy', np.zeros_like(xpl_0[..., 0])), cmap='gray')
    axes[4].set_title('Anisotropy')
    
    axes[5].imshow(maps_dict.get('pleochroism', np.zeros_like(xpl_0[..., 0])), cmap='gray')
    axes[5].set_title('Pleochroism')
    
    for ax in axes:
        ax.axis('off')
    
    plt.suptitle(f'Maps QC: {site_id}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_segment_overlay(ppl_0, xpl_max, label_img, region_list, site_id, save_path):
    """Boundaries on PPL and XPL"""
    import skimage.segmentation as seg
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    bound_ppl = seg.mark_boundaries(ppl_0, label_img, color=(1,0,0))
    axes[0].imshow(bound_ppl)
    axes[0].set_title('Segments on PPL')
    axes[0].axis('off')
    
    bound_xpl = seg.mark_boundaries(xpl_max, label_img, color=(1,0,0))
    axes[1].imshow(bound_xpl)
    axes[1].set_title('Segments on XPL Max')
    axes[1].axis('off')
    
    plt.suptitle(f'Segment Overlay: {site_id}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_feature_distributions(df, save_dir):
    """Histogram per feature column"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        plt.figure(figsize=(6,4))
        df[col].hist(bins=50)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig(save_dir / f"{col}_dist.png")
        plt.close()

def plot_contact_sheet(images, label, save_path, n=30):
    """Grid of 30 random crops"""
    n_images = len(images)
    if n_images == 0:
        return
    n = min(n, n_images)
    idx = np.random.choice(n_images, n, replace=False)
    selected = [images[i] for i in idx]
    
    cols = 6
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axes = np.atleast_2d(axes) if rows > 1 else axes.reshape(1, -1)
    
    for i, ax in enumerate(axes.flatten()):
        if i < n:
            ax.imshow(selected[i])
        ax.axis('off')
        
    plt.suptitle(f'Contact Sheet: {label}')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
