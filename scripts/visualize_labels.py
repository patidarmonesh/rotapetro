import argparse
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
import cv2

def main():
    parser = argparse.ArgumentParser(description="Visualize labels overlaid on a thin section")
    parser.add_argument("--site", type=str, required=True, help="Site ID (e.g., 11agt17_r1)")
    parser.add_argument("--outdir", type=str, default="data/interim/qc/labeled_slides", help="Output directory")
    args = parser.parse_args()

    site_id = args.site
    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, f"{site_id}_labeled.jpg")

    # 1. Load the original PPL frame (000.jpg)
    parts = site_id.split('_r')
    if len(parts) != 2:
        print(f"Error: Invalid site ID {site_id}")
        return
    code = parts[0]
    r_num = "r" + parts[1]
    
    # Try finding the raw frame case-insensitively like in io.py
    raw_dir = Path(f"data/raw/ukvm_dataset/specimens/{code}/rot/{r_num}")
    img_path = None
    if raw_dir.exists():
        for ppl in ['ppl', 'PPL']:
            d = raw_dir / ppl
            if d.exists():
                for f in d.iterdir():
                    if f.name.lower() == '000.jpg':
                        img_path = f
                        break
            if img_path: break
            
    if not img_path or not img_path.exists():
        print(f"Could not find 000.jpg for site {site_id}")
        return

    print(f"Loading image {img_path}...")
    img = cv2.imread(str(img_path))
    if img is None:
        print("Failed to load image.")
        return
        
    h, w = img.shape[:2]

    # 2. Load segmentation mask
    seg_path = Path(f"data/interim/segments/{site_id}.png")
    if not seg_path.exists():
        print(f"Segmentation mask not found: {seg_path}")
        return
    
    print(f"Loading mask {seg_path}...")
    mask = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED) # 16-bit
    if mask is None:
        print("Failed to load mask.")
        return
        
    # Resize mask if it differs from raw image (though they should match or be 490x490 mapped)
    # Actually, segments are output at the original image size, or aligned size?
    # Usually segments are at the target resolution, let's just resize the mask to the image if needed.
    if mask.shape[:2] != (h, w):
        print(f"Resizing mask {mask.shape} to match image {(h, w)}")
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. Load labels
    labels_path = "data/processed/labels.parquet"
    if not os.path.exists(labels_path):
        print("labels.parquet not found!")
        return
        
    print("Loading labels...")
    df = pd.read_parquet(labels_path)
    # Filter for this site
    df_site = df[df['region_id'].str.startswith(site_id + "_")]
    
    if df_site.empty:
        print(f"No labels found for site {site_id}")
        return

    # Create a color mapping for different minerals
    unique_minerals = df_site['label_fine'].dropna().unique()
    colors = {}
    np.random.seed(42)
    for m in unique_minerals:
        colors[m] = (int(np.random.randint(50, 255)), 
                     int(np.random.randint(50, 255)), 
                     int(np.random.randint(50, 255)))

    # Draw outlines and labels
    print("Drawing overlays...")
    overlay = img.copy()
    
    for _, row in df_site.iterrows():
        label = row['label_fine']
        if pd.isna(label):
            # Try group label
            label = row['label_group']
            if pd.isna(label):
                continue
                
        # Get region ID and mask value. region_id format: {site_id}_g{val:03d}
        try:
            val_str = row['region_id'].split('_g')[-1]
            val = int(val_str)
        except ValueError:
            continue
            
        # Find pixels for this region
        region_mask = (mask == val).astype(np.uint8) * 255
        
        # Find contours
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
            
        c = colors.get(label, (0, 255, 255)) # Default yellow if not in fine colors
        
        # Draw contour
        cv2.drawContours(overlay, contours, -1, c, 2)
        
        # Put text at centroid
        M = cv2.moments(contours[0])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Draw text background
            text = str(label)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            
            cv2.rectangle(overlay, (cX - text_size[0]//2 - 5, cY - text_size[1]//2 - 5),
                          (cX + text_size[0]//2 + 5, cY + text_size[1]//2 + 5), (0, 0, 0), -1)
            cv2.putText(overlay, text, (cX - text_size[0]//2, cY + text_size[1]//2), 
                        font, font_scale, (255, 255, 255), thickness)

    # Blend overlay with original image
    alpha = 0.7
    final_img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    print(f"Saving to {out_path}...")
    cv2.imwrite(out_path, final_img)
    print("Done!")

if __name__ == "__main__":
    main()
