import argparse
import os
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm

def get_color(label):
    """Generate a consistent, bright color for a given string label using MD5 hash."""
    h = hashlib.md5(str(label).encode()).hexdigest()
    # Ensure colors are bright enough by setting a minimum floor of 50
    r = max(int(h[0:2], 16), 50)
    g = max(int(h[2:4], 16), 50)
    b = max(int(h[4:6], 16), 50)
    return (b, g, r)  # OpenCV uses BGR

def process_site(site_id, df, sites_df, outdir):
    # Filter labels for this site
    df_site = df[df['region_id'].str.startswith(site_id + "_")]
    
    # Only keep rows that actually have a label (fine or group)
    df_site = df_site[df_site['label_fine'].notna() | df_site['label_group'].notna()]
    
    if df_site.empty:
        return False

    # Find the exact raw folder path from sites.csv
    try:
        raw_folder = sites_df[sites_df['site_id'] == site_id].iloc[0]['raw_folder']
    except IndexError:
        print(f"Could not find {site_id} in sites.csv")
        return False

    # Find the 000.jpg image case-insensitively
    raw_dir = Path(raw_folder)
    img_path = None
    for ppl in ['ppl', 'PPL']:
        d = raw_dir / ppl
        if d.exists():
            for ext in ['.jpg', '.JPG']:
                p = d / f"000{ext}"
                if p.exists():
                    img_path = p
                    break
        if img_path: break
        
    if not img_path:
        print(f"Missing 000.jpg for {site_id}")
        return False
        
    img = cv2.imread(str(img_path))
    if img is None: 
        return False
    
    # Load segmentation mask
    seg_path = Path(f"data/interim/segments/{site_id}.png")
    if not seg_path.exists(): 
        return False
    mask = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
    if mask is None: 
        return False
    
    # Ensure dimensions match
    h, w = img.shape[:2]
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = img.copy()
    legend_labels = set()
    
    for _, row in df_site.iterrows():
        # Prefer fine label, fallback to group label
        label = row['label_fine'] if pd.notna(row['label_fine']) else row['label_group']
        if pd.isna(label): continue
            
        try:
            val = int(row['region_id'].split('_g')[-1])
        except ValueError:
            continue
            
        region_mask = mask == val
        if not np.any(region_mask): continue
        
        c = get_color(label)
        legend_labels.add((label, c))
        
        # Colorize the mask area
        overlay[region_mask] = c
        
        # Draw a crisp white boundary contour
        region_mask_uint8 = (region_mask).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)

    # Blend the colored overlay with the original image (45% overlay opacity)
    alpha = 0.45
    blended = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    
    # Expand canvas to the right for the legend
    legend_w = 400
    final_img = np.ones((h, w + legend_w, 3), dtype=np.uint8) * 240 # Light gray background
    final_img[:, :w] = blended
    
    # Draw the legend
    y = 50
    cv2.putText(final_img, f"Site: {site_id}", (w + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    y += 60
    cv2.putText(final_img, "Legend", (w + 20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    y += 40
    
    for label, c in sorted(list(legend_labels)):
        # Color swatch
        cv2.rectangle(final_img, (w + 20, y - 20), (w + 60, y + 20), c, -1)
        cv2.rectangle(final_img, (w + 20, y - 20), (w + 60, y + 20), (100,100,100), 1)
        # Text label
        cv2.putText(final_img, str(label), (w + 80, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
        y += 50
        
    out_path = os.path.join(outdir, f"{site_id}_color_labeled.jpg")
    cv2.imwrite(out_path, final_img)
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", type=str, help="Process a specific site ID")
    parser.add_argument("--all", action="store_true", help="Process all sites that have labels")
    parser.add_argument("--outdir", type=str, default="data/interim/qc/labeled_slides_clean", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    
    print("Loading labels.parquet and sites.csv...")
    df = pd.read_parquet("data/processed/labels.parquet")
    sites_df = pd.read_csv("data/interim/sites.csv")
    
    if args.site:
        print(f"Processing {args.site}...")
        success = process_site(args.site, df, sites_df, args.outdir)
        if success:
            print(f"Done! Saved to {args.outdir}/{args.site}_color_labeled.jpg")
        else:
            print("Failed to process site.")
            
    elif args.all:
        labeled_df = df[df['label_fine'].notna() | df['label_group'].notna()]
        # Extract site IDs correctly by dropping the last part (the _gXXX grain ID suffix)
        site_ids = labeled_df['region_id'].apply(lambda x: "_".join(x.split('_')[:-1])).unique()
        
        print(f"Found {len(site_ids)} sites with labels. Generating colored overlays for all...")
        success_count = 0
        for s in tqdm(site_ids):
            if process_site(s, df, sites_df, args.outdir):
                success_count += 1
                
        print(f"Done! Successfully generated {success_count} overlaid images in {args.outdir}/")
    else:
        print("Please specify --site <id> or use --all")

if __name__ == "__main__":
    main()
