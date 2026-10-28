#!/usr/bin/env python3
"""Phase 2 — Optical property maps (§8).

Script: scripts/02_maps.py
Usage:  python scripts/02_maps.py --config configs/default.yaml [--site 11agt25_r1]

Reads  : data/interim/aligned/<site_id>.npz
Writes : data/interim/maps/<site_id>.npz
         data/interim/qc/maps/<site_id>.png
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config, load_aligned_stack, save_maps  # noqa: E402
from src.rotapetro.maps import PropertyMapper                            # noqa: E402
from src.rotapetro.michel_levy import generate_michel_levy_lut           # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: compute optical property maps (§8)")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML config file")
    parser.add_argument(
        "--site", type=str, default=None,
        help="Optional: process only this site_id")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-compute maps even if the .npz already exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("phase2")

    logger.info("─── Phase 2: Optical Property Maps ───")
    cfg = load_config(args.config)

    interim_dir = Path(cfg["paths"]["interim"])
    aligned_dir = interim_dir / "aligned"
    maps_dir = interim_dir / "maps"
    qc_dir = interim_dir / "qc" / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # ── Generate Michel-Lévy LUT if it doesn't exist ──
    knowledge_dir = Path(cfg["paths"].get("knowledge", "knowledge"))
    lut_path = knowledge_dir / "michel_levy_lut.npz"
    if not lut_path.is_file():
        logger.info("Generating Michel-Lévy LUT → %s", lut_path)
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        generate_michel_levy_lut(str(lut_path))
    else:
        logger.info("Michel-Lévy LUT exists at %s", lut_path)

    mapper = PropertyMapper(cfg, lut_path=str(lut_path))

    # ── Load sites to process ──
    sites_csv = interim_dir / "sites.csv"
    if not sites_csv.is_file():
        logger.error("sites.csv not found at %s — run Phase 0 first", sites_csv)
        sys.exit(1)

    sites_df = pd.read_csv(sites_csv)
    usable = sites_df[sites_df["usable"] == True].copy()

    # If align_summary.csv exists, skip sites that failed Phase 1
    align_summary = interim_dir / "qc" / "align_summary.csv"
    if align_summary.is_file():
        align_df = pd.read_csv(align_summary)
        passed_ids = set(
            align_df.loc[align_df["passed"] == True, "site_id"])
        before = len(usable)
        usable = usable[usable["site_id"].isin(passed_ids)]
        skipped = before - len(usable)
        if skipped > 0:
            logger.info("Skipped %d sites that failed Phase 1 alignment", skipped)

    if args.site:
        usable = usable[usable["site_id"] == args.site]
        if usable.empty:
            logger.error("Site %s not found or not usable/aligned", args.site)
            sys.exit(1)

    logger.info("Processing %d sites", len(usable))

    n_ok = 0
    n_err = 0

    for _, row in usable.iterrows():
        site_id = row["site_id"]
        aligned_path = aligned_dir / f"{site_id}.npz"
        out_path = maps_dir / f"{site_id}.npz"

        if out_path.is_file() and not args.force:
            logger.info("  %s: maps exist, skipping (use --force)", site_id)
            n_ok += 1
            continue

        if not aligned_path.is_file():
            logger.warning("  %s: aligned .npz not found, skipping", site_id)
            n_err += 1
            continue

        logger.info("  %s: computing maps…", site_id)

        try:
            aligned = load_aligned_stack(str(aligned_path))
            maps_dict = mapper.compute_maps(aligned)
            save_maps(str(out_path), maps_dict)
        except Exception as exc:
            logger.error("  %s: map computation failed: %s", site_id, exc,
                         exc_info=True)
            n_err += 1
            continue

        # QC PNG (§8.7): 6-panel figure
        try:
            _save_qc_png(maps_dict, aligned, site_id, qc_dir / f"{site_id}.png")
        except Exception as exc:
            logger.warning("  %s: QC PNG failed: %s", site_id, exc)

        n_ok += 1

    logger.info("─── Phase 2 complete: %d OK, %d errors ───", n_ok, n_err)


def _save_qc_png(maps_dict: dict, aligned: dict,
                 site_id: str, save_path: Path) -> None:
    """§8.7: 6-panel QC figure.

    Panels: PPL 0°, XPL 0°, XPL max projection, extinction colour map,
    anisotropy, pleochroism.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ppl_0 = aligned["ppl"][0]   # [H,W,3] uint8 RGB
    xpl_0 = aligned["xpl"][0]
    valid = maps_dict["valid"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Panel 1: PPL 0°
    axes[0, 0].imshow(ppl_0)
    axes[0, 0].set_title("PPL 0°")
    axes[0, 0].axis("off")

    # Panel 2: XPL 0°
    axes[0, 1].imshow(xpl_0)
    axes[0, 1].set_title("XPL 0°")
    axes[0, 1].axis("off")

    # Panel 3: XPL max projection
    xpl_maxproj = maps_dict["xpl_maxproj_rgb"]
    if xpl_maxproj.dtype != np.uint8:
        xpl_maxproj = np.clip(xpl_maxproj, 0, 255).astype(np.uint8)
    axes[0, 2].imshow(xpl_maxproj)
    axes[0, 2].set_title("XPL max projection")
    axes[0, 2].axis("off")

    # Panel 4: Extinction colour map (ext_deg as HSV hue)
    ext = np.nan_to_num(maps_dict["ext_deg"].astype(np.float32), nan=0)
    aniso = np.nan_to_num(maps_dict["aniso"].astype(np.float32), nan=0)
    # OpenCV 8-bit hue runs 0-179 (not 0-255).  Hue = extinction angle,
    # brightness = anisotropy (isotropic / opaque pixels appear dark).
    import cv2
    shown = maps_dict.get("aniso_mask", valid).astype(bool)
    hsv = np.zeros((*ext.shape, 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(ext / 90.0 * 179.0, 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = (np.clip(aniso * 1.5, 0, 1) * 255 * shown).astype(np.uint8)
    ext_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    axes[1, 0].imshow(ext_rgb)
    axes[1, 0].set_title("Extinction position")
    axes[1, 0].axis("off")

    # Panel 5: Anisotropy
    aniso_show = np.where(valid, aniso, 0)
    axes[1, 1].imshow(aniso_show, cmap="inferno", vmin=0, vmax=1)
    axes[1, 1].set_title("Anisotropy")
    axes[1, 1].axis("off")

    # Panel 6: Pleochroism
    pleo = np.nan_to_num(maps_dict["pleo_dE"].astype(np.float32), nan=0)
    pleo_show = np.where(valid, pleo, 0)
    axes[1, 2].imshow(pleo_show, cmap="magma", vmin=0, vmax=30)
    axes[1, 2].set_title("Pleochroism ΔE")
    axes[1, 2].axis("off")

    fig.suptitle(site_id, fontsize=14)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
