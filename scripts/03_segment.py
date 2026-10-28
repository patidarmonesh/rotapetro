#!/usr/bin/env python3
"""Phase 3 — Automatic grain segmentation (§9).

Script: scripts/03_segment.py
Usage:  python scripts/03_segment.py --config configs/default.yaml [--site 11agt25_r1]

Reads  : data/interim/aligned/<site_id>.npz
         data/interim/maps/<site_id>.npz
Writes : data/interim/segments/<site_id>.png  (uint16 label image)
         data/interim/segments/<site_id>.json (region list)
         data/interim/qc/segment/<site_id>.png
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import (load_config, load_aligned_stack,  # noqa: E402
                               load_maps, save_segments)
from src.rotapetro.segment import GrainSegmenter, save_segment_qc_png  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: automatic grain segmentation (§9)")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML config file")
    parser.add_argument(
        "--site", type=str, default=None,
        help="Optional: process only this site_id")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-segment even if outputs already exist")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("phase3")

    logger.info("─── Phase 3: Grain Segmentation ───")
    cfg = load_config(args.config)
    segmenter = GrainSegmenter(cfg)

    interim_dir = Path(cfg["paths"]["interim"])
    aligned_dir = interim_dir / "aligned"
    maps_dir = interim_dir / "maps"
    seg_dir = interim_dir / "segments"
    qc_dir = interim_dir / "qc" / "segment"
    seg_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # ── Determine which sites to process ──
    sites_csv = interim_dir / "sites.csv"
    if not sites_csv.is_file():
        logger.error("sites.csv not found at %s — run Phase 0 first", sites_csv)
        sys.exit(1)

    sites_df = pd.read_csv(sites_csv)
    usable = sites_df[sites_df["usable"] == True].copy()

    # Skip sites that failed Phase 1 alignment (same check as Phase 2)
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

    # Only process sites that have maps (Phase 2 completed)
    map_ids = {p.stem for p in maps_dir.glob("*.npz")}
    usable = usable[usable["site_id"].isin(map_ids)]

    if args.site:
        usable = usable[usable["site_id"] == args.site]
        if usable.empty:
            logger.error("Site %s not found or maps not available", args.site)
            sys.exit(1)

    logger.info("Processing %d sites", len(usable))

    n_ok = 0
    n_err = 0
    summary_rows: list[dict] = []

    for _, row in usable.iterrows():
        site_id = row["site_id"]
        aligned_path = aligned_dir / f"{site_id}.npz"
        maps_path = maps_dir / f"{site_id}.npz"
        seg_png = seg_dir / f"{site_id}.png"
        seg_json = seg_dir / f"{site_id}.json"

        if seg_png.is_file() and seg_json.is_file() and not args.force:
            logger.info("  %s: segments exist, skipping (use --force)", site_id)
            n_ok += 1
            continue

        if not aligned_path.is_file() or not maps_path.is_file():
            logger.warning("  %s: aligned/maps not found, skipping", site_id)
            n_err += 1
            continue

        logger.info("  %s: segmenting…", site_id)

        try:
            aligned = load_aligned_stack(str(aligned_path))
            maps_dict = load_maps(str(maps_path))

            label_img, region_list = segmenter.segment_site(
                aligned, maps_dict, site_id)

            # Build the §9.5 JSON structure
            um_per_px = cfg.get("acquisition", {}).get("um_per_px_rot", 4.0)
            seg_data = {
                "site_id": site_id,
                "um_per_px": um_per_px,
                "regions": region_list,
            }

            # Save
            save_segments(str(seg_png), str(seg_json), label_img, seg_data)

        except Exception as exc:
            logger.error("  %s: segmentation failed: %s", site_id, exc,
                         exc_info=True)
            n_err += 1
            continue

        # Count types
        type_counts: dict[str, int] = {}
        for r in region_list:
            t = r["type"]
            type_counts[t] = type_counts.get(t, 0) + 1

        n_total = len(region_list)
        n_grain = type_counts.get("grain", 0)
        n_opaque = type_counts.get("opaque", 0)
        n_agg = type_counts.get("aggregate", 0)
        n_hole = type_counts.get("hole", 0)
        n_partial = sum(1 for r in region_list if r.get("partial"))
        n_small = sum(1 for r in region_list if r.get("small"))

        logger.info(
            "  %s: %d regions (%d grain, %d opaque, %d aggregate, %d hole) "
            "| %d partial, %d small",
            site_id, n_total, n_grain, n_opaque, n_agg, n_hole,
            n_partial, n_small)

        summary_rows.append({
            "site_id": site_id,
            "n_regions": n_total,
            "n_grain": n_grain,
            "n_opaque": n_opaque,
            "n_aggregate": n_agg,
            "n_hole": n_hole,
            "n_partial": n_partial,
            "n_small": n_small,
        })

        # QC overlay PNG (§9.7)
        try:
            ppl_0 = aligned["ppl"][0]  # uint8 RGB
            xpl_maxproj = maps_dict["xpl_maxproj_rgb"]
            save_segment_qc_png(
                ppl_0, xpl_maxproj, label_img, region_list,
                site_id, qc_dir / f"{site_id}.png")
        except Exception as exc:
            logger.warning("  %s: QC PNG failed: %s", site_id, exc)

        n_ok += 1

    # Write summary CSV
    if summary_rows:
        summary_path = interim_dir / "qc" / "segment_summary.csv"
        new = pd.DataFrame(summary_rows)
        if summary_path.is_file() and args.site:      # keep other sites' rows
            old = pd.read_csv(summary_path)
            new = pd.concat([old[~old["site_id"].isin(new["site_id"])], new])
        new.to_csv(summary_path, index=False)
        logger.info("Segment summary written to %s", summary_path)

    logger.info("─── Phase 3 complete: %d OK, %d errors ───", n_ok, n_err)


if __name__ == "__main__":
    main()
