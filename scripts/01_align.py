#!/usr/bin/env python3
"""Phase 1 — Align rotation frames (§7).

Script: scripts/01_align.py
Usage:  python scripts/01_align.py --config configs/default.yaml [--site 11agt25_r1]

Reads  : data/interim/sites.csv + raw frames
Writes : data/interim/aligned/<site_id>.npz
         data/interim/qc/align/<site_id>.png
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

from src.rotapetro.io import (load_config, save_aligned_stack,   # noqa: E402
                               load_aligned_stack)
from src.rotapetro.align import FrameAligner, save_qc_png        # noqa: E402


def _verify_existing_qc(result: dict, raw_folder: str,
                         aligner: FrameAligner) -> dict:
    """Load the un-folded frames 000 and 036, warp 036, and run QC."""
    import cv2
    raw_folder = Path(raw_folder)

    ppl_000 = cv2.cvtColor(
        cv2.imread(str(raw_folder / "ppl" / "000.jpg")), cv2.COLOR_BGR2RGB)
    ppl_036_raw = cv2.cvtColor(
        cv2.imread(str(raw_folder / "ppl" / "036.jpg")), cv2.COLOR_BGR2RGB)

    M_036 = result["M"][36]
    bgr_036 = cv2.cvtColor(ppl_036_raw, cv2.COLOR_RGB2BGR)
    size = (aligner.frame_size, aligner.frame_size)
    warped_036 = cv2.warpAffine(
        bgr_036, M_036, size,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    ppl_warped_036 = cv2.cvtColor(warped_036, cv2.COLOR_BGR2RGB)

    return aligner.compute_qc(result, ppl_000, ppl_warped_036)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 1: align rotation frames (§7)")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML config file")
    parser.add_argument(
        "--site", type=str, default=None,
        help="Optional: process only this site_id (e.g. 11agt25_r1)")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run alignment even if the .npz already exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("phase1")

    logger.info("─── Phase 1: Align Rotation Frames ───")
    cfg = load_config(args.config)
    aligner = FrameAligner(cfg)

    interim_dir = Path(cfg["paths"]["interim"])
    aligned_dir = interim_dir / "aligned"
    qc_dir = interim_dir / "qc" / "align"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    # Load sites.csv and filter to usable
    sites_csv = interim_dir / "sites.csv"
    if not sites_csv.is_file():
        logger.error("sites.csv not found at %s — run Phase 0 first", sites_csv)
        sys.exit(1)

    sites_df = pd.read_csv(sites_csv)
    usable = sites_df[sites_df["usable"] == True].copy()

    if args.site:
        usable = usable[usable["site_id"] == args.site]
        if usable.empty:
            logger.error("Site %s not found or not usable", args.site)
            sys.exit(1)

    logger.info("Processing %d usable sites", len(usable))

    n_passed = 0
    n_failed = 0
    qc_rows: list[dict] = []

    for _, row in usable.iterrows():
        site_id = row["site_id"]
        raw_folder = row["raw_folder"]
        out_path = aligned_dir / f"{site_id}.npz"

        if out_path.is_file() and not args.force:
            # Load existing NPZ and verify QC instead of blindly passing
            logger.info("  %s: .npz exists, verifying QC…", site_id)
            try:
                result = load_aligned_stack(str(out_path))
                qc = _verify_existing_qc(result, raw_folder, aligner)
                qc_rows.append({"site_id": site_id, **qc})
                if qc["passed"]:
                    n_passed += 1
                    logger.info("  %s: existing PASS", site_id)
                else:
                    n_failed += 1
                    logger.warning(
                        "  %s: existing FAIL (angle_err=%.2f° fallbacks=%d "
                        "fold_diff=%.1f shift=%.2fpx) — re-run with --force",
                        site_id, qc["median_angle_err"],
                        qc["n_fallback"], qc["fold_diff_mean"],
                        qc["residual_shift_px"])
            except Exception as exc:
                logger.warning("  %s: could not verify existing .npz: %s",
                               site_id, exc)
                n_failed += 1
            continue

        logger.info("  %s: aligning…", site_id)

        try:
            result = aligner.align_site(raw_folder)
        except Exception as exc:
            logger.error("  %s: alignment failed: %s", site_id, exc)
            n_failed += 1
            continue

        # Save NPZ (§7.2)
        save_aligned_stack(
            out_path,
            ppl=result["ppl"],
            xpl=result["xpl"],
            theta_deg=result["theta_deg"],
            valid=result["valid"],
            M=result["M"],
            rot_measured_deg=result["rot_measured_deg"],
            rot_sign=result["rot_sign"],
            center_xy=result["center_xy"],
            align_fallback=result["align_fallback"],
            ecc_cc=result["ecc_cc"],
            ecc_ok=result["ecc_ok"],
        )

        # QC metrics (§7.3) — same helper as for existing files
        try:
            qc = _verify_existing_qc(result, raw_folder, aligner)
            qc_rows.append({"site_id": site_id, **qc})
            status = "PASS" if qc["passed"] else "FAIL"
            logger.info(
                "  %s: %s  angle_err=%.2f°  fallbacks=%d  fold_diff=%.1f  "
                "shift=%.2fpx  ecc_ok=%d/71",
                site_id, status, qc["median_angle_err"], qc["n_fallback"],
                qc["fold_diff_mean"], qc["residual_shift_px"],
                int(result["ecc_ok"][1:].sum()))
            if qc["passed"]:
                n_passed += 1
            else:
                n_failed += 1
        except Exception as exc:
            logger.warning("  %s: QC computation failed: %s", site_id, exc)
            qc_rows.append({"site_id": site_id, "passed": False})
            n_failed += 1  # QC failure = unverified = count as failed

        # Save QC PNG (§7.3)
        try:
            save_qc_png(result["ppl"], result["valid"], site_id,
                        qc_dir / f"{site_id}.png")
        except Exception as exc:
            logger.warning("  %s: QC PNG failed: %s", site_id, exc)

    # Phase 2 reads this file and skips sites with passed == False
    if qc_rows:
        summary_path = interim_dir / "qc" / "align_summary.csv"
        new = pd.DataFrame(qc_rows)
        if summary_path.is_file() and args.site:
            old = pd.read_csv(summary_path)
            new = pd.concat([old[~old["site_id"].isin(new["site_id"])], new])
        new.to_csv(summary_path, index=False)
        logger.info("QC summary written to %s", summary_path)

    total = n_passed + n_failed
    fail_pct = (n_failed / max(total, 1)) * 100
    logger.info("─── Phase 1 complete: %d passed, %d failed (%.1f%%) ───",
                n_passed, n_failed, fail_pct)

    if fail_pct > 5:
        logger.warning("  >5% failures — inspect QC images before continuing!")


if __name__ == "__main__":
    main()
