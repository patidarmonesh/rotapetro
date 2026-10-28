#!/usr/bin/env python3
"""Phase 4 — Grain property cards (§10).

Script: scripts/04_features.py
Usage:  python scripts/04_features.py --config configs/default.yaml [--site 11agt25_r1]

Reads  : data/interim/aligned/<site_id>.npz
         data/interim/maps/<site_id>.npz
         data/interim/segments/<site_id>.json  (+ .png)
Writes : data/processed/regions.parquet
         data/interim/qc/features/*.png  (distribution plots)
"""
from __future__ import annotations

import argparse
import hashlib
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
                               load_maps, load_segments, validate_v1_config)
from src.rotapetro.features import FeatureExtractor, FEATURE_VERSION  # noqa: E402


def _site_input_signature(site_id: str, aligned_dir: Path, maps_dir: Path, seg_dir: Path) -> str:
    """Compute signature for a site's input files based on mtime and size."""
    files = [
        aligned_dir / f"{site_id}.npz",
        maps_dir / f"{site_id}.npz",
        seg_dir / f"{site_id}.png",
        seg_dir / f"{site_id}.json",
    ]
    parts = []
    for f in files:
        if f.is_file():
            st = f.stat()
            parts.append(f"{f.name}:{st.st_mtime_ns}:{st.st_size}")
        else:
            parts.append(f"{f.name}:missing")
    return hashlib.sha256(";".join(parts).encode("utf-8")).hexdigest()[:16]


def _compute_features_fingerprint(cfg: dict) -> tuple[str, str, str]:
    """Compute feature extraction provenance fingerprint across config and code."""
    # 1. Config hash (features and v1 sections)
    feat_cfg = cfg.get("features", {})
    v1_cfg = cfg.get("v1", {})
    cfg_bytes = json.dumps({"features": feat_cfg, "v1": v1_cfg}, sort_keys=True, default=str).encode("utf-8")
    cfg_hash = hashlib.sha256(cfg_bytes).hexdigest()[:16]

    # 2. Code hash (src/rotapetro/features.py, scripts/04_features.py)
    ch = hashlib.sha256()
    for f in [
        _PROJECT_ROOT / "scripts" / "04_features.py",
        _PROJECT_ROOT / "src" / "rotapetro" / "features.py",
    ]:
        if f.is_file():
            ch.update(f.name.encode("utf-8"))
            ch.update(f.read_bytes().replace(b"\r\n", b"\n"))
    ch.update(FEATURE_VERSION.encode("utf-8"))
    code_hash = ch.hexdigest()[:16]

    fp = hashlib.sha256(f"{FEATURE_VERSION}:{cfg_hash}:{code_hash}".encode("utf-8")).hexdigest()[:16]
    return fp, cfg_hash, code_hash


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: extract grain property cards (§10)")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML config file")
    parser.add_argument(
        "--site", type=str, default=None,
        help="Optional: process only this site_id")
    parser.add_argument(
        "--force", action="store_true",
        help="Re-extract even if regions.parquet exists")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("phase4")

    logger.info("─── Phase 4: Feature Extraction ───")
    cfg = load_config(args.config)
    validate_v1_config(cfg)
    extractor = FeatureExtractor(cfg)

    interim_dir = Path(cfg["paths"]["interim"])
    processed_dir = _PROJECT_ROOT / cfg["v1"]["paths"]["processed"]
    aligned_dir = interim_dir / "aligned"
    maps_dir = interim_dir / "maps"
    seg_dir = interim_dir / "segments"
    qc_dir = interim_dir / "qc" / "features_v1"
    processed_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)

    out_path = processed_dir / "regions.parquet"
    legacy = _PROJECT_ROOT / "data/processed/regions.parquet"
    assert out_path.resolve() != legacy.resolve(), "refusing to overwrite legacy regions.parquet"

    # ── Determine which sites to process ──
    sites_csv = interim_dir / "sites.csv"
    if not sites_csv.is_file():
        logger.error("sites.csv not found at %s — run Phase 0 first", sites_csv)
        sys.exit(1)

    sites_df = pd.read_csv(sites_csv)
    usable = sites_df[sites_df["usable"] == True].copy()

    # Skip sites that failed Phase 1
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

    # Only process sites that have segments
    seg_ids = {p.stem for p in seg_dir.glob("*.json")}
    usable = usable[usable["site_id"].isin(seg_ids)]

    if args.site:
        usable = usable[usable["site_id"] == args.site]
        if usable.empty:
            logger.error("Site %s not found or segments not available", args.site)
            sys.exit(1)

    logger.info("Processing %d sites", len(usable))

    # Provenance and fingerprinting
    prov_path = processed_dir / "features_provenance.json"
    fingerprint, cfg_hash, code_hash = _compute_features_fingerprint(cfg)
    logger.info("Features fingerprint: %s (cfg: %s, code: %s)", fingerprint, cfg_hash, code_hash)

    # Load existing provenance if available
    existing_prov: dict = {}
    if prov_path.is_file():
        try:
            with open(prov_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    existing_prov = loaded
        except Exception as e:
            logger.warning("Could not read features_provenance.json: %s", e)

    stored_fp = existing_prov.get("fingerprint")

    # Guard --site: if --site given, regions.parquet exists, and stored fingerprint missing or != current
    if args.site and out_path.is_file():
        if not stored_fp or stored_fp != fingerprint:
            sys.exit(
                "Feature code/config changed since regions.parquet was built; "
                "a single-site update would mix versions. Do a full rebuild."
            )

    # Load existing parquet if not --force (incremental mode with provenance check)
    existing_df = None
    if out_path.is_file() and not args.force and not args.site:
        if not prov_path.is_file() or not existing_prov or not stored_fp or stored_fp != fingerprint:
            logger.warning(
                "Features provenance missing, malformed, or fingerprint mismatch (stored=%s, current=%s); re-extracting all sites...",
                stored_fp, fingerprint,
            )
            existing_df = None
        else:
            try:
                existing_df = pd.read_parquet(out_path)
                site_sigs = existing_prov.get("site_signatures", {})
                site_fps = existing_prov.get("site_fingerprints", {})

                # Check each site for input freshness & fingerprint match
                stale_sites = set()
                existing_sites = set(existing_df["site_id"])
                for s in existing_sites:
                    cur_sig = _site_input_signature(s, aligned_dir, maps_dir, seg_dir)
                    recorded_sig = site_sigs.get(s)
                    recorded_fp = site_fps.get(s)
                    if recorded_fp != fingerprint or recorded_sig != cur_sig:
                        stale_sites.add(s)

                if stale_sites:
                    logger.info("Found %d stale sites with updated input files or outdated fingerprint; re-extracting them", len(stale_sites))
                    existing_df = existing_df[~existing_df["site_id"].isin(stale_sites)]
                    valid_existing = existing_sites - stale_sites
                else:
                    valid_existing = existing_sites

                before = len(usable)
                usable = usable[~usable["site_id"].isin(valid_existing)]
                logger.info(
                    "Skipping %d already-extracted up-to-date sites (%d remaining to process)",
                    before - len(usable), len(usable),
                )
            except Exception as e:
                logger.warning("Failed to load existing regions.parquet: %s; re-extracting all", e)
                existing_df = None


    all_rows: list[dict] = []
    n_ok = 0
    n_err = 0
    failed_sites: list[str] = []

    for _, row in usable.iterrows():
        site_id = row["site_id"]
        code = row.get("code", site_id.rsplit("_r", 1)[0])
        aligned_path = aligned_dir / f"{site_id}.npz"
        maps_path = maps_dir / f"{site_id}.npz"
        seg_png = seg_dir / f"{site_id}.png"
        seg_json = seg_dir / f"{site_id}.json"

        if not all(p.is_file() for p in (aligned_path, maps_path, seg_png, seg_json)):
            logger.warning("  %s: missing input files, skipping", site_id)
            n_err += 1
            failed_sites.append(site_id)
            continue

        logger.info("  %s: extracting features…", site_id)

        try:
            aligned = load_aligned_stack(str(aligned_path))
            maps_dict = load_maps(str(maps_path))
            label_img, seg_data = load_segments(str(seg_png), str(seg_json))

            # seg_data can be either the full dict or just the regions list
            if isinstance(seg_data, dict):
                region_list = seg_data.get("regions", [])
                # §Fix 7: site-specific um_per_px from Phase 3 JSON
                site_um = seg_data.get("um_per_px", None)
            else:
                region_list = seg_data
                site_um = None

            rows = extractor.extract_site(
                label_img, region_list, aligned, maps_dict,
                site_id, code, um_per_px=site_um)

            # Task D1.1: Extract single-angle feature columns
            k1_frame_idx = cfg.get("v1", {}).get("training", {}).get("primary_test_angle_index", 1)
            k1_rows = extractor.extract_single_angle(
                label_img, region_list, aligned, maps_dict,
                site_id, frame_idx=k1_frame_idx)
            k1_by_id = {r["region_id"]: r for r in k1_rows}
            for r in rows:
                rid = r["region_id"]
                if rid in k1_by_id:
                    for k, v in k1_by_id[rid].items():
                        if k != "region_id":
                            r[k] = v

            all_rows.extend(rows)
            logger.info("  %s: %d regions extracted", site_id, len(rows))
            n_ok += 1

        except Exception as exc:
            logger.error("  %s: feature extraction failed: %s", site_id, exc,
                         exc_info=True)
            n_err += 1
            failed_sites.append(site_id)

    # ── Build and save parquet ──
    if all_rows:
        new_df = pd.DataFrame(all_rows)

        from src.rotapetro.features import V1_STATUS_COLUMNS, V1_SINGLE_ANGLE_COLUMNS
        missing = [c for c in V1_STATUS_COLUMNS + V1_SINGLE_ANGLE_COLUMNS + ["eligible_v1", "reject_reason_v1", "interior_px"]
                   if c not in new_df.columns]
        if missing:
            raise RuntimeError(f"V1 feature columns missing: {missing}")

        # Merge with existing if incremental
        if existing_df is not None:
            existing_df = existing_df[~existing_df["site_id"].isin(new_df["site_id"])]
            new_df = pd.concat([existing_df, new_df], ignore_index=True)

        # If --site, merge into existing
        if args.site and out_path.is_file():
            old_df = pd.read_parquet(out_path)
            new_df = pd.concat(
                [old_df[~old_df["site_id"].isin(new_df["site_id"])], new_df],
                ignore_index=True)

        new_df.to_parquet(out_path, index=False)
        logger.info("Saved %d regions to %s", len(new_df), out_path)

        # Save provenance record
        extracted_sites = set(r["site_id"] for r in all_rows)
        all_final_sites = set(new_df["site_id"].unique())

        old_sigs = existing_prov.get("site_signatures", {}) if isinstance(existing_prov, dict) else {}
        old_fps = existing_prov.get("site_fingerprints", {}) if isinstance(existing_prov, dict) else {}

        site_signatures = {}
        site_fingerprints = {}
        for s in sorted(all_final_sites):
            if s in extracted_sites or s not in old_sigs:
                site_signatures[s] = _site_input_signature(s, aligned_dir, maps_dir, seg_dir)
            else:
                site_signatures[s] = old_sigs[s]

            if s in extracted_sites or s not in old_fps:
                site_fingerprints[s] = fingerprint
            else:
                site_fingerprints[s] = old_fps[s]

        prov_record = {
            "feature_version": FEATURE_VERSION,
            "fingerprint": fingerprint,
            "config_hash": cfg_hash,
            "code_hash": code_hash,
            "site_signatures": site_signatures,
            "site_fingerprints": site_fingerprints,
        }
        with open(prov_path, "w", encoding="utf-8") as f:
            json.dump(prov_record, f, indent=2)
        logger.info("Saved features provenance to %s", prov_path)

        # ── QC: distribution plots ──
        try:
            _save_distribution_plots(new_df, qc_dir)
            logger.info("Distribution plots saved to %s", qc_dir)
        except Exception as exc:
            logger.warning("QC plots failed: %s", exc)

    logger.info("─── Phase 4 complete: %d OK, %d errors ───", n_ok, n_err)

    is_v1 = bool(cfg.get("v1"))
    if is_v1 and n_err > 0:
        logger.error(
            "Feature extraction failed on %d sites in V1: %s",
            n_err,
            failed_sites,
        )
        sys.exit(1)


def _save_distribution_plots(df: pd.DataFrame, qc_dir: Path) -> None:
    """Generate histogram per numeric feature column."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Skip columns that are all NaN or constant
    plot_cols = [c for c in numeric_cols
                 if df[c].notna().sum() > 5 and df[c].nunique() > 1]

    # Limit to interesting feature columns
    skip = {"label_value", "feature_version"}
    plot_cols = [c for c in plot_cols if c not in skip]

    n = len(plot_cols)
    if n == 0:
        return

    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).flatten()

    for i, col in enumerate(plot_cols):
        ax = axes[i]
        vals = df[col].dropna()
        ax.hist(vals, bins=30, edgecolor="none", alpha=0.7)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=6)

    # Hide unused
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(str(qc_dir / "feature_distributions.png"), dpi=100)
    plt.close(fig)

    # Per-feature individual plots for key columns
    key_cols = ["biref_cls", "aniso_med", "gamma_med_nm", "ppl_chroma",
                "pleo_dE_med", "ext_std_deg", "aspect", "boundary_contrast"]
    for col in key_cols:
        if col not in df.columns:
            continue
        vals = df[col].dropna()
        if len(vals) < 5:
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(vals, bins=40, edgecolor="none", alpha=0.7)
        ax.set_xlabel(col)
        ax.set_ylabel("Count")
        ax.set_title(f"{col} (n={len(vals)})")
        plt.tight_layout()
        plt.savefig(str(qc_dir / f"{col}.png"), dpi=100)
        plt.close(fig)


if __name__ == "__main__":
    main()
