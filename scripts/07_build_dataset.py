#!/usr/bin/env python3
"""Phase 7 — Build training dataset and splits (§13).

Script: scripts/07_build_dataset.py
Inputs:  data/processed/regions.parquet, data/processed/labels.parquet,
         data/processed/classes.yaml, data/interim/specimens_clean.csv,
         data/interim/sites.csv
Outputs: data/processed/splits/folds.json,
         data/processed/splits/class_counts.json,
         data/processed/samples.parquet
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config, validate_v1_config
from src.rotapetro.dataset import (
    add_segment_geometry,
    build_folds,
    compute_class_counts,
    filter_samples,
)
from src.rotapetro.knowledge import KnowledgeBase

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 7: build dataset splits and sample index")
    parser.add_argument("--config", type=str,
                        default=str(_PROJECT_ROOT / "configs" / "default.yaml"),
                        help="Path to config YAML")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing splits")
    parser.add_argument("--v1", action="store_true",
                        help="Run V1 dataset build (reads data/processed/v1/regions.parquet, writes samples.parquet)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    validate_v1_config(cfg)

    if args.v1:
        _run_v1_build_dataset(cfg, args)
        return

    paths = cfg.get("paths", {})

    processed_dir = _PROJECT_ROOT / paths.get("processed", "data/processed")
    interim_dir = _PROJECT_ROOT / paths.get("interim", "data/interim")
    splits_dir = processed_dir / "splits"

    # ── Load inputs ─────────────────────────────────────────────────
    regions_path = processed_dir / "regions.parquet"
    if not regions_path.is_file():
        logger.error("regions.parquet not found — run Phase 4 first")
        sys.exit(1)
    regions_df = pd.read_parquet(regions_path)
    logger.info("Loaded %d regions", len(regions_df))

    labels_path = processed_dir / "labels.parquet"
    if not labels_path.is_file():
        logger.error("labels.parquet not found — run Phase 6 first")
        sys.exit(1)
    labels_df = pd.read_parquet(labels_path)
    logger.info("Loaded %d labels", len(labels_df))

    classes_path = processed_dir / "classes.yaml"
    if not classes_path.is_file():
        logger.error("classes.yaml not found — run Phase 6 first")
        sys.exit(1)
    with open(classes_path, "r", encoding="utf-8") as f:
        classes = yaml.safe_load(f)
    logger.info("Classes: %d fine_classes", len(classes.get("fine_classes", [])))

    specimens_path = interim_dir / "specimens_clean.csv"
    if not specimens_path.is_file():
        logger.error("specimens_clean.csv not found — run Phase 0 first")
        sys.exit(1)
    specimens_df = pd.read_csv(specimens_path)
    logger.info("Loaded %d specimens", len(specimens_df))

    sites_path = interim_dir / "sites.csv"
    if not sites_path.is_file():
        logger.error("sites.csv not found — run Phase 0 first")
        sys.exit(1)
    sites_df = pd.read_csv(sites_path)
    site_to_code = {
        str(row["site_id"]): str(row["code"])
        for _, row in sites_df.iterrows()
    }

    # ── 1. Filter samples (§13.1) ──────────────────────────────────
    samples_df = filter_samples(regions_df, labels_df)

    # Add code column from site_to_code
    if "code" not in samples_df.columns:
        samples_df["code"] = samples_df["site_id"].map(
            lambda sid: site_to_code.get(str(sid), ""))

    # label_value / bbox_xywh / centroid_xy are needed for crops & masks
    samples_df = add_segment_geometry(samples_df, interim_dir / "segments")

    n_labelled = samples_df["label_fine"].notna().sum()
    n_unlabelled = samples_df["label_fine"].isna().sum()
    logger.info("Samples: %d total (%d labelled, %d unlabelled)",
                len(samples_df), n_labelled, n_unlabelled)

    # Save samples index
    samples_path = processed_dir / "samples.parquet"
    samples_df.to_parquet(samples_path, index=False)
    logger.info("Saved samples.parquet (%d rows)", len(samples_df))

    # ── 2. Build folds (§13.4) ─────────────────────────────────────
    folds_path = splits_dir / "folds.json"
    if folds_path.is_file() and not args.force:
        logger.info("folds.json already exists — loading (use --force "
                     "to regenerate)")
        with open(folds_path, "r", encoding="utf-8") as f:
            folds = json.load(f)
    else:
        # Only specimens that actually have samples (a specimen whose
        # sites all failed alignment/segmentation would make a fold empty)
        have = set(samples_df["code"].astype(str))
        spec_used = specimens_df[specimens_df["code"].astype(str).isin(have)]
        dropped = sorted(set(specimens_df["code"].astype(str)) - have)
        if dropped:
            logger.warning("%d specimens have no samples and are left out of "
                           "the folds: %s", len(dropped), dropped[:20])
        folds = build_folds(spec_used, cfg, output_path=folds_path)

    # Report fold sizes
    for fold_name, split in folds.items():
        logger.info("  %s: train=%d, val=%d, test=%d specimens",
                     fold_name,
                     len(split["train"]),
                     len(split["val"]),
                     len(split["test"]))

    # Validate: no overlap between train/val/test within each fold
    for fold_name, split in folds.items():
        train_set = set(split["train"])
        val_set = set(split["val"])
        test_set = set(split["test"])
        if train_set & val_set:
            logger.error("%s: train/val overlap: %s",
                         fold_name, train_set & val_set)
        if train_set & test_set:
            logger.error("%s: train/test overlap: %s",
                         fold_name, train_set & test_set)
        if val_set & test_set:
            logger.error("%s: val/test overlap: %s",
                         fold_name, val_set & test_set)

    # ── 3. Class imbalance bookkeeping (§13.5) ─────────────────────
    # Use the filtered samples_df (not raw labels_df) to ensure only
    # eligible training regions are counted.
    fine_classes = classes.get("fine_classes", [])

    # ── Group vocabulary + members (for the group head and the
    #    partial-credit loss on group-only labels, §15.1) ──
    kb = KnowledgeBase(_PROJECT_ROOT / paths.get("knowledge", "knowledge"), cfg)
    relabel = classes.get("relabel_map", {})
    used_groups = set(samples_df["label_group"].dropna().astype(str))
    used_groups |= {kb.get_group(c) for c in fine_classes}
    used_groups = sorted(g for g in used_groups if g and g != "unknown")
    group_members: dict[str, list[str]] = {}
    for g in used_groups:
        members = [g] if g == "other" else (kb.expand_group(g) or [g])
        finals = sorted({relabel.get(m, m) for m in members} & set(fine_classes))
        group_members[g] = finals
        if not finals:
            logger.warning("group '%s' has no member among fine classes", g)
    groups_path = processed_dir / "groups.yaml"
    with open(groups_path, "w", encoding="utf-8") as f:
        yaml.dump({"group_classes": used_groups, "group_members": group_members},
                  f, sort_keys=False)
    logger.info("Saved %d groups to %s", len(used_groups), groups_path)
    groups_set = used_groups

    fold_counts = compute_class_counts(
        samples_df, folds, fine_classes, groups_set, site_to_code)

    counts_path = splits_dir / "class_counts.json"
    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(counts_path, "w", encoding="utf-8") as f:
        json.dump(fold_counts, f, indent=2)
    logger.info("Saved per-fold class counts to %s", counts_path)

    # Report per-fold training class distribution
    for fold_name, counts in fold_counts.items():
        fine_total = sum(counts["fine_counts"].values())
        group_total = sum(counts["group_counts"].values())
        logger.info("  %s: %d fine-labelled training regions, "
                     "%d group-labelled",
                     fold_name, fine_total, group_total)
        # Show top-5 fine classes
        sorted_fine = sorted(counts["fine_counts"].items(),
                             key=lambda x: -x[1])[:5]
        for cls, n in sorted_fine:
            if n > 0:
                logger.info("    %-20s %d (%.1f%%)",
                            cls, n, 100 * n / max(fine_total, 1))

    # ── Summary ─────────────────────────────────────────────────────
    logger.info("Phase 7 complete.")
    logger.info("  samples.parquet:   %s (%d rows)", samples_path,
                len(samples_df))
    logger.info("  folds.json:        %s (%d folds)", folds_path,
                len(folds))
    logger.info("  class_counts.json: %s", counts_path)


def _run_v1_build_dataset(cfg: dict, args: argparse.Namespace) -> None:
    """Build V1 samples.parquet and copy frozen folds.json (§7a)."""
    import hashlib
    import shutil
    import numpy as np
    from src.rotapetro.io import resolve_v1_paths

    v1_paths = resolve_v1_paths(cfg)
    processed_v1 = _PROJECT_ROOT / v1_paths["processed"]
    processed_v1.mkdir(parents=True, exist_ok=True)
    splits_v1 = processed_v1 / "splits"
    splits_v1.mkdir(parents=True, exist_ok=True)

    interim_dir = _PROJECT_ROOT / cfg.get("paths", {}).get("interim", "data/interim")
    segments_dir = interim_dir / "segments"

    # 1. Read data/processed/v1/regions.parquet
    regions_v1_path = processed_v1 / "regions.parquet"
    if not regions_v1_path.is_file():
        logger.error("V1 regions not found at %s — run Phase 4 with --v1 first", regions_v1_path)
        sys.exit(1)
    regions_df = pd.read_parquet(regions_v1_path)
    logger.info("Loaded %d V1 regions from %s", len(regions_df), regions_v1_path)

    # 2. Keep eligible_v1 == True rows only
    samples_df = regions_df[regions_df["eligible_v1"] == True].copy()
    n_eligible = len(samples_df)
    logger.info("Kept %d eligible_v1 == True regions", n_eligible)

    # 3. Add geometry (label_value, bbox_xywh, centroid_xy) with add_segment_geometry
    samples_df = add_segment_geometry(samples_df, segments_dir)
    assert len(samples_df) == n_eligible, (
        f"Row count changed after adding geometry: {len(samples_df)} != {n_eligible}"
    )

    # 4. Assert region_id is unique
    assert samples_df["region_id"].is_unique, "region_id is not unique in samples_df"

    # 5. Assert that every sample has a valid bbox (4 finite ints, w > 0, h > 0)
    #    and that label_value matches the segment JSON
    for _, r in samples_df.iterrows():
        rid = r["region_id"]
        bbox = r["bbox_xywh"]
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            raise AssertionError(f"{rid}: bbox must be 4 elements, got {bbox}")
        if not all(isinstance(x, (int, float, np.integer)) and np.isfinite(x) for x in bbox):
            raise AssertionError(f"{rid}: non-finite bbox elements: {bbox}")
        if int(bbox[2]) <= 0 or int(bbox[3]) <= 0:
            raise AssertionError(f"{rid}: non-positive bbox width/height: {bbox}")
        lv = r["label_value"]
        if pd.isna(lv) or int(lv) <= 0:
            raise AssertionError(f"{rid}: invalid label_value: {lv}")

    logger.info("Validated all %d samples have unique region_id and valid bbox/label_value", len(samples_df))

    # 6. Ensure no label columns exist (targets are per fold and get joined at training time)
    label_cols = [c for c in ["label_fine", "label_group", "label_source", "label_conf",
                             "label_fine_original", "label_group_original"] if c in samples_df.columns]
    if label_cols:
        samples_df = samples_df.drop(columns=label_cols)
        logger.info("Dropped legacy label columns: %s", label_cols)

    # 7. Write data/processed/v1/samples.parquet
    samples_path = processed_v1 / "samples.parquet"
    samples_df.to_parquet(samples_path, index=False)
    logger.info("Saved V1 samples.parquet (%d rows) to %s", len(samples_df), samples_path)

    # 8. Copy splits/folds.json to data/processed/v1/splits/folds.json
    src_folds = _PROJECT_ROOT / "data/processed/splits/folds.json"
    if not src_folds.is_file():
        logger.error("Frozen folds not found at %s", src_folds)
        sys.exit(1)
    dst_folds = splits_v1 / "folds.json"
    shutil.copy2(src_folds, dst_folds)
    logger.info("Copied frozen folds to %s", dst_folds)

    # Assert SHA-256 equals FOLDS_SHA256.txt
    sha_file = processed_v1 / "FOLDS_SHA256.txt"
    if not sha_file.is_file():
        raise AssertionError(f"FOLDS_SHA256.txt missing at {sha_file}")
    expected_sha = sha_file.read_text(encoding="utf-8").strip()
    with open(dst_folds, "rb") as f:
        actual_sha = hashlib.sha256(f.read()).hexdigest()
    assert actual_sha == expected_sha, (
        f"Folds SHA-256 mismatch! Expected {expected_sha}, got {actual_sha}"
    )
    logger.info("Verified folds SHA-256 equals FOLDS_SHA256.txt (%s)", actual_sha)

    # 9. QC Crop Panel for 20 random samples (CHECK 7)
    qc_dir = interim_dir / "qc" / "samples_v1"
    qc_dir.mkdir(parents=True, exist_ok=True)
    _generate_qc_crop_panel(samples_df, cfg, qc_dir, n_samples=20)
    logger.info("V1 Phase 7 build complete.")


def _generate_qc_crop_panel(
    samples_df: pd.DataFrame,
    cfg: dict,
    qc_dir: Path,
    n_samples: int = 20,
) -> None:
    """Save crop panel (PPL, XPL, mask overlay) for random samples to verify mask sits on right grain."""
    import cv2
    import numpy as np
    from src.rotapetro.dataset import extract_crops

    interim_dir = _PROJECT_ROOT / cfg.get("paths", {}).get("interim", "data/interim")
    aligned_dir = interim_dir / "aligned"
    segments_dir = interim_dir / "segments"

    sample = samples_df.sample(n=min(n_samples, len(samples_df)), random_state=42)

    panels = []
    for _, r in sample.iterrows():
        site_id = r["site_id"]
        al_path = aligned_dir / f"{site_id}.npz"
        seg_path = segments_dir / f"{site_id}.png"
        if not (al_path.is_file() and seg_path.is_file()):
            continue
        try:
            aligned = np.load(al_path)
            label_img = cv2.imread(str(seg_path), cv2.IMREAD_UNCHANGED)
            bbox = tuple(r["bbox_xywh"])
            lv = int(r["label_value"])

            crops = extract_crops(aligned, label_img, lv, bbox, cfg)
            ppl0 = cv2.resize(crops["ppl"][0], (128, 128))
            xpl0 = cv2.resize(crops["xpl"][0], (128, 128))
            mask = cv2.resize(crops["mask"].astype(np.uint8) * 255, (128, 128), interpolation=cv2.INTER_NEAREST)

            overlay = ppl0.copy()
            green = np.zeros_like(ppl0)
            green[:, :, 1] = 255
            mask_bool = mask > 128
            overlay[mask_bool] = (0.5 * overlay[mask_bool] + 0.5 * green[mask_bool]).astype(np.uint8)

            panel = np.hstack([ppl0, xpl0, overlay])
            panels.append(panel)
        except Exception as e:
            logger.warning("Failed crop for sample %s: %s", r["region_id"], e)
            continue

    if not panels:
        return

    grid = np.vstack(panels)
    out_path = qc_dir / "crop_panel_samples.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    logger.info("Saved QC crop panel (%d samples) to %s", len(panels), out_path)


if __name__ == "__main__":
    main()
