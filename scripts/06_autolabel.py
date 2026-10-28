#!/usr/bin/env python3
"""Phase 6 — Automatic labelling (§12).

Script: scripts/06_autolabel.py
Inputs:  data/processed/regions.parquet, data/interim/specimens_clean.csv,
         data/interim/sites.csv, knowledge/*
Outputs: data/processed/labels.parquet, data/processed/classes.yaml,
         qc/labels/*.png (contact sheets per label)

Four label sources: S1 rule engine, S2 monomineralic, S3 titles, S4 MIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config, validate_v1_config, resolve_v1_paths
from src.rotapetro.knowledge import KnowledgeBase
from src.rotapetro.rules import RuleEngine, _parse_semicolon, _safe_float

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 6: automatic labelling via rule engine")
    parser.add_argument("--config", type=str,
                        default=str(_PROJECT_ROOT / "configs" / "default.yaml"),
                        help="Path to config YAML")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing labels.parquet")
    parser.add_argument("--v1", action="store_true",
                        help="Run V1 per-fold teacher pipeline (writes to data/processed/v1/teachers/)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    validate_v1_config(cfg)

    if args.v1:
        _run_v1_teachers(cfg, args)
        return

    paths = cfg.get("paths", {})

    processed_dir = _PROJECT_ROOT / paths.get("processed", "data/processed")
    interim_dir = _PROJECT_ROOT / paths.get("interim", "data/interim")
    knowledge_dir = _PROJECT_ROOT / paths.get("knowledge", "knowledge")
    segments_dir = interim_dir / "segments"
    qc_dir = interim_dir / "qc" / "labels"

    # ── Load inputs ─────────────────────────────────────────────────
    regions_path = processed_dir / "regions.parquet"
    if not regions_path.is_file():
        logger.error("regions.parquet not found at %s — run Phase 4 first",
                      regions_path)
        sys.exit(1)
    regions_df = pd.read_parquet(regions_path)
    logger.info("Loaded %d regions from %s", len(regions_df), regions_path)
    # regions.parquet has no label_value / bbox / centroid -> take them from
    # the Phase-3 JSONs (needed for title lookup and contact sheets)
    regions_df = _add_segment_geometry(regions_df, segments_dir)

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
    logger.info("Loaded %d sites", len(sites_df))

    # ── Check existing output ───────────────────────────────────────
    labels_path = processed_dir / "labels.parquet"
    classes_path = processed_dir / "classes.yaml"

    if labels_path.is_file() and not args.force:
        logger.info("labels.parquet already exists (use --force to "
                     "overwrite)")
        sys.exit(0)

    # ── Load knowledge base ─────────────────────────────────────────
    kb = KnowledgeBase(knowledge_dir, cfg)
    logger.info("Knowledge base: %s", repr(kb))

    # ── Run rule engine (S1 + S2 + S3) ─────────────────────────────
    engine = RuleEngine(kb, cfg)
    engine.calibrate_thresholds(regions_df)
    logger.info("Labelling %d regions ...", len(regions_df))

    labels_df = engine.label_all(
        regions_df, specimens_df, sites_df,
        segments_dir=segments_dir if segments_dir.is_dir() else None,
    )

    # ── Build class list (§12.6) and apply relabelling (Fix #3) ────
    class_info, relabel_map = engine.build_class_list(
        labels_df, regions_df, cfg)

    labels_df = RuleEngine.apply_relabel(labels_df, relabel_map)

    logger.info("Class list: %d classes: %s",
                len(class_info["fine_classes"]),
                class_info["fine_classes"])
    logger.info("Relabel map: %s", relabel_map)

    # ── Statistics ──────────────────────────────────────────────────
    n_total = len(labels_df)
    n_fine = labels_df["label_fine"].notna().sum()
    n_group = (labels_df["label_group"].notna()
               & labels_df["label_fine"].isna()).sum()
    n_none = (labels_df["label_source"] == "none").sum()

    source_counts = labels_df["label_source"].value_counts()

    logger.info("Labelling complete:")
    logger.info("  Total regions:    %d", n_total)
    logger.info("  Fine labels:      %d (%.1f%%)",
                n_fine, 100 * n_fine / max(n_total, 1))
    logger.info("  Group-only:       %d (%.1f%%)",
                n_group, 100 * n_group / max(n_total, 1))
    logger.info("  Unlabelled:       %d (%.1f%%)",
                n_none, 100 * n_none / max(n_total, 1))
    for src, count in source_counts.items():
        logger.info("  Source %-10s: %d", src, count)

    # ── Fine label distribution ─────────────────────────────────────
    if n_fine > 0:
        fine_dist = labels_df["label_fine"].value_counts()
        logger.info("Fine label distribution:")
        for mineral, count in fine_dist.items():
            if mineral is not None and not pd.isna(mineral):
                logger.info("  %-20s %d", mineral, count)

    # ── Validate: all label_fine values are in classes.yaml ─────────
    valid_classes = set(class_info["fine_classes"])
    fine_vals = set(labels_df["label_fine"].dropna().unique())
    invalid = fine_vals - valid_classes
    if invalid:
        logger.error("Labels reference classes not in classes.yaml: %s",
                     sorted(invalid))
    else:
        logger.info("All label_fine values are in classes.yaml")

    # ── Save outputs ────────────────────────────────────────────────
    processed_dir.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(labels_path, index=False)
    logger.info("Saved labels.parquet (%d rows) to %s",
                len(labels_df), labels_path)

    with open(classes_path, "w", encoding="utf-8") as f:
        yaml.dump(class_info, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved classes.yaml to %s", classes_path)

    # ── QC: contact sheets per class ────────────────────────────────
    try:
        _generate_contact_sheets(
            labels_df, regions_df, sites_df, cfg, qc_dir)
    except Exception as e:
        logger.warning("Contact sheet generation failed: %s", e)

    # ── Summary report ──────────────────────────────────────────────
    report_path = processed_dir / "labelling_report.md"
    _write_report(report_path, labels_df, class_info, relabel_map,
                  n_total, n_fine, n_group, n_none, source_counts)
    logger.info("Report written to %s", report_path)

    logger.info("Phase 6 complete.")


def _add_segment_geometry(regions_df: pd.DataFrame, seg_dir: Path) -> pd.DataFrame:
    import json as _json
    rows = []
    for site_id in regions_df["site_id"].unique():
        p = seg_dir / f"{site_id}.json"
        if not p.is_file():
            continue
        data = _json.loads(p.read_text(encoding="utf-8"))
        for r in (data["regions"] if isinstance(data, dict) else data):
            rows.append({"region_id": r["region_id"],
                         "label_value": r["label_value"],
                         "bbox_xywh": r["bbox_xywh"],
                         "centroid_xy": r["centroid_xy"]})
    if not rows:
        return regions_df
    geo = pd.DataFrame(rows)
    keep = [c for c in regions_df.columns if c not in ("label_value", "bbox_xywh", "centroid_xy")]
    return regions_df[keep].merge(geo, on="region_id", how="left")


def _generate_contact_sheets(
    labels_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    cfg: dict,
    qc_dir: Path,
    n_per_sheet: int = 30,
) -> None:
    """Generate QC contact sheet PNGs: 30 random regions per class."""
    import cv2
    import numpy as np
    import json as _json
    import math as _math

    qc_dir.mkdir(parents=True, exist_ok=True)
    paths_cfg = cfg.get("paths", {})
    interim_dir = Path(__file__).resolve().parents[1] / paths_cfg.get(
        "interim", "data/interim")

    # Merge labels with regions
    merged = labels_df.merge(
        regions_df[["region_id", "site_id", "bbox_xywh"]].drop_duplicates(
            subset="region_id"),
        on="region_id", how="left",
    )

    # Show the mineral the rule engine chose (before merging rare classes
    # into groups / "other"), otherwise the sheets say nothing.
    col = ("label_fine_original" if "label_fine_original" in merged.columns
           else "label_fine")
    fine_labels = merged[merged[col].notna()]
    classes = fine_labels[col].unique()

    for cls in classes:
        cls_regions = fine_labels[fine_labels[col] == cls]
        sample = cls_regions.sample(
            n=min(n_per_sheet, len(cls_regions)), random_state=42)

        crops: list[np.ndarray] = []
        for _, r in sample.iterrows():
            site_id = r["site_id"]
            maps_path = interim_dir / "maps" / f"{site_id}.npz"
            if not maps_path.is_file():
                continue
            try:
                maps_data = np.load(maps_path)
                xpl_max = maps_data.get("xpl_maxproj_rgb")
                ppl_mean = maps_data.get("ppl_mean_rgb")
                if xpl_max is None:
                    continue

                bbox = r.get("bbox_xywh")
                if bbox is None:
                    continue
                if isinstance(bbox, float) and _math.isnan(bbox):
                    continue
                if isinstance(bbox, str):
                    bbox = _json.loads(bbox)
                bx, by, bw, bh = (int(bbox[0]), int(bbox[1]),
                                  int(bbox[2]), int(bbox[3]))

                crop = xpl_max[by:by+bh, bx:bx+bw]
                if crop.size == 0:
                    continue
                pair = [cv2.resize(crop, (64, 64))]
                if ppl_mean is not None:
                    pair.insert(0, cv2.resize(ppl_mean[by:by+bh, bx:bx+bw], (64, 64)))
                crops.append(np.hstack(pair))
            except Exception:
                continue

        if not crops:
            continue

        # Arrange in grid
        n = len(crops)
        cols = min(6, n)
        rows = (n + cols - 1) // cols
        cw = crops[0].shape[1] + 2          # 64 (XPL) or 128 (PPL|XPL) + gap
        sheet = np.zeros((rows * 66, cols * cw, 3), dtype=np.uint8)

        for i, crop in enumerate(crops):
            r_idx = i // cols
            c_idx = i % cols
            y0 = r_idx * 66 + 1
            x0 = c_idx * cw + 1
            sheet[y0:y0+64, x0:x0+crop.shape[1]] = crop

        out_path = qc_dir / f"{cls}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))

    logger.info("Contact sheets saved to %s (%d classes)",
                qc_dir, len(classes))


def _write_report(
    path: Path,
    labels_df: pd.DataFrame,
    class_info: dict,
    relabel_map: dict[str, str],
    n_total: int,
    n_fine: int,
    n_group: int,
    n_none: int,
    source_counts: pd.Series,
) -> None:
    """Write a labelling summary report in Markdown."""
    lines = [
        "# Phase 6 - Labelling Report\n\n",
        f"Total regions: {n_total}\n\n",
        f"Fine labels: {n_fine} ({100*n_fine/max(n_total,1):.1f}%)\n\n",
        f"Group-only: {n_group} ({100*n_group/max(n_total,1):.1f}%)\n\n",
        f"Unlabelled: {n_none} ({100*n_none/max(n_total,1):.1f}%)\n\n",
        "## Label sources\n\n",
    ]
    for src, count in source_counts.items():
        lines.append(f"- {src}: {count}\n")

    lines.append("\n## Class list\n\n")
    for cls in class_info.get("fine_classes", []):
        counts = class_info.get("mineral_counts", {}).get(cls, {})
        n_r = counts.get("n_regions", 0)
        n_s = counts.get("n_specimens", 0)
        lines.append(f"- {cls}: {n_r} regions from {n_s} specimens\n")

    lines.append("\n## Relabel mapping\n\n")
    for orig, final in sorted(relabel_map.items()):
        if orig != final:
            lines.append(f"- {orig} -> {final}\n")

    lines.append("\n## Fine label distribution\n\n")
    fine_dist = labels_df["label_fine"].value_counts()
    for mineral, count in fine_dist.items():
        if mineral is not None and not pd.isna(mineral):
            lines.append(f"- {mineral}: {count}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


from src.rotapetro.provenance import (
    compute_code_hash as _compute_code_hash,
    compute_knowledge_hash as _compute_knowledge_hash,
    compute_teacher_fingerprint as _compute_teacher_fingerprint,
    current_teacher_fingerprint,
    current_teacher_provenance,
    get_folds_sha256 as _get_folds_sha256,
    hash_file as _hash_file,
)


def _run_v1_teachers(cfg: dict, args: argparse.Namespace) -> None:
    """Run V1 per-fold teacher calibration and target generation across all folds."""
    import time

    v1_paths = resolve_v1_paths(cfg)
    teachers_dir = _PROJECT_ROOT / v1_paths["teachers"]
    teachers_dir.mkdir(parents=True, exist_ok=True)

    interim_dir = _PROJECT_ROOT / cfg.get("paths", {}).get("interim", "data/interim")
    knowledge_dir = _PROJECT_ROOT / cfg.get("paths", {}).get("knowledge", "knowledge")

    # 1. Load V1 regions
    regions_v1_path = _PROJECT_ROOT / v1_paths["processed"] / "regions.parquet"
    if not regions_v1_path.is_file():
        logger.error("V1 regions not found at %s — run Phase 4 with --v1 first", regions_v1_path)
        sys.exit(1)
    regions_df = pd.read_parquet(regions_v1_path)
    logger.info("Loaded %d V1 regions from %s", len(regions_df), regions_v1_path)

    # Assert V1 columns exist
    from src.rotapetro.features import V1_STATUS_COLUMNS
    req_cols = V1_STATUS_COLUMNS + ["eligible_v1", "reject_reason_v1", "interior_px"]
    missing = [c for c in req_cols if c not in regions_df.columns]
    if missing:
        raise RuntimeError(f"V1 feature columns missing from regions.parquet: {missing}")

    # 2. Load splits/folds.json — prefer the V1 copy (hash-checked)
    folds_path = _PROJECT_ROOT / v1_paths["processed"] / "splits" / "folds.json"
    if not folds_path.is_file():
        folds_path = _PROJECT_ROOT / cfg.get("paths", {}).get("processed", "data/processed") / "splits" / "folds.json"
    if not folds_path.is_file():
        logger.error("splits/folds.json not found at %s", folds_path)
        sys.exit(1)
    with open(folds_path, "r", encoding="utf-8") as f:
        folds = json.load(f)
    logger.info("Loaded %d folds from %s", len(folds), folds_path)

    # 3. Load specimens & build candidate sets
    specimens_path = interim_dir / "specimens_clean.csv"
    if not specimens_path.is_file():
        logger.error("specimens_clean.csv not found at %s", specimens_path)
        sys.exit(1)
    specimens_df = pd.read_csv(specimens_path)

    kb = KnowledgeBase(knowledge_dir, cfg)
    logger.info("Loaded KnowledgeBase: %s", repr(kb))

    spec_candidates: dict[str, set[str]] = {}
    for _, row in specimens_df.iterrows():
        c = str(row["code"]).strip()
        rf = _parse_semicolon(row.get("rock_forming", ""))
        ac = _parse_semicolon(row.get("accessory", ""))
        spec_candidates[c] = kb.get_candidate_set(rf, ac)

    all_minerals = set(kb.all_minerals())
    families = list(cfg.get("v1", {}).get("families", {}).keys())

    # 4. Compute composite fingerprint hash and teacher version
    fingerprint_hash, provenance_dict = current_teacher_provenance(cfg, project_root=_PROJECT_ROOT)
    v1_cfg = cfg.get("v1", {})
    folds_sha = provenance_dict.get("folds_sha256", "")
    regions_sha = provenance_dict.get("regions_sha256", "")
    specimens_sha = provenance_dict.get("specimens_sha256", "")
    knowledge_hash = provenance_dict.get("knowledge_hash", "")
    code_hash = provenance_dict.get("code_hash", "")

    v1_cfg_bytes = json.dumps(v1_cfg, sort_keys=True).encode("utf-8")
    cfg_hash = hashlib.sha256(v1_cfg_bytes).hexdigest()[:8]
    teacher_version = f"{v1_cfg.get('version', 'v1.0')}-{fingerprint_hash}"
    logger.info(
        "Teacher version: %s (fingerprint: %s, config hash: %s, folds sha: %s)",
        teacher_version, fingerprint_hash, cfg_hash, folds_sha[:8] if folds_sha else "none",
    )

    records = regions_df.to_dict("records")

    # 5. Process each fold
    for fold_name, split in sorted(folds.items()):
        fold_dir = teachers_dir / fold_name
        fold_dir.mkdir(parents=True, exist_ok=True)
        targets_path = fold_dir / "targets.parquet"
        params_path = fold_dir / "teacher_params.json"

        # Check existing targets & fingerprint
        if targets_path.is_file() and not args.force:
            match = False
            targets_changed = False
            if params_path.is_file():
                try:
                    with open(params_path, "r", encoding="utf-8") as f:
                        old_p = json.load(f)
                    old_fp = old_p.get("fingerprint")
                    old_tv = old_p.get("teacher_version")
                    old_sha = old_p.get("targets_sha256")
                    cur_sha = _hash_file(targets_path)
                    if not old_sha or old_sha != cur_sha:
                        targets_changed = True
                    elif old_fp == fingerprint_hash and old_tv == teacher_version:
                        match = True
                except Exception as e:
                    logger.warning("Could not read teacher_params.json for %s: %s", fold_name, e)
                    targets_changed = True
            else:
                targets_changed = True

            if match:
                logger.info(
                    "Fold %s already up to date at %s (matches fingerprint %s; use --force to overwrite)",
                    fold_name, targets_path, fingerprint_hash,
                )
                continue
            elif targets_changed:
                logger.warning(
                    "Fold %s: targets file changed since generation; regenerating",
                    fold_name,
                )
            else:
                logger.warning(
                    "Fold %s existing targets stale or fingerprint mismatch; regenerating...",
                    fold_name,
                )

        t0 = time.time()
        train_codes = set(split.get("train", []))
        val_codes = set(split.get("val", []))
        test_codes = set(split.get("test", []))

        engine = RuleEngine(kb, cfg)
        engine.calibrate_thresholds(regions_df, train_codes=train_codes)

        rows = []
        for r in records:
            rid = r["region_id"]
            sid = r["site_id"]
            code = r["code"]
            if code in train_codes:
                partition = "train"
            elif code in val_codes:
                partition = "val"
            elif code in test_codes:
                partition = "test"
            else:
                partition = "unknown"

            candidates = spec_candidates.get(code, all_minerals)
            ev = engine.evaluate_region_v1(r, candidates)

            state = ev["state"]
            allowed = ev["allowed"]
            reason = ev["reason"]
            fam_p_dict = ev.get("fam_p", {})
            fam_p = {f: float(fam_p_dict.get(f, 0.0)) for f in families}
            p_out = float(ev.get("p_out", 0.0))
            n_info = int(ev.get("n_info", 0))

            optics_state = ev["optics_state"]
            optics_allowed = ev["optics_allowed"]
            optics_reason = ev.get("optics_reason", "")
            optics_list_conflict = bool(ev.get("optics_list_conflict", False))
            metadata_dependent = bool(ev.get("metadata_dependent", False))

            deciding_evidence = ev.get("deciding_evidence")
            deciding_terms = ev.get("info_terms", []) if state in ("singleton", "partial_set") else []

            top_species = ev.get("top1")
            top_species_p = float(ev.get("p1", 0.0))

            # Single-angle evaluation (D1.3)
            # Compute k1 only for regions that passed the full quality gate
            if state != "excluded":
                r_k1 = engine.single_angle_view(r)
                ev_k1 = engine.evaluate_region_v1(r_k1, candidates, skip_gate=True)
                k1_state = ev_k1["state"]
                k1_allowed = ev_k1["allowed"]
                k1_reason = ev_k1["reason"]
                k1_optics_state = ev_k1["optics_state"]
                k1_optics_allowed = ev_k1["optics_allowed"]
            else:
                k1_state = "excluded"
                k1_allowed = []
                k1_reason = reason
                k1_optics_state = "excluded"
                k1_optics_allowed = []

            static_category = engine.compute_static_category(
                state, allowed, k1_state, k1_allowed
            )
            static_category_optics = engine.compute_static_category(
                optics_state, optics_allowed, k1_optics_state, k1_optics_allowed
            )

            row_data = {
                "region_id": rid,
                "site_id": sid,
                "code": code,
                "fold": fold_name,
                "partition": partition,
                "state": state,
                "allowed": allowed,
                "reason": reason,
                **{f"fam_p_{f}": fam_p[f] for f in families},
                "p_out": p_out,
                "n_info": n_info,
                "optics_state": optics_state,
                "optics_allowed": optics_allowed,
                "optics_reason": optics_reason,
                "optics_list_conflict": optics_list_conflict,
                "metadata_dependent": metadata_dependent,
                "deciding_evidence": deciding_evidence,
                "deciding_terms": deciding_terms,
                "top_species": top_species,
                "top_species_p": top_species_p,
                "k1_state": k1_state,
                "k1_allowed": k1_allowed,
                "k1_reason": k1_reason,
                "k1_optics_state": k1_optics_state,
                "k1_optics_allowed": k1_optics_allowed,
                "static_category": static_category,
                "static_category_optics": static_category_optics,
                "teacher_version": teacher_version,
            }
            rows.append(row_data)

        targets_df = pd.DataFrame(rows)

        params = {
            "fold": fold_name,
            "undulose_high": engine.undulose_high,
            "twinkle_high": engine.twinkle_high,
            "n_train_specimens": len(split.get("train", [])),
            "train_codes": list(split.get("train", [])),
            "config_hash": cfg_hash,
            "fingerprint": fingerprint_hash,
            "folds_sha256": folds_sha,
            "regions_sha256": regions_sha,
            "specimens_sha256": specimens_sha,
            "knowledge_hash": knowledge_hash,
            "code_hash": code_hash,
            "teacher_version": teacher_version,
        }

        targets_df.to_parquet(targets_path, index=False)
        params["targets_sha256"] = _hash_file(targets_path)
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2)

        dt = time.time() - t0
        state_counts = targets_df["state"].value_counts().to_dict()
        n_meta_dep = int(targets_df["metadata_dependent"].sum())
        static_cat_counts = targets_df["static_category"].value_counts().to_dict()
        logger.info(
            "Saved %s targets (%d rows) in %.2fs: states=%s, metadata_dependent=%d, static_categories=%s",
            fold_name, len(targets_df), dt, state_counts, n_meta_dep, static_cat_counts,
        )

    logger.info("V1 per-fold teacher generation finished successfully.")


if __name__ == "__main__":
    main()
