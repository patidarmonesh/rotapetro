#!/usr/bin/env python3
"""V1 Label Feasibility Report — Step 6.

Script: scripts/v1_label_feasibility.py

Reads:
- data/processed/v1/regions.parquet
- data/processed/v1/teachers/<fold>/targets.parquet (all 5 folds)

Does NOT run the rule engine. Evaluates the 5-family gate and produces:
1. label_feasibility_report.md
2. target_support_by_fold.csv
3. partial_set_support.csv
4. label_evidence_breakdown.csv
5. label_rejections.csv
6. provenance_feasibility.json
7. data/interim/qc/labels_v1/<family>_singletons.png (20 random singletons per family)
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.features import V1_STATUS_COLUMNS
from src.rotapetro.io import load_config, resolve_v1_paths, save_provenance, validate_v1_config
from src.rotapetro.provenance import current_teacher_fingerprint

logger = logging.getLogger(__name__)


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file, returning empty string if missing."""
    if not path.is_file():
        return ""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


STATIC_CATEGORIES: list[str] = [
    "reproduced", "k1_broader", "k1_narrower", "conflict", "full_only", "k1_only", "unresolved_both", "n/a"
]


def _compute_static_category(
    state: str,
    allowed: object,
    k1_state: str,
    k1_allowed: object,
) -> str:
    """Compute static category per CONTRACTS.md Section 3."""
    st = str(state).strip().lower()
    k1_st = str(k1_state).strip().lower()
    if st in ("excluded", "mixed"):
        return "n/a"
    full_labelled = st in ("singleton", "partial_set")
    if full_labelled and k1_st == "mixed":
        return "conflict"
    k1_labelled = k1_st in ("singleton", "partial_set")
    s_full = set(allowed) if isinstance(allowed, (list, np.ndarray, set)) else set()
    s_k1 = set(k1_allowed) if isinstance(k1_allowed, (list, np.ndarray, set)) else set()

    if full_labelled and k1_labelled:
        if st == k1_st and s_full == s_k1:
            return "reproduced"
        elif s_full < s_k1:
            return "k1_broader"
        elif s_k1 < s_full:
            return "k1_narrower"
        elif s_full.isdisjoint(s_k1):
            return "conflict"
        else:
            return "conflict"
    elif full_labelled and k1_st == "abstain":
        return "full_only"
    elif st == "abstain" and k1_labelled:
        return "k1_only"
    elif st == "abstain" and k1_st == "abstain":
        return "unresolved_both"
    return "n/a"


def _build_static_category_table(
    df: pd.DataFrame,
    cat_col: str,
    family_col: str,
    primary_classes: list[str],
) -> pd.DataFrame:
    """Build count matrix for static_category x family."""
    families = list(primary_classes) + ["other"]
    matrix = {cat: {fam: 0 for fam in families} for cat in STATIC_CATEGORIES}

    for _, row in df.iterrows():
        cat = str(row.get(cat_col, "n/a")).strip()
        if cat not in matrix:
            cat = "n/a"
        fam = str(row.get(family_col, "other")).strip()
        if fam not in primary_classes:
            fam = "other"
        matrix[cat][fam] += 1

    table_df = pd.DataFrame.from_dict(matrix, orient="index")[families]
    table_df["Total"] = table_df.sum(axis=1)
    return table_df


def _format_markdown_table(df: pd.DataFrame, index_name: str = "Category") -> str:
    """Format DataFrame as a clean Markdown table without external dependencies."""
    cols = [index_name] + list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for idx, row in df.iterrows():
        vals = [str(idx)] + [str(v) for v in row.values]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _add_segment_geometry(regions_df: pd.DataFrame, seg_dir: Path) -> pd.DataFrame:
    """Add label_value / bbox_xywh / centroid_xy from Phase 3 JSONs."""
    rows = []
    for site_id in regions_df["site_id"].unique():
        p = seg_dir / f"{site_id}.json"
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in (data["regions"] if isinstance(data, dict) else data):
            rows.append({
                "region_id": r["region_id"],
                "label_value": r["label_value"],
                "bbox_xywh": r["bbox_xywh"],
                "centroid_xy": r["centroid_xy"],
            })
    if not rows:
        return regions_df
    geo = pd.DataFrame(rows)
    keep = [c for c in regions_df.columns if c not in ("label_value", "bbox_xywh", "centroid_xy")]
    return regions_df[keep].merge(geo, on="region_id", how="left")


def _generate_singleton_panels(
    targets_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    cfg: dict,
    qc_dir: Path,
    families: list[str],
    n_per_family: int = 20,
) -> None:
    """Generate visual PPL/XPL contact sheet panels for random singletons per family."""
    import cv2

    qc_dir.mkdir(parents=True, exist_ok=True)
    interim_dir = _PROJECT_ROOT / cfg.get("paths", {}).get("interim", "data/interim")

    merged = targets_df[targets_df["state"] == "singleton"].merge(
        regions_df[["region_id", "bbox_xywh"]].drop_duplicates(subset="region_id"),
        on="region_id",
        how="left",
    )

    for fam in families:
        fam_singletons = merged[merged["allowed"].apply(
            lambda a: fam in a if isinstance(a, (list, np.ndarray)) else a == fam
        )]
        if fam_singletons.empty:
            logger.warning("No singletons found for family '%s' to create panel", fam)
            continue

        sample = fam_singletons.sample(
            n=min(n_per_family, len(fam_singletons)),
            random_state=42,
        )

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
                if isinstance(bbox, float) and np.isnan(bbox):
                    continue
                if isinstance(bbox, str):
                    bbox = json.loads(bbox)
                bx, by, bw, bh = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                crop_xpl = xpl_max[by:by + bh, bx:bx + bw]
                if crop_xpl.size == 0:
                    continue
                pair = [cv2.resize(crop_xpl, (64, 64))]
                if ppl_mean is not None:
                    crop_ppl = ppl_mean[by:by + bh, bx:bx + bw]
                    pair.insert(0, cv2.resize(crop_ppl, (64, 64)))
                crops.append(np.hstack(pair))
            except Exception:
                continue

        if not crops:
            continue

        n = len(crops)
        cols = min(5, n)
        rows = (n + cols - 1) // cols
        cw = crops[0].shape[1] + 2
        sheet = np.zeros((rows * 66, cols * cw, 3), dtype=np.uint8)

        for i, crop in enumerate(crops):
            r_idx = i // cols
            c_idx = i % cols
            y0 = r_idx * 66 + 1
            x0 = c_idx * cw + 1
            sheet[y0:y0 + 64, x0:x0 + crop.shape[1]] = crop

        out_path = qc_dir / f"{fam}_singletons.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        logger.info("Saved QC panel for %s (%d crops) to %s", fam, len(crops), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V1 Label Feasibility Report — Step 6 reporting and gate evaluation"
    )
    parser.add_argument("--config", type=str,
                        default=str(_PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--qc-dir", type=str, default="",
                        help="Directory to save singleton QC panels")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    validate_v1_config(cfg)
    v1_paths = resolve_v1_paths(cfg)
    v1_cfg = cfg.get("v1", {})
    primary_classes = v1_cfg.get("taxonomy", {}).get("primary_classes", [])
    min_train_specimens = int(v1_cfg.get("training", {}).get("min_train_specimens_per_class", 5))

    out_dir = _PROJECT_ROOT / v1_paths["processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    teachers_dir = _PROJECT_ROOT / v1_paths["teachers"]

    interim_dir = _PROJECT_ROOT / cfg.get("paths", {}).get("interim", "data/interim")
    segments_dir = interim_dir / "segments"
    qc_dir = Path(args.qc_dir) if args.qc_dir else interim_dir / "qc" / "labels_v1"

    # ── 1. Load V1 regions ──────────────────────────────────────────
    regions_path = out_dir / "regions.parquet"
    if not regions_path.is_file():
        logger.error("V1 regions.parquet not found at %s", regions_path)
        sys.exit(1)
    regions_v1 = pd.read_parquet(regions_path)
    logger.info("Loaded %d V1 regions from %s", len(regions_v1), regions_path)

    # Hard Assert 1: status columns exist
    missing_status = set(V1_STATUS_COLUMNS) - set(regions_v1.columns)
    assert set(regions_v1.columns) >= set(V1_STATUS_COLUMNS), (
        f"Missing V1 status columns in regions_v1: {missing_status}"
    )
    logger.info("Hard assert passed: all V1_STATUS_COLUMNS present in regions_v1")

    # ── 2. Load 5 teachers/<fold>/targets.parquet ─────────────────────
    fold_dirs = sorted([d for d in teachers_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")])
    if len(fold_dirs) != 5:
        logger.error("Expected 5 fold directories under %s, found %d", teachers_dir, len(fold_dirs))
        sys.exit(1)

    targets_by_fold: dict[str, pd.DataFrame] = {}
    params_by_fold: dict[str, dict] = {}
    teacher_fingerprints: dict[str, str] = {}
    targets_checksums: dict[str, str] = {}
    all_decided_terms: Counter = Counter()

    folds_path = out_dir / "splits" / "folds.json"
    if not folds_path.is_file():
        folds_path = _PROJECT_ROOT / "data/processed/splits/folds.json"
    if not folds_path.is_file():
        logger.error("splits/folds.json not found at %s. Please rerun 06_autolabel.py.", folds_path)
        sys.exit(1)
    with open(folds_path, "r", encoding="utf-8") as ff:
        folds_data = json.load(ff)

    cur_teacher_fp = current_teacher_fingerprint(cfg, project_root=_PROJECT_ROOT)

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        t_path = fold_dir / "targets.parquet"
        p_path = fold_dir / "teacher_params.json"
        if not t_path.is_file():
            logger.error("targets.parquet missing for %s at %s. Please rerun 06_autolabel.py.", fold_name, t_path)
            sys.exit(1)
        if not p_path.is_file():
            logger.error("teacher_params.json missing for %s at %s. Please rerun 06_autolabel.py.", fold_name, p_path)
            sys.exit(1)

        try:
            with open(p_path, "r", encoding="utf-8") as pf:
                params = json.load(pf)
        except Exception as exc:
            logger.error("Could not read teacher_params.json for %s: %s. Please rerun 06_autolabel.py.", fold_name, exc)
            sys.exit(1)

        fold_fp = str(params.get("fingerprint", ""))
        if fold_fp != cur_teacher_fp:
            logger.error(
                "Teacher fingerprint mismatch for %s (stored=%s, current=%s). Please rerun 06_autolabel.py.",
                fold_name, fold_fp, cur_teacher_fp,
            )
            sys.exit(1)

        # Check targets_sha256 matches teacher_params
        t_sha = _hash_file(t_path)
        expected_t_sha = params.get("targets_sha256", "")
        if not expected_t_sha or t_sha != expected_t_sha:
            logger.error(
                "targets.parquet sha256 mismatch for %s (actual=%s, expected=%s). rerun 06_autolabel.py --v1 (or --force)",
                fold_name, t_sha, expected_t_sha,
            )
            sys.exit(1)
        targets_checksums[fold_name] = t_sha

        params_by_fold[fold_name] = params
        teacher_fingerprints[fold_name] = fold_fp

        targets = pd.read_parquet(t_path)
        assert len(targets) == len(regions_v1), (
            f"{fold_name} targets has {len(targets)} rows, expected {len(regions_v1)}"
        )

        expected_tv = params.get("teacher_version", "")
        if "teacher_version" not in targets.columns or not (targets["teacher_version"] == expected_tv).all():
            logger.error(
                "targets.parquet for %s has mismatched teacher_version (expected %s). Please rerun 06_autolabel.py.",
                fold_name, expected_tv,
            )
            sys.exit(1)

        if not targets["region_id"].is_unique:
            logger.error("targets.parquet for %s has non-unique region_id values. Please rerun 06_autolabel.py.", fold_name)
            sys.exit(1)

        if set(targets["region_id"]) != set(regions_v1["region_id"]):
            logger.error(
                "targets.parquet for %s region_id set does not equal regions.parquet. Please rerun 06_autolabel.py.",
                fold_name,
            )
            sys.exit(1)

        if fold_name not in folds_data:
            logger.error("Fold %s missing from folds.json. Please rerun 06_autolabel.py.", fold_name)
            sys.exit(1)

        split = folds_data[fold_name]
        train_codes = set(split.get("train", []))
        val_codes = set(split.get("val", []))
        test_codes = set(split.get("test", []))
        bad_train = targets.loc[targets["code"].isin(train_codes), "partition"].ne("train").any()
        bad_val = targets.loc[targets["code"].isin(val_codes), "partition"].ne("val").any()
        bad_test = targets.loc[targets["code"].isin(test_codes), "partition"].ne("test").any()
        if bad_train or bad_val or bad_test:
            logger.error(
                "targets.parquet for %s has partition assignments not matching folds.json. Please rerun 06_autolabel.py.",
                fold_name,
            )
            sys.exit(1)

        # Hard Assert 2: fine_aggregate never mapped to singleton
        sing_species = targets.loc[targets["state"] == "singleton", "top_species"]
        assert not any("fine_aggregate" == t for t in sing_species), (
            f"{fold_name} has 'fine_aggregate' as singleton top_species"
        )

        # Collect decided terms for hard assert 3
        if "deciding_terms" in targets.columns:
            fold_decided = Counter(
                t for terms in targets.loc[targets["state"].isin(["singleton", "partial_set"]), "deciding_terms"]
                for t in terms
            )
        else:
            fold_decided = Counter()

        # Hard Assert 3: birefringence and extinction_type disabled
        assert fold_decided.get("birefringence", 0) == 0 and fold_decided.get("extinction_type", 0) == 0, (
            f"{fold_name} has quarantined terms in deciding_terms: {fold_decided}"
        )

        all_decided_terms.update(fold_decided)
        targets_by_fold[fold_name] = targets

    logger.info("Hard asserts passed across all %d folds:", len(targets_by_fold))
    logger.info("  - %d targets.parquet files verified with %d rows each", len(targets_by_fold), len(regions_v1))
    logger.info("  - No 'fine_aggregate' as singleton top_species")
    logger.info("  - birefringence=0 and extinction_type=0 in deciding terms")

    # ── 3. Gate Result: singleton training specimens per family ───────
    gate_table_rows = []
    overall_gate_pass = True

    for fold_name, targets in targets_by_fold.items():
        train_sing = targets[(targets["partition"] == "train") & (targets["state"] == "singleton")].copy()
        train_sing["family"] = train_sing["allowed"].apply(lambda a: a[0] if len(a) > 0 else "")

        for fam in primary_classes:
            fam_rows = train_sing[train_sing["family"] == fam]
            n_regs = len(fam_rows)
            n_specs = fam_rows["code"].nunique()
            fam_pass = (n_specs >= min_train_specimens)
            if not fam_pass:
                overall_gate_pass = False

            gate_table_rows.append({
                "fold": fold_name,
                "family": fam,
                "train_regions": n_regs,
                "train_specimens": n_specs,
                "gate_threshold": min_train_specimens,
                "gate_status": "PASS" if fam_pass else "FAIL",
            })

    gate_df = pd.DataFrame(gate_table_rows)

    logger.info("=" * 60)
    logger.info("LABEL GATE RESULT: %s", "PASS" if overall_gate_pass else "FAIL")
    logger.info("=" * 60)
    for fold_name in sorted(targets_by_fold.keys()):
        fold_gate = gate_df[gate_df["fold"] == fold_name]
        logger.info(
            "  %s: %s",
            fold_name,
            ", ".join(f"{r['family']}={r['train_specimens']} specs ({r['gate_status']})" for _, r in fold_gate.iterrows())
        )

    # ── 4. Detailed Statistics by Fold & Partition ────────────────────
    support_rows = []
    partial_set_counts_by_fold: dict[str, dict[str, int]] = {}
    abstain_reasons_by_fold: dict[str, dict[str, int]] = {}
    meta_dep_by_fold: dict[str, float] = {}
    evidence_terms_by_fold: dict[str, dict[str, int]] = {}
    conflict_stats_by_fold: dict[str, dict[str, dict[str, int]]] = {}
    static_tables_by_fold: dict[str, dict[str, tuple[pd.DataFrame, pd.DataFrame]]] = {}
    static_dist_rows: list[dict] = []

    for fold_name, targets in targets_by_fold.items():
        conflict_stats_by_fold[fold_name] = {}
        static_tables_by_fold[fold_name] = {}

        # Ensure static_category and static_category_optics are present
        if "static_category" not in targets.columns:
            if "k1_state" in targets.columns and "k1_allowed" in targets.columns:
                targets["static_category"] = [
                    _compute_static_category(r["state"], r.get("allowed", []), r["k1_state"], r.get("k1_allowed", []))
                    for _, r in targets.iterrows()
                ]
            else:
                targets["static_category"] = "n/a"

        if "static_category_optics" not in targets.columns:
            if "k1_optics_state" in targets.columns and "k1_optics_allowed" in targets.columns and "optics_state" in targets.columns:
                targets["static_category_optics"] = [
                    _compute_static_category(r["optics_state"], r.get("optics_allowed", []), r["k1_optics_state"], r.get("k1_optics_allowed", []))
                    for _, r in targets.iterrows()
                ]
            else:
                targets["static_category_optics"] = targets["static_category"]

        targets["assigned_family"] = targets["allowed"].apply(
            lambda a: a[0] if (isinstance(a, (list, np.ndarray)) and len(a) > 0) else "other"
        )
        if "optics_allowed" in targets.columns:
            targets["assigned_family_optics"] = targets["optics_allowed"].apply(
                lambda a: a[0] if (isinstance(a, (list, np.ndarray)) and len(a) > 0) else "other"
            )
        else:
            targets["assigned_family_optics"] = targets["assigned_family"]

        # Overall metadata dependence (over trainable rows only: singleton + partial_set)
        trainable = targets[targets["state"].isin(["singleton", "partial_set"])]
        n_meta_dep = trainable["metadata_dependent"].sum() if len(trainable) > 0 else 0
        pct_meta_dep = 100.0 * n_meta_dep / max(len(trainable), 1)
        meta_dep_by_fold[fold_name] = pct_meta_dep

        # Abstain reasons
        abstain_sub = targets[targets["state"] == "abstain"]
        abstain_reasons_by_fold[fold_name] = abstain_sub["reason"].value_counts().to_dict()

        # Partial set counts
        partial_sub = targets[targets["state"] == "partial_set"]
        ps_sets = partial_sub["allowed"].apply(lambda a: "{" + ", ".join(sorted(a)) + "}")
        partial_set_counts_by_fold[fold_name] = ps_sets.value_counts().to_dict()

        # Evidence terms
        if "deciding_terms" in targets.columns:
            ev_terms = Counter(
                t for terms in targets.loc[targets["state"].isin(["singleton", "partial_set"]), "deciding_terms"]
                for t in terms
            )
            evidence_terms_by_fold[fold_name] = dict(ev_terms)
        else:
            evidence_terms_by_fold[fold_name] = {}

        # Partition breakdown
        for partition in ["train", "val", "test", "all"]:
            p_df = targets if partition == "all" else targets[targets["partition"] == partition]
            state_counts = p_df["state"].value_counts().to_dict()

            p_sing = p_df[p_df["state"] == "singleton"].copy()
            p_sing["family"] = p_sing["allowed"].apply(lambda a: a[0] if len(a) > 0 else "")

            # Static category cross-tabulation matrices
            table_meta = _build_static_category_table(p_df, "static_category", "assigned_family", primary_classes)
            table_optics = _build_static_category_table(p_df, "static_category_optics", "assigned_family_optics", primary_classes)
            static_tables_by_fold[fold_name][partition] = (table_meta, table_optics)

            for cat_name, r_vals in table_meta.iterrows():
                for f_name, cnt in r_vals.items():
                    if f_name != "Total":
                        static_dist_rows.append({
                            "fold": fold_name,
                            "partition": partition,
                            "mode": "metadata_assisted",
                            "static_category": cat_name,
                            "family": f_name,
                            "count": int(cnt),
                        })

            for cat_name, r_vals in table_optics.iterrows():
                for f_name, cnt in r_vals.items():
                    if f_name != "Total":
                        static_dist_rows.append({
                            "fold": fold_name,
                            "partition": partition,
                            "mode": "optics_only",
                            "static_category": cat_name,
                            "family": f_name,
                            "count": int(cnt),
                        })

            # Counts for conflicts, contradictions, quality gate, and metadata dependence
            n_list_conflict = int((p_df["reason"] == "list_conflict").sum()) if "reason" in p_df.columns else 0
            n_optics_list_conflict = int((p_df["optics_list_conflict"] == True).sum()) if "optics_list_conflict" in p_df.columns else 0
            n_hard_contradiction = int((p_df["reason"] == "hard_contradiction").sum()) if "reason" in p_df.columns else 0
            n_low_r2_4 = int((p_df["reason"] == "low_r2_4").sum()) if "reason" in p_df.columns else 0

            n_sing_meta_true = int((p_sing["metadata_dependent"] == True).sum()) if "metadata_dependent" in p_sing.columns else 0
            n_sing_meta_false = int((p_sing["metadata_dependent"] == False).sum()) if "metadata_dependent" in p_sing.columns else 0

            conflict_stats_by_fold[fold_name][partition] = {
                "n_list_conflict": n_list_conflict,
                "n_optics_list_conflict": n_optics_list_conflict,
                "n_hard_contradiction": n_hard_contradiction,
                "n_low_r2_4": n_low_r2_4,
                "singleton_meta_dependent_true": n_sing_meta_true,
                "singleton_meta_dependent_false": n_sing_meta_false,
            }

            for fam in primary_classes:
                fam_df = p_sing[p_sing["family"] == fam]
                support_rows.append({
                    "fold": fold_name,
                    "partition": partition,
                    "class": fam,
                    "singleton_regions": len(fam_df),
                    "singleton_specimens": fam_df["code"].nunique(),
                    "total_partition_regions": len(p_df),
                    "n_singleton": state_counts.get("singleton", 0),
                    "n_partial": state_counts.get("partial_set", 0),
                    "n_abstain": state_counts.get("abstain", 0),
                    "n_mixed": state_counts.get("mixed", 0),
                    "n_excluded": state_counts.get("excluded", 0),
                    "pct_metadata_dependent": round(100.0 * p_df[p_df["state"].isin(["singleton", "partial_set"])]["metadata_dependent"].sum() / max(len(p_df[p_df["state"].isin(["singleton", "partial_set"])]), 1), 2),
                    "n_list_conflict": n_list_conflict,
                    "n_optics_list_conflict": n_optics_list_conflict,
                    "n_hard_contradiction": n_hard_contradiction,
                    "n_low_r2_4": n_low_r2_4,
                    "singleton_meta_dependent_true": n_sing_meta_true,
                    "singleton_meta_dependent_false": n_sing_meta_false,
                })

    support_df = pd.DataFrame(support_rows)
    support_df.to_csv(out_dir / "target_support_by_fold.csv", index=False)
    logger.info("Saved target_support_by_fold.csv")

    static_dist_df = pd.DataFrame(static_dist_rows)
    static_dist_df.to_csv(out_dir / "static_category_distributions.csv", index=False)
    logger.info("Saved static_category_distributions.csv (%d records)", len(static_dist_df))

    # ── 5. Partial set support CSV ────────────────────────────────────
    partial_rows = []
    ref_targets = targets_by_fold["fold_0"]
    ref_partial = ref_targets[ref_targets["state"] == "partial_set"]
    for _, row in ref_partial.iterrows():
        partial_rows.append({
            "region_id": row["region_id"],
            "site_id": row["site_id"],
            "code": row["code"],
            "partition": row["partition"],
            "allowed_set": "{" + ", ".join(sorted(row["allowed"])) + "}",
            "set_size": len(row["allowed"]),
            "top_species": row.get("top_species"),
            "top_species_p": row.get("top_species_p"),
            "metadata_dependent": row.get("metadata_dependent"),
        })
    partial_df = pd.DataFrame(partial_rows)
    partial_df.to_csv(out_dir / "partial_set_support.csv", index=False)
    logger.info("Saved partial_set_support.csv (%d regions)", len(partial_df))

    # ── 6. Label evidence breakdown CSV ──────────────────────────────
    evidence_rows = []
    labelled_ref = ref_targets[ref_targets["state"].isin(["singleton", "partial_set"])]
    for _, row in labelled_ref.iterrows():
        evidence_rows.append({
            "region_id": row["region_id"],
            "state": row["state"],
            "allowed": "{" + ", ".join(sorted(row["allowed"])) + "}",
            "top_species": row.get("top_species"),
            "top_species_p": row.get("top_species_p"),
            "n_info": row.get("n_info"),
            "deciding_terms": ", ".join(row.get("deciding_terms", [])),
            "metadata_dependent": row.get("metadata_dependent"),
            "optics_state": row.get("optics_state"),
            "optics_allowed": "{" + ", ".join(sorted(row.get("optics_allowed", []))) + "}",
        })
    evidence_df = pd.DataFrame(evidence_rows)
    evidence_df.to_csv(out_dir / "label_evidence_breakdown.csv", index=False)
    logger.info("Saved label_evidence_breakdown.csv (%d rows)", len(evidence_df))

    # ── 7. Label rejections CSV ──────────────────────────────────────
    rejections_rows = []
    unlabelled_ref = ref_targets[~ref_targets["state"].isin(["singleton", "partial_set"])]
    for _, row in unlabelled_ref.iterrows():
        rejections_rows.append({
            "region_id": row["region_id"],
            "site_id": row["site_id"],
            "code": row["code"],
            "state": row["state"],
            "reason": row["reason"],
            "p_out": row.get("p_out"),
            "top_species": row.get("top_species"),
        })
    rejections_df = pd.DataFrame(rejections_rows)
    rejections_df.to_csv(out_dir / "label_rejections.csv", index=False)
    logger.info("Saved label_rejections.csv (%d rows)", len(rejections_df))

    # ── 8. Check Opaque Darkness (ppl_L and xpl_rel_L) ────────────────
    ref_merged = ref_targets.merge(
        regions_v1[["region_id", "ppl_L", "xpl_rel_L"]], on="region_id", how="left"
    )
    opaque_sing = ref_merged[(ref_merged["state"] == "singleton") & ref_merged["allowed"].apply(lambda a: "opaque_material" in a)]
    other_sing = ref_merged[(ref_merged["state"] == "singleton") & ~ref_merged["allowed"].apply(lambda a: "opaque_material" in a)]

    op_ppl_med = float(opaque_sing["ppl_L"].median()) if len(opaque_sing) > 0 else float("nan")
    op_xpl_med = float(opaque_sing["xpl_rel_L"].median()) if len(opaque_sing) > 0 else float("nan")
    oth_ppl_med = float(other_sing["ppl_L"].median()) if len(other_sing) > 0 else float("nan")
    oth_xpl_med = float(other_sing["xpl_rel_L"].median()) if len(other_sing) > 0 else float("nan")

    logger.info("Opaque lightness verification:")
    logger.info("  opaque_material singletons: n=%d, ppl_L median=%.2f, xpl_rel_L median=%.2f",
                len(opaque_sing), op_ppl_med, op_xpl_med)
    logger.info("  other singletons: n=%d, ppl_L median=%.2f, xpl_rel_L median=%.2f",
                len(other_sing), oth_ppl_med, oth_xpl_med)

    # ── 9. Generate QC Contact Sheets for Singletons ──────────────────
    try:
        regions_geo = _add_segment_geometry(regions_v1, segments_dir)
        _generate_singleton_panels(
            ref_targets, regions_geo, cfg, qc_dir, primary_classes, n_per_family=20
        )
    except Exception as e:
        logger.warning("QC contact sheet panel generation encountered an error: %s", e)

    # ── 10. Write label_feasibility_report.md ─────────────────────────
    n_total_regions = len(regions_v1)
    n_sites = regions_v1["site_id"].nunique() if "site_id" in regions_v1.columns else regions_v1["code"].nunique()
    list_conflict_policy = v1_cfg.get("rules", {}).get("list_conflict_policy", "abstain")
    rep_lines = [
        "# V1 Label Feasibility Report\n\n",
        f"**Generated by `v1_label_feasibility.py` across all {len(targets_by_fold)} frozen folds.**\n\n",
        f"**Total regions:** {n_total_regions:,} across {n_sites:,} sites\n\n",
        f"**Active list conflict policy:** `{list_conflict_policy}` (from `v1.rules.list_conflict_policy`).\n\n",
        "## 1. Five-Class Gate Result\n\n",
        f"**Gate Criterion:** Every family must have $\\ge {min_train_specimens}$ singleton training specimens in every fold.\n\n",
        f"### **OVERALL GATE STATUS: {'✅ PASS' if overall_gate_pass else '❌ FAIL'}**\n\n",
        "| Fold | Family | Train Regions | Train Specimens | Threshold | Status |\n",
        "|------|--------|---------------|-----------------|-----------|--------|\n",
    ]
    for _, r in gate_df.iterrows():
        rep_lines.append(
            f"| {r['fold']} | {r['family']} | {r['train_regions']} | {r['train_specimens']} | "
            f"$\\ge {r['gate_threshold']}$ | {r['gate_status']} |\n"
        )

    rep_lines.append("\n## 2. State Distribution by Fold and Partition\n\n")
    rep_lines.append("| Fold | Partition | Singleton | Partial Set | Abstain | Mixed | Excluded | % Meta-Dep (trainable) |\n")
    rep_lines.append("|------|-----------|-----------|-------------|---------|-------|----------|------------------------|\n")

    for fold_name in sorted(targets_by_fold.keys()):
        targets = targets_by_fold[fold_name]
        for part in ["train", "val", "test", "all"]:
            p_df = targets if part == "all" else targets[targets["partition"] == part]
            sc = p_df["state"].value_counts().to_dict()
            p_trainable = p_df[p_df["state"].isin(["singleton", "partial_set"])]
            pct_md = 100.0 * p_trainable["metadata_dependent"].sum() / max(len(p_trainable), 1)
            rep_lines.append(
                f"| {fold_name} | {part} | {sc.get('singleton', 0)} | {sc.get('partial_set', 0)} | "
                f"{sc.get('abstain', 0)} | {sc.get('mixed', 0)} | {sc.get('excluded', 0)} | {pct_md:.1f}% |\n"
            )

    rep_lines.append("\n## 3. Conflict, Contradiction, and Quality Gate Counts by Fold and Partition\n\n")
    rep_lines.append("| Fold | Partition | `list_conflict` | `optics_list_conflict` | `hard_contradiction` | `low_r2_4` | Singleton Meta-Dep=True | Singleton Meta-Dep=False |\n")
    rep_lines.append("|------|-----------|-----------------|------------------------|----------------------|------------|-------------------------|--------------------------|\n")
    for fold_name in sorted(targets_by_fold.keys()):
        for part in ["train", "val", "test", "all"]:
            st = conflict_stats_by_fold[fold_name][part]
            rep_lines.append(
                f"| {fold_name} | {part} | {st['n_list_conflict']} | {st['n_optics_list_conflict']} | "
                f"{st['n_hard_contradiction']} | {st['n_low_r2_4']} | {st['singleton_meta_dependent_true']} | "
                f"{st['singleton_meta_dependent_false']} |\n"
            )

    rep_lines.append("\n## 4. Partial Set Distribution (Fold 0)\n\n")
    rep_lines.append("| Candidate Set | Region Count |\n")
    rep_lines.append("|---------------|--------------|\n")
    for s_name, count in sorted(partial_set_counts_by_fold["fold_0"].items(), key=lambda x: -x[1]):
        rep_lines.append(f"| {s_name} | {count} |\n")

    rep_lines.append("\n## 5. Abstain Reasons (Fold 0)\n\n")
    rep_lines.append("| Reason | Region Count |\n")
    rep_lines.append("|--------|--------------|\n")
    for r_name, count in sorted(abstain_reasons_by_fold["fold_0"].items(), key=lambda x: -x[1]):
        rep_lines.append(f"| {r_name} | {count} |\n")

    rep_lines.append("\n## 6. Deciding Evidence Terms (Fold 0)\n\n")
    rep_lines.append("Evidence terms contributing to `n_info` on singletons and partial sets:\n\n")
    rep_lines.append("| Evidence Term | Occurrences |\n")
    rep_lines.append("|---------------|-------------|\n")
    for term, count in sorted(evidence_terms_by_fold["fold_0"].items(), key=lambda x: -x[1]):
        rep_lines.append(f"| {term} | {count} |\n")

    rep_lines.append("\n## 7. Opaque Material Lightness Verification\n\n")
    rep_lines.append(f"- **Opaque Material Singletons:** $N={len(opaque_sing)}$, median `ppl_L` = {op_ppl_med:.2f}, median `xpl_rel_L` = {op_xpl_med:.2f}\n")
    rep_lines.append(f"- **Other Singletons:** $N={len(other_sing)}$, median `ppl_L` = {oth_ppl_med:.2f}, median `xpl_rel_L` = {oth_xpl_med:.2f}\n\n")

    # ── 8. Hard Assertion Summary ────────────────────────────────────
    n_eligible = int(regions_v1["eligible_v1"].sum()) if "eligible_v1" in regions_v1.columns else len(regions_v1)
    n_low_r2_4 = int((ref_targets["reason"] == "low_r2_4").sum()) if "reason" in ref_targets.columns else 0
    pct_low_r2_4 = (100.0 * n_low_r2_4 / n_eligible) if n_eligible > 0 else 0.0

    rep_lines.append("## 8. Hard Assertion Summary\n\n")
    rep_lines.append("- `set(regions_v1.columns) >= set(V1_STATUS_COLUMNS)`: **PASSED**\n")
    rep_lines.append("- `not any('fine_aggregate' == t for t in targets.top_species[targets.state=='singleton'])`: **PASSED**\n")
    rep_lines.append("- `decided_terms.get('birefringence', 0) == 0 and decided_terms.get('extinction_type', 0) == 0`: **PASSED**\n")
    rep_lines.append(f"- `low_r2_4 extinction quality gate`: Anisotropic regions with $R^2_{{2,4}} < 0.4$ are gated out by RuleEngine to prevent noisy extinction signals from corrupting label assignments.\n\n")
    rep_lines.append(f"low_r2_4 grains ({n_low_r2_4}, {pct_low_r2_4:.1f}% of eligible) are excluded as a declared population restriction; results are conditional on this population.\n")

    rep_lines.append("\n## 9. Static Category Distributions by Fold and Partition (Task D1.4)\n\n")
    rep_lines.append("Comparison between full rotation and single-angle k1 targets per CONTRACTS.md Section 3.\n\n")

    for fold_name in sorted(targets_by_fold.keys()):
        for part in ["train", "val", "test", "all"]:
            table_meta, table_optics = static_tables_by_fold[fold_name][part]
            rep_lines.append(f"### {fold_name} — Partition: `{part}`\n\n")
            rep_lines.append("#### Metadata-Assisted: `static_category` x Family\n\n")
            rep_lines.append(_format_markdown_table(table_meta, index_name="Static Category") + "\n\n")
            rep_lines.append("#### Optics-Only: `static_category_optics` x Family\n\n")
            rep_lines.append(_format_markdown_table(table_optics, index_name="Static Category (Optics)") + "\n\n")

    report_path = out_dir / "label_feasibility_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(rep_lines)
    logger.info("Saved label_feasibility_report.md to %s", report_path)

    # ── 11. Save provenance ──────────────────────────────────────────
    save_provenance(
        out_dir / "provenance_feasibility.json", cfg,
        extra={
            "overall_gate_pass": overall_gate_pass,
            "teacher_fingerprints": teacher_fingerprints,
            "targets_sha256": targets_checksums,
            "regions_sha256": _hash_file(regions_path),
            "folds_sha256": _hash_file(folds_path),
            "min_train_specimens": min_train_specimens,
            "primary_classes": primary_classes,
            "n_folds": len(targets_by_fold),
            "op_ppl_med": op_ppl_med,
            "op_xpl_med": op_xpl_med,
        },
    )
    logger.info("Saved provenance_feasibility.json")

    # ── 12. Enforce Feasibility Gate ─────────────────────────────────
    # The low_r2_4 gate in RuleEngine.check_quality_gate ensures that anisotropic
    # regions with harmonic fit r2_4 < min_r2_4 (typically 0.4) are rejected/abstained
    # so that noisy extinction angles do not corrupt label evidence.
    if not overall_gate_pass:
        logger.error(
            "LABEL FEASIBILITY GATE FAILED: One or more primary classes do not meet the minimum training specimen threshold (>= %d) across all folds.",
            min_train_specimens,
        )
        sys.exit(1)

    logger.info("LABEL FEASIBILITY GATE PASSED across all folds.")


if __name__ == "__main__":
    main()
