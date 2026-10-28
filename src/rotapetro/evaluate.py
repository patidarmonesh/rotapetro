"""V1 Evaluation — specimen-balanced macro-F1 with paired bootstrap (§10).

Features:
- Specimen-balanced macro-F1: sample_weight = 1 / n_singleton_rows_in_that_specimen
- Point estimate: pooled over 5 folds, averaged over seeds (not bootstrap mean)
- Paired bootstrap: 5,000 replicates resampling specimens within each fold (same draw for both arms and all seeds)
- 95% CI (2.5 - 97.5 percentiles), p-value, and class disappearance tracking
- Secondary metrics:
  - Partial-set compatibility (top-1 in set and probability mass in set)
  - Unordered vs static and angle-aware vs unordered comparisons
  - Static angle sensitivity (all 12 K12 angles + index 0)
  - Results split by metadata_dependent
- Strict region_id joins: duplicate or missing IDs raise ValueError; shuffling invariant
- Dummy predictor power check on targets
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, accuracy_score

from src.rotapetro.dataset import TEST_PREDICTED_STATES

logger = logging.getLogger(__name__)

DEFAULT_PRIMARY_CLASSES = ["quartz", "feldspar", "mica", "pyroxene", "opaque_material"]


def compute_weighted_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    n_classes: int,
    class_names: list[str],
) -> dict[str, Any]:
    """Compute weighted macro-F1, per-class metrics, accuracy, and confusion matrix."""
    pair_indices = y_true * n_classes + y_pred
    cm = np.bincount(pair_indices, weights=weights, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    p_denom = tp + fp
    precision = np.zeros(n_classes, dtype=float)
    p_nz = p_denom > 0
    precision[p_nz] = tp[p_nz] / p_denom[p_nz]

    r_denom = tp + fn
    recall = np.zeros(n_classes, dtype=float)
    r_nz = r_denom > 0
    recall[r_nz] = tp[r_nz] / r_denom[r_nz]

    f1_denom = 2.0 * tp + fp + fn
    f1 = np.zeros(n_classes, dtype=float)
    f1_nz = f1_denom > 0
    f1[f1_nz] = (2.0 * tp[f1_nz]) / f1_denom[f1_nz]

    macro_f1 = float(np.mean(f1))
    total_weight = float(np.sum(weights))
    accuracy = float(np.sum(tp) / total_weight) if total_weight > 0 else 0.0

    per_class = {}
    for c in range(n_classes):
        c_name = class_names[c] if c < len(class_names) else str(c)
        per_class[c_name] = {
            "precision": round(float(precision[c]), 4),
            "recall": round(float(recall[c]), 4),
            "f1": round(float(f1[c]), 4),
            "support_weighted": round(float(r_denom[c]), 4),
            "support_unweighted": int(np.sum(y_true == c)),
        }

    return {
        "macro_f1": float(macro_f1),
        "accuracy": float(accuracy),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def fast_macro_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: np.ndarray,
    n_classes: int,
) -> float:
    """Compute weighted macro-F1 in microseconds using numpy bincount."""
    pair_indices = y_true * n_classes + y_pred
    cm = np.bincount(pair_indices, weights=weights, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2.0 * tp + fp + fn
    f1 = np.zeros(n_classes, dtype=float)
    nz = denom > 0
    f1[nz] = (2.0 * tp[nz]) / denom[nz]
    return float(np.mean(f1))


def _normalize_metadata_val(val: Any) -> Any:
    """Normalize metadata value for comparison."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, (list, tuple, np.ndarray)):
        return tuple(str(x) for x in val)
    if isinstance(val, str):
        s = val.strip()
        if (s.startswith("[") and s.endswith("]")) or (s.startswith("(") and s.endswith(")")):
            import ast
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, set)):
                    return tuple(str(x) for x in parsed)
            except Exception:
                pass
        return s
    return val


def _metadata_values_equal(v1: Any, v2: Any, is_fold: bool = False) -> bool:
    """Check equality between prediction metadata and authoritative target metadata."""
    if is_fold:
        s1 = str(v1).replace("fold_", "").strip()
        s2 = str(v2).replace("fold_", "").strip()
        return s1 == s2

    n1 = _normalize_metadata_val(v1)
    n2 = _normalize_metadata_val(v2)
    if n1 is None and n2 is None:
        return True
    if n1 is None or n2 is None:
        return False
    if isinstance(n1, tuple) and isinstance(n2, tuple):
        return sorted(n1) == sorted(n2)
    if type(n1) != type(n2):
        return str(n1) == str(n2)
    return n1 == n2


class Evaluator:
    """V1 Evaluator (§10)."""

    def __init__(
        self,
        cfg: dict | None = None,
        classes: list[str] | None = None,
        output_dir: str | Path | None = None,
        n_bootstrap: int = 5000,
        seed: int = 42,
    ):
        self.cfg = cfg or {}
        v1_cfg = self.cfg.get("v1", {})
        v1_train = v1_cfg.get("training", {})
        v1_tax = v1_cfg.get("taxonomy", {})
        v1_paths = v1_cfg.get("paths", {})

        if classes is not None:
            self.classes = list(classes)
        elif "primary_classes" in v1_tax:
            self.classes = list(v1_tax["primary_classes"])
        else:
            self.classes = list(DEFAULT_PRIMARY_CLASSES)

        self.n_classes = len(self.classes)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        out_path = output_dir or v1_paths.get("results", "outputs/v1")
        self.output_dir = Path(out_path)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.n_bootstrap = int(v1_train.get("n_bootstrap", n_bootstrap))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    # ──────────────────────────────────────────────────────────────────
    #  1. Specimen-balanced F1 (§10.1)
    # ──────────────────────────────────────────────────────────────────

    def compute_specimen_balanced_f1(
        self,
        predictions_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compute specimen-balanced macro-F1 on singleton rows.

        sample_weight = 1 / n_singleton_rows_in_that_specimen
        """
        if "region_id" in predictions_df.columns and predictions_df["region_id"].duplicated().any():
            raise ValueError("Duplicate region_id found in predictions_df")

        df = predictions_df.copy()
        if "state" in df.columns:
            df = df[df["state"] == "singleton"].copy()

        if len(df) == 0:
            return {
                "macro_f1": 0.0,
                "accuracy": 0.0,
                "n_regions": 0,
                "n_specimens": 0,
                "per_class": {c: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support_weighted": 0.0, "support_unweighted": 0} for c in self.classes},
                "confusion_matrix": [[0] * self.n_classes for _ in range(self.n_classes)],
            }

        # Format y_true and y_pred as integer class indices
        if "y_true" in df.columns:
            y_true_raw = df["y_true"].values
        elif "allowed" in df.columns:
            y_true_raw = [a[0] if (isinstance(a, (list, tuple, np.ndarray)) and len(a) > 0) else a for a in df["allowed"]]
        else:
            raise KeyError("predictions_df must contain 'y_true' or 'allowed' column")

        y_true = np.array([self.class_to_idx.get(str(y), -1) for y in y_true_raw], dtype=int)
        y_pred_raw = df["y_pred"].values
        y_pred = np.array([self.class_to_idx.get(str(y), -1) if isinstance(y, str) else int(y) for y in y_pred_raw], dtype=int)

        if np.any(y_true < 0):
            raise ValueError(f"Unrecognized true class labels found: {set(y_true_raw[y_true < 0])}")
        if np.any(y_pred < 0):
            raise ValueError(f"Unrecognized pred class labels found: {set(y_pred_raw[y_pred < 0])}")

        spec_col = "specimen" if "specimen" in df.columns else ("code" if "code" in df.columns else None)
        if spec_col is None:
            raise KeyError("predictions_df must contain 'specimen' or 'code' column")
        specimens = df[spec_col].values

        # Specimen-balanced weights: 1 / n_singleton_rows_in_that_specimen
        counts = pd.Series(specimens).value_counts()
        weights = np.array([1.0 / counts[s] for s in specimens], dtype=float)

        metrics = compute_weighted_metrics(
            y_true=y_true,
            y_pred=y_pred,
            weights=weights,
            n_classes=self.n_classes,
            class_names=self.classes,
        )
        metrics["n_regions"] = len(df)
        metrics["n_specimens"] = int(pd.Series(specimens).nunique())
        return metrics

    compute_macro_f1 = compute_specimen_balanced_f1

    # ──────────────────────────────────────────────────────────────────
    #  2. Joining by region_id (§10.5)
    # ──────────────────────────────────────────────────────────────────

    def join_predictions_with_targets(
        self,
        preds_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        require_two_way: bool = False,
    ) -> pd.DataFrame:
        """Strictly join predictions with targets by region_id (§10.5).

        Targets are the authoritative source of truth:
        - state
        - allowed
        - specimen (or code)
        - fold

        If preds_df contains any of these columns and their values differ from
        targets_df, raises ValueError("Prediction metadata mismatch with target...").

        Raises ValueError on duplicate or missing region_ids.
        Row order invariance is guaranteed.
        """
        if preds_df["region_id"].duplicated().any():
            dups = preds_df["region_id"][preds_df["region_id"].duplicated()].unique().tolist()
            raise ValueError(f"Duplicate region_id in predictions: {dups[:5]}")
        if targets_df["region_id"].duplicated().any():
            dups = targets_df["region_id"][targets_df["region_id"].duplicated()].unique().tolist()
            raise ValueError(f"Duplicate region_id in targets: {dups[:5]}")

        target_rids = set(targets_df["region_id"])
        pred_rids = set(preds_df["region_id"])
        missing = pred_rids - target_rids
        if missing:
            raise ValueError(f"Predictions contain {len(missing)} region_ids not present in targets: {list(missing)[:5]}")

        if require_two_way:
            missing_preds = target_rids - pred_rids
            if missing_preds:
                raise ValueError(
                    f"Two-way match failed: missing predictions for {len(missing_preds)} target regions: {list(missing_preds)[:5]}"
                )

        # Train/val contamination guard if partition is present in targets
        if "partition" in targets_df.columns:
            train_val_rids = set(targets_df[targets_df["partition"].isin(["train", "val"])]["region_id"])
            tv_leak = pred_rids & train_val_rids
            if tv_leak:
                raise ValueError(
                    f"Predictions contain {len(tv_leak)} extra rows from train or val: {list(tv_leak)[:5]}"
                )

        # Check metadata consistency between predictions and targets
        # Targets are authoritative: state, allowed, specimen, fold
        metadata_cols = ["state", "allowed", "specimen", "fold"]
        targets_indexed = targets_df.set_index("region_id")

        for col in metadata_cols:
            p_col = col
            if p_col not in preds_df.columns and col == "specimen" and "code" in preds_df.columns:
                p_col = "code"

            t_col = col
            if t_col not in targets_df.columns and col == "specimen" and "code" in targets_df.columns:
                t_col = "code"

            if p_col in preds_df.columns and t_col in targets_df.columns:
                target_vals = targets_indexed.loc[preds_df["region_id"], t_col].values
                pred_vals = preds_df[p_col].values
                for rid, p_val, t_val in zip(preds_df["region_id"].values, pred_vals, target_vals):
                    if not _metadata_values_equal(p_val, t_val, is_fold=(col == "fold")):
                        raise ValueError(
                            f"Prediction metadata mismatch with target for column '{col}' on region '{rid}': "
                            f"prediction has {p_val!r}, target has {t_val!r}"
                        )

        # Retain necessary target fields; targets are authoritative
        # Remove any metadata columns from preds_df so they come from targets_df
        drop_from_preds = {"state", "allowed", "specimen", "code", "fold"} & set(preds_df.columns)
        clean_preds = preds_df.drop(columns=list(drop_from_preds)) if drop_from_preds else preds_df

        target_cols = [c for c in targets_df.columns if c not in clean_preds.columns or c == "region_id"]
        joined = clean_preds.merge(targets_df[target_cols], on="region_id", how="inner")
        if len(joined) != len(preds_df):
            raise ValueError(f"Join mismatch: preds has {len(preds_df)} rows, joined has {len(joined)} rows")

        # Ensure specimen column exists if code was present
        if "specimen" not in joined.columns and "code" in joined.columns:
            joined["specimen"] = joined["code"]

        return joined

    # ──────────────────────────────────────────────────────────────────
    #  3. Paired bootstrap within folds (§10.2, §10.3)
    # ──────────────────────────────────────────────────────────────────

    def paired_bootstrap(
        self,
        df_a: pd.DataFrame | list[pd.DataFrame] | dict[int, pd.DataFrame],
        df_b: pd.DataFrame | list[pd.DataFrame] | dict[int, pd.DataFrame],
    ) -> dict[str, Any]:
        """Paired bootstrap comparison (Arm A - Arm B) with 5,000 replicates.

        Resamples specimens with replacement WITHIN each fold.
        Uses the exact same specimen draw for both arms and all seeds (§10.3).
        Returns point estimate difference, 95% CI, p-value, and class disappearance rates.
        """
        # Normalize input to list of DataFrames (one per seed)
        if isinstance(df_a, pd.DataFrame):
            dfs_a = [df_a]
        elif isinstance(df_a, dict):
            dfs_a = [df_a[k] for k in sorted(df_a.keys())]
        else:
            dfs_a = list(df_a)

        if isinstance(df_b, pd.DataFrame):
            dfs_b = [df_b]
        elif isinstance(df_b, dict):
            dfs_b = [df_b[k] for k in sorted(df_b.keys())]
        else:
            dfs_b = list(df_b)

        n_seeds = len(dfs_a)
        if len(dfs_b) != n_seeds:
            raise ValueError(
                f"Arms have different numbers of seeds: A={n_seeds}, B={len(dfs_b)}")

        # Filter to singletons
        singletons_a = [df[df["state"] == "singleton"].sort_values("region_id").reset_index(drop=True) for df in dfs_a]
        singletons_b = [df[df["state"] == "singleton"].sort_values("region_id").reset_index(drop=True) for df in dfs_b]

        # Validate identical region_id sequences across all seeds and between arms
        ref_rids = singletons_a[0]["region_id"].values
        for s in range(len(singletons_a)):
            if not np.array_equal(singletons_a[s]["region_id"].values, ref_rids):
                raise ValueError(f"Arm A seed {s} region_id sequence differs from reference (seed 0)")
        for s in range(len(singletons_b)):
            if not np.array_equal(singletons_b[s]["region_id"].values, ref_rids):
                raise ValueError(f"Arm B seed {s} region_id sequence differs from reference")

        if any(len(df) == 0 for df in singletons_a):
            raise ValueError("Arm A has no singleton rows in one or more seeds — cannot compute bootstrap")
        if any(len(df) == 0 for df in singletons_b):
            raise ValueError("Arm B has no singleton rows in one or more seeds — cannot compute bootstrap")

        ref_df = singletons_a[0]
        fold_col = "fold" if "fold" in ref_df.columns else None
        folds = sorted(ref_df[fold_col].unique()) if fold_col else ["all"]
        spec_col = "specimen" if "specimen" in ref_df.columns else "code"

        # Unique specimens per fold
        fold_specimens: dict[str, list[str]] = {}
        for f in folds:
            sub = ref_df[ref_df[fold_col] == f] if fold_col else ref_df
            fold_specimens[f] = sorted(sub[spec_col].unique().tolist())

        # Precalculate per-specimen singleton count and row index mappings
        spec_row_indices_a = {s: np.where(ref_df[spec_col].values == s)[0] for s in ref_df[spec_col].unique()}
        spec_row_indices_b = {s: np.where(singletons_b[0][spec_col].values == s)[0] for s in singletons_b[0][spec_col].unique()}
        spec_counts = {s: len(spec_row_indices_a[s]) for s in ref_df[spec_col].unique()}

        def extract_labels(df):
            if "y_true" in df.columns:
                yt = df["y_true"].values
            else:
                yt = [a[0] if isinstance(a, (list, tuple, np.ndarray)) else a for a in df["allowed"]]
            y_t = np.array([self.class_to_idx[str(y)] for y in yt], dtype=int)
            y_p = np.array([self.class_to_idx[str(y)] if isinstance(y, str) else int(y) for y in df["y_pred"]], dtype=int)
            return y_t, y_p

        yt_a_list, yp_a_list = [], []
        yt_b_list, yp_b_list = [], []
        for s in range(n_seeds):
            y_t_a, y_p_a = extract_labels(singletons_a[s])
            y_t_b, y_p_b = extract_labels(singletons_b[s])
            yt_a_list.append(y_t_a)
            yp_a_list.append(y_p_a)
            yt_b_list.append(y_t_b)
            yp_b_list.append(y_p_b)

        # Baseline point estimate calculation (§10.2): average across seeds
        base_weights_a = np.array([1.0 / spec_counts[s] for s in ref_df[spec_col].values], dtype=float)
        base_weights_b = np.array([1.0 / spec_counts[s] for s in singletons_b[0][spec_col].values], dtype=float)

        seed_diffs_base = []
        f1_a_base_list = []
        f1_b_base_list = []
        for s in range(n_seeds):
            f1_a = fast_macro_f1(yt_a_list[s], yp_a_list[s], base_weights_a, self.n_classes)
            f1_b = fast_macro_f1(yt_b_list[s], yp_b_list[s], base_weights_b, self.n_classes)
            f1_a_base_list.append(f1_a)
            f1_b_base_list.append(f1_b)
            seed_diffs_base.append(f1_a - f1_b)

        point_diff = float(np.mean(seed_diffs_base))

        # 5,000 bootstrap replicates (§10.3)
        boot_diffs = np.zeros(self.n_bootstrap, dtype=float)
        class_disappeared_count = np.zeros(self.n_classes, dtype=int)

        n_rows_a = len(ref_df)
        n_rows_b = len(singletons_b[0])
        rng = np.random.default_rng(self.seed)

        for b in range(self.n_bootstrap):
            w_a = np.zeros(n_rows_a, dtype=float)
            w_b = np.zeros(n_rows_b, dtype=float)

            # Draw specimens with replacement per fold (exact same draw for both arms and all seeds)
            for f, specs in fold_specimens.items():
                if len(specs) == 0:
                    continue
                drawn = rng.choice(specs, size=len(specs), replace=True)
                counts = pd.Series(drawn).value_counts()
                for s_id, c_s in counts.items():
                    factor = float(c_s) / spec_counts[s_id]
                    idx_a = spec_row_indices_a.get(s_id, [])
                    if len(idx_a) > 0:
                        w_a[idx_a] = factor
                    idx_b = spec_row_indices_b.get(s_id, [])
                    if len(idx_b) > 0:
                        w_b[idx_b] = factor

            # Check class disappearance
            active_a = w_a > 0
            if np.any(active_a):
                supp = np.bincount(yt_a_list[0][active_a], weights=w_a[active_a], minlength=self.n_classes)
                for c in range(self.n_classes):
                    if supp[c] <= 0:
                        class_disappeared_count[c] += 1

            # Recompute weighted F1 difference for each seed, average across seeds
            diffs_b = []
            for s in range(n_seeds):
                m_a = fast_macro_f1(yt_a_list[s], yp_a_list[s], w_a, self.n_classes)
                m_b = fast_macro_f1(yt_b_list[s], yp_b_list[s], w_b, self.n_classes)
                diffs_b.append(m_a - m_b)

            boot_diffs[b] = float(np.mean(diffs_b))

        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))
        p_val = float(np.mean(boot_diffs <= 0))

        disappear_rates = {
            self.classes[c]: round(float(class_disappeared_count[c] / self.n_bootstrap), 5)
            for c in range(self.n_classes)
        }

        return {
            "point_estimate_diff": round(float(point_diff), 5),
            "bootstrap_mean_diff": round(float(np.mean(boot_diffs)), 5),
            "mean_diff": round(float(point_diff), 5),
            "ci_lower": round(ci_lower, 5),
            "ci_upper": round(ci_upper, 5),
            "p_value_approx": round(p_val, 5),
            "class_disappearance_rate": disappear_rates,
            "n_bootstrap": self.n_bootstrap,
            "f1_a_base": round(float(np.mean(f1_a_base_list)), 5),
            "f1_b_base": round(float(np.mean(f1_b_base_list)), 5),
        }

    bootstrap_comparison = paired_bootstrap

    # ──────────────────────────────────────────────────────────────────
    #  4. Partial-set compatibility (§10.4)
    # ──────────────────────────────────────────────────────────────────

    def evaluate_partial_sets(
        self,
        predictions_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compute specimen-balanced partial-set compatibility and mass."""
        df = predictions_df.copy()
        if "state" in df.columns:
            df = df[df["state"] == "partial_set"].copy()

        if len(df) == 0:
            return {"top1_compatibility": 0.0, "probability_mass": 0.0, "n_regions": 0, "n_specimens": 0, "stratified": {}}

        spec_col = "specimen" if "specimen" in df.columns else "code"
        specimens = df[spec_col].values
        counts = pd.Series(specimens).value_counts()
        weights = np.array([1.0 / counts[s] for s in specimens], dtype=float)

        is_compat = []
        masses = []
        allowed_sets = []

        for _, row in df.iterrows():
            pred = str(row["y_pred"])
            allowed = row["allowed"]
            if isinstance(allowed, str):
                import ast
                try:
                    parsed = ast.literal_eval(allowed)
                    if isinstance(parsed, (list, tuple, set)):
                        allowed_list = [str(x) for x in parsed]
                    else:
                        allowed_list = [allowed]
                except Exception:
                    allowed_list = [allowed]
            elif isinstance(allowed, (list, tuple, np.ndarray)):
                allowed_list = [str(x) for x in allowed]
            else:
                allowed_list = [str(allowed)]
            compat = 1.0 if (pred in allowed_list) else 0.0
            is_compat.append(compat)

            # Sum of probabilities in allowed set
            p_mass = sum(float(row.get(f"probs_{f}", 0.0)) for f in allowed_list)
            masses.append(p_mass)
            allowed_sets.append(tuple(sorted(allowed_list)))

        is_compat = np.array(is_compat, dtype=float)
        masses = np.array(masses, dtype=float)
        tot_w = float(np.sum(weights))

        overall_compat = float(np.sum(weights * is_compat) / tot_w) if tot_w > 0 else 0.0
        overall_mass = float(np.sum(weights * masses) / tot_w) if tot_w > 0 else 0.0

        # Stratified by allowed set
        stratified = {}
        df["_set_tuple"] = allowed_sets
        for st, grp in df.groupby("_set_tuple"):
            st_idx = grp.index
            w_sub = weights[df.index.get_indexer(st_idx)]
            c_sub = is_compat[df.index.get_indexer(st_idx)]
            m_sub = masses[df.index.get_indexer(st_idx)]
            sub_w = float(np.sum(w_sub))
            stratified[str(list(st))] = {
                "n_regions": len(grp),
                "n_specimens": int(grp[spec_col].nunique()),
                "top1_compatibility": round(float(np.sum(w_sub * c_sub) / sub_w), 4) if sub_w > 0 else 0.0,
                "probability_mass": round(float(np.sum(w_sub * m_sub) / sub_w), 4) if sub_w > 0 else 0.0,
            }

        return {
            "top1_compatibility": round(overall_compat, 4),
            "probability_mass": round(overall_mass, 4),
            "n_regions": len(df),
            "n_specimens": int(pd.Series(specimens).nunique()),
            "stratified": stratified,
        }

    # ──────────────────────────────────────────────────────────────────
    #  5. Static angle sensitivity (§10.4)
    # ──────────────────────────────────────────────────────────────────

    def _find_folds_json(self, teach_dir: Path | str | None = None) -> Path | None:
        """Find folds.json path if available."""
        candidates = []
        if teach_dir is not None:
            td = Path(teach_dir)
            candidates.extend([
                td / "folds.json",
                td.parent / "folds.json",
                td.parent / "splits" / "folds.json",
            ])
        v1_paths = self.cfg.get("v1", {}).get("paths", {})
        if "folds" in v1_paths:
            candidates.append(Path(v1_paths["folds"]))
        ds_cfg = self.cfg.get("dataset", {})
        if "folds_path" in ds_cfg:
            candidates.append(Path(ds_cfg["folds_path"]))
        candidates.extend([
            Path("data/processed/v1/splits/folds.json"),
            Path("data/processed/splits/folds.json"),
        ])
        for cand in candidates:
            if cand.is_file():
                return cand
        return None

    @staticmethod
    def _recursive_average(dict_list: list[dict[str, Any]]) -> dict[str, Any]:
        """Recursively average numeric fields across a list of dictionaries.

        Recurses through arbitrary levels of nested dicts (e.g. metadata_split -> per_class -> f1).
        Preserves non-numeric values and exact integer counts.
        """
        if not dict_list:
            return {}
        base = dict_list[0]
        out: dict[str, Any] = {}
        for k, v in base.items():
            if isinstance(v, dict):
                sub_list = [d[k] for d in dict_list if isinstance(d.get(k), dict)]
                out[k] = Evaluator._recursive_average(sub_list)
            elif isinstance(v, (int, float, np.integer, np.floating)):
                vals = [d[k] for d in dict_list if isinstance(d.get(k), (int, float, np.integer, np.floating))]
                if vals:
                    if isinstance(v, (int, np.integer)) and all(d.get(k) == v for d in dict_list):
                        out[k] = int(v)
                    else:
                        out[k] = float(np.mean(vals))
                else:
                    out[k] = v
            elif isinstance(v, list):
                try:
                    arrs = [np.array(d[k], dtype=float) for d in dict_list if k in d]
                    out[k] = np.mean(arrs, axis=0).tolist()
                except Exception:
                    out[k] = v
            else:
                out[k] = v
        return out

    @staticmethod
    def _is_stratum_available(stratum_metrics: dict[str, Any]) -> tuple[bool, str]:
        """Check if stratum meets minimum requirements:
        >= 30 singleton regions and >= 3 classes with support.
        Returns (is_available, display_str).
        """
        n_reg = stratum_metrics.get("n_regions", 0)
        per_class = stratum_metrics.get("per_class", {})
        n_supported = sum(
            1 for c_info in per_class.values()
            if isinstance(c_info, dict) and (c_info.get("support_unweighted", 0) > 0 or c_info.get("support_weighted", 0) > 0)
        )
        if n_reg < 30 or n_supported < 3:
            return False, f"unavailable (n={n_reg})"
        f1 = stratum_metrics.get("macro_f1", 0.0)
        return True, f"{f1:.4f}"

    def _get_held_out_test_targets(
        self,
        targets_df: pd.DataFrame,
        fold: str,
        teach_dir: Path | str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter fold targets to held-out test rows and train/val rows.

        Returns (test_targets, train_val_targets).
        """
        # 1. Direct partition column in targets
        if "partition" in targets_df.columns:
            test_mask = targets_df["partition"] == "test"
            test_df = targets_df[test_mask & targets_df["state"].isin(TEST_PREDICTED_STATES)].copy()
            train_val_df = targets_df[~test_mask].copy()
            if len(test_df) == 0 and len(targets_df) > 0:
                raise ValueError(f"No held-out test rows found for fold '{fold}' with partition == 'test'")
            return test_df, train_val_df

        # 2. Split column in targets
        if "split" in targets_df.columns:
            test_mask = targets_df["split"] == "test"
            test_df = targets_df[test_mask & targets_df["state"].isin(TEST_PREDICTED_STATES)].copy()
            train_val_df = targets_df[~test_mask].copy()
            if len(test_df) == 0 and len(targets_df) > 0:
                raise ValueError(f"No held-out test rows found for fold '{fold}' with split == 'test'")
            return test_df, train_val_df

        # 3. is_test boolean column in targets
        if "is_test" in targets_df.columns:
            test_mask = targets_df["is_test"] == True
            test_df = targets_df[test_mask & targets_df["state"].isin(TEST_PREDICTED_STATES)].copy()
            train_val_df = targets_df[~test_mask].copy()
            if len(test_df) == 0 and len(targets_df) > 0:
                raise ValueError(f"No held-out test rows found for fold '{fold}' with is_test == True")
            return test_df, train_val_df

        # 4. Check folds.json for test split codes
        spec_col = "specimen" if "specimen" in targets_df.columns else ("code" if "code" in targets_df.columns else None)
        if spec_col is not None:
            folds_path = self._find_folds_json(teach_dir)
            if folds_path is not None and folds_path.is_file():
                try:
                    with open(folds_path, "r", encoding="utf-8") as f:
                        folds_data = json.load(f)
                    fold_entry = (
                        folds_data.get(fold)
                        or folds_data.get(fold.replace("fold_", ""))
                        or (folds_data.get(int(fold.replace("fold_", ""))) if fold.replace("fold_", "").isdigit() else None)
                    )
                    if fold_entry and "test" in fold_entry:
                        all_fold_codes = (
                            set(str(c) for c in fold_entry.get("train", []))
                            | set(str(c) for c in fold_entry.get("val", []))
                            | set(str(c) for c in fold_entry.get("test", []))
                        )
                        target_specs = set(targets_df[spec_col].astype(str))
                        if target_specs & all_fold_codes:
                            test_codes = set(str(c) for c in fold_entry["test"])
                            test_mask = targets_df[spec_col].astype(str).isin(test_codes)
                            test_df = targets_df[test_mask & targets_df["state"].isin(TEST_PREDICTED_STATES)].copy()
                            train_val_df = targets_df[~test_mask].copy()
                            if len(test_df) == 0 and len(targets_df) > 0:
                                raise ValueError(
                                    f"No held-out test rows matched test split codes for fold '{fold}'"
                                )
                            return test_df, train_val_df
                except Exception as e:
                    if isinstance(e, ValueError):
                        raise
                    logger.warning("Could not read folds.json at %s: %s", folds_path, e)

        # 5. Fallback must raise ValueError
        raise ValueError(f"targets have no partition/split/is_test column for fold '{fold}'")

    def evaluate_static_sensitivity(
        self,
        static_dir: Path | str,
        folds: list[str],
        seeds: list[int],
        fold_targets: dict[str, pd.DataFrame] | None = None,
        teach_dir: Path | str | None = None,
        samples_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Evaluate static model predictions across all angles (§10.4).

        Requires that all expected folds and seeds are present, matching primary
        comparison rigor.
        """
        static_path = Path(static_dir)
        if not static_path.is_dir():
            return {}

        if fold_targets:
            s_cand = samples_path
            if s_cand is None and teach_dir:
                cand_path = Path(teach_dir).parent / "samples.parquet"
                if cand_path.is_file():
                    s_cand = cand_path
            if s_cand and Path(s_cand).is_file():
                s_df = pd.read_parquet(s_cand)
                s_rids = set(s_df["region_id"])
                for f in folds:
                    if f in fold_targets:
                        t_targets, _ = self._get_held_out_test_targets(fold_targets[f], f, teach_dir=teach_dir)
                        missing = set(t_targets["region_id"]) - s_rids
                        if missing:
                            raise ValueError(
                                f"Cross-check against samples.parquet failed for fold '{f}': "
                                f"{len(missing)} required test regions missing from samples.parquet: "
                                f"{sorted(missing)[:5]}"
                            )

        k12_indices = [0, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]
        results_by_angle: dict[int, float] = {}

        # Check if sensitivity angle files exist
        has_angle_files = False
        for f in folds:
            for s in seeds:
                s_dir = static_path / f / f"seed_{s}"
                if s_dir.is_dir() and any(s_dir.glob("predictions_angle_*.parquet")):
                    has_angle_files = True
                    break
            if has_angle_files:
                break

        if has_angle_files:
            angles_to_eval = k12_indices
        else:
            # Check if primary angle (index 1) predictions exist
            has_primary = any(
                (static_path / f / f"seed_{s}" / "predictions.parquet").is_file()
                for f in folds for s in seeds
            )
            angles_to_eval = [1] if has_primary else []

        if not angles_to_eval:
            return {}

        for ang in angles_to_eval:
            seed_f1s = []
            for s in seeds:
                fold_dfs = []
                for f in folds:
                    p = static_path / f / f"seed_{s}" / f"predictions_angle_{ang}.parquet"
                    if not p.is_file() and ang == 1:
                        p_alt = static_path / f / f"seed_{s}" / "predictions.parquet"
                        if p_alt.is_file():
                            p = p_alt
                    if not p.is_file():
                        raise FileNotFoundError(
                            f"Static sensitivity requires all expected folds and seeds to be present. "
                            f"Missing predictions file for angle {ang}, fold '{f}', seed {s}: {p}"
                        )
                    p_df = pd.read_parquet(p)
                    if "fold" not in p_df.columns:
                        p_df["fold"] = f

                    if fold_targets and f in fold_targets:
                        test_targets, train_val_targets = self._get_held_out_test_targets(
                            fold_targets[f], f, teach_dir=teach_dir
                        )
                        pred_rids = set(p_df["region_id"])
                        test_rids = set(test_targets["region_id"])
                        train_val_rids = set(train_val_targets["region_id"])

                        tv_overlap = pred_rids & train_val_rids
                        if tv_overlap:
                            raise ValueError(
                                f"Static sensitivity angle {ang}, fold '{f}', seed {s} contains "
                                f"{len(tv_overlap)} extra rows from train or val: {list(tv_overlap)[:5]}"
                            )
                        extra_rids = pred_rids - test_rids
                        if extra_rids:
                            raise ValueError(
                                f"Static sensitivity angle {ang}, fold '{f}', seed {s} contains "
                                f"{len(extra_rids)} extra rows not in held-out test targets: {list(extra_rids)[:5]}"
                            )
                        missing_test_rids = test_rids - pred_rids
                        if missing_test_rids:
                            raise ValueError(
                                f"Static sensitivity angle {ang}, fold '{f}', seed {s} missing "
                                f"predictions for {len(missing_test_rids)} held-out test regions: {list(missing_test_rids)[:5]}"
                            )
                        p_df = self.join_predictions_with_targets(p_df, test_targets, require_two_way=True)

                    fold_dfs.append(p_df)

                if len(fold_dfs) != len(folds):
                    raise ValueError(
                        f"Missing folds for static sensitivity angle {ang}, seed {s}: "
                        f"expected {len(folds)}, got {len(fold_dfs)}"
                    )
                pooled = pd.concat(fold_dfs, ignore_index=True)
                m = self.compute_specimen_balanced_f1(pooled)
                seed_f1s.append(m["macro_f1"])

            if len(seed_f1s) != len(seeds):
                raise ValueError(
                    f"Missing seeds for static sensitivity angle {ang}: "
                    f"expected {len(seeds)}, got {len(seed_f1s)}"
                )
            results_by_angle[ang] = round(float(np.mean(seed_f1s)), 4)

        if not results_by_angle:
            return {}

        vals = list(results_by_angle.values())
        return {
            "by_angle_index": results_by_angle,
            "mean_macro_f1": round(float(np.mean(vals)), 4),
            "std_macro_f1": round(float(np.std(vals)), 4),
            "min_macro_f1": round(float(np.min(vals)), 4),
            "max_macro_f1": round(float(np.max(vals)), 4),
            "primary_angle_index_1": results_by_angle.get(1),
        }

    # ──────────────────────────────────────────────────────────────────
    #  6. Metadata-dependent split (§10.4)
    # ──────────────────────────────────────────────────────────────────

    def evaluate_metadata_split(
        self,
        preds_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Evaluate metrics split by metadata_dependent flag."""
        if "metadata_dependent" not in preds_df.columns:
            return {"note": "metadata_dependent column missing from dataframe"}

        df = preds_df[preds_df["state"] == "singleton"].copy()
        meta_dep = df[df["metadata_dependent"] == True]
        optics_only = df[df["metadata_dependent"] == False]

        return {
            "metadata_dependent": self.compute_specimen_balanced_f1(meta_dep),
            "optics_only_reproducible": self.compute_specimen_balanced_f1(optics_only),
        }

    def evaluate_static_strata(
        self,
        pooled_preds: dict[str, dict[int, pd.DataFrame]],
        strata: tuple[str, ...] = ("reproduced", "rotation_dependent"),
    ) -> dict[str, Any]:
        """Evaluate static strata performance and paired bootstrap (§10, Task D1.5).

        For held-out singleton rows, evaluate strata static_category == 'reproduced'
        and rotation_dependent (full_only ∪ k1_broader).
        Computes specimen-balanced macro-F1 per arm, paired bootstrap (angle_aware_k12 vs static_k1),
        n regions, n specimens. Applies unavailable rule (< 30 regions or < 3 supported classes).
        """
        results: dict[str, Any] = {}

        for stratum in strata:
            stratum_res: dict[str, Any] = {
                "arm_metrics": {},
                "paired_bootstrap": None,
                "status": "available",
            }

            arm_seed_dfs: dict[str, list[pd.DataFrame]] = {}
            for arm, seed_dict in pooled_preds.items():
                dfs = []
                for s in sorted(seed_dict.keys()):
                    df = seed_dict[s]
                    if "static_category" in df.columns:
                        if stratum == "rotation_dependent":
                            mask = df["static_category"].isin(["full_only", "k1_broader"])
                        else:
                            mask = (df["static_category"] == stratum)
                        if "state" in df.columns:
                            mask &= (df["state"] == "singleton")
                        s_df = df[mask].copy()
                        if "state" not in s_df.columns:
                            s_df["state"] = "singleton"
                    else:
                        s_df = df.iloc[0:0].copy()
                        if "state" not in s_df.columns:
                            s_df["state"] = "singleton"
                    dfs.append(s_df)
                arm_seed_dfs[arm] = dfs

            n_reg = 0
            n_spec = 0
            for arm, dfs in arm_seed_dfs.items():
                if dfs and len(dfs) > 0 and len(dfs[0]) > 0:
                    first_df = dfs[0]
                    n_reg = len(first_df)
                    spec_col = "specimen" if "specimen" in first_df.columns else ("code" if "code" in first_df.columns else None)
                    n_spec = int(first_df[spec_col].nunique()) if spec_col else 0
                    break

            stratum_res["n_regions"] = n_reg
            stratum_res["n_specimens"] = n_spec

            for arm, dfs in arm_seed_dfs.items():
                seed_metrics = [self.compute_specimen_balanced_f1(df) for df in dfs]
                avg_m = self._recursive_average(seed_metrics)
                stratum_res["arm_metrics"][arm] = avg_m

            # Apply unavailable rule
            sample_m = stratum_res["arm_metrics"].get("angle_aware_k12") or (
                next(iter(stratum_res["arm_metrics"].values())) if stratum_res["arm_metrics"] else {"n_regions": n_reg}
            )
            avail, reason = self._is_stratum_available(sample_m)

            if not avail:
                stratum_res["status"] = "unavailable"
                stratum_res["unavailable_reason"] = reason
            else:
                stratum_res["status"] = "available"
                # Paired bootstrap between angle_aware_k12 and static_k1
                if "angle_aware_k12" in arm_seed_dfs and "static_k1" in arm_seed_dfs:
                    dfs_a = arm_seed_dfs["angle_aware_k12"]
                    dfs_b = arm_seed_dfs["static_k1"]
                    if (
                        dfs_a
                        and dfs_b
                        and len(dfs_a) == len(dfs_b)
                        and all(len(d) > 0 for d in dfs_a)
                        and all(len(d) > 0 for d in dfs_b)
                    ):
                        try:
                            stratum_res["paired_bootstrap"] = self.paired_bootstrap(dfs_a, dfs_b)
                        except Exception as e:
                            logger.warning("Could not compute paired bootstrap for stratum '%s': %s", stratum, e)
                            stratum_res["paired_bootstrap"] = {"error": str(e)}

            results[stratum] = stratum_res

        return results

    # ──────────────────────────────────────────────────────────────────
    #  7. Dummy predictor power check (§11)
    # ──────────────────────────────────────────────────────────────────

    def run_dummy_power_check(
        self,
        targets_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """Run evaluator with dummy majority-class predictor on real targets."""
        df = targets_df.copy()
        if "partition" in df.columns:
            df = df[df["partition"] == "test"]
        singletons = df[df["state"] == "singleton"].copy()
        if len(singletons) == 0:
            return {"error": "No singleton targets found"}

        # Extract true class
        y_true_raw = [a[0] if isinstance(a, (list, tuple, np.ndarray)) else a for a in singletons["allowed"]]
        singletons["y_true"] = y_true_raw

        majority_cls = singletons["y_true"].value_counts().index[0]
        logger.info("Dummy predictor power check using majority class: %s", majority_cls)

        dummy_preds = singletons.copy()
        dummy_preds["y_pred"] = majority_cls

        m_dummy = self.compute_specimen_balanced_f1(dummy_preds)

        # Bootstrap on dummy predictor vs random predictor
        rand_preds = singletons.copy()
        rand_preds["y_pred"] = self.rng.choice(self.classes, size=len(rand_preds))
        boot = self.paired_bootstrap(dummy_preds, rand_preds)

        return {
            "majority_class": majority_cls,
            "dummy_macro_f1": m_dummy["macro_f1"],
            "bootstrap_ci": [boot["ci_lower"], boot["ci_upper"]],
            "ci_width": round(boot["ci_upper"] - boot["ci_lower"], 5),
            "n_regions": len(singletons),
            "n_specimens": int(singletons["code"].nunique() if "code" in singletons.columns else 0),
        }

    # ──────────────────────────────────────────────────────────────────
    #  8. Full experiment evaluation & report generation (§10)
    # ──────────────────────────────────────────────────────────────────

    def evaluate_experiment(
        self,
        results_dir: Path | str | None = None,
        teachers_dir: Path | str | None = None,
        folds: list[str] | None = None,
        seeds: list[int] | None = None,
        arms: list[str] | None = None,
        samples_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """Run full evaluation over all arms, folds, and seeds."""
        res_dir = Path(results_dir or self.output_dir)
        teach_dir = Path(teachers_dir or "data/processed/v1/teachers")

        # Discover arms if not provided
        if arms is None:
            arms = [p.name for p in res_dir.iterdir() if p.is_dir() and p.name in ("static_k1", "unordered_k12", "angle_aware_k12")]
            if not arms:
                # Check if current directory itself contains fold_*
                if any(p.name.startswith("fold_") for p in res_dir.iterdir() if p.is_dir()):
                    arms = ["custom_arm"]
                    res_dir = res_dir.parent

        logger.info("Found arms to evaluate: %s", arms)

        # Discover folds
        if folds is None:
            discovered_folds = set()
            for arm in arms:
                arm_path = res_dir / arm
                if arm_path.is_dir():
                    for f in arm_path.iterdir():
                        if f.is_dir() and f.name.startswith("fold_"):
                            discovered_folds.add(f.name)
            folds = sorted(discovered_folds)

        logger.info("Found folds: %s", folds)

        # Discover seeds
        if seeds is None:
            discovered_seeds = set()
            for arm in arms:
                for f in folds:
                    f_path = res_dir / arm / f
                    if f_path.is_dir():
                        for s in f_path.iterdir():
                            if s.is_dir() and s.name.startswith("seed_"):
                                try:
                                    discovered_seeds.add(int(s.name.replace("seed_", "")))
                                except ValueError:
                                    pass
            seeds = sorted(discovered_seeds) if discovered_seeds else [0]

        logger.info("Found seeds: %s", seeds)

        # Load targets per fold — must exist for all evaluated folds
        fold_targets: dict[str, pd.DataFrame] = {}
        for f in folds:
            tp = teach_dir / f / "targets.parquet"
            if not tp.is_file():
                raise FileNotFoundError(f"Required targets file not found for fold '{f}': {tp}")
            t_df = pd.read_parquet(tp)
            if "fold" not in t_df.columns:
                t_df["fold"] = f
            fold_targets[f] = t_df

        # Cross-check against samples.parquet (CONTRACTS.md §4, Task C1)
        if samples_path is None:
            cand = teach_dir.parent / "samples.parquet"
            if cand.is_file():
                samples_path = cand
            else:
                proc = Path(self.cfg.get("v1", {}).get("paths", {}).get("processed", "data/processed/v1"))
                cand2 = proc / "samples.parquet"
                if cand2.is_file():
                    samples_path = cand2
                else:
                    samples_path = cand

        s_p = Path(samples_path)
        if not s_p.is_file():
            raise FileNotFoundError(f"Required samples.parquet not found: {s_p}")
        samples_df = pd.read_parquet(s_p)
        samples_rids = set(samples_df["region_id"])

        for f in folds:
            test_targets, _ = self._get_held_out_test_targets(
                fold_targets[f], f, teach_dir=teach_dir
            )
            missing_samples = set(test_targets["region_id"]) - samples_rids
            if missing_samples:
                raise ValueError(
                    f"Cross-check against samples.parquet failed for fold '{f}': "
                    f"{len(missing_samples)} required test regions missing from samples.parquet: "
                    f"{sorted(missing_samples)[:5]}"
                )

        # Load and pool predictions per arm and seed
        # arm -> seed -> DataFrame
        pooled_preds: dict[str, dict[int, pd.DataFrame]] = {arm: {} for arm in arms}
        primary_arms = {"angle_aware_k12", "static_k1"}
        for arm in arms:
            for s in seeds:
                dfs = []
                for f in folds:
                    preds_p = res_dir / arm / f / f"seed_{s}" / "predictions.parquet"
                    if not preds_p.is_file():
                        if arm in primary_arms:
                            raise FileNotFoundError(
                                f"Required predictions file missing for primary arm '{arm}', fold '{f}', seed {s}: {preds_p}"
                            )
                        logger.warning("Predictions file missing: %s", preds_p)
                        continue
                    p_df = pd.read_parquet(preds_p)
                    if "fold" not in p_df.columns:
                        p_df["fold"] = f

                    # Two-way test population check
                    test_targets, train_val_targets = self._get_held_out_test_targets(
                        fold_targets[f], f, teach_dir=teach_dir
                    )

                    pred_rids = set(p_df["region_id"])
                    test_rids = set(test_targets["region_id"])
                    train_val_rids = set(train_val_targets["region_id"])

                    # 1. Prediction must not contain train or val rows
                    tv_overlap = pred_rids & train_val_rids
                    if tv_overlap:
                        raise ValueError(
                            f"Two-way population check failed for arm '{arm}', fold '{f}', seed {s}: "
                            f"predictions contain {len(tv_overlap)} extra rows from train or val: {list(tv_overlap)[:5]}"
                        )

                    # 2. Prediction must not contain extra rows
                    extra_rids = pred_rids - test_rids
                    if extra_rids:
                        raise ValueError(
                            f"Two-way population check failed for arm '{arm}', fold '{f}', seed {s}: "
                            f"predictions contain {len(extra_rids)} extra rows not in held-out test targets: {list(extra_rids)[:5]}"
                        )

                    # 3. Every held-out test region in targets must have a prediction
                    missing_test_rids = test_rids - pred_rids
                    if missing_test_rids:
                        raise ValueError(
                            f"Two-way population check failed for arm '{arm}', fold '{f}', seed {s}: "
                            f"missing predictions for {len(missing_test_rids)} held-out test regions: {list(missing_test_rids)[:5]}"
                        )

                    # Strict join with targets (authoritative target truth & metadata match verification)
                    p_df = self.join_predictions_with_targets(p_df, test_targets, require_two_way=True)
                    dfs.append(p_df)
                if dfs:
                    pooled_preds[arm][s] = pd.concat(dfs, ignore_index=True)

        summary: dict[str, Any] = {
            "arms": arms,
            "folds": folds,
            "seeds": seeds,
            "point_estimates": {},
            "pairwise_comparisons": {},
            "partial_set_compatibility": {},
            "static_sensitivity": {},
            "metadata_split": {},
            "static_strata": {},
        }

        # 1. Point estimates per arm (averaged over seeds)
        for arm in arms:
            seed_metrics = []
            for s in seeds:
                if s in pooled_preds[arm]:
                    m = self.compute_specimen_balanced_f1(pooled_preds[arm][s])
                    seed_metrics.append(m)
            if seed_metrics:
                macro_f1s = [m["macro_f1"] for m in seed_metrics]
                accuracies = [m["accuracy"] for m in seed_metrics]
                per_class_avg = {}
                for c in self.classes:
                    per_class_avg[c] = {
                        "precision": round(float(np.mean([m["per_class"][c]["precision"] for m in seed_metrics])), 4),
                        "recall": round(float(np.mean([m["per_class"][c]["recall"] for m in seed_metrics])), 4),
                        "f1": round(float(np.mean([m["per_class"][c]["f1"] for m in seed_metrics])), 4),
                        "support_weighted": seed_metrics[0]["per_class"][c]["support_weighted"],
                        "support_unweighted": seed_metrics[0]["per_class"][c]["support_unweighted"],
                    }
                summary["point_estimates"][arm] = {
                    "macro_f1_mean": round(float(np.mean(macro_f1s)), 4),
                    "macro_f1_std": round(float(np.std(macro_f1s)), 4),
                    "accuracy_mean": round(float(np.mean(accuracies)), 4),
                    "n_regions": seed_metrics[0]["n_regions"],
                    "n_specimens": seed_metrics[0]["n_specimens"],
                    "per_class": per_class_avg,
                    "per_seed_macro_f1": [round(x, 4) for x in macro_f1s],
                }

        # 2. Pairwise bootstrap comparisons
        pairs = [
            ("angle_aware_k12", "static_k1", "primary_comparison"),
            ("unordered_k12", "static_k1", "unordered_vs_static"),
            ("angle_aware_k12", "unordered_k12", "angle_aware_vs_unordered"),
        ]
        for a, b, comp_name in pairs:
            if comp_name == "primary_comparison":
                if a not in arms:
                    raise ValueError(f"Primary comparison requires arm '{a}' in evaluated arms ({arms})")
                if b not in arms:
                    raise ValueError(f"Primary comparison requires arm '{b}' in evaluated arms ({arms})")
                dfs_a = [pooled_preds[a][s] for s in seeds if s in pooled_preds[a]]
                dfs_b = [pooled_preds[b][s] for s in seeds if s in pooled_preds[b]]
                if len(dfs_a) == 0:
                    raise ValueError(f"Arm '{a}' has 0 seed predictions for primary comparison")
                if len(dfs_b) == 0:
                    raise ValueError(f"Arm '{b}' has 0 seed predictions for primary comparison")
                if len(dfs_a) != len(seeds) or len(dfs_b) != len(seeds):
                    raise ValueError(
                        f"Missing seeds in primary comparison: {a} has {len(dfs_a)}/{len(seeds)} seeds, "
                        f"{b} has {len(dfs_b)}/{len(seeds)} seeds"
                    )
                res = self.paired_bootstrap(dfs_a, dfs_b)
                summary["pairwise_comparisons"][comp_name] = res
            else:
                if a in arms and b in arms:
                    dfs_a = [pooled_preds[a][s] for s in seeds if s in pooled_preds[a]]
                    dfs_b = [pooled_preds[b][s] for s in seeds if s in pooled_preds[b]]
                    if dfs_a and dfs_b:
                        if len(dfs_a) != len(dfs_b):
                            raise ValueError(
                                f"Seed count mismatch in comparison '{comp_name}': {a} has {len(dfs_a)}, {b} has {len(dfs_b)}"
                            )
                        res = self.paired_bootstrap(dfs_a, dfs_b)
                        summary["pairwise_comparisons"][comp_name] = res

        # 3. Partial set compatibility (averaged over all seeds)
        for arm in arms:
            seed_results = []
            for s in seeds:
                if s in pooled_preds[arm]:
                    seed_results.append(self.evaluate_partial_sets(pooled_preds[arm][s]))
            if seed_results:
                # Average numeric fields across seeds
                avg = {}
                for key in seed_results[0]:
                    vals = [r[key] for r in seed_results if isinstance(r.get(key), (int, float))]
                    avg[key] = float(np.mean(vals)) if vals else seed_results[0][key]
                summary["partial_set_compatibility"][arm] = avg

        # 4. Static angle sensitivity
        if "static_k1" in arms:
            summary["static_sensitivity"] = self.evaluate_static_sensitivity(
                res_dir / "static_k1", folds, seeds, fold_targets=fold_targets, teach_dir=teach_dir, samples_path=samples_path
            )

        # 5. Metadata split (averaged over all seeds)
        for arm in arms:
            seed_results = []
            for s in seeds:
                if s in pooled_preds[arm]:
                    seed_results.append(self.evaluate_metadata_split(pooled_preds[arm][s]))
            if seed_results:
                summary["metadata_split"][arm] = self._recursive_average(seed_results)

        # 6. Static strata evaluation (averaged over seeds, with paired bootstrap)
        summary["static_strata"] = self.evaluate_static_strata(pooled_preds)

        # Save summary JSON
        summary_path = res_dir / "evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Saved evaluation summary to %s", summary_path)

        # Generate and save Markdown report
        report_md = self.generate_markdown_report(summary)
        report_path = res_dir / "evaluation_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_md)
        logger.info("Saved evaluation markdown report to %s", report_path)

        return summary

    def generate_markdown_report(self, summary: dict[str, Any]) -> str:
        """Generate GitHub-flavored markdown report tables (§10)."""
        lines = [
            "# RotaPetro V1 Evaluation Report",
            "",
            "## 1. Primary Endpoint: Specimen-Balanced Macro-F1 on Singletons",
            "",
            "| Arm | Macro-F1 (mean ± std) | Accuracy | Regions | Specimens | Per-Seed F1 |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for arm, pe in summary.get("point_estimates", {}).items():
            f1_str = f"{pe['macro_f1_mean']:.4f} ± {pe['macro_f1_std']:.4f}"
            acc_str = f"{pe['accuracy_mean']:.4f}"
            seeds_str = ", ".join(f"{x:.4f}" for x in pe.get("per_seed_macro_f1", []))
            lines.append(f"| {arm} | {f1_str} | {acc_str} | {pe['n_regions']} | {pe['n_specimens']} | [{seeds_str}] |")

        lines.extend([
            "",
            "## 2. Paired Bootstrap Comparisons (5,000 replicates)",
            "",
            "| Comparison | Point Diff | 95% Bootstrap CI | p-value (approx) |",
            "|---|---:|---:|---:|",
        ])
        for name, comp in summary.get("pairwise_comparisons", {}).items():
            ci_str = f"[{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]"
            lines.append(f"| {name} | {comp['point_estimate_diff']:+.4f} | {ci_str} | {comp['p_value_approx']:.4f} |")

        lines.extend([
            "",
            "## 3. Partial-Set Compatibility & Probability Mass",
            "",
            "| Arm | Top-1 Compatibility | Probability Mass in Set | Regions | Specimens |",
            "|---|---:|---:|---:|---:|",
        ])
        for arm, psc in summary.get("partial_set_compatibility", {}).items():
            lines.append(f"| {arm} | {psc.get('top1_compatibility', 0):.4f} | {psc.get('probability_mass', 0):.4f} | {psc.get('n_regions', 0)} | {psc.get('n_specimens', 0)} |")

        if summary.get("static_sensitivity"):
            sens = summary["static_sensitivity"]
            lines.extend([
                "",
                "## 4. Static Arm Angle Sensitivity",
                "",
                f"- **Mean Macro-F1 across angles**: {sens.get('mean_macro_f1', 0):.4f} ± {sens.get('std_macro_f1', 0):.4f}",
                f"- **Range [min, max]**: [{sens.get('min_macro_f1', 0):.4f}, {sens.get('max_macro_f1', 0):.4f}]",
                f"- **Preregistered Primary Test Angle (Index 1 = 5°)**: {sens.get('primary_angle_index_1', 0):.4f}",
                "",
                "| Angle Index | Angle (deg) | Macro-F1 |",
                "|---:|---:|---:|",
            ])
            for idx_str, f1_val in sens.get("by_angle_index", {}).items():
                deg = int(idx_str) * 5.0
                lines.append(f"| {idx_str} | {deg:.1f}° | {f1_val:.4f} |")

        if summary.get("metadata_split"):
            lines.extend([
                "",
                "## 5. Performance Split by Metadata Dependency",
                "",
                "| Arm | Stratum | Macro-F1 | Regions | Specimens | n regions / n specimens |",
                "|---|---|---:|---:|---:|---:|",
            ])
            for arm, strata in summary["metadata_split"].items():
                if isinstance(strata, dict):
                    for stratum_name in ("metadata_dependent", "optics_only_reproducible"):
                        if stratum_name in strata and isinstance(strata[stratum_name], dict):
                            sm = strata[stratum_name]
                            avail, f1_str = self._is_stratum_available(sm)
                            n_reg = sm.get("n_regions", 0)
                            n_spec = sm.get("n_specimens", 0)
                            lines.append(f"| {arm} | {stratum_name} | {f1_str} | {n_reg} | {n_spec} | {n_reg} / {n_spec} |")

        if summary.get("static_strata"):
            lines.extend([
                "",
                "## 6. Static Strata Performance & Comparisons",
                "",
                "| Stratum | Arm | Macro-F1 | Regions | Specimens | n regions / n specimens | Paired Bootstrap Diff vs static_k1 (95% CI, p) |",
                "|---|---|---:|---:|---:|---:|---|",
            ])
            for stratum_name, s_data in summary["static_strata"].items():
                status = s_data.get("status", "available")
                n_reg = s_data.get("n_regions", 0)
                n_spec = s_data.get("n_specimens", 0)
                boot = s_data.get("paired_bootstrap")
                boot_str = "-"
                if boot and isinstance(boot, dict) and "point_estimate_diff" in boot:
                    boot_str = f"{boot['point_estimate_diff']:+.4f} [{boot['ci_lower']:.4f}, {boot['ci_upper']:.4f}] (p={boot['p_value_approx']:.4f})"

                if status == "unavailable":
                    reason = s_data.get("unavailable_reason", f"unavailable (n={n_reg})")
                    lines.append(f"| {stratum_name} | all | {reason} | {n_reg} | {n_spec} | {n_reg} / {n_spec} | - |")
                else:
                    for arm, m in s_data.get("arm_metrics", {}).items():
                        f1_val = m.get("macro_f1", 0.0)
                        b_str = boot_str if arm == "angle_aware_k12" else "-"
                        lines.append(f"| {stratum_name} | {arm} | {f1_val:.4f} | {n_reg} | {n_spec} | {n_reg} / {n_spec} | {b_str} |")

        lines.append("")
        return "\n".join(lines)
