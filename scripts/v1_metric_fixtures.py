#!/usr/bin/env python3
"""Simulation only — not evidence for the proposal.

V1 Metric Fixtures — correctness checks for the evaluation pipeline.

Script: scripts/v1_metric_fixtures.py

Tests the evaluation infrastructure with deterministic dummy predictors:
1. Majority predictor (always predicts the most common class)
2. Random predictor (uniform random)
3. Perfect predictor (y_pred == y_true)
4. Identical paired predictions (zero-width bootstrap CI)
5. Shuffled-row predictor

Also runs conditional detectability scenarios per plan section 5.9.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config
from src.rotapetro.evaluate import Evaluator

logger = logging.getLogger(__name__)


def _make_synthetic_data(
    classes: list[str] | None = None,
    n_specimens: int = 30,
    n_regions_per_specimen: int = 10,
    n_classes: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic prediction data for fixture tests."""
    rng = np.random.RandomState(seed)
    if classes is None:
        classes = [f"class_{i}" for i in range(n_classes)]
    else:
        n_classes = len(classes)
    rows = []
    for s in range(n_specimens):
        specimen = f"spec_{s:03d}"
        fold = f"fold_{s % 5}"
        # Assign specimen to a dominant class with some noise
        dominant_idx = s % n_classes
        for r in range(n_regions_per_specimen):
            # 80% dominant class, 20% random
            if rng.rand() < 0.8:
                c_idx = dominant_idx
            else:
                c_idx = rng.randint(0, n_classes)
            y_true = classes[c_idx]
            rows.append({
                "specimen": specimen,
                "region_id": f"{specimen}_r{r:03d}",
                "fold": fold,
                "state": "singleton",
                "allowed": [y_true],
                "y_true": y_true,
            })
    return pd.DataFrame(rows)


def test_majority_predictor(evaluator: Evaluator, data: pd.DataFrame) -> dict:
    """Majority predictor: always predicts the most common class."""
    majority_class = data["y_true"].value_counts().idxmax()
    data_copy = data.copy()
    data_copy["y_pred"] = majority_class
    metrics = evaluator.compute_specimen_balanced_f1(data_copy)
    logger.info("Majority predictor: macro_f1=%.4f, accuracy=%.4f",
                metrics["macro_f1"], metrics["accuracy"])
    return {"fixture": "majority", **metrics}


def test_random_predictor(evaluator: Evaluator, data: pd.DataFrame,
                          seed: int = 123) -> dict:
    """Random predictor: uniform random class."""
    rng = np.random.RandomState(seed)
    data_copy = data.copy()
    data_copy["y_pred"] = rng.choice(evaluator.classes, size=len(data))
    metrics = evaluator.compute_specimen_balanced_f1(data_copy)
    logger.info("Random predictor: macro_f1=%.4f, accuracy=%.4f",
                metrics["macro_f1"], metrics["accuracy"])
    return {"fixture": "random", **metrics}


def test_perfect_predictor(evaluator: Evaluator, data: pd.DataFrame) -> dict:
    """Perfect predictor: y_pred == y_true."""
    data_copy = data.copy()
    data_copy["y_pred"] = data_copy["y_true"]
    metrics = evaluator.compute_specimen_balanced_f1(data_copy)
    assert abs(metrics["macro_f1"] - 1.0) < 1e-6, f"Perfect predictor F1 != 1.0: {metrics['macro_f1']}"
    assert abs(metrics["accuracy"] - 1.0) < 1e-6, f"Perfect predictor acc != 1.0: {metrics['accuracy']}"
    logger.info("Perfect predictor: macro_f1=%.4f ✓", metrics["macro_f1"])
    return {"fixture": "perfect", **metrics}


def test_identical_bootstrap(evaluator: Evaluator, data: pd.DataFrame) -> dict:
    """Identical paired predictions → zero-width CI."""
    data_copy = data.copy()
    data_copy["y_pred"] = data_copy["y_true"]
    result = evaluator.paired_bootstrap(data_copy, data_copy)
    diff = result.get("mean_diff", result.get("point_estimate_diff", 0.0))
    assert abs(diff) < 1e-10, \
        f"Identical bootstrap diff != 0: {diff}"
    assert abs(result["ci_lower"]) < 1e-10, \
        f"Identical bootstrap CI lower != 0: {result['ci_lower']}"
    assert abs(result["ci_upper"]) < 1e-10, \
        f"Identical bootstrap CI upper != 0: {result['ci_upper']}"
    logger.info("Identical bootstrap: mean_diff=%.6f, CI=[%.6f, %.6f] ✓",
                diff, result["ci_lower"], result["ci_upper"])
    return {"fixture": "identical_bootstrap", **result}


def test_shuffled_predictor(evaluator: Evaluator, data: pd.DataFrame,
                            seed: int = 456) -> dict:
    """Shuffled-row predictor: labels randomly permuted within each specimen."""
    rng = np.random.RandomState(seed)
    data_copy = data.copy()
    shuffled = []
    for specimen, group in data_copy.groupby("specimen"):
        perm = rng.permutation(len(group))
        shuffled.extend(group["y_true"].values[perm].tolist())
    data_copy["y_pred"] = shuffled
    metrics = evaluator.compute_specimen_balanced_f1(data_copy)
    logger.info("Shuffled predictor: macro_f1=%.4f, accuracy=%.4f",
                metrics["macro_f1"], metrics["accuracy"])
    return {"fixture": "shuffled", **metrics}


def _simulate_model_predictions(
    data: pd.DataFrame,
    n_flip: int,
    clustering: str,
    class_list: list[str],
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Build model predictions with specified clustering procedure.

    Simulation only — not evidence for the proposal.
    For concentrated: shuffle specimens with scenario RNG, add specimens until
    containing >= n_flip regions, flip exactly n_flip regions among them.
    Flip always picks class != y_true.
    """
    df = data.copy()
    df["y_pred"] = df["y_true"].copy()
    if n_flip <= 0:
        return df

    if clustering == "concentrated":
        # Shuffle specimens with scenario RNG
        unique_specimens = list(df["specimen"].unique())
        shuffled_specimens = rng.permutation(unique_specimens)
        candidate_indices: list[int] = []
        # Add specimens until containing >= n_flip regions
        for spec in shuffled_specimens:
            spec_idx = df[df["specimen"] == spec].index.tolist()
            candidate_indices.extend(spec_idx)
            if len(candidate_indices) >= n_flip:
                break
        # Flip exactly n_flip regions among them
        flip_idx = rng.choice(candidate_indices, size=n_flip, replace=False)
    else:
        # Uniform: select n_flip regions across the whole dataset
        flip_idx = rng.choice(df.index.values, size=n_flip, replace=False)

    # Flip always picks class != y_true
    for idx in flip_idx:
        y_true = df.loc[idx, "y_true"]
        alt_classes = [c for c in class_list if c != y_true]
        df.loc[idx, "y_pred"] = rng.choice(alt_classes)

    return df


def run_detectability_scenarios(
    evaluator: Evaluator,
    data: pd.DataFrame,
    classes: list[str] | None = None,
    n_classes: int = 5,
    seed: int = 789,
) -> list[dict]:
    """Conditional detectability: simulate paired models with known gains.

    Simulation only — not evidence for the proposal.
    """
    results = []
    class_list = classes or evaluator.classes

    for gain_pct in [0, 2, 5, 10, 15]:
        for clustering in ["uniform", "concentrated"]:
            scenario_seed = seed + gain_pct * 10 + (0 if clustering == "uniform" else 1)
            rng_a = np.random.RandomState(scenario_seed)
            rng_b = np.random.RandomState(scenario_seed + 1000)

            base_acc = 0.55
            n_flip_a = int(len(data) * (1 - base_acc))
            improved_acc = base_acc + gain_pct / 100.0
            n_flip_b = int(len(data) * (1 - improved_acc))

            # Build model A and B with the same clustering procedure
            data_a = _simulate_model_predictions(data, n_flip_a, clustering, class_list, rng_a)
            data_b = _simulate_model_predictions(data, n_flip_b, clustering, class_list, rng_b)

            errors_a = int((data_a["y_pred"] != data_a["y_true"]).sum())
            errors_b = int((data_b["y_pred"] != data_b["y_true"]).sum())

            bootstrap = evaluator.paired_bootstrap(data_b, data_a)
            diff = bootstrap.get("mean_diff", bootstrap.get("point_estimate_diff", 0.0))

            results.append({
                "gain_pct": gain_pct,
                "clustering": clustering,
                "mean_diff": diff,
                "ci_lower": bootstrap["ci_lower"],
                "ci_upper": bootstrap["ci_upper"],
                "excludes_zero": bootstrap["ci_lower"] > 0,
                "p_value": bootstrap["p_value_approx"],
                "errors_a": errors_a,
                "errors_b": errors_b,
            })

            logger.info(
                "Gain=%d%% clustering=%s: errors_a=%d errors_b=%d diff=%.4f CI=[%.4f, %.4f] "
                "excludes_zero=%s",
                gain_pct, clustering,
                errors_a, errors_b,
                diff,
                bootstrap["ci_lower"],
                bootstrap["ci_upper"],
                bootstrap["ci_lower"] > 0,
            )

    return results


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(str(_PROJECT_ROOT / "configs" / "default.yaml"))
    v1_cfg = cfg.get("v1", {})
    primary_classes = v1_cfg.get("taxonomy", {}).get("primary_classes", [])
    n_bootstrap = v1_cfg.get("training", {}).get("n_bootstrap", 5000)

    out_dir = _PROJECT_ROOT / v1_cfg.get("paths", {}).get(
        "processed", "data/processed/v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(
        classes=primary_classes,
        output_dir=out_dir / "metric_fixtures",
        n_bootstrap=n_bootstrap,
    )

    data = _make_synthetic_data(
        classes=primary_classes,
        n_specimens=30,
        n_regions_per_specimen=10,
    )

    logger.info("=" * 60)
    logger.info("METRIC FIXTURE TESTS")
    logger.info("=" * 60)

    fixture_results = []
    fixture_results.append(test_majority_predictor(evaluator, data))
    fixture_results.append(test_random_predictor(evaluator, data))
    fixture_results.append(test_perfect_predictor(evaluator, data))
    fixture_results.append(test_identical_bootstrap(evaluator, data))
    fixture_results.append(test_shuffled_predictor(evaluator, data))

    logger.info("=" * 60)
    logger.info("CONDITIONAL DETECTABILITY SCENARIOS")
    logger.info("=" * 60)

    detect_results = run_detectability_scenarios(
        evaluator, data, classes=primary_classes)

    # Save results
    fixtures_path = out_dir / "metric_fixtures" / "fixture_results.json"
    fixtures_path.parent.mkdir(parents=True, exist_ok=True)
    with open(fixtures_path, "w", encoding="utf-8") as f:
        json.dump({
            "fixtures": fixture_results,
            "detectability": detect_results,
        }, f, indent=2, default=str)
    logger.info("Saved fixture results to %s", fixtures_path)

    logger.info("All fixture tests PASSED ✓")


if __name__ == "__main__":
    main()
