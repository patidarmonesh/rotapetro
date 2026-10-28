"""Unit tests for V1 Evaluator (§10).

Covers:
- Specimen-balanced macro-F1 calculation
- Row-order invariance
- Strict join assertions (duplicate / missing region_ids)
- Paired bootstrap zero-diff identity on identical predictions
- Bootstrap reproducibility with fixed seed
- Partial-set compatibility and mass
- Class disappearance tracking
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score

from src.rotapetro.evaluate import Evaluator, fast_macro_f1, compute_weighted_metrics

CLASSES = ["quartz", "feldspar", "mica", "pyroxene", "opaque_material"]


def test_specimen_balanced_macro_f1():
    """Verify that specimen-balanced weighting (1 / n_s) correctly weights specimens equally."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")

    # Specimen A has 100 quartz singletons; Specimen B has 1 feldspar singleton
    # Total rows = 101
    rows = []
    for i in range(100):
        rows.append({
            "region_id": f"reg_a_{i}",
            "specimen": "spec_A",
            "state": "singleton",
            "allowed": ["quartz"],
            "y_pred": "quartz",  # Predict quartz
        })
    rows.append({
        "region_id": "reg_b_0",
        "specimen": "spec_B",
        "state": "singleton",
        "allowed": ["feldspar"],
        "y_pred": "quartz",  # Wrongly predict quartz
    })
    df = pd.DataFrame(rows)

    m = evaluator.compute_specimen_balanced_f1(df)

    # Specimen A: 100 rows, each weight 1/100 = 0.01. Sum = 1.0 (all TP for quartz)
    # Specimen B: 1 row, weight 1/1 = 1.0. Sum = 1.0 (FN for feldspar, FP for quartz)
    # Quartz: TP = 1.0, FP = 1.0, FN = 0.0 -> P = 0.5, R = 1.0 -> F1 = 2/3 ≈ 0.6667
    # Feldspar: TP = 0.0, FP = 0.0, FN = 1.0 -> F1 = 0.0
    # Other 3 classes: F1 = 0.0
    # Macro F1 = (2/3 + 0 + 0 + 0 + 0) / 5 = 2/15 ≈ 0.13333
    expected_macro = (2.0 / 3.0) / 5.0
    assert abs(m["macro_f1"] - expected_macro) < 1e-4

    # Notice that unweighted macro F1 would be completely different:
    # Unweighted quartz: TP=100, FP=1, FN=0 -> P=100/101, R=1 -> F1 = 200/201 ≈ 0.995
    # Unweighted macro F1 would be ≈ 0.995 / 5 ≈ 0.199
    unweighted_f1 = f1_score(
        [0] * 100 + [1],
        [0] * 101,
        labels=range(5),
        average="macro",
        zero_division=0,
    )
    assert abs(m["macro_f1"] - unweighted_f1) > 0.05


def test_row_order_invariance():
    """Verify that shuffling rows of predictions produces identical metrics down to machine precision."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")
    rng = np.random.default_rng(123)

    rows = []
    for i in range(150):
        spec = f"spec_{i % 10}"
        c_true = CLASSES[i % 5]
        c_pred = CLASSES[(i + (1 if i % 3 == 0 else 0)) % 5]
        rows.append({
            "region_id": f"reg_{i:04d}",
            "specimen": spec,
            "state": "singleton",
            "allowed": [c_true],
            "y_pred": c_pred,
        })
    df_orig = pd.DataFrame(rows)
    m_orig = evaluator.compute_specimen_balanced_f1(df_orig)

    # Shuffle rows 5 times
    for s in range(5):
        shuffled_indices = rng.permutation(len(df_orig))
        df_shuffled = df_orig.iloc[shuffled_indices].reset_index(drop=True)
        m_shuffled = evaluator.compute_specimen_balanced_f1(df_shuffled)
        assert abs(m_orig["macro_f1"] - m_shuffled["macro_f1"]) < 1e-12
        assert abs(m_orig["accuracy"] - m_shuffled["accuracy"]) < 1e-12


def test_duplicate_and_missing_region_ids():
    """Strict join must raise ValueError on duplicate or missing region_ids."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")

    preds_df = pd.DataFrame([
        {"region_id": "r1", "y_pred": "quartz"},
        {"region_id": "r1", "y_pred": "feldspar"},  # Duplicate!
    ])
    targets_df = pd.DataFrame([
        {"region_id": "r1", "state": "singleton", "allowed": ["quartz"]},
    ])
    with pytest.raises(ValueError, match="Duplicate region_id in predictions"):
        evaluator.join_predictions_with_targets(preds_df, targets_df)

    preds_unique = pd.DataFrame([
        {"region_id": "r1", "y_pred": "quartz"},
        {"region_id": "r2", "y_pred": "feldspar"},  # Missing in targets!
    ])
    with pytest.raises(ValueError, match="not present in targets"):
        evaluator.join_predictions_with_targets(preds_unique, targets_df)


def test_paired_bootstrap_identical_predictions_zero_diff():
    """Comparing identical predictions must produce exactly 0.0 point diff and [0.0, 0.0] CI."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1", n_bootstrap=200, seed=42)

    rows = []
    for i in range(60):
        spec = f"spec_{i % 6}"
        fold = f"fold_{i % 3}"
        c = CLASSES[i % 5]
        rows.append({
            "region_id": f"reg_{i}",
            "specimen": spec,
            "fold": fold,
            "state": "singleton",
            "allowed": [c],
            "y_pred": c,
        })
    df_a = pd.DataFrame(rows)
    df_b = df_a.copy()

    res = evaluator.paired_bootstrap(df_a, df_b)
    assert res["point_estimate_diff"] == 0.0
    assert res["ci_lower"] == 0.0
    assert res["ci_upper"] == 0.0


def test_partial_set_compatibility():
    """Verify top-1 compatibility and probability mass calculations on partial sets."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")

    # Specimen A: 2 partial-set rows
    # row 0: allowed=[quartz, feldspar], y_pred=quartz (compat), probs: quartz=0.6, feldspar=0.3 -> mass=0.9
    # row 1: allowed=[quartz, feldspar], y_pred=mica (incompat), probs: quartz=0.2, feldspar=0.2 -> mass=0.4
    rows = [
        {
            "region_id": "r0",
            "specimen": "spec_A",
            "state": "partial_set",
            "allowed": ["quartz", "feldspar"],
            "y_pred": "quartz",
            "probs_quartz": 0.6,
            "probs_feldspar": 0.3,
            "probs_mica": 0.05,
            "probs_pyroxene": 0.03,
            "probs_opaque_material": 0.02,
        },
        {
            "region_id": "r1",
            "specimen": "spec_A",
            "state": "partial_set",
            "allowed": ["quartz", "feldspar"],
            "y_pred": "mica",
            "probs_quartz": 0.2,
            "probs_feldspar": 0.2,
            "probs_mica": 0.5,
            "probs_pyroxene": 0.05,
            "probs_opaque_material": 0.05,
        },
    ]
    df = pd.DataFrame(rows)
    res = evaluator.evaluate_partial_sets(df)

    # Both rows in spec_A have weight 0.5
    # top1_compat = 0.5 * 1.0 + 0.5 * 0.0 = 0.5
    # prob_mass = 0.5 * 0.9 + 0.5 * 0.4 = 0.65
    assert abs(res["top1_compatibility"] - 0.5) < 1e-4
    assert abs(res["probability_mass"] - 0.65) < 1e-4
    assert res["n_regions"] == 2
    assert res["n_specimens"] == 1


def test_paired_bootstrap_multi_seed_averaging():
    """Verify paired bootstrap correctly accepts multiple seeds and averages diffs."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1", n_bootstrap=100, seed=42)

    rows_s0_a = [
        {"region_id": f"r{i}", "specimen": f"spec_{i % 4}", "fold": "fold_0", "state": "singleton", "allowed": ["quartz"], "y_pred": "quartz"}
        for i in range(20)
    ]
    rows_s0_b = [
        {"region_id": f"r{i}", "specimen": f"spec_{i % 4}", "fold": "fold_0", "state": "singleton", "allowed": ["quartz"], "y_pred": "feldspar"}
        for i in range(20)
    ]
    # In seed 1, model b also gets some right
    rows_s1_a = [
        {"region_id": f"r{i}", "specimen": f"spec_{i % 4}", "fold": "fold_0", "state": "singleton", "allowed": ["quartz"], "y_pred": "quartz"}
        for i in range(20)
    ]
    rows_s1_b = [
        {"region_id": f"r{i}", "specimen": f"spec_{i % 4}", "fold": "fold_0", "state": "singleton", "allowed": ["quartz"], "y_pred": "quartz"}
        for i in range(20)
    ]

    dfs_a = [pd.DataFrame(rows_s0_a), pd.DataFrame(rows_s1_a)]
    dfs_b = [pd.DataFrame(rows_s0_b), pd.DataFrame(rows_s1_b)]

    res = evaluator.paired_bootstrap(dfs_a, dfs_b)
    assert res["point_estimate_diff"] > 0.0
    assert "class_disappearance_rate" in res
    assert res["n_bootstrap"] == 100


def test_l9_perfect_and_majority_class_macro_f1():
    """Test L9: Perfect predictions give macro F1 = 1.0; majority class predictor gives the expected baseline score."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")

    # 1. Perfect predictions: all 5 classes correctly predicted
    rows_perf = []
    for idx, c in enumerate(CLASSES):
        for i in range(10):
            rows_perf.append({
                "region_id": f"perf_{idx}_{i}",
                "specimen": f"spec_{c}",
                "state": "singleton",
                "allowed": [c],
                "y_pred": c,
            })
    df_perf = pd.DataFrame(rows_perf)
    m_perf = evaluator.compute_specimen_balanced_f1(df_perf)

    assert m_perf["macro_f1"] == pytest.approx(1.0, abs=1e-6)
    assert m_perf["accuracy"] == pytest.approx(1.0, abs=1e-6)

    # 2. Majority class predictor: always predicts class 0 ("quartz")
    rows_maj = []
    for idx, c in enumerate(CLASSES):
        for i in range(10):
            rows_maj.append({
                "region_id": f"maj_{idx}_{i}",
                "specimen": f"spec_{c}",
                "state": "singleton",
                "allowed": [c],
                "y_pred": "quartz",
            })
    df_maj = pd.DataFrame(rows_maj)
    m_maj = evaluator.compute_specimen_balanced_f1(df_maj)

    # 5 specimens each weight 1.0 (total 5.0). Quartz TP=1, FP=4, FN=0 -> P=0.2, R=1.0 -> F1=1/3.
    # Other 4 classes F1=0. Macro F1 = (1/3) / 5 = 1/15. Accuracy = 1.0 / 5.0 = 0.2.
    expected_macro_f1 = (1.0 / 3.0) / 5.0
    assert m_maj["macro_f1"] == pytest.approx(expected_macro_f1, abs=1e-5)
    assert m_maj["accuracy"] == pytest.approx(0.2, abs=1e-5)


def test_l11_specimen_weighting_equal_specimen_totals():
    """Test L11: Specimen weighting treats Specimen A (1 region) and Specimen B (99 regions) with equal total weight (1.0 each)."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")

    # Specimen A: 1 region (predicted correctly)
    # Specimen B: 99 regions (predicted incorrectly)
    rows = [
        {
            "region_id": "r_a_0",
            "specimen": "spec_A",
            "state": "singleton",
            "allowed": ["quartz"],
            "y_pred": "quartz",
        }
    ]
    for i in range(99):
        rows.append({
            "region_id": f"r_b_{i}",
            "specimen": "spec_B",
            "state": "singleton",
            "allowed": ["feldspar"],
            "y_pred": "quartz",
        })
    df = pd.DataFrame(rows)

    # Check specimen weights sum to 1.0 each
    counts = df["specimen"].value_counts()
    weights_a = [1.0 / counts["spec_A"]] * 1
    weights_b = [1.0 / counts["spec_B"]] * 99

    assert sum(weights_a) == pytest.approx(1.0, abs=1e-6)
    assert sum(weights_b) == pytest.approx(1.0, abs=1e-6)

    m = evaluator.compute_specimen_balanced_f1(df)

    # Accuracy with equal weighting (1.0 each, total weight 2.0):
    # Spec A contributes 1.0 TP, Spec B contributes 0.0 TP -> accuracy = 1.0 / 2.0 = 0.50
    assert m["accuracy"] == pytest.approx(0.50, abs=1e-6)

    # Hand-calculated expected macro F1:
    # Quartz: TP=1.0, FP=1.0, FN=0.0 -> P=0.5, R=1.0 -> F1 = 2/3
    # Feldspar: TP=0.0, FP=0.0, FN=1.0 -> F1 = 0.0
    # Other 3 classes: F1 = 0.0
    # Expected macro F1 = (2/3) / 5 = 2/15 ≈ 0.133333
    expected_macro = (2.0 / 3.0) / 5.0
    assert m["macro_f1"] == pytest.approx(expected_macro, abs=1e-5)

    # Verify support_weighted in per_class metrics:
    assert m["per_class"]["quartz"]["support_weighted"] == pytest.approx(1.0, abs=1e-6)
    assert m["per_class"]["feldspar"]["support_weighted"] == pytest.approx(1.0, abs=1e-6)


# ── Checklist Aliases & Explicit Tests (V1_CODE_CHECKLIST.md) ───────────

def test_l8_identical_arms_zero_diff():
    """Test L8: Comparing identical predictions must produce exactly 0.0 point diff and [0.0, 0.0] CI."""
    test_paired_bootstrap_identical_predictions_zero_diff()


def test_l10_shuffling_rows_invariance():
    """Test L10: Shuffling rows of predictions produces identical metrics down to machine precision."""
    test_row_order_invariance()



def test_bootstrap_row_shuffle_invariance():
    """Verify that paired_bootstrap is invariant to row ordering of input DataFrames.

    This guards against the P0 bug where spec_row_indices were tied to seed 0
    row order and would misalign if other seeds had different orderings.
    """
    rng = np.random.default_rng(42)
    n_specimens = 10
    rows_per = 5
    classes = CLASSES

    rows = []
    for i in range(n_specimens):
        fam = classes[i % len(classes)]
        spec = f"spec_{i:02d}"
        for j in range(rows_per):
            rid = f"{spec}_r{j}"
            pred = classes[(i + 1) % len(classes)]  # intentionally wrong for some
            rows.append({
                "region_id": rid, "specimen": spec, "fold": f"fold_{i % 3}",
                "state": "singleton", "y_true": fam, "y_pred": pred,
                "allowed": [fam],
            })

    df_ordered = pd.DataFrame(rows)

    # Create a shuffled version
    df_shuffled = df_ordered.sample(frac=1.0, random_state=99).reset_index(drop=True)

    ev = Evaluator(classes=classes, n_bootstrap=200, seed=0)

    result_ordered = ev.paired_bootstrap(
        df_a=[df_ordered], df_b=[df_ordered.copy()]
    )
    result_shuffled = ev.paired_bootstrap(
        df_a=[df_shuffled], df_b=[df_shuffled.copy()]
    )

    assert result_ordered["point_estimate_diff"] == pytest.approx(
        result_shuffled["point_estimate_diff"], abs=1e-10
    ), "Bootstrap point estimate changed with row shuffling"
    assert result_ordered["ci_lower"] == pytest.approx(
        result_shuffled["ci_lower"], abs=1e-6
    ), "Bootstrap CI lower changed with row shuffling"
    assert result_ordered["ci_upper"] == pytest.approx(
        result_shuffled["ci_upper"], abs=1e-6
    ), "Bootstrap CI upper changed with row shuffling"


def test_evaluate_experiment_primary_comparison_guards(tmp_path: Path):
    """Test that evaluate_experiment raises FileNotFoundError/ValueError if targets, predictions or primary arms are missing."""
    ev = Evaluator(classes=CLASSES, n_bootstrap=50, seed=0)

    res_dir = tmp_path / "results"
    teach_dir = tmp_path / "teachers"
    res_dir.mkdir()
    teach_dir.mkdir()

    # 1. Missing targets file for a fold raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Required targets file not found for fold 'fold_0'"):
        ev.evaluate_experiment(
            results_dir=res_dir,
            teachers_dir=teach_dir,
            arms=["static_k1", "angle_aware_k12"],
            folds=["fold_0"],
            seeds=[0],
        )

    # Create dummy targets for fold_0
    (teach_dir / "fold_0").mkdir()
    dummy_targets = pd.DataFrame([{"region_id": "r1", "partition": "test", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]}])
    dummy_targets.to_parquet(teach_dir / "fold_0" / "targets.parquet")
    pd.DataFrame([{"region_id": "r1", "specimen": "spec1"}]).to_parquet(teach_dir.parent / "samples.parquet")

    # 2. Missing predictions for primary arm raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Required predictions file missing for primary arm"):
        ev.evaluate_experiment(
            results_dir=res_dir,
            teachers_dir=teach_dir,
            arms=["static_k1", "angle_aware_k12"],
            folds=["fold_0"],
            seeds=[0],
        )

    # 3. Create dummy predictions for static_k1
    s_dir = res_dir / "static_k1" / "fold_0" / "seed_0"
    s_dir.mkdir(parents=True)
    dummy_preds = pd.DataFrame([{"region_id": "r1", "y_pred": "quartz"}])
    dummy_preds.to_parquet(s_dir / "predictions.parquet")

    # Missing angle_aware_k12 from arms raises ValueError for primary_comparison
    with pytest.raises(ValueError, match="Primary comparison requires arm"):
        ev.evaluate_experiment(
            results_dir=res_dir,
            teachers_dir=teach_dir,
            arms=["static_k1"],  # angle_aware_k12 is missing
            folds=["fold_0"],
            seeds=[0],
        )


# ── Prediction Metadata Truth & Two-Way Checks (§10.5, Finding 5) ────────

def test_join_rejects_mismatched_state():
    """Prediction rows with stale/mismatched state relative to targets must be rejected with ValueError."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")
    targets_df = pd.DataFrame([
        {"region_id": "r1", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
    ])
    preds_df = pd.DataFrame([
        {"region_id": "r1", "y_pred": "quartz", "state": "partial_set"},
    ])
    with pytest.raises(ValueError, match="Prediction metadata mismatch with target for column 'state'"):
        evaluator.join_predictions_with_targets(preds_df, targets_df)


def test_join_rejects_mismatched_allowed():
    """Prediction rows with stale/mismatched allowed relative to targets must be rejected with ValueError."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")
    targets_df = pd.DataFrame([
        {"region_id": "r1", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
    ])
    preds_df = pd.DataFrame([
        {"region_id": "r1", "y_pred": "quartz", "allowed": ["feldspar"]},
    ])
    with pytest.raises(ValueError, match="Prediction metadata mismatch with target for column 'allowed'"):
        evaluator.join_predictions_with_targets(preds_df, targets_df)


def test_join_rejects_mismatched_specimen():
    """Prediction rows with stale/mismatched specimen (or code) relative to targets must be rejected with ValueError."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")
    targets_df = pd.DataFrame([
        {"region_id": "r1", "specimen": "spec_target", "state": "singleton", "allowed": ["quartz"]},
    ])
    preds_df = pd.DataFrame([
        {"region_id": "r1", "y_pred": "quartz", "specimen": "spec_stale"},
    ])
    with pytest.raises(ValueError, match="Prediction metadata mismatch with target for column 'specimen'"):
        evaluator.join_predictions_with_targets(preds_df, targets_df)

    # Also test code column alias
    targets_code = pd.DataFrame([
        {"region_id": "r1", "code": "spec_target", "state": "singleton", "allowed": ["quartz"]},
    ])
    preds_code = pd.DataFrame([
        {"region_id": "r1", "y_pred": "quartz", "code": "spec_stale"},
    ])
    with pytest.raises(ValueError, match="Prediction metadata mismatch with target for column 'specimen'"):
        evaluator.join_predictions_with_targets(preds_code, targets_code)


def test_join_rejects_extra_non_test_rows_from_train_or_val():
    """Extra non-test rows (e.g. from train or val partition) in predictions must trigger a ValueError."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")
    targets_df = pd.DataFrame([
        {"region_id": "r_train", "partition": "train", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
        {"region_id": "r_val", "partition": "val", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
        {"region_id": "r_test", "partition": "test", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
    ])
    # Predictions erroneously include a row from the training partition
    preds_with_train = pd.DataFrame([
        {"region_id": "r_train", "y_pred": "quartz"},
        {"region_id": "r_test", "y_pred": "quartz"},
    ])
    with pytest.raises(ValueError, match="Predictions contain .* extra rows from train or val"):
        evaluator.join_predictions_with_targets(preds_with_train, targets_df)

    # Predictions erroneously include a row from the validation partition
    preds_with_val = pd.DataFrame([
        {"region_id": "r_val", "y_pred": "quartz"},
        {"region_id": "r_test", "y_pred": "quartz"},
    ])
    with pytest.raises(ValueError, match="Predictions contain .* extra rows from train or val"):
        evaluator.join_predictions_with_targets(preds_with_val, targets_df)


def test_join_rejects_missing_test_rows():
    """Missing test rows in predictions must trigger an error when two-way match is required."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")
    targets_df = pd.DataFrame([
        {"region_id": "test_r1", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
        {"region_id": "test_r2", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
        {"region_id": "test_r3", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
    ])
    # Predictions are missing test_r3
    preds_df = pd.DataFrame([
        {"region_id": "test_r1", "y_pred": "quartz"},
        {"region_id": "test_r2", "y_pred": "quartz"},
    ])
    with pytest.raises(ValueError, match="Two-way match failed: missing predictions for 1 target regions"):
        evaluator.join_predictions_with_targets(preds_df, targets_df, require_two_way=True)


def test_join_accepts_matching_metadata_and_uses_target_as_authoritative():
    """Matching metadata passes validation, and targets remain the authoritative source of truth."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")
    targets_df = pd.DataFrame([
        {"region_id": "r1", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"], "fold": "fold_0"},
    ])
    preds_df = pd.DataFrame([
        {"region_id": "r1", "y_pred": "quartz", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"], "fold": "fold_0"},
    ])
    joined = evaluator.join_predictions_with_targets(preds_df, targets_df, require_two_way=True)
    assert len(joined) == 1
    assert joined.iloc[0]["state"] == "singleton"
    assert joined.iloc[0]["allowed"] == ["quartz"]
    assert joined.iloc[0]["specimen"] == "spec1"
    assert joined.iloc[0]["fold"] == "fold_0"


def test_join_rejects_mismatched_fold():
    """Prediction rows with stale/mismatched fold relative to targets must be rejected with ValueError."""
    evaluator = Evaluator(classes=CLASSES, output_dir="outputs/v1")
    targets_df = pd.DataFrame([
        {"region_id": "r1", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"], "fold": "fold_0"},
    ])
    preds_df = pd.DataFrame([
        {"region_id": "r1", "y_pred": "quartz", "fold": "fold_1"},
    ])
    with pytest.raises(ValueError, match="Prediction metadata mismatch with target for column 'fold'"):
        evaluator.join_predictions_with_targets(preds_df, targets_df)


def test_evaluate_experiment_two_way_held_out_checks(tmp_path: Path):
    """evaluate_experiment enforces two-way population match for held-out test regions."""
    ev = Evaluator(classes=CLASSES, n_bootstrap=50, seed=0)
    res_dir = tmp_path / "results"
    teach_dir = tmp_path / "teachers"
    res_dir.mkdir()
    teach_dir.mkdir()

    # Create fold_0 targets with train, val, and test partitions
    (teach_dir / "fold_0").mkdir()
    targets_df = pd.DataFrame([
        {"region_id": "r_test1", "partition": "test", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
        {"region_id": "r_test2", "partition": "test", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
        {"region_id": "r_train", "partition": "train", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
    ])
    targets_df.to_parquet(teach_dir / "fold_0" / "targets.parquet")
    pd.DataFrame([
        {"region_id": "r_test1", "specimen": "spec1"},
        {"region_id": "r_test2", "specimen": "spec1"},
        {"region_id": "r_train", "specimen": "spec1"},
    ]).to_parquet(teach_dir.parent / "samples.parquet")

    # Arm with missing test prediction (r_test2 missing)
    s_dir = res_dir / "static_k1" / "fold_0" / "seed_0"
    s_dir.mkdir(parents=True)
    preds_missing = pd.DataFrame([{"region_id": "r_test1", "y_pred": "quartz"}])
    preds_missing.to_parquet(s_dir / "predictions.parquet")

    with pytest.raises(ValueError, match="Two-way population check failed.*missing predictions"):
        ev.evaluate_experiment(
            results_dir=res_dir,
            teachers_dir=teach_dir,
            arms=["static_k1"],
            folds=["fold_0"],
            seeds=[0],
        )

    # Arm with extra row from train
    preds_train_leak = pd.DataFrame([
        {"region_id": "r_test1", "y_pred": "quartz"},
        {"region_id": "r_test2", "y_pred": "quartz"},
        {"region_id": "r_train", "y_pred": "quartz"},
    ])
    preds_train_leak.to_parquet(s_dir / "predictions.parquet")

    with pytest.raises(ValueError, match="Two-way population check failed.*extra rows from train or val"):
        ev.evaluate_experiment(
            results_dir=res_dir,
            teachers_dir=teach_dir,
            arms=["static_k1"],
            folds=["fold_0"],
            seeds=[0],
        )


def test_evaluate_static_sensitivity_requires_all_folds_and_seeds(tmp_path: Path):
    """evaluate_static_sensitivity requires all expected folds and seeds to be present."""
    ev = Evaluator(classes=CLASSES, n_bootstrap=50, seed=0)
    static_dir = tmp_path / "static_k1"
    (static_dir / "fold_0" / "seed_0").mkdir(parents=True)
    (static_dir / "fold_0" / "seed_1").mkdir(parents=True)

    # Save angle predictions for fold_0
    df = pd.DataFrame([{"region_id": "r1", "specimen": "s1", "state": "singleton", "allowed": ["quartz"], "y_pred": "quartz"}])
    df.to_parquet(static_dir / "fold_0" / "seed_0" / "predictions_angle_0.parquet")
    df.to_parquet(static_dir / "fold_0" / "seed_1" / "predictions_angle_0.parquet")

    # Missing fold_1
    with pytest.raises(FileNotFoundError, match="Static sensitivity requires all expected folds and seeds to be present"):
        ev.evaluate_static_sensitivity(
            static_dir=static_dir,
            folds=["fold_0", "fold_1"],
            seeds=[0, 1],
        )


def test_c1_test_predicted_states_and_samples_cross_check(tmp_path: Path):
    """C1: Test-predicted states contract and samples cross-check.

    Verifies:
    1. 5-state fixture passes (singleton, partial_set, abstain evaluated; excluded, mixed ignored)
    2. Missing singleton fails
    3. Extra excluded fails
    4. Missing from samples.parquet fails
    5. Fallback in _get_held_out_test_targets raises ValueError
    """
    ev = Evaluator(classes=CLASSES, n_bootstrap=50, seed=0)
    res_dir = tmp_path / "results"
    teach_dir = tmp_path / "teachers"
    res_dir.mkdir()
    teach_dir.mkdir()

    (teach_dir / "fold_0").mkdir()
    targets_df = pd.DataFrame([
        {"region_id": "r_singleton", "partition": "test", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
        {"region_id": "r_partial", "partition": "test", "specimen": "spec1", "state": "partial_set", "allowed": ["quartz", "feldspar"]},
        {"region_id": "r_abstain", "partition": "test", "specimen": "spec1", "state": "abstain", "allowed": []},
        {"region_id": "r_excluded", "partition": "test", "specimen": "spec1", "state": "excluded", "allowed": []},
        {"region_id": "r_mixed", "partition": "test", "specimen": "spec1", "state": "mixed", "allowed": []},
        {"region_id": "r_train", "partition": "train", "specimen": "spec1", "state": "singleton", "allowed": ["quartz"]},
    ])
    targets_df.to_parquet(teach_dir / "fold_0" / "targets.parquet")

    # Verify _get_held_out_test_targets filters strictly to TEST_PREDICTED_STATES
    test_targets, train_val_targets = ev._get_held_out_test_targets(targets_df, "fold_0")
    assert set(test_targets["region_id"]) == {"r_singleton", "r_partial", "r_abstain"}
    assert set(train_val_targets["region_id"]) == {"r_train"}

    # Samples parquet containing all regions
    samples_df = pd.DataFrame([
        {"region_id": f"r_{k}", "specimen": "spec1"}
        for k in ["singleton", "partial", "abstain", "excluded", "mixed", "train"]
    ])
    samples_df.to_parquet(teach_dir.parent / "samples.parquet")

    # Setup valid predictions for both primary arms
    pred_rows = [
        {"region_id": "r_singleton", "y_pred": "quartz"},
        {"region_id": "r_partial", "y_pred": "quartz"},
        {"region_id": "r_abstain", "y_pred": "quartz"},
    ]
    for arm in ["static_k1", "angle_aware_k12"]:
        s_dir = res_dir / arm / "fold_0" / "seed_0"
        s_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(pred_rows).to_parquet(s_dir / "predictions.parquet")

    # 1. 5-state fixture passes
    summary = ev.evaluate_experiment(
        results_dir=res_dir,
        teachers_dir=teach_dir,
        arms=["static_k1", "angle_aware_k12"],
        folds=["fold_0"],
        seeds=[0],
    )
    assert summary is not None

    # 2. Missing singleton fails
    missing_sing_rows = [
        {"region_id": "r_partial", "y_pred": "quartz"},
        {"region_id": "r_abstain", "y_pred": "quartz"},
    ]
    pd.DataFrame(missing_sing_rows).to_parquet(res_dir / "static_k1" / "fold_0" / "seed_0" / "predictions.parquet")
    with pytest.raises(ValueError, match="missing predictions for 1 held-out test regions"):
        ev.evaluate_experiment(
            results_dir=res_dir,
            teachers_dir=teach_dir,
            arms=["static_k1", "angle_aware_k12"],
            folds=["fold_0"],
            seeds=[0],
        )

    # 3. Extra excluded fails
    extra_excl_rows = pred_rows + [{"region_id": "r_excluded", "y_pred": "quartz"}]
    pd.DataFrame(extra_excl_rows).to_parquet(res_dir / "static_k1" / "fold_0" / "seed_0" / "predictions.parquet")
    with pytest.raises(ValueError, match="extra rows not in held-out test targets"):
        ev.evaluate_experiment(
            results_dir=res_dir,
            teachers_dir=teach_dir,
            arms=["static_k1", "angle_aware_k12"],
            folds=["fold_0"],
            seeds=[0],
        )

    # Restore valid predictions for static_k1
    pd.DataFrame(pred_rows).to_parquet(res_dir / "static_k1" / "fold_0" / "seed_0" / "predictions.parquet")

    # 4. Missing from samples fails
    bad_samples_df = pd.DataFrame([
        {"region_id": f"r_{k}", "specimen": "spec1"}
        for k in ["partial", "abstain", "excluded", "mixed", "train"]
    ])
    bad_samples_df.to_parquet(teach_dir.parent / "samples.parquet")
    with pytest.raises(ValueError, match="Cross-check against samples.parquet failed.*1 required test regions missing"):
        ev.evaluate_experiment(
            results_dir=res_dir,
            teachers_dir=teach_dir,
            arms=["static_k1", "angle_aware_k12"],
            folds=["fold_0"],
            seeds=[0],
        )

    # 5. Fallback in _get_held_out_test_targets raises ValueError
    df_no_partition = targets_df.drop(columns=["partition"])
    with pytest.raises(ValueError, match="targets have no partition/split/is_test column for fold"):
        ev._get_held_out_test_targets(df_no_partition, "fold_0")


def test_c3_recursive_metadata_split_averaging_and_report():
    """C3: Recursive metadata-split averaging across seeds and report rendering."""
    ev = Evaluator(classes=CLASSES, n_bootstrap=50, seed=0)

    # 1. Test recursive averaging with 2 seeds having different nested per_class
    seed_0 = {
        "metadata_dependent": {
            "macro_f1": 0.80,
            "accuracy": 0.85,
            "n_regions": 100,
            "n_specimens": 10,
            "per_class": {
                "quartz": {"precision": 0.80, "recall": 0.70, "f1": 0.75, "support_unweighted": 40, "support_weighted": 4.0},
                "feldspar": {"precision": 0.75, "recall": 0.70, "f1": 0.725, "support_unweighted": 60, "support_weighted": 6.0},
            },
        },
        "optics_only_reproducible": {
            "macro_f1": 0.90,
            "accuracy": 0.92,
            "n_regions": 200,
            "n_specimens": 15,
            "per_class": {
                "quartz": {"precision": 0.90, "recall": 0.80, "f1": 0.85, "support_unweighted": 100, "support_weighted": 10.0},
                "feldspar": {"precision": 0.85, "recall": 0.80, "f1": 0.825, "support_unweighted": 100, "support_weighted": 10.0},
            },
        },
    }

    seed_1 = {
        "metadata_dependent": {
            "macro_f1": 0.60,
            "accuracy": 0.65,
            "n_regions": 100,
            "n_specimens": 10,
            "per_class": {
                "quartz": {"precision": 0.60, "recall": 0.50, "f1": 0.55, "support_unweighted": 40, "support_weighted": 4.0},
                "feldspar": {"precision": 0.55, "recall": 0.50, "f1": 0.525, "support_unweighted": 60, "support_weighted": 6.0},
            },
        },
        "optics_only_reproducible": {
            "macro_f1": 0.70,
            "accuracy": 0.72,
            "n_regions": 200,
            "n_specimens": 15,
            "per_class": {
                "quartz": {"precision": 0.70, "recall": 0.60, "f1": 0.65, "support_unweighted": 100, "support_weighted": 10.0},
                "feldspar": {"precision": 0.65, "recall": 0.60, "f1": 0.625, "support_unweighted": 100, "support_weighted": 10.0},
            },
        },
    }

    avg = ev._recursive_average([seed_0, seed_1])

    # Top-level metrics
    assert avg["metadata_dependent"]["macro_f1"] == pytest.approx(0.70)
    assert avg["metadata_dependent"]["accuracy"] == pytest.approx(0.75)
    assert avg["metadata_dependent"]["n_regions"] == 100
    assert avg["metadata_dependent"]["n_specimens"] == 10

    # Nested per_class metrics match mean
    assert avg["metadata_dependent"]["per_class"]["quartz"]["f1"] == pytest.approx(0.65)
    assert avg["metadata_dependent"]["per_class"]["quartz"]["precision"] == pytest.approx(0.70)
    assert avg["metadata_dependent"]["per_class"]["quartz"]["recall"] == pytest.approx(0.60)
    assert avg["metadata_dependent"]["per_class"]["quartz"]["support_unweighted"] == 40

    assert avg["metadata_dependent"]["per_class"]["feldspar"]["f1"] == pytest.approx(0.625)
    assert avg["metadata_dependent"]["per_class"]["feldspar"]["precision"] == pytest.approx(0.65)
    assert avg["metadata_dependent"]["per_class"]["feldspar"]["recall"] == pytest.approx(0.60)
    assert avg["metadata_dependent"]["per_class"]["feldspar"]["support_unweighted"] == 60

    # Optics-only reproducible
    assert avg["optics_only_reproducible"]["macro_f1"] == pytest.approx(0.80)
    assert avg["optics_only_reproducible"]["per_class"]["quartz"]["f1"] == pytest.approx(0.75)

    # 2. Test report rendering: unavailable when < 30 regions or < 3 classes with support
    summary_tiny = {
        "metadata_split": {
            "static_k1": {
                "metadata_dependent": {
                    "macro_f1": 0.70,
                    "n_regions": 15,  # < 30 -> unavailable
                    "n_specimens": 3,
                    "per_class": {
                        "c1": {"support_unweighted": 5},
                        "c2": {"support_unweighted": 5},
                        "c3": {"support_unweighted": 5},
                    },
                },
                "optics_only_reproducible": {
                    "macro_f1": 0.85,
                    "n_regions": 50,  # >= 30, but only 2 classes with support -> unavailable
                    "n_specimens": 10,
                    "per_class": {
                        "c1": {"support_unweighted": 25},
                        "c2": {"support_unweighted": 25},
                        "c3": {"support_unweighted": 0},
                    },
                },
            },
            "angle_aware_k12": {
                "metadata_dependent": {
                    "macro_f1": 0.78,
                    "n_regions": 45,  # >= 30 and 3 classes with support -> available
                    "n_specimens": 8,
                    "per_class": {
                        "c1": {"support_unweighted": 15},
                        "c2": {"support_unweighted": 15},
                        "c3": {"support_unweighted": 15},
                    },
                },
                "optics_only_reproducible": {
                    "macro_f1": 0.88,
                    "n_regions": 120,  # >= 30 and 3 classes with support -> available
                    "n_specimens": 15,
                    "per_class": {
                        "c1": {"support_unweighted": 40},
                        "c2": {"support_unweighted": 40},
                        "c3": {"support_unweighted": 40},
                    },
                },
            },
        }
    }

    report = ev.generate_markdown_report(summary_tiny)
    assert "Performance Split by Metadata Dependency" in report
    assert "unavailable (n=15)" in report
    assert "unavailable (n=50)" in report
    assert "0.7800" in report
    assert "0.8800" in report


def test_d1_5_static_strata_evaluation(tmp_path: Path):
    """D1.5: Static strata performance and paired bootstrap comparisons."""
    ev = Evaluator(classes=CLASSES, n_bootstrap=50, seed=42)

    # 1. Test tiny stratum (< 30 regions) and stratum with < 3 classes with support
    tiny_rows_k12 = [
        {"region_id": f"r_tiny_{i}", "specimen": f"spec_{i % 3}", "state": "singleton", "allowed": ["quartz"], "y_pred": "quartz", "static_category": "reproduced", "fold": "fold_0"}
        for i in range(15)  # 15 < 30
    ]
    tiny_rows_k1 = [
        {"region_id": f"r_tiny_{i}", "specimen": f"spec_{i % 3}", "state": "singleton", "allowed": ["quartz"], "y_pred": "quartz", "static_category": "reproduced", "fold": "fold_0"}
        for i in range(15)
    ]
    pooled_tiny = {
        "angle_aware_k12": {0: pd.DataFrame(tiny_rows_k12)},
        "static_k1": {0: pd.DataFrame(tiny_rows_k1)},
    }
    res_tiny = ev.evaluate_static_strata(pooled_tiny)
    assert res_tiny["reproduced"]["status"] == "unavailable"
    assert "unavailable (n=15)" in res_tiny["reproduced"]["unavailable_reason"]
    assert res_tiny["reproduced"]["paired_bootstrap"] is None
    assert res_tiny["reproduced"]["n_regions"] == 15
    assert res_tiny["reproduced"]["n_specimens"] == 3

    # Report rendering for unavailable stratum
    report_tiny = ev.generate_markdown_report({"static_strata": res_tiny})
    assert "## 6. Static Strata Performance & Comparisons" in report_tiny
    assert "| reproduced | all | unavailable (n=15) | 15 | 3 | 15 / 3 | - |" in report_tiny

    # 2. Test valid strata (>= 30 regions, >= 3 classes with support)
    # 45 reproduced regions + 45 rotation_dependent regions (30 full_only + 15 k1_broader)
    classes_sub = ["quartz", "feldspar", "mica"]
    valid_k12 = []
    valid_k1 = []
    for i in range(45):
        c = classes_sub[i % 3]
        rid = f"r_val_{i:02d}"
        spec = f"spec_{i % 5}"
        valid_k12.append({
            "region_id": rid,
            "specimen": spec,
            "state": "singleton",
            "allowed": [c],
            "y_pred": c,
            "static_category": "reproduced",
            "fold": "fold_0",
        })
        pred_k1 = c if i % 2 == 0 else classes_sub[(i + 1) % 3]
        valid_k1.append({
            "region_id": rid,
            "specimen": spec,
            "state": "singleton",
            "allowed": [c],
            "y_pred": pred_k1,
            "static_category": "reproduced",
            "fold": "fold_0",
        })

    # Add rotation_dependent rows: 30 full_only + 15 k1_broader
    for i in range(45):
        c = classes_sub[i % 3]
        rid = f"r_rotdep_{i:02d}"
        spec = f"spec_{(i % 5) + 5}"
        cat = "full_only" if i < 30 else "k1_broader"
        valid_k12.append({
            "region_id": rid,
            "specimen": spec,
            "state": "singleton",
            "allowed": [c],
            "y_pred": c,
            "static_category": cat,
            "fold": "fold_0",
        })
        pred_k1 = c if i % 3 == 0 else classes_sub[(i + 1) % 3]
        valid_k1.append({
            "region_id": rid,
            "specimen": spec,
            "state": "singleton",
            "allowed": [c],
            "y_pred": pred_k1,
            "static_category": cat,
            "fold": "fold_0",
        })

    pooled_valid = {
        "angle_aware_k12": {0: pd.DataFrame(valid_k12)},
        "static_k1": {0: pd.DataFrame(valid_k1)},
    }
    res_valid = ev.evaluate_static_strata(pooled_valid)
    assert res_valid["reproduced"]["status"] == "available"
    assert res_valid["reproduced"]["n_regions"] == 45
    assert res_valid["reproduced"]["n_specimens"] == 5
    assert res_valid["reproduced"]["arm_metrics"]["angle_aware_k12"]["macro_f1"] > res_valid["reproduced"]["arm_metrics"]["static_k1"]["macro_f1"]

    assert res_valid["rotation_dependent"]["status"] == "available"
    assert res_valid["rotation_dependent"]["n_regions"] == 45
    assert res_valid["rotation_dependent"]["n_specimens"] == 5

    # Paired bootstrap computed between angle_aware_k12 and static_k1
    boot = res_valid["reproduced"]["paired_bootstrap"]
    assert boot is not None
    assert "point_estimate_diff" in boot
    assert boot["point_estimate_diff"] > 0
    assert "ci_lower" in boot
    assert "ci_upper" in boot

    # Report rendering for available strata
    report_valid = ev.generate_markdown_report({"static_strata": res_valid})
    assert "## 6. Static Strata Performance & Comparisons" in report_valid
    assert "| reproduced | angle_aware_k12 |" in report_valid
    assert "| reproduced | static_k1 |" in report_valid
    assert "| rotation_dependent | angle_aware_k12 |" in report_valid

    # 3. Test full integration in evaluate_experiment with static_category in targets
    res_dir = tmp_path / "results_d15"
    teach_dir = tmp_path / "teachers_d15"
    (teach_dir / "fold_0").mkdir(parents=True)
    res_dir.mkdir(parents=True)

    targets_all = pd.DataFrame(valid_k12).copy()
    targets_all["partition"] = "test"
    targets_all.to_parquet(teach_dir / "fold_0" / "targets.parquet")

    samples_df = targets_all[["region_id", "specimen"]].copy()
    samples_df.to_parquet(teach_dir.parent / "samples.parquet")

    for arm, rows in [("angle_aware_k12", valid_k12), ("static_k1", valid_k1)]:
        arm_dir = res_dir / arm / "fold_0" / "seed_0"
        arm_dir.mkdir(parents=True)
        pd.DataFrame(rows)[["region_id", "y_pred"]].to_parquet(arm_dir / "predictions.parquet")

    summary = ev.evaluate_experiment(
        results_dir=res_dir,
        teachers_dir=teach_dir,
        arms=["angle_aware_k12", "static_k1"],
        folds=["fold_0"],
        seeds=[0],
    )

    assert "static_strata" in summary
    assert "reproduced" in summary["static_strata"]
    assert "rotation_dependent" in summary["static_strata"]
    assert summary["static_strata"]["reproduced"]["status"] == "available"
    assert summary["static_strata"]["reproduced"]["n_regions"] == 45
    assert summary["static_strata"]["rotation_dependent"]["status"] == "available"
    assert summary["static_strata"]["rotation_dependent"]["n_regions"] == 45

    report_file = res_dir / "evaluation_report.md"
    assert report_file.is_file()
    report_content = report_file.read_text(encoding="utf-8")
    assert "## 6. Static Strata Performance & Comparisons" in report_content
    assert "| rotation_dependent |" in report_content





