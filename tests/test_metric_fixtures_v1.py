"""Tests for V1 metric fixtures detectability simulation balance at zero gain.

Simulation only — not evidence for the proposal.
Test file: tests/test_metric_fixtures_v1.py
Task: D2
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.v1_metric_fixtures import (
    _make_synthetic_data,
    _simulate_model_predictions,
    run_detectability_scenarios,
)
from src.rotapetro.evaluate import Evaluator


def test_zero_gain_equal_errors_uniform_and_concentrated():
    """At gain=0, Model A and Model B must have exactly equal error counts for both uniform and concentrated clustering."""
    data = _make_synthetic_data(n_specimens=30, n_regions_per_specimen=10, seed=42)
    classes = [f"class_{i}" for i in range(5)]
    base_acc = 0.55
    n_flip = int(len(data) * (1 - base_acc))

    # Test uniform clustering
    rng_a = np.random.RandomState(100)
    rng_b = np.random.RandomState(200)
    df_a_uni = _simulate_model_predictions(data, n_flip, "uniform", classes, rng_a)
    df_b_uni = _simulate_model_predictions(data, n_flip, "uniform", classes, rng_b)

    errors_a_uni = (df_a_uni["y_pred"] != df_a_uni["y_true"]).sum()
    errors_b_uni = (df_b_uni["y_pred"] != df_b_uni["y_true"]).sum()

    assert errors_a_uni == n_flip
    assert errors_b_uni == n_flip
    assert errors_a_uni == errors_b_uni

    # Test concentrated clustering
    rng_c = np.random.RandomState(300)
    rng_d = np.random.RandomState(400)
    df_a_conc = _simulate_model_predictions(data, n_flip, "concentrated", classes, rng_c)
    df_b_conc = _simulate_model_predictions(data, n_flip, "concentrated", classes, rng_d)

    errors_a_conc = (df_a_conc["y_pred"] != df_a_conc["y_true"]).sum()
    errors_b_conc = (df_b_conc["y_pred"] != df_b_conc["y_true"]).sum()

    assert errors_a_conc == n_flip
    assert errors_b_conc == n_flip
    assert errors_a_conc == errors_b_conc


def test_flip_always_picks_different_class():
    """Flipped predictions must strictly differ from y_true (never accidentally matching)."""
    data = _make_synthetic_data(n_specimens=20, n_regions_per_specimen=5, seed=123)
    classes = [f"class_{i}" for i in range(5)]
    rng = np.random.RandomState(999)

    for clustering in ["uniform", "concentrated"]:
        n_flip = 40
        sim = _simulate_model_predictions(data, n_flip, clustering, classes, rng)
        # Exactly n_flip errors should exist
        mismatches = (sim["y_pred"] != sim["y_true"]).sum()
        assert mismatches == n_flip


def test_concentrated_clustering_specimen_count():
    """Concentrated clustering adds specimens until containing >= n_flip regions."""
    data = _make_synthetic_data(n_specimens=30, n_regions_per_specimen=10, seed=77)
    classes = [f"class_{i}" for i in range(5)]
    rng = np.random.RandomState(88)

    n_flip = 50  # 5 specimens of 10 regions each
    sim = _simulate_model_predictions(data, n_flip, "concentrated", classes, rng)

    # Errors should be confined to at most ceil(50/10) = 5 specimens
    error_mask = sim["y_pred"] != sim["y_true"]
    error_specimens = sim.loc[error_mask, "specimen"].unique()
    assert len(error_specimens) <= 5


def test_detectability_scenarios_zero_gain():
    """Verify run_detectability_scenarios outputs equal errors at gain=0 for both clusterings."""
    evaluator = Evaluator(
        classes=[f"class_{i}" for i in range(5)],
        n_bootstrap=20,
    )
    data = _make_synthetic_data(n_specimens=15, n_regions_per_specimen=8, seed=42)

    scenarios = run_detectability_scenarios(evaluator, data, seed=42)
    zero_gain_results = [s for s in scenarios if s["gain_pct"] == 0]

    assert len(zero_gain_results) == 2  # uniform and concentrated
    for res in zero_gain_results:
        assert res["errors_a"] == res["errors_b"]

