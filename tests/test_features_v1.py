"""Unit tests for V1 feature extraction fixes (Step 2 / Fix 2 & Fix 10).

Tests:
1. Absorption circular mean:
   - empty phases -> NaN
   - 179° and 1° aggregate near 0° / 180° (not Euclidean median 90°)
   - dispersed phases (e.g. [0°, 45°, 90°, 135°] where R < 0.3) -> NaN
2. FeatureExtractor absorption angle behavior:
   - empty absorption phases -> NaN absorption_vs_long_deg and absorption_status == 'unresolved'
   - [179°, 1°] phases aggregate near 0°/180° -> absorption_vs_long_deg near 0° (NOT 90°)
   - dispersed phases (R < 0.3) -> NaN absorption_vs_long_deg and absorption_status == 'unresolved'
3. Twin classification (_classify_twin):
   - Two domains (n_domains=2) with no detected Hough lines returns 'none' (not 'simple').
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest
from unittest.mock import patch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config
from src.rotapetro.features import FeatureExtractor


@pytest.fixture(scope="module")
def extractor() -> FeatureExtractor:
    cfg = load_config("configs/default.yaml")
    return FeatureExtractor(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Circular Mean Mathematical Properties (Period-180° Axial Symmetry)
# ─────────────────────────────────────────────────────────────────────────────

def test_absorption_circular_mean_empty():
    """Empty or all-NaN phases must yield NaN for pleo_dark_med."""
    pl_dark = np.array([], dtype=np.float32)
    if len(pl_dark) > 0:
        rad2 = np.radians(2.0 * pl_dark)
        mean_z = np.mean(np.exp(1j * rad2))
        pleo_R = float(np.abs(mean_z))
        result = float((np.degrees(np.angle(mean_z)) / 2.0) % 180.0) if pleo_R >= 0.3 else float("nan")
    else:
        result = float("nan")
    assert np.isnan(result)


def test_absorption_circular_mean_boundary_aggregation():
    """179° and 1° must aggregate near 0°/180° under period-180° circular mean, NOT Euclidean 90°."""
    pl_dark = np.array([179.0, 1.0], dtype=np.float32)

    # Euclidean median (the old bug)
    euclidean_median = float(np.median(pl_dark))
    assert euclidean_median == pytest.approx(90.0)

    # Period-180° circular mean (the fix)
    rad2 = np.radians(2.0 * pl_dark)
    mean_z = np.mean(np.exp(1j * rad2))
    pleo_R = float(np.abs(mean_z))
    assert pleo_R >= 0.3
    circ_mean = float((np.degrees(np.angle(mean_z)) / 2.0) % 180.0)

    # 0° and 180° are identical mod 180. Distance to 0°/180° should be <= 1°
    diff_from_zero = min(circ_mean, abs(180.0 - circ_mean))
    assert diff_from_zero <= 1.0
    # Must NOT be near Euclidean median 90°
    assert abs(circ_mean - 90.0) > 80.0


def test_absorption_circular_mean_dispersed_phases_nan():
    """Dispersed phases (e.g. uniform [0°, 45°, 90°, 135°] where R < 0.3) must yield NaN."""
    pl_dark = np.array([0.0, 45.0, 90.0, 135.0], dtype=np.float32)
    rad2 = np.radians(2.0 * pl_dark)
    mean_z = np.mean(np.exp(1j * rad2))
    pleo_R = float(np.abs(mean_z))

    # Resultant length R is near 0 (< 0.3 threshold)
    assert pleo_R < 0.3
    result = float((np.degrees(np.angle(mean_z)) / 2.0) % 180.0) if pleo_R >= 0.3 else float("nan")
    assert np.isnan(result)


# ─────────────────────────────────────────────────────────────────────────────
# 2. End-to-End Extraction Verification with FeatureExtractor
# ─────────────────────────────────────────────────────────────────────────────

def _make_synthetic_site_for_pleo(pl_dark_values: np.ndarray | None):
    """Create a minimal synthetic site with one elongated pleochroic grain."""
    H, W = 50, 50
    label_img = np.zeros((H, W), dtype=np.int32)
    # Elongated horizontal grain (rows 20:30, cols 10:40 -> aspect ~ 3.0, long_axis_deg ~ 0°)
    label_img[20:30, 10:40] = 1

    region_list = [{"region_id": "r_pleo", "label_value": 1, "type": "grain"}]
    valid = np.ones((H, W), dtype=bool)
    aligned = {
        "valid": valid,
        "ppl": np.full((36, H, W, 3), 200, dtype=np.uint8),
        "theta_deg": np.linspace(0, 175, 36).astype(np.float32),
        "rot_sign": 1,
    }

    maps = {
        "aniso": np.zeros((H, W), dtype=np.float32),
        "ext_deg": np.full((H, W), np.nan, dtype=np.float32),
        "r2_4": np.zeros((H, W), dtype=np.float32),
        "ppl_max_L": np.full((H, W), 200.0, dtype=np.float32),
        "ppl_Lab": np.zeros((H, W, 3), dtype=np.float32),
        "pleo_dE": np.zeros((H, W), dtype=np.float32),
        "pleo_dark_deg": np.full((H, W), np.nan, dtype=np.float32),
        "relief_mod": np.zeros((H, W), dtype=np.float32),
        "ppl_tex": np.zeros((H, W), dtype=np.float32),
        "gamma_est_nm": np.full((H, W), np.nan, dtype=np.float32),
        "biref_cls": np.zeros((H, W), dtype=np.int8),
        "ppl_mean_rgb": np.full((H, W, 3), 200, dtype=np.uint8),
        "white_level": 255.0,
        "xpl_rel_rgb": np.zeros((H, W, 3), dtype=np.float32),
        "aniso_mask": np.zeros((H, W), dtype=bool),
    }

    maps["aniso"][20:30, 10:40] = 0.6
    maps["aniso_mask"][20:30, 10:40] = True
    maps["r2_4"][20:30, 10:40] = 0.9
    maps["ext_deg"][20:30, 10:40] = 30.0
    maps["pleo_dE"][20:30, 10:40] = 15.0  # strong/moderate pleochroism

    if pl_dark_values is not None:
        grain_pts = np.where(label_img == 1)
        n_pts = len(grain_pts[0])
        # Repeat or tile values across grain pixels
        vals = np.resize(pl_dark_values, n_pts)
        maps["pleo_dark_deg"][grain_pts] = vals

    return label_img, region_list, aligned, maps


def test_extractor_absorption_empty_phases_yields_nan(extractor: FeatureExtractor):
    """When pleo_dark_deg has no measurements (all NaN / empty), absorption is NaN & unresolved."""
    label_img, region_list, aligned, maps = _make_synthetic_site_for_pleo(None)
    extracted = extractor.extract_site(
        label_img=label_img, region_list=region_list, aligned=aligned, maps=maps,
        site_id="s_empty", code="c_empty",
    )
    r = extracted[0]
    assert np.isnan(r["absorption_vs_long_deg"]), "Expected NaN absorption_vs_long_deg for empty phases"
    assert r["absorption_status"] == "unresolved", (
        f"Expected 'unresolved' absorption_status for pleochroic grain with NaN dark angle, got {r['absorption_status']}"
    )


def test_extractor_absorption_boundary_circular_mean(extractor: FeatureExtractor):
    """When pleo_dark_deg contains 179° and 1°, circular mean places axis near 0° (NOT Euclidean 90°)."""
    # 179° and 1° interleaved
    pl_dark = np.array([179.0, 1.0], dtype=np.float32)
    label_img, region_list, aligned, maps = _make_synthetic_site_for_pleo(pl_dark)
    extracted = extractor.extract_site(
        label_img=label_img, region_list=region_list, aligned=aligned, maps=maps,
        site_id="s_circ", code="c_circ",
    )
    r = extracted[0]
    assert r["absorption_status"] == "observed"
    assert not np.isnan(r["absorption_vs_long_deg"])
    # The grain is horizontally elongated (long axis ~ 0° / 180°).
    # Since circular mean of 179° and 1° is ~0°, absorption_vs_long_deg should be near 0° (< 5°)
    # If Euclidean median were used (90°), absorption_vs_long_deg would be ~90°!
    assert r["absorption_vs_long_deg"] < 5.0, (
        f"Expected absorption_vs_long_deg near 0° (< 5°), got {r['absorption_vs_long_deg']} (Euclidean median bug = ~90°)"
    )


def test_extractor_absorption_dispersed_phases_yields_nan(extractor: FeatureExtractor):
    """When pleo_dark_deg phases are dispersed (R < 0.3), absorption is NaN & unresolved."""
    pl_dark = np.array([0.0, 45.0, 90.0, 135.0], dtype=np.float32)
    label_img, region_list, aligned, maps = _make_synthetic_site_for_pleo(pl_dark)
    extracted = extractor.extract_site(
        label_img=label_img, region_list=region_list, aligned=aligned, maps=maps,
        site_id="s_disp", code="c_disp",
    )
    r = extracted[0]
    assert np.isnan(r["absorption_vs_long_deg"]), "Expected NaN absorption_vs_long_deg for dispersed phases"
    assert r["absorption_status"] == "unresolved", (
        f"Expected 'unresolved' absorption_status for dispersed phases, got {r['absorption_status']}"
    )


def test_extractor_absorption_minimum_support_400px_yields_nan(extractor: FeatureExtractor):
    """5 finite phases in 400px interior must yield NaN (min support = max(20, 0.10*400)=40) -> unresolved."""
    # 1. Direct method test
    five_phases = np.array([45.0, 45.0, 45.0, 45.0, 45.0], dtype=np.float32)
    dark_med = extractor._compute_absorption_dark_angle(five_phases, n_interior_px=400)
    assert np.isnan(dark_med), f"Expected NaN for 5 phases in 400px interior, got {dark_med}"

    # 2. End-to-end site extraction test:
    # 24x24 square grain with border_px=2 gives 20x20 = 400 interior pixels
    H, W = 30, 30
    label_img = np.zeros((H, W), dtype=np.int32)
    label_img[3:27, 3:27] = 1

    region_list = [{"region_id": "r_400px", "label_value": 1, "type": "grain"}]
    valid = np.ones((H, W), dtype=bool)
    aligned = {
        "valid": valid,
        "ppl": np.full((36, H, W, 3), 200, dtype=np.uint8),
        "theta_deg": np.linspace(0, 175, 36).astype(np.float32),
        "rot_sign": 1,
    }

    maps = {
        "aniso": np.full((H, W), 0.6, dtype=np.float32),
        "ext_deg": np.full((H, W), 30.0, dtype=np.float32),
        "r2_4": np.full((H, W), 0.9, dtype=np.float32),
        "ppl_max_L": np.full((H, W), 200.0, dtype=np.float32),
        "ppl_Lab": np.zeros((H, W, 3), dtype=np.float32),
        "pleo_dE": np.full((H, W), 15.0, dtype=np.float32),  # moderate/strong pleochroism
        "pleo_dark_deg": np.full((H, W), np.nan, dtype=np.float32),
        "relief_mod": np.zeros((H, W), dtype=np.float32),
        "ppl_tex": np.zeros((H, W), dtype=np.float32),
        "gamma_est_nm": np.full((H, W), np.nan, dtype=np.float32),
        "biref_cls": np.zeros((H, W), dtype=np.int8),
        "ppl_mean_rgb": np.full((H, W, 3), 200, dtype=np.uint8),
        "white_level": 255.0,
        "xpl_rel_rgb": np.zeros((H, W, 3), dtype=np.float32),
        "aniso_mask": np.ones((H, W), dtype=bool),
    }

    # Place exactly 5 finite values in the interior (rows 5:10, col 5)
    maps["pleo_dark_deg"][5:10, 5] = 45.0

    extracted = extractor.extract_site(
        label_img=label_img, region_list=region_list, aligned=aligned, maps=maps,
        site_id="s_400px", code="c_400px",
    )
    r = extracted[0]
    assert r["interior_px"] == 400
    assert np.isnan(r["absorption_vs_long_deg"]), (
        f"Expected NaN absorption_vs_long_deg for 5 phases in 400px interior, got {r['absorption_vs_long_deg']}"
    )
    assert r["absorption_status"] == "unresolved", (
        f"Expected 'unresolved' absorption_status for insufficient support, got {r['absorption_status']}"
    )



# ─────────────────────────────────────────────────────────────────────────────
# 3. Twin Classification: two domains with no Hough lines returns 'none'
# ─────────────────────────────────────────────────────────────────────────────

def test_classify_twin_two_domains_no_hough_lines_returns_none(extractor: FeatureExtractor):
    """When n_domains=2 but Hough line detector finds no linear boundary, return 'none' (not 'simple')."""
    H, W = 40, 40
    region_mask = np.zeros((H, W), dtype=bool)
    region_mask[5:35, 5:35] = True

    ext_deg_img = np.full((H, W), np.nan, dtype=np.float32)
    # Domain 1 (left half) at 10°, Domain 2 (right half) at 55° (separated by > 8° in 90° space)
    ext_deg_img[5:35, 5:20] = 10.0
    ext_deg_img[5:35, 20:35] = 55.0

    aniso_img = np.full((H, W), 0.8, dtype=np.float32)
    conf_mask = np.zeros((H, W), dtype=bool)
    conf_mask[5:35, 5:35] = True

    # 1. When Hough returns lines None or empty list -> MUST return 'none'
    with patch("src.rotapetro.features.probabilistic_hough_line", return_value=[]):
        twin_cls = extractor._classify_twin(
            n_domains=2,
            region_mask=region_mask,
            ext_deg_img=ext_deg_img,
            aniso_img=aniso_img,
            conf_mask=conf_mask,
        )
        assert twin_cls == "none", f"Expected 'none' when no Hough lines detected for n_domains=2, got '{twin_cls}'"

    # Also test lines=None
    with patch("src.rotapetro.features.probabilistic_hough_line", return_value=None):
        twin_cls = extractor._classify_twin(
            n_domains=2,
            region_mask=region_mask,
            ext_deg_img=ext_deg_img,
            aniso_img=aniso_img,
            conf_mask=conf_mask,
        )
        assert twin_cls == "none", f"Expected 'none' when Hough returns None for n_domains=2, got '{twin_cls}'"


def test_classify_twin_two_domains_with_valid_line_returns_simple(extractor: FeatureExtractor):
    """When n_domains=2 and Hough finds a single dominant straight line boundary, return 'simple'."""
    H, W = 40, 40
    region_mask = np.zeros((H, W), dtype=bool)
    region_mask[5:35, 5:35] = True

    ext_deg_img = np.full((H, W), np.nan, dtype=np.float32)
    ext_deg_img[5:35, 5:20] = 10.0
    ext_deg_img[5:35, 20:35] = 55.0

    aniso_img = np.full((H, W), 0.8, dtype=np.float32)
    conf_mask = np.zeros((H, W), dtype=bool)
    conf_mask[5:35, 5:35] = True

    # Mock Hough returning a clean vertical line spanning 25 pixels along the boundary (x=14, y=5..30)
    fake_lines = [((14, 5), (14, 30))]
    with patch("src.rotapetro.features.probabilistic_hough_line", return_value=fake_lines):
        twin_cls = extractor._classify_twin(
            n_domains=2,
            region_mask=region_mask,
            ext_deg_img=ext_deg_img,
            aniso_img=aniso_img,
            conf_mask=conf_mask,
        )
        assert twin_cls == "simple", f"Expected 'simple' when single straight line detected, got '{twin_cls}'"


def test_twin_detection_deterministic_5_times(extractor: FeatureExtractor):
    """Synthetic grain twin detection must be strictly identical across 5 runs (deterministic via rng=0)."""
    H, W = 40, 40
    region_mask = np.zeros((H, W), dtype=bool)
    region_mask[5:35, 5:35] = True

    ext_deg_img = np.full((H, W), np.nan, dtype=np.float32)
    ext_deg_img[5:35, 5:20] = 10.0
    ext_deg_img[5:35, 20:35] = 55.0

    aniso_img = np.full((H, W), 0.8, dtype=np.float32)
    conf_mask = np.zeros((H, W), dtype=bool)
    conf_mask[5:35, 5:35] = True

    results = []
    for _ in range(5):
        cls = extractor._classify_twin(
            n_domains=2,
            region_mask=region_mask,
            ext_deg_img=ext_deg_img,
            aniso_img=aniso_img,
            conf_mask=conf_mask,
            ppl_grad=None,
        )
        results.append(cls)

    assert len(results) == 5
    assert len(set(results)) == 1, f"Twin classification varied across 5 runs: {results}"


def test_cleavage_detection_deterministic_5_times(extractor: FeatureExtractor):
    """Synthetic cleavage detection must produce identical results across 5 runs (deterministic via rng=0)."""
    H, W = 50, 50
    region_mask = np.zeros((H, W), dtype=bool)
    region_mask[5:45, 5:45] = True
    eroded = region_mask.copy()

    ridge_img = np.zeros((H, W), dtype=np.float32)
    # Add parallel straight ridge lines simulating cleavage cracks
    for row in (15, 25, 35):
        ridge_img[row, 10:40] = 10.0
    ridge_thresh = 1.0
    ppl_gray = np.full((H, W), 200.0, dtype=np.float32)

    results = []
    for _ in range(5):
        c = extractor._extract_cleavage(
            region_mask=region_mask,
            eroded=eroded,
            ppl_gray=ppl_gray,
            ridge_img=ridge_img,
            ridge_thresh=ridge_thresh,
            ext_mean_deg=0.0,
            rot_sign=1,
        )
        results.append((c["cleavage_strength"], c["n_cleavage_dirs"], c["cleavage_status"], c["cleavage_angle_deg"]))

    assert len(results) == 5
    assert len(set(results)) == 1, f"Cleavage detection varied across 5 runs: {results}"
    assert results[0][1] >= 1, f"Expected at least 1 cleavage direction detected, got {results[0]}"


def test_extract_single_angle_dark_in_frame_1_unresolved(extractor: FeatureExtractor):
    """Grain dark in XPL frame 1 but bright in other frames must have k1_optic_class == 'unresolved' (not 'isotropic')."""
    H, W = 30, 30
    label_img = np.zeros((H, W), dtype=np.int32)
    label_img[5:25, 5:25] = 1

    region_list = [{"region_id": "r_single_angle", "label_value": 1, "type": "grain"}]
    valid = np.ones((H, W), dtype=bool)

    # 36 frames: in frame 1, grain is dark (XPL=5).
    # In other frames (e.g. frame 0, frame 5, etc.), grain is bright (XPL=200).
    ppl = np.full((36, H, W, 3), 200, dtype=np.uint8)
    xpl = np.zeros((36, H, W, 3), dtype=np.uint8)
    xpl[0, 5:25, 5:25] = 200
    xpl[5, 5:25, 5:25] = 200
    xpl[1, 5:25, 5:25] = 5  # dark in primary test angle (frame 1)

    aligned = {
        "valid": valid,
        "ppl": ppl,
        "xpl": xpl,
        "theta_deg": np.linspace(0, 175, 36).astype(np.float32),
        "rot_sign": 1,
    }

    maps_dict = {
        "white_level": 255.0,
    }

    # Extract single angle at frame_idx=1
    rows = extractor.extract_single_angle(
        label_img=label_img,
        region_list=region_list,
        aligned=aligned,
        maps_dict=maps_dict,
        site_id="s_test",
        frame_idx=1,
    )

    assert len(rows) == 1
    r = rows[0]
    assert r["region_id"] == "r_single_angle"
    # Transparent in PPL, dark in XPL in frame 1
    assert r["k1_opaque_frac"] <= 0.8
    assert r["k1_xpl_bright_frac"] < 0.2
    # Single-angle optic class must be "unresolved", NOT "isotropic"!
    assert r["k1_optic_class"] == "unresolved", (
        f"Expected 'unresolved' k1_optic_class for grain dark in frame 1, got '{r['k1_optic_class']}'"
    )
    assert r["k1_optic_class_status"] == "unresolved", (
        f"Expected 'unresolved' status, got '{r['k1_optic_class_status']}'"
    )

    # Verify all 10 required contract columns are present and properly typed
    expected_cols = [
        "k1_ppl_L", "k1_ppl_a", "k1_ppl_b", "k1_ppl_chroma", "k1_ppl_hue_deg",
        "k1_ppl_colour", "k1_opaque_frac", "k1_xpl_bright_frac",
        "k1_optic_class", "k1_optic_class_status",
    ]
    for col in expected_cols:
        assert col in r, f"Missing required contract column {col}"
        if col.endswith("_status") or col.endswith("_colour") or col.endswith("_class"):
            assert isinstance(r[col], str), f"Column {col} should be str, got {type(r[col])}"
        else:
            assert isinstance(r[col], float), f"Column {col} should be float, got {type(r[col])}"


def test_extract_single_angle_colourless_with_colour_cast(extractor: FeatureExtractor):
    """Uniform colourless synthetic grain with a colour cast must be classified as 'colourless'."""
    H, W = 40, 40
    label_img = np.zeros((H, W), dtype=np.int32)
    label_img[10:30, 10:30] = 1

    region_list = [{"region_id": "r_cast", "label_value": 1, "type": "grain"}]
    valid = np.ones((H, W), dtype=bool)

    # Synthetic image with bluish colour cast: R=185, G=195, B=215 (L > 70)
    ppl = np.full((36, H, W, 3), [185, 195, 215], dtype=np.uint8)
    xpl = np.full((36, H, W, 3), 50, dtype=np.uint8)

    wb_gains = np.array([1.02, 1.0, 0.98], dtype=np.float32)

    aligned = {
        "valid": valid,
        "ppl": ppl,
        "xpl": xpl,
        "theta_deg": np.linspace(0, 175, 36).astype(np.float32),
        "rot_sign": 1,
    }
    maps_dict = {
        "white_level": 255.0,
        "wb_gains": wb_gains,
    }

    rows = extractor.extract_single_angle(
        label_img=label_img,
        region_list=region_list,
        aligned=aligned,
        maps_dict=maps_dict,
        site_id="s_cast",
        frame_idx=1,
    )
    assert len(rows) == 1
    r = rows[0]
    assert r["k1_ppl_chroma"] < 2.5, f"Expected chroma < 2.5, got {r['k1_ppl_chroma']}"
    assert r["k1_ppl_colour"] == "colourless", f"Expected 'colourless', got '{r['k1_ppl_colour']}'"


def test_extract_single_angle_isolated_from_other_frames(extractor: FeatureExtractor):
    """Zeroing out every other frame in the aligned stack leaves all k1_* outputs identical."""
    H, W = 40, 40
    label_img = np.zeros((H, W), dtype=np.int32)
    label_img[10:30, 10:30] = 1

    region_list = [{"region_id": "r_iso", "label_value": 1, "type": "grain"}]
    valid = np.ones((H, W), dtype=bool)

    # Multi-angle stack with varying values across frames
    ppl = np.random.RandomState(42).randint(50, 240, size=(36, H, W, 3)).astype(np.uint8)
    xpl = np.random.RandomState(43).randint(10, 100, size=(36, H, W, 3)).astype(np.uint8)

    aligned = {
        "valid": valid,
        "ppl": ppl,
        "xpl": xpl,
        "theta_deg": np.linspace(0, 175, 36).astype(np.float32),
        "rot_sign": 1,
    }

    # Extract at frame 1 with full stack
    rows_orig = extractor.extract_single_angle(
        label_img=label_img,
        region_list=region_list,
        aligned=aligned,
        site_id="s_iso",
        frame_idx=1,
    )

    # Zero out all frames other than frame 1
    ppl_mod = ppl.copy()
    xpl_mod = xpl.copy()
    for idx in range(36):
        if idx != 1:
            ppl_mod[idx] = 0
            xpl_mod[idx] = 0

    aligned_mod = {
        "valid": valid,
        "ppl": ppl_mod,
        "xpl": xpl_mod,
        "theta_deg": np.linspace(0, 175, 36).astype(np.float32),
        "rot_sign": 1,
    }

    rows_mod = extractor.extract_single_angle(
        label_img=label_img,
        region_list=region_list,
        aligned=aligned_mod,
        site_id="s_iso",
        frame_idx=1,
    )

    assert len(rows_orig) == 1 and len(rows_mod) == 1
    r_orig, r_mod = rows_orig[0], rows_mod[0]

    # Every k1_* output must stay identical
    k1_keys = [k for k in r_orig if k.startswith("k1_")]
    assert len(k1_keys) >= 10, f"Expected at least 10 k1_* keys, got {len(k1_keys)}"
    for k in k1_keys:
        assert r_orig[k] == r_mod[k], f"Key {k} differs after zeroing other frames: {r_orig[k]} vs {r_mod[k]}"



