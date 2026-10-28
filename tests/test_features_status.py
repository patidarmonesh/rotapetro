"""Unit tests for V1 feature extraction status conventions (Test C11).

Tests:
- Test C11: Build a tiny synthetic 50x50 test region with synthetic maps (anisotropic,
  isotropic, opaque). Verify that all 6 columns in V1_STATUS_COLUMNS are extracted and
  non-empty. Verify that an anisotropic grain with twin_type == 'none' gets
  twin_status == 'unresolved'. Verify absorption_status behavior.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config
from src.rotapetro.features import FeatureExtractor, V1_STATUS_COLUMNS


@pytest.fixture
def synthetic_site_data():
    """Create a 50x50 synthetic site with anisotropic, isotropic, opaque, and pleochroic regions."""
    H, W = 50, 50
    label_img = np.zeros((H, W), dtype=np.int32)

    # 1. Anisotropic grain, no pleochroism, equant
    label_img[2:18, 2:18] = 1  # 16x16, area 256

    # 2. Isotropic grain
    label_img[2:18, 24:40] = 2  # 16x16, area 256

    # 3. Opaque grain
    label_img[24:40, 2:18] = 3  # 16x16, area 256

    # 4. Anisotropic elongated grain with pleochroism (aspect > 1.3)
    label_img[24:48, 22:30] = 4  # 24x8, aspect ~ 3.0

    # 5. Anisotropic equant grain with pleochroism (aspect ~ 1.0)
    label_img[24:40, 34:50] = 5  # 16x16, aspect ~ 1.0

    region_list = [
        {"region_id": "r_aniso", "label_value": 1, "type": "grain"},
        {"region_id": "r_iso", "label_value": 2, "type": "grain"},
        {"region_id": "r_opaque", "label_value": 3, "type": "opaque"},
        {"region_id": "r_pleo_elongated", "label_value": 4, "type": "grain"},
        {"region_id": "r_pleo_equant", "label_value": 5, "type": "grain"},
    ]

    white_level = 255.0
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
        "white_level": white_level,
        "xpl_rel_rgb": np.zeros((H, W, 3), dtype=np.float32),
        "aniso_mask": np.zeros((H, W), dtype=bool),
    }

    # Region 1 (anisotropic, equant, colourless)
    maps["aniso"][2:18, 2:18] = 0.6
    maps["aniso_mask"][2:18, 2:18] = True
    maps["r2_4"][2:18, 2:18] = 0.9
    maps["ext_deg"][2:18, 2:18] = 30.0
    maps["gamma_est_nm"][2:18, 2:18] = 200.0
    maps["biref_cls"][2:18, 2:18] = 1

    # Region 2 (isotropic)
    maps["aniso"][2:18, 24:40] = 0.05
    maps["aniso_mask"][2:18, 24:40] = False
    maps["r2_4"][2:18, 24:40] = 0.1

    # Region 3 (opaque)
    maps["ppl_max_L"][24:40, 2:18] = 10.0  # < 0.15 * white_level
    maps["aniso"][24:40, 2:18] = 0.0
    maps["aniso_mask"][24:40, 2:18] = False

    # Region 4 (pleochroic, elongated)
    maps["aniso"][24:48, 22:30] = 0.6
    maps["aniso_mask"][24:48, 22:30] = True
    maps["r2_4"][24:48, 22:30] = 0.9
    maps["ext_deg"][24:48, 22:30] = 30.0
    maps["pleo_dE"][24:48, 22:30] = 12.0  # pleo_cls != 'none'
    maps["pleo_dark_deg"][24:48, 22:30] = 45.0

    # Region 5 (pleochroic, equant)
    maps["aniso"][24:40, 34:50] = 0.6
    maps["aniso_mask"][24:40, 34:50] = True
    maps["r2_4"][24:40, 34:50] = 0.9
    maps["ext_deg"][24:40, 34:50] = 30.0
    maps["pleo_dE"][24:40, 34:50] = 12.0
    maps["pleo_dark_deg"][24:40, 34:50] = 45.0

    return label_img, region_list, aligned, maps


def test_c11_features_status_extraction(synthetic_site_data):
    """Test C11: verify V1_STATUS_COLUMNS extraction, anisotropic twin unresolved, and absorption_status."""
    label_img, region_list, aligned, maps = synthetic_site_data
    cfg = load_config("configs/default.yaml")
    extractor = FeatureExtractor(cfg)

    extracted = extractor.extract_site(
        label_img=label_img,
        region_list=region_list,
        aligned=aligned,
        maps=maps,
        site_id="synthetic_site",
        code="synth_code",
        um_per_px=4.0,
    )

    assert len(extracted) == len(region_list)
    by_id = {r["region_id"]: r for r in extracted}

    # 1. Verify all 6 status columns are extracted and non-empty for all regions
    assert len(V1_STATUS_COLUMNS) == 6
    for rid, r in by_id.items():
        for col in V1_STATUS_COLUMNS:
            assert col in r, f"Missing status column {col} in region {rid}"
            val = r[col]
            assert val is not None and val != "", f"Empty status column {col} in region {rid}"
            assert isinstance(val, str), f"Status column {col} must be string, got {type(val)}"

    # 2. Region 1: Anisotropic grain with twin_type == 'none' gets twin_status == 'unresolved'
    r1 = by_id["r_aniso"]
    assert r1["optic_class"] == "anisotropic", f"Expected anisotropic, got {r1['optic_class']}"
    assert r1["twin_type"] == "none", f"Expected twin_type none, got {r1['twin_type']}"
    assert r1["twin_status"] == "unresolved", (
        f"Expected twin_status unresolved for anisotropic grain with twin_type none, got {r1['twin_status']}"
    )

    # 3. Region 2: Isotropic grain
    r2 = by_id["r_iso"]
    assert r2["optic_class"] == "isotropic"
    assert r2["twin_status"] == "not_applicable"
    assert r2["extinction_status"] == "not_applicable"

    # 4. Region 3: Opaque grain
    r3 = by_id["r_opaque"]
    assert r3["optic_class"] == "opaque"
    assert r3["absorption_status"] == "not_applicable"
    assert r3["twin_status"] == "not_applicable"

    # 5. Absorption status behavior:
    # - Non-pleochroic: not_applicable
    assert r1["absorption_status"] == "not_applicable"

    # - Pleochroic with long axis (elongated): observed
    r4 = by_id["r_pleo_elongated"]
    assert r4["pleo_cls"] != "none"
    assert not np.isnan(r4["long_axis_deg"])
    assert r4["absorption_status"] == "observed", (
        f"Expected absorption_status observed for elongated pleochroic grain, got {r4['absorption_status']}"
    )

    # - Pleochroic equant (long_axis is NaN): unresolved
    r5 = by_id["r_pleo_equant"]
    assert r5["pleo_cls"] != "none"
    assert np.isnan(r5["long_axis_deg"])
    assert r5["absorption_status"] == "unresolved", (
        f"Expected absorption_status unresolved for equant pleochroic grain, got {r5['absorption_status']}"
    )
