"""Tests for features provenance hardening and site updates (Task A2).

Requirements tested:
(a) Missing provenance re-extracts all sites.
(b) Malformed JSON in features_provenance.json rebuilds all sites.
(c) --site on fingerprint mismatch exits with error message.
(d) No provenance written without extraction.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _PROJECT_ROOT / "scripts" / "04_features.py"


def _create_synthetic_dataset(tmp_path: Path):
    """Create a minimal runnable dataset for 04_features.py."""
    interim_dir = tmp_path / "interim"
    processed_dir = tmp_path / "processed"
    aligned_dir = interim_dir / "aligned"
    maps_dir = interim_dir / "maps"
    seg_dir = interim_dir / "segments"
    qc_dir = interim_dir / "qc"

    for d in (aligned_dir, maps_dir, seg_dir, qc_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Base config loaded from default.yaml with paths redirected
    from src.rotapetro.io import load_config
    cfg = load_config(_PROJECT_ROOT / "configs" / "default.yaml")
    cfg["paths"]["interim"] = str(interim_dir.as_posix())
    cfg["v1"]["paths"]["processed"] = str(processed_dir.as_posix())
    cfg_path = tmp_path / "test_config.yaml"
    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)

    # 2. sites.csv
    sites_df = pd.DataFrame([
        {"site_id": "site_1", "usable": True, "code": "site_1"},
        {"site_2": "site_2", "usable": True, "code": "site_2", "site_id": "site_2"},
    ])
    sites_df.to_csv(interim_dir / "sites.csv", index=False)

    # 3. Create dummy data for site_1 and site_2
    H, W = 20, 20
    for sid in ("site_1", "site_2"):
        ppl = np.full((36, H, W, 3), 200, dtype=np.uint8)
        xpl = np.full((36, H, W, 3), 50, dtype=np.uint8)
        valid = np.ones((H, W), dtype=bool)
        theta_deg = np.linspace(0, 175, 36).astype(np.float32)
        np.savez_compressed(
            aligned_dir / f"{sid}.npz",
            ppl=ppl,
            xpl=xpl,
            theta_deg=theta_deg,
            valid=valid,
        )

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
            "white_level": np.float32(255.0),
            "xpl_rel_rgb": np.zeros((H, W, 3), dtype=np.float32),
            "aniso_mask": np.zeros((H, W), dtype=bool),
        }
        np.savez_compressed(maps_dir / f"{sid}.npz", **maps)

        label_img = np.zeros((H, W), dtype=np.uint8)
        label_img[2:18, 2:18] = 1
        cv2.imwrite(str(seg_dir / f"{sid}.png"), label_img)

        seg_json = {
            "regions": [
                {"region_id": f"{sid}_r1", "label_value": 1, "type": "grain"}
            ]
        }
        with open(seg_dir / f"{sid}.json", "w", encoding="utf-8") as f:
            json.dump(seg_json, f)

    return cfg_path, interim_dir, processed_dir


def test_missing_provenance_re_extracts(tmp_path: Path):
    """(a) If features_provenance.json is missing, all sites must be re-extracted."""
    cfg_path, interim_dir, processed_dir = _create_synthetic_dataset(tmp_path)
    out_parquet = processed_dir / "regions.parquet"
    prov_json = processed_dir / "features_provenance.json"

    # Run initial extraction
    res = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0, f"Extraction failed: {res.stderr}\n{res.stdout}"
    assert out_parquet.is_file()
    assert prov_json.is_file()

    # Remove provenance file
    prov_json.unlink()
    assert not prov_json.is_file()

    # Run again: should notice missing provenance and re-extract all sites
    res2 = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert res2.returncode == 0, f"Re-extraction failed: {res2.stderr}\n{res2.stdout}"
    # Verify both sites were processed rather than skipped
    assert "site_1: extracting features" in res2.stderr or "site_1: extracting features" in res2.stdout
    assert "site_2: extracting features" in res2.stderr or "site_2: extracting features" in res2.stdout
    assert prov_json.is_file()


def test_malformed_json_rebuilds(tmp_path: Path):
    """(b) If features_provenance.json is malformed or lacks fingerprint, rebuild all sites."""
    cfg_path, interim_dir, processed_dir = _create_synthetic_dataset(tmp_path)
    out_parquet = processed_dir / "regions.parquet"
    prov_json = processed_dir / "features_provenance.json"

    # Initial extraction
    res = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    assert prov_json.is_file()

    # Write malformed JSON
    with open(prov_json, "w", encoding="utf-8") as f:
        f.write("{corrupted json contents")

    res2 = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert res2.returncode == 0, f"Run failed: {res2.stderr}\n{res2.stdout}"
    assert "site_1: extracting features" in res2.stderr or "site_1: extracting features" in res2.stdout
    assert "site_2: extracting features" in res2.stderr or "site_2: extracting features" in res2.stdout

    # Now write valid JSON but with no "fingerprint" key
    with open(prov_json, "w", encoding="utf-8") as f:
        json.dump({"other_key": 123}, f)

    res3 = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert res3.returncode == 0
    assert "site_1: extracting features" in res3.stderr or "site_1: extracting features" in res3.stdout
    assert "site_2: extracting features" in res3.stderr or "site_2: extracting features" in res3.stdout


def test_site_guard_fingerprint_mismatch_exits(tmp_path: Path):
    """(c) --site on fingerprint mismatch exits with error."""
    cfg_path, interim_dir, processed_dir = _create_synthetic_dataset(tmp_path)
    out_parquet = processed_dir / "regions.parquet"
    prov_json = processed_dir / "features_provenance.json"

    # Initial extraction
    res = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert res.returncode == 0
    assert out_parquet.is_file()

    # Tamper fingerprint to an obsolete value
    with open(prov_json, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": "obsolete_fp_123"}, f)

    # Attempt single-site update
    res_site = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--config", str(cfg_path), "--site", "site_1"],
        capture_output=True, text=True,
    )
    assert res_site.returncode != 0
    expected_msg = "Feature code/config changed since regions.parquet was built; a single-site update would mix versions. Do a full rebuild."
    combined_output = res_site.stderr + "\n" + res_site.stdout
    assert expected_msg in combined_output, f"Expected error message not found in output: {combined_output}"


def test_no_provenance_written_without_extraction(tmp_path: Path):
    """(d) No provenance written without extraction."""
    cfg_path, interim_dir, processed_dir = _create_synthetic_dataset(tmp_path)
    out_parquet = processed_dir / "regions.parquet"
    prov_json = processed_dir / "features_provenance.json"

    # Create empty regions.parquet directly, with NO extraction occurring
    df = pd.DataFrame([{"region_id": "dummy_1", "site_id": "site_1"}])
    df.to_parquet(out_parquet, index=False)
    assert out_parquet.is_file()
    assert not prov_json.is_file()

    # Clear sites.csv so usable sites is empty -> 0 sites extracted
    empty_sites = pd.DataFrame(columns=["site_id", "usable", "code"])
    empty_sites.to_csv(interim_dir / "sites.csv", index=False)

    # Run script: 0 sites processed, no rows extracted
    res = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    # Provenance file MUST NOT be created / stamped
    assert not prov_json.is_file(), "Provenance was written even though no features were extracted!"


def test_v1_features_n_err_exits_code_1(tmp_path: Path):
    """In V1, if n_err > 0 at end of 04_features.py, exits with code 1 and lists failed sites."""
    cfg_path, interim_dir, processed_dir = _create_synthetic_dataset(tmp_path)

    # Intentionally corrupt site_2 by deleting its aligned file
    corrupt_aligned = interim_dir / "aligned" / "site_2.npz"
    corrupt_aligned.unlink()

    res = subprocess.run(
        [sys.executable, str(_SCRIPT_PATH), "--config", str(cfg_path)],
        capture_output=True, text=True,
    )
    assert res.returncode == 1, f"Expected returncode 1, got {res.returncode}. Output:\n{res.stdout}\n{res.stderr}"
    combined_output = res.stderr + "\n" + res.stdout
    assert "site_2" in combined_output, f"Expected failed site 'site_2' in output:\n{combined_output}"
    assert "Feature extraction failed on 1 sites in V1" in combined_output
