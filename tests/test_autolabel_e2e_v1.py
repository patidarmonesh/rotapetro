"""End-to-end integration test for V1 autolabel / teacher generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

import importlib.util
from src.rotapetro.io import load_config
from src.rotapetro.provenance import current_teacher_fingerprint
from src.rotapetro.features import V1_STATUS_COLUMNS

def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

autolabel_mod = _load_script("autolabel_06", _PROJECT_ROOT / "scripts" / "06_autolabel.py")
_run_v1_teachers = autolabel_mod._run_v1_teachers


def test_autolabel_e2e_v1_run(tmp_path: Path, monkeypatch):
    """A tiny fixture (3 regions, 1 fold, temporary project root) running _run_v1_teachers.
    Asserts targets.parquet and teacher_params.json are written with matching fingerprint and teacher_version.
    """
    # 1. Setup temporary directory structure
    processed_v1 = tmp_path / "data" / "processed" / "v1"
    splits_dir = processed_v1 / "splits"
    splits_dir.mkdir(parents=True)
    interim_dir = tmp_path / "data" / "interim"
    interim_dir.mkdir(parents=True)
    teachers_dir = processed_v1 / "teachers"
    teachers_dir.mkdir(parents=True)

    # Copy knowledge files from real project root
    knowledge_dir = tmp_path / "knowledge"
    shutil.copytree(_PROJECT_ROOT / "knowledge", knowledge_dir)

    # 2. Setup specimens_clean.csv
    specimens_csv = interim_dir / "specimens_clean.csv"
    specimens_df = pd.DataFrame([
        {"code": "s1", "rock_forming": "quartz;biotite", "accessory": "zircon"},
        {"code": "s2", "rock_forming": "plagioclase;augite", "accessory": "apatite"},
    ])
    specimens_df.to_csv(specimens_csv, index=False)

    # 3. Setup folds.json and FOLDS_SHA256.txt
    folds_file = splits_dir / "folds.json"
    folds_data = {
        "fold_0": {
            "train": ["s1"],
            "val": [],
            "test": ["s2"],
        }
    }
    folds_json_bytes = json.dumps(folds_data, indent=2).encode("utf-8")
    folds_file.write_bytes(folds_json_bytes)
    folds_sha = hashlib.sha256(folds_json_bytes).hexdigest()
    (processed_v1 / "FOLDS_SHA256.txt").write_text(folds_sha, encoding="utf-8")

    # 4. Setup 3 dummy regions
    base_region = {
        "optic_class_status": "observed",
        "pleo_status": "unresolved",
        "absorption_status": "unresolved",
        "twin_status": "unresolved",
        "extinction_status": "unresolved",
        "cleavage_status": "unresolved",
        "small": False,
        "eligible_v1": True,
        "reject_reason_v1": "",
        "interior_px": 300,
        "area_px": 500,
        "aspect": 1.2,
        "r2_4_med": 0.8,
        "ppl_L": 70.0,
        "ppl_a": 0.0,
        "ppl_b": 0.0,
        "ppl_chroma": 1.0,
        "ppl_hue_deg": 0.0,
        "k1_ppl_L": 70.0,
        "k1_ppl_a": 0.0,
        "k1_ppl_b": 0.0,
        "k1_ppl_chroma": 1.0,
        "k1_ppl_hue_deg": 0.0,
        "k1_ppl_colour": "colourless",
        "k1_opaque_frac": 0.0,
        "k1_xpl_bright_frac": 0.5,
        "k1_optic_class": "anisotropic",
        "k1_optic_class_status": "observed",
        "pleo_cls": "none",
        "twin_type": "none",
        "ext_type": "parallel",
        "absorption_vs_long_deg": float("nan"),
    }

    r1 = dict(base_region, region_id="r1", site_id="s1", code="s1", type="grain", optic_class="anisotropic", ppl_colour="colourless")
    r2 = dict(base_region, region_id="r2", site_id="s1", code="s1", type="grain", optic_class="isotropic", ppl_colour="colourless", k1_optic_class="isotropic", r2_4_med=0.0)
    r3 = dict(base_region, region_id="r3", site_id="s2", code="s2", type="grain", optic_class="opaque", ppl_colour="black", k1_optic_class="opaque", k1_ppl_colour="black", ppl_L=5.0, k1_ppl_L=5.0, r2_4_med=0.0)

    regions_df = pd.DataFrame([r1, r2, r3])
    # Ensure all V1_STATUS_COLUMNS exist
    for col in V1_STATUS_COLUMNS:
        if col not in regions_df.columns:
            regions_df[col] = "unresolved"

    regions_file = processed_v1 / "regions.parquet"
    regions_df.to_parquet(regions_file, index=False)

    # 5. Load and update config
    cfg = load_config(_PROJECT_ROOT / "configs" / "default.yaml")
    cfg["paths"]["interim"] = str(interim_dir)
    cfg["paths"]["knowledge"] = str(_PROJECT_ROOT / "knowledge")
    cfg["v1"]["paths"]["processed"] = str(processed_v1)
    cfg["v1"]["paths"]["teachers"] = str(teachers_dir)

    # 6. Run _run_v1_teachers
    args = argparse.Namespace(force=True)
    _run_v1_teachers(cfg, args)

    # 7. Assert outputs exist
    fold_0_targets = teachers_dir / "fold_0" / "targets.parquet"
    fold_0_params = teachers_dir / "fold_0" / "teacher_params.json"
    assert fold_0_targets.is_file(), f"targets.parquet missing at {fold_0_targets}"
    assert fold_0_params.is_file(), f"teacher_params.json missing at {fold_0_params}"

    # 8. Assert matching fingerprint and teacher_version
    targets_out = pd.read_parquet(fold_0_targets)
    assert len(targets_out) == 3
    assert set(targets_out["region_id"]) == {"r1", "r2", "r3"}

    with open(fold_0_params, "r", encoding="utf-8") as pf:
        params_out = json.load(pf)

    cur_fp = current_teacher_fingerprint(cfg, project_root=_PROJECT_ROOT)
    assert params_out["fingerprint"] == cur_fp
    assert params_out["teacher_version"] == targets_out["teacher_version"].iloc[0]
    assert all(tv == params_out["teacher_version"] for tv in targets_out["teacher_version"])
    assert params_out["targets_sha256"] == hashlib.sha256(fold_0_targets.read_bytes()).hexdigest()


def test_autolabel_targets_checksum_mismatch_regenerates_fold(tmp_path: Path, caplog):
    """Generate a small teacher, edit only its targets.parquet, rerun _run_v1_teachers with force=False
    -> that fold is regenerated and its checksum matches again.
    """
    import logging

    # 1. Setup temporary directory structure
    processed_v1 = tmp_path / "data" / "processed" / "v1"
    splits_dir = processed_v1 / "splits"
    splits_dir.mkdir(parents=True)
    interim_dir = tmp_path / "data" / "interim"
    interim_dir.mkdir(parents=True)
    teachers_dir = processed_v1 / "teachers"
    teachers_dir.mkdir(parents=True)

    # Copy knowledge files from real project root
    knowledge_dir = tmp_path / "knowledge"
    shutil.copytree(_PROJECT_ROOT / "knowledge", knowledge_dir)

    # 2. Setup specimens_clean.csv
    specimens_csv = interim_dir / "specimens_clean.csv"
    specimens_df = pd.DataFrame([
        {"code": "s1", "rock_forming": "quartz;biotite", "accessory": "zircon"},
        {"code": "s2", "rock_forming": "plagioclase;augite", "accessory": "apatite"},
    ])
    specimens_df.to_csv(specimens_csv, index=False)

    # 3. Setup folds.json and FOLDS_SHA256.txt
    folds_file = splits_dir / "folds.json"
    folds_data = {
        "fold_0": {
            "train": ["s1"],
            "val": [],
            "test": ["s2"],
        }
    }
    folds_json_bytes = json.dumps(folds_data, indent=2).encode("utf-8")
    folds_file.write_bytes(folds_json_bytes)
    folds_sha = hashlib.sha256(folds_json_bytes).hexdigest()
    (processed_v1 / "FOLDS_SHA256.txt").write_text(folds_sha, encoding="utf-8")

    # 4. Setup 3 dummy regions
    base_region = {
        "optic_class_status": "observed",
        "pleo_status": "unresolved",
        "absorption_status": "unresolved",
        "twin_status": "unresolved",
        "extinction_status": "unresolved",
        "cleavage_status": "unresolved",
        "small": False,
        "eligible_v1": True,
        "reject_reason_v1": "",
        "interior_px": 300,
        "area_px": 500,
        "aspect": 1.2,
        "r2_4_med": 0.8,
        "ppl_L": 70.0,
        "ppl_a": 0.0,
        "ppl_b": 0.0,
        "ppl_chroma": 1.0,
        "ppl_hue_deg": 0.0,
        "k1_ppl_L": 70.0,
        "k1_ppl_a": 0.0,
        "k1_ppl_b": 0.0,
        "k1_ppl_chroma": 1.0,
        "k1_ppl_hue_deg": 0.0,
        "k1_ppl_colour": "colourless",
        "k1_opaque_frac": 0.0,
        "k1_xpl_bright_frac": 0.5,
        "k1_optic_class": "anisotropic",
        "k1_optic_class_status": "observed",
        "pleo_cls": "none",
        "twin_type": "none",
        "ext_type": "parallel",
        "absorption_vs_long_deg": float("nan"),
    }

    r1 = dict(base_region, region_id="r1", site_id="s1", code="s1", type="grain", optic_class="anisotropic", ppl_colour="colourless")
    r2 = dict(base_region, region_id="r2", site_id="s1", code="s1", type="grain", optic_class="isotropic", ppl_colour="colourless", k1_optic_class="isotropic", r2_4_med=0.0)
    r3 = dict(base_region, region_id="r3", site_id="s2", code="s2", type="grain", optic_class="opaque", ppl_colour="black", k1_optic_class="opaque", k1_ppl_colour="black", ppl_L=5.0, k1_ppl_L=5.0, r2_4_med=0.0)

    regions_df = pd.DataFrame([r1, r2, r3])
    for col in V1_STATUS_COLUMNS:
        if col not in regions_df.columns:
            regions_df[col] = "unresolved"

    regions_file = processed_v1 / "regions.parquet"
    regions_df.to_parquet(regions_file, index=False)

    # 5. Load and update config
    cfg = load_config(_PROJECT_ROOT / "configs" / "default.yaml")
    cfg["paths"]["interim"] = str(interim_dir)
    cfg["paths"]["knowledge"] = str(_PROJECT_ROOT / "knowledge")
    cfg["v1"]["paths"]["processed"] = str(processed_v1)
    cfg["v1"]["paths"]["teachers"] = str(teachers_dir)

    # 6. Generate initial teacher
    args_force = argparse.Namespace(force=True)
    _run_v1_teachers(cfg, args_force)

    fold_0_targets = teachers_dir / "fold_0" / "targets.parquet"
    fold_0_params = teachers_dir / "fold_0" / "teacher_params.json"
    initial_sha = hashlib.sha256(fold_0_targets.read_bytes()).hexdigest()
    with open(fold_0_params, "r", encoding="utf-8") as pf:
        initial_params = json.load(pf)
    assert initial_params["targets_sha256"] == initial_sha

    # 7. Edit ONLY its targets.parquet
    targets_df = pd.read_parquet(fold_0_targets)
    orig_state = targets_df.loc[0, "state"]
    tampered_state = "partial_set" if orig_state != "partial_set" else "singleton"
    targets_df.loc[0, "state"] = tampered_state
    targets_df.to_parquet(fold_0_targets, index=False)
    tampered_sha = hashlib.sha256(fold_0_targets.read_bytes()).hexdigest()
    assert tampered_sha != initial_sha

    # 8. Rerun _run_v1_teachers with force=False
    with caplog.at_level(logging.WARNING):
        args_no_force = argparse.Namespace(force=False)
        _run_v1_teachers(cfg, args_no_force)

    # Assert regeneration logged
    assert "targets file changed since generation; regenerating" in caplog.text

    # 9. Assert that fold is regenerated and its checksum matches again
    regenerated_sha = hashlib.sha256(fold_0_targets.read_bytes()).hexdigest()
    with open(fold_0_params, "r", encoding="utf-8") as pf:
        regenerated_params = json.load(pf)

    assert regenerated_params["targets_sha256"] == regenerated_sha
    assert regenerated_sha == initial_sha

    # And verify the edit was overwritten
    targets_regenerated = pd.read_parquet(fold_0_targets)
    assert targets_regenerated.loc[0, "state"] == orig_state

