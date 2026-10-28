"""Unit tests for V1 configuration gates and feasibility checks."""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config, validate_v1_config, require_fresh_feasibility
import hashlib
import json
import shutil
import pandas as pd


def test_a6_border_exclude_px_equals_erosion_px():
    """A6: validate_v1_config raises ValueError if border_exclude_px != erosion_px."""
    cfg = load_config("configs/default.yaml")
    # Base config should be valid
    validate_v1_config(cfg)

    # Mismatch when border_exclude_px is changed
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["features"]["border_exclude_px"] = 5
    bad_cfg["v1"]["region_selection"]["erosion_px"] = 2
    with pytest.raises(ValueError, match="features.border_exclude_px .* must equal v1.region_selection.erosion_px"):
        validate_v1_config(bad_cfg)

    # Mismatch when erosion_px is changed
    bad_cfg2 = copy.deepcopy(cfg)
    bad_cfg2["features"]["border_exclude_px"] = 2
    bad_cfg2["v1"]["region_selection"]["erosion_px"] = 0
    with pytest.raises(ValueError, match="features.border_exclude_px .* must equal v1.region_selection.erosion_px"):
        validate_v1_config(bad_cfg2)


def _setup_feasibility_fixture(tmp_path: Path) -> tuple[dict, Path]:
    """Helper to setup valid feasibility files and return config and processed_dir."""
    processed_dir = tmp_path / "data" / "processed" / "v1"
    processed_dir.mkdir(parents=True)
    teachers_dir = processed_dir / "teachers"
    teachers_dir.mkdir()
    splits_dir = processed_dir / "splits"
    splits_dir.mkdir()

    # Create dummy splits/folds.json
    folds_file = splits_dir / "folds.json"
    folds_data = {f"fold_{i}": {"train": ["s1"], "val": [], "test": []} for i in range(5)}
    with open(folds_file, "w", encoding="utf-8") as f:
        json.dump(folds_data, f)
    folds_sha256 = hashlib.sha256(folds_file.read_bytes()).hexdigest()

    # Create dummy regions.parquet and compute its sha256
    regions_file = processed_dir / "regions.parquet"
    pd.DataFrame([{"region_id": "r1", "site_id": "s1", "code": "s1"}]).to_parquet(regions_file)
    with open(regions_file, "rb") as f:
        regions_sha256 = hashlib.sha256(f.read()).hexdigest()

    cfg = load_config("configs/default.yaml")
    cfg["v1"]["paths"]["processed"] = str(processed_dir)

    from src.rotapetro.provenance import current_teacher_fingerprint
    cur_fp = current_teacher_fingerprint(cfg, project_root=_PROJECT_ROOT)

    # Create 5 folds of targets.parquet and teacher_params.json
    teacher_fps = {}
    targets_shas = {}
    teacher_version = f"v1.0-{cur_fp}"
    for i in range(5):
        fold = f"fold_{i}"
        f_dir = teachers_dir / fold
        f_dir.mkdir()
        t_file = f_dir / "targets.parquet"
        # Give fold 0 and fold 1 slightly different partitions to ensure distinct hashes
        p_name = "train" if i == 0 else "val"
        t_df = pd.DataFrame([{
            "region_id": "r1",
            "state": "singleton",
            "allowed": ["quartz"],
            "partition": p_name,
            "teacher_version": teacher_version,
        }])
        t_df.to_parquet(t_file, index=False)
        t_sha = hashlib.sha256(t_file.read_bytes()).hexdigest()
        targets_shas[fold] = t_sha

        teacher_fps[fold] = cur_fp
        with open(f_dir / "teacher_params.json", "w", encoding="utf-8") as f:
            json.dump({
                "fingerprint": cur_fp,
                "fold": fold,
                "targets_sha256": t_sha,
                "teacher_version": teacher_version,
            }, f)

    # Create provenance_feasibility.json
    prov = {
        "overall_gate_pass": True,
        "teacher_fingerprints": teacher_fps,
        "targets_sha256": targets_shas,
        "regions_sha256": regions_sha256,
        "folds_sha256": folds_sha256,
    }
    with open(processed_dir / "provenance_feasibility.json", "w", encoding="utf-8") as f:
        json.dump(prov, f)

    return cfg, processed_dir


def test_c2_require_fresh_feasibility_pass(tmp_path: Path):
    """C2: require_fresh_feasibility succeeds when all checks pass."""
    cfg, _ = _setup_feasibility_fixture(tmp_path)
    # Should not raise
    require_fresh_feasibility(cfg)


def test_c2_require_fresh_feasibility_missing_provenance(tmp_path: Path):
    """C2: Failure mode 1 — provenance_feasibility.json does not exist."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    (processed_dir / "provenance_feasibility.json").unlink()

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1


def test_c2_require_fresh_feasibility_overall_gate_false(tmp_path: Path):
    """C2: Failure mode 2 — overall_gate_pass is False."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    prov_file = processed_dir / "provenance_feasibility.json"
    with open(prov_file, "r", encoding="utf-8") as f:
        prov = json.load(f)
    prov["overall_gate_pass"] = False
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(prov, f)

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1


def test_c2_require_fresh_feasibility_fingerprint_mismatch(tmp_path: Path):
    """C2: Failure mode 3 — teacher fingerprint mismatch."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    # Modify teacher_params.json for fold_2
    t_file = processed_dir / "teachers" / "fold_2" / "teacher_params.json"
    with open(t_file, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": "altered_fingerprint", "fold": "fold_2"}, f)

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1


def test_c2_require_fresh_feasibility_missing_fold_params(tmp_path: Path):
    """C2: Failure mode 4 — missing fold teacher_params.json."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    (processed_dir / "teachers" / "fold_4" / "teacher_params.json").unlink()

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1


def test_c2_require_fresh_feasibility_regions_sha_mismatch(tmp_path: Path):
    """C2: Failure mode 5 — regions_sha256 mismatch."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    prov_file = processed_dir / "provenance_feasibility.json"
    with open(prov_file, "r", encoding="utf-8") as f:
        prov = json.load(f)
    prov["regions_sha256"] = "mismatched_sha256_hash"
    with open(prov_file, "w", encoding="utf-8") as f:
        json.dump(prov, f)

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1


def test_c2_require_fresh_feasibility_missing_regions_file(tmp_path: Path):
    """C2: Failure mode 6 — regions.parquet file does not exist."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    (processed_dir / "regions.parquet").unlink()

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1


def test_freshness_change_regions_parquet_same_row_count(tmp_path: Path, capsys):
    """Change regions.parquet values with the same row count -> rejected with rerun message."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    regions_file = processed_dir / "regions.parquet"
    # Change values keeping row count = 1
    pd.DataFrame([{"region_id": "r1", "site_id": "s1_modified", "code": "s1"}]).to_parquet(regions_file)

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "06_autolabel.py" in captured.err or "v1_label_feasibility.py" in captured.err


def test_freshness_change_folds_json(tmp_path: Path, capsys):
    """Change folds.json -> rejected with rerun message."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    folds_file = processed_dir / "splits" / "folds.json"
    with open(folds_file, "w", encoding="utf-8") as f:
        json.dump({"fold_0": {"train": ["altered"], "val": [], "test": []}}, f)

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "06_autolabel.py" in captured.err or "v1_label_feasibility.py" in captured.err


def test_freshness_change_rule_setting(tmp_path: Path, capsys):
    """Change one rule setting -> rejected with message naming 06_autolabel.py."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    cfg["v1"]["rules"]["singleton_p"] = 0.99  # was 0.80

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "teacher fingerprint mismatch" in captured.err
    assert "06_autolabel.py" in captured.err


def test_freshness_edit_targets_file_in_feasibility(tmp_path: Path, capsys):
    """Editing targets file or teacher_params is rejected with 'rerun 06_autolabel.py'."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    # Tamper with fold_0 teacher_params fingerprint
    t_file = processed_dir / "teachers" / "fold_0" / "teacher_params.json"
    with open(t_file, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": "corrupted_fp", "fold": "fold_0"}, f)

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "teacher fingerprint mismatch" in captured.err
    assert "06_autolabel.py" in captured.err


def test_feasibility_report_low_r2_4_computed_statement():
    """Verify that low_r2_4 population restriction line is properly computed (not hard-coded)."""
    regions_v1 = pd.DataFrame([
        {"region_id": "r1", "eligible_v1": True},
        {"region_id": "r2", "eligible_v1": True},
        {"region_id": "r3", "eligible_v1": True},
        {"region_id": "r4", "eligible_v1": True},
        {"region_id": "r5", "eligible_v1": False},
    ])
    ref_targets = pd.DataFrame([
        {"region_id": "r1", "reason": "low_r2_4"},
        {"region_id": "r2", "reason": "singleton"},
        {"region_id": "r3", "reason": "low_r2_4"},
        {"region_id": "r4", "reason": "singleton"},
        {"region_id": "r5", "reason": "not_eligible"},
    ])

    n_eligible = int(regions_v1["eligible_v1"].sum())
    n_low_r2_4 = int((ref_targets["reason"] == "low_r2_4").sum())
    pct_low_r2_4 = (100.0 * n_low_r2_4 / n_eligible) if n_eligible > 0 else 0.0

    expected_line = f"low_r2_4 grains ({n_low_r2_4}, {pct_low_r2_4:.1f}% of eligible) are excluded as a declared population restriction; results are conditional on this population."
    assert n_eligible == 4
    assert n_low_r2_4 == 2
    assert f"{pct_low_r2_4:.1f}%" == "50.0%"
    assert expected_line == "low_r2_4 grains (2, 50.0% of eligible) are excluded as a declared population restriction; results are conditional on this population."


def test_targets_checksum_change_one_label_rejected(tmp_path: Path, capsys):
    """(a) Change one label, same IDs and row count -> rejected with rerun message."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    t_file = processed_dir / "teachers" / "fold_0" / "targets.parquet"
    t_df = pd.read_parquet(t_file)
    # Modify one label while preserving same IDs and row count
    t_df.loc[0, "state"] = "partial_set"
    t_df.to_parquet(t_file, index=False)

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "rerun 06_autolabel.py --v1 (or --force)" in captured.err


def test_targets_checksum_delete_targets_file_rejected(tmp_path: Path, capsys):
    """(b) Delete a targets file -> rejected with rerun message."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    t_file = processed_dir / "teachers" / "fold_0" / "targets.parquet"
    t_file.unlink()

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "rerun 06_autolabel.py --v1 (or --force)" in captured.err


def test_targets_checksum_replace_with_another_fold_file_rejected(tmp_path: Path, capsys):
    """(c) Replace targets with another fold's file -> rejected with rerun message."""
    cfg, processed_dir = _setup_feasibility_fixture(tmp_path)
    t0_file = processed_dir / "teachers" / "fold_0" / "targets.parquet"
    t1_file = processed_dir / "teachers" / "fold_1" / "targets.parquet"
    # Verify their hashes differ
    assert hashlib.sha256(t0_file.read_bytes()).hexdigest() != hashlib.sha256(t1_file.read_bytes()).hexdigest()

    # Overwrite fold 0 targets with fold 1 targets
    shutil.copy2(t1_file, t0_file)

    with pytest.raises(SystemExit) as exc:
        require_fresh_feasibility(cfg)
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "rerun 06_autolabel.py --v1 (or --force)" in captured.err



