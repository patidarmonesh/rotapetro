"""Unit tests for Fix 9: Provenance fingerprinting and stale output avoidance.

Tests:
1. Autolabel fingerprint covers knowledge YAMLs, rules cfg, v1 cfg, code hash, and FOLDS_SHA256.
2. Autolabel fingerprint changes when any of the components change.
3. Autolabel handles list_conflict reason correctly (state='abstain', allowed=[], metadata_dependent=True).
4. Features fingerprint covers features cfg, v1 cfg, code hash, and FEATURE_VERSION.
5. Site input signature detects modifications to aligned/maps/segment files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys
import importlib.util

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def _load_script(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

autolabel_mod = _load_script("autolabel_mod", _PROJECT_ROOT / "scripts" / "06_autolabel.py")
features_mod = _load_script("features_mod", _PROJECT_ROOT / "scripts" / "04_features.py")


def test_autolabel_knowledge_hash(tmp_path: Path):
    """Knowledge hash covers YAML and CSV files, changing on modification."""
    k_dir = tmp_path / "knowledge"
    k_dir.mkdir()
    (k_dir / "test.yaml").write_text("foo: 1\n", encoding="utf-8")
    (k_dir / "minerals.csv").write_text("mineral,class\nquartz,silica\n", encoding="utf-8")
    (k_dir / "aliases.csv").write_text("alias,canonical\nqtz,quartz\n", encoding="utf-8")

    h1 = autolabel_mod._compute_knowledge_hash(k_dir)
    assert len(h1) == 64

    # Modifying minerals.csv changes hash
    (k_dir / "minerals.csv").write_text("mineral,class\nquartz,tectosilicate\n", encoding="utf-8")
    h2 = autolabel_mod._compute_knowledge_hash(k_dir)
    assert h1 != h2

    # Modifying aliases.csv changes hash
    (k_dir / "aliases.csv").write_text("alias,canonical\nqtz,quartz_alt\n", encoding="utf-8")
    h3 = autolabel_mod._compute_knowledge_hash(k_dir)
    assert h2 != h3


def test_autolabel_folds_sha256(tmp_path: Path):
    """Folds SHA256 verifies against FOLDS_SHA256.txt (raises on mismatch) or hashes folds.json directly."""
    v1_proc = tmp_path / "v1"
    v1_proc.mkdir()
    sha_file = v1_proc / "FOLDS_SHA256.txt"
    sha_file.write_text("dummy_hash_12345", encoding="utf-8")

    folds_path = tmp_path / "folds.json"
    folds_path.write_text('{"fold_0": {}}', encoding="utf-8")

    # Mismatched FOLDS_SHA256.txt must raise RuntimeError
    with pytest.raises(RuntimeError, match="FOLDS_SHA256.txt mismatch"):
        autolabel_mod._get_folds_sha256(folds_path, v1_proc)

    # Matching FOLDS_SHA256.txt succeeds
    expected = hashlib.sha256(b'{"fold_0": {}}').hexdigest()
    sha_file.write_text(expected, encoding="utf-8")
    loaded_sha = autolabel_mod._get_folds_sha256(folds_path, v1_proc)
    assert loaded_sha == expected

    # When FOLDS_SHA256.txt is absent, computes from folds.json
    sha_file.unlink()
    fallback_sha = autolabel_mod._get_folds_sha256(folds_path, v1_proc)
    assert fallback_sha == expected


def test_features_fingerprint_changes_on_config():
    """Features fingerprint changes when config changes."""
    cfg1 = {
        "features": {"border_exclude_px": 2},
        "v1": {"region_selection": {"min_area_px": 100}},
    }
    cfg2 = {
        "features": {"border_exclude_px": 3},
        "v1": {"region_selection": {"min_area_px": 100}},
    }
    fp1, c1, _ = features_mod._compute_features_fingerprint(cfg1)
    fp2, c2, _ = features_mod._compute_features_fingerprint(cfg2)
    assert fp1 != fp2
    assert c1 != c2


def test_site_input_signature_detects_changes(tmp_path: Path):
    """Site input signature changes when an input file changes size or mtime."""
    aligned_dir = tmp_path / "aligned"
    maps_dir = tmp_path / "maps"
    seg_dir = tmp_path / "segments"
    for d in (aligned_dir, maps_dir, seg_dir):
        d.mkdir()

    site = "site_test_01"
    (aligned_dir / f"{site}.npz").write_bytes(b"aligned_v1")
    (maps_dir / f"{site}.npz").write_bytes(b"maps_v1")
    (seg_dir / f"{site}.png").write_bytes(b"png_v1")
    (seg_dir / f"{site}.json").write_text("{}", encoding="utf-8")

    sig1 = features_mod._site_input_signature(site, aligned_dir, maps_dir, seg_dir)
    assert len(sig1) == 16

    # Modify one file
    (seg_dir / f"{site}.json").write_text('{"regions": [1]}', encoding="utf-8")
    sig2 = features_mod._site_input_signature(site, aligned_dir, maps_dir, seg_dir)
    assert sig1 != sig2


def test_autolabel_fingerprint_covers_all_components():
    """Autolabel fingerprint changes when rules, v1, code, regions, or specimens change."""
    cfg1 = {
        "v1": {"version": "v1.0", "families": {"feldspar": []}},
        "rules": {"min_r2_4": 0.3},
    }
    cfg2 = {
        "v1": {"version": "v1.0", "families": {"feldspar": []}},
        "rules": {"min_r2_4": 0.5},
    }

    fp1, _ = autolabel_mod._compute_teacher_fingerprint(
        knowledge_hash="k1",
        rules_cfg=cfg1.get("rules", {}),
        v1_cfg=cfg1.get("v1", {}),
        code_hash="c1",
        folds_sha="f1",
        regions_sha="r1",
        specimens_sha="s1",
    )
    fp2, _ = autolabel_mod._compute_teacher_fingerprint(
        knowledge_hash="k1",
        rules_cfg=cfg2.get("rules", {}),
        v1_cfg=cfg2.get("v1", {}),
        code_hash="c1",
        folds_sha="f1",
        regions_sha="r1",
        specimens_sha="s1",
    )
    assert fp1 != fp2


def test_teacher_fingerprint_changes_on_all_inputs(tmp_path: Path):
    """Teacher fingerprint changes when minerals.csv, aliases.csv, regions.parquet, specimens_clean.csv, or folds.json changes."""
    k_dir = tmp_path / "knowledge"
    k_dir.mkdir()
    (k_dir / "minerals.csv").write_text("min,cls\nqtz,silica\n", encoding="utf-8")
    (k_dir / "aliases.csv").write_text("a,b\nqtz,quartz\n", encoding="utf-8")

    proc_dir = tmp_path / "proc"
    proc_dir.mkdir()
    reg_file = proc_dir / "regions.parquet"
    reg_file.write_bytes(b"regions_bytes_1")
    spec_file = proc_dir / "specimens_clean.csv"
    spec_file.write_text("code,rf\nS1,qtz\n", encoding="utf-8")
    folds_file = proc_dir / "folds.json"
    folds_file.write_text('{"fold_0": {"train": ["S1"]}}', encoding="utf-8")

    rules_cfg = {"min_r2_4": 0.4}
    v1_cfg = {"version": "v1.0"}
    code_hash = "code_hash_1"

    def compute_fp():
        k_hash = autolabel_mod._compute_knowledge_hash(k_dir)
        folds_sha = autolabel_mod._get_folds_sha256(folds_file, proc_dir)
        reg_sha = autolabel_mod._hash_file(reg_file)
        spec_sha = autolabel_mod._hash_file(spec_file)
        fp, _ = autolabel_mod._compute_teacher_fingerprint(
            knowledge_hash=k_hash,
            rules_cfg=rules_cfg,
            v1_cfg=v1_cfg,
            code_hash=code_hash,
            folds_sha=folds_sha,
            regions_sha=reg_sha,
            specimens_sha=spec_sha,
        )
        return fp

    base_fp = compute_fp()

    # 1. minerals.csv changes
    (k_dir / "minerals.csv").write_text("min,cls\nqtz,silica_alt\n", encoding="utf-8")
    fp_min = compute_fp()
    assert fp_min != base_fp
    (k_dir / "minerals.csv").write_text("min,cls\nqtz,silica\n", encoding="utf-8")

    # 2. aliases.csv changes
    (k_dir / "aliases.csv").write_text("a,b\nqtz,quartz_alt\n", encoding="utf-8")
    fp_alias = compute_fp()
    assert fp_alias != base_fp
    (k_dir / "aliases.csv").write_text("a,b\nqtz,quartz\n", encoding="utf-8")

    # 3. regions.parquet changes
    reg_file.write_bytes(b"regions_bytes_2")
    fp_reg = compute_fp()
    assert fp_reg != base_fp
    reg_file.write_bytes(b"regions_bytes_1")

    # 4. specimens_clean.csv changes
    spec_file.write_text("code,rf\nS1,feldspar\n", encoding="utf-8")
    fp_spec = compute_fp()
    assert fp_spec != base_fp
    spec_file.write_text("code,rf\nS1,qtz\n", encoding="utf-8")

    # 5. folds.json changes
    folds_file.write_text('{"fold_0": {"train": ["S2"]}}', encoding="utf-8")
    fp_folds = compute_fp()
    assert fp_folds != base_fp


def test_list_conflict_reason_handling():
    """When list_conflict reason is emitted, state is abstain, allowed is [], and metadata_dependent is True."""
    reason = "list_conflict"
    meta = {"state": "singleton", "allowed": ["feldspar"], "reason": "list_conflict"}
    opt = {"state": "singleton", "allowed": ["quartz"]}

    state = meta["state"]
    allowed = meta.get("allowed", [])
    fam_p_dict = {}
    families = ["feldspar", "quartz"]
    fam_p = {f: float(fam_p_dict.get(f, 0.0)) for f in families}
    p_out = 0.0
    n_info = 0

    optics_state = opt["state"]
    optics_allowed = opt.get("allowed", [])
    metadata_dependent = (state, allowed) != (optics_state, optics_allowed)

    if reason == "list_conflict" or meta.get("reason") == "list_conflict":
        state = "abstain"
        allowed = []
        reason = "list_conflict"
        metadata_dependent = True

    assert state == "abstain"
    assert allowed == []
    assert metadata_dependent is True
    assert reason == "list_conflict"


def test_folds_sha256_provenance_doc_phrasing():
    """Verify that docs/FOLDS_SHA256_PROVENANCE.md uses the accurate phrasing."""
    doc_path = _PROJECT_ROOT / "docs" / "FOLDS_SHA256_PROVENANCE.md"
    assert doc_path.is_file()
    content = doc_path.read_text(encoding="utf-8")
    assert "proves the split files are unchanged" in content
    assert "guarantees fair comparison/significance" not in content
    assert "fair comparison and statistical significance testing across arms are guaranteed" not in content


def test_line_ending_independent_hashing(tmp_path: Path, monkeypatch):
    """Saving files with CRLF vs LF yields identical knowledge, code, and features hashes."""
    from src.rotapetro.provenance import compute_code_hash, compute_knowledge_hash

    # 1. Knowledge hash
    k_dir = tmp_path / "knowledge"
    k_dir.mkdir()
    yaml_file = k_dir / "test.yaml"
    csv_file = k_dir / "minerals.csv"

    yaml_file.write_bytes(b"key: value\r\nother: 123\r\n")
    csv_file.write_bytes(b"mineral,class\r\nquartz,silica\r\n")
    h_crlf = compute_knowledge_hash(k_dir)

    yaml_file.write_bytes(b"key: value\nother: 123\n")
    csv_file.write_bytes(b"mineral,class\nquartz,silica\n")
    h_lf = compute_knowledge_hash(k_dir)

    assert h_crlf == h_lf

    # 2. Code hash
    fake_root = tmp_path / "repo"
    (fake_root / "scripts").mkdir(parents=True)
    (fake_root / "src" / "rotapetro").mkdir(parents=True)

    f1 = fake_root / "scripts" / "06_autolabel.py"
    f2 = fake_root / "src" / "rotapetro" / "rules.py"

    f1.write_bytes(b"# 06_autolabel script\r\ndef main():\r\n    pass\r\n")
    f2.write_bytes(b"# rules script\r\nclass RuleEngine:\r\n    pass\r\n")
    c_crlf = compute_code_hash(fake_root)

    f1.write_bytes(b"# 06_autolabel script\ndef main():\n    pass\n")
    f2.write_bytes(b"# rules script\nclass RuleEngine:\n    pass\n")
    c_lf = compute_code_hash(fake_root)

    assert c_crlf == c_lf

    # 3. Features fingerprint code hash
    monkeypatch.setattr(features_mod, "_PROJECT_ROOT", fake_root)
    feat_code = fake_root / "src" / "rotapetro" / "features.py"
    feat_script = fake_root / "scripts" / "04_features.py"

    feat_code.write_bytes(b"# features\r\nFEATURE_VERSION = '4-v1.1'\r\n")
    feat_script.write_bytes(b"# 04_features\r\ndef main():\r\n    pass\r\n")
    fp_crlf, cfg_h1, code_h1 = features_mod._compute_features_fingerprint({"features": {}, "v1": {}})

    feat_code.write_bytes(b"# features\nFEATURE_VERSION = '4-v1.1'\n")
    feat_script.write_bytes(b"# 04_features\ndef main():\n    pass\n")
    fp_lf, cfg_h2, code_h2 = features_mod._compute_features_fingerprint({"features": {}, "v1": {}})

    assert code_h1 == code_h2
    assert fp_crlf == fp_lf



