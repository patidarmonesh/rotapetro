"""Provenance and fingerprinting utilities for V1 pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from src.rotapetro.io import resolve_v1_paths

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def hash_file(path: str | Path) -> str:
    """Compute SHA-256 hex digest of file contents (returns '' if missing)."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def compute_knowledge_hash(knowledge_dir: str | Path) -> str:
    """Hash all knowledge YAML, YML, and CSV files in knowledge_dir."""
    k_dir = Path(knowledge_dir)
    h = hashlib.sha256()
    target_files: list[Path] = []
    if k_dir.is_dir():
        for p in k_dir.iterdir():
            if p.is_file() and p.suffix.lower() in (".yaml", ".yml", ".csv"):
                target_files.append(p)
    target_files.sort(key=lambda p: p.name)
    for p in target_files:
        h.update(p.name.encode("utf-8"))
        h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def compute_code_hash(project_root: Path | None = None) -> str:
    """Hash core rule/autolabel source files and package version."""
    root = project_root if project_root is not None else _PROJECT_ROOT
    h = hashlib.sha256()
    try:
        from src.rotapetro import __version__
        h.update(str(__version__).encode("utf-8"))
    except Exception:
        pass

    code_files = [
        root / "scripts" / "06_autolabel.py",
        root / "src" / "rotapetro" / "rules.py",
        root / "src" / "rotapetro" / "knowledge.py",
        root / "src" / "rotapetro" / "features.py",
        root / "src" / "rotapetro" / "provenance.py",
    ]
    for p in code_files:
        if p.is_file():
            h.update(p.name.encode("utf-8"))
            h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def get_folds_sha256(
    folds_path: str | Path,
    v1_processed_dir: str | Path | None = None,
    project_root: Path | None = None,
) -> str:
    """Calculate SHA256 of folds split and verify against FOLDS_SHA256.txt."""
    f_path = Path(folds_path)
    if not f_path.is_file():
        return ""
    actual_hash = hashlib.sha256(f_path.read_bytes()).hexdigest()

    root = project_root if project_root is not None else _PROJECT_ROOT
    v1_p_dir = Path(v1_processed_dir) if v1_processed_dir is not None else None

    sha_file = None
    if v1_p_dir is not None and (v1_p_dir / "FOLDS_SHA256.txt").is_file():
        sha_file = v1_p_dir / "FOLDS_SHA256.txt"
    elif (f_path.parent / "FOLDS_SHA256.txt").is_file():
        sha_file = f_path.parent / "FOLDS_SHA256.txt"
    else:
        default_sha_file = root / "data" / "processed" / "v1" / "FOLDS_SHA256.txt"
        if default_sha_file.is_file() and (v1_p_dir is None or v1_p_dir.resolve() == default_sha_file.parent.resolve()):
            sha_file = default_sha_file

    if sha_file is not None and sha_file.is_file():
        expected_hash = sha_file.read_text(encoding="utf-8").strip()
        if expected_hash and expected_hash != actual_hash:
            raise RuntimeError(
                f"FOLDS_SHA256.txt mismatch: expected {expected_hash}, got {actual_hash}"
            )
    return actual_hash


def compute_teacher_fingerprint(
    knowledge_hash: str,
    rules_cfg: dict,
    v1_cfg: dict,
    code_hash: str,
    folds_sha: str,
    regions_sha: str,
    specimens_sha: str,
) -> tuple[str, dict]:
    """Compute 12-char fingerprint and return (fingerprint, provenance_dict)."""
    provenance_dict = {
        "knowledge_hash": knowledge_hash,
        "rules_config": rules_cfg,
        "v1_config": v1_cfg,
        "code_hash": code_hash,
        "folds_sha256": folds_sha,
        "regions_sha256": regions_sha,
        "specimens_sha256": specimens_sha,
    }
    fp_bytes = json.dumps(provenance_dict, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(fp_bytes).hexdigest()[:12], provenance_dict


def current_teacher_provenance(cfg: dict, project_root: Path | None = None) -> tuple[str, dict]:
    """Compute the authoritative teacher fingerprint and provenance dict for current inputs."""
    root = project_root if project_root is not None else _PROJECT_ROOT
    v1_paths = resolve_v1_paths(cfg)

    proc_path = Path(v1_paths["processed"])
    processed_dir = proc_path if proc_path.is_absolute() else root / proc_path

    interim_path = Path(cfg.get("paths", {}).get("interim", "data/interim"))
    interim_dir = interim_path if interim_path.is_absolute() else root / interim_path

    k_path = Path(cfg.get("paths", {}).get("knowledge", "knowledge"))
    knowledge_dir = k_path if k_path.is_absolute() else root / k_path

    regions_v1_path = processed_dir / "regions.parquet"
    specimens_path = interim_dir / "specimens_clean.csv"

    folds_path = processed_dir / "splits" / "folds.json"
    if not folds_path.is_file():
        legacy_p = Path(cfg.get("paths", {}).get("processed", "data/processed")) / "splits" / "folds.json"
        folds_path = legacy_p if legacy_p.is_absolute() else root / legacy_p

    k_hash = compute_knowledge_hash(knowledge_dir)
    c_hash = compute_code_hash(root)
    f_sha = get_folds_sha256(folds_path, processed_dir, root)
    r_sha = hash_file(regions_v1_path)
    s_sha = hash_file(specimens_path)

    v1_cfg = cfg.get("v1", {})
    rules_cfg = cfg.get("rules", {})

    return compute_teacher_fingerprint(
        knowledge_hash=k_hash,
        rules_cfg=rules_cfg,
        v1_cfg=v1_cfg,
        code_hash=c_hash,
        folds_sha=f_sha,
        regions_sha=r_sha,
        specimens_sha=s_sha,
    )


def current_teacher_fingerprint(cfg: dict, project_root: Path | None = None) -> str:
    """Return the 12-char authoritative teacher fingerprint for the current config and data."""
    fp, _ = current_teacher_provenance(cfg, project_root)
    return fp
