import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import yaml

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (returns a new dict).

    For every key in *override*:
    - if both values are dicts, merge recursively;
    - otherwise the override value wins.
    """
    merged = dict(base)
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _deep_merge(merged[k], v)
        else:
            merged[k] = v
    return merged


def load_config(path: str | Path) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    if 'base' in config:
        base_path = Path(path).parent / config['base']
        if not base_path.exists():
            base_path = Path(config['base'])
        base_config = load_config(base_path)
        config = _deep_merge(base_config, config)
        config.pop('base', None)
    return config

def resolve_pol_dir(raw_folder: str | Path, pol: str) -> Path | None:
    """Find ppl or xpl directory case-insensitively (e.g. ppl, PPL, Ppl)."""
    base = Path(raw_folder)
    if not base.is_dir():
        return None
    d = base / pol
    if d.is_dir():
        return d
    d = base / pol.upper()
    if d.is_dir():
        return d
    for sub in base.iterdir():
        if sub.is_dir() and sub.name.lower() == pol.lower():
            return sub
    return None


def resolve_frame_path(pol_dir: str | Path, frame_idx: int) -> Path | None:
    """Find frame image (e.g. 000.jpg, 000.JPG, 000.jpeg) case-insensitively."""
    base = Path(pol_dir)
    if not base.is_dir():
        return None
    stem = f"{frame_idx:03d}"
    for ext in (".jpg", ".JPG", ".jpeg", ".JPEG", ".png", ".PNG"):
        cand = base / f"{stem}{ext}"
        if cand.is_file():
            return cand
    for f in base.iterdir():
        if f.is_file() and f.stem == stem:
            return f
    return None


def load_raw_frames(specimen_dir: str | Path, rot_folder: str, mode: str = 'ppl') -> np.ndarray:
    """Returns shape [72,490,490,3] array of frames, case-insensitively."""
    base = Path(specimen_dir) / rot_folder
    dir_path = resolve_pol_dir(base, mode)
    if dir_path is None or not dir_path.is_dir():
        raise FileNotFoundError(f"Directory {base / mode} not found.")
    
    # Try finding sequential frames 000..071 first
    frames = []
    has_indexed = True
    for i in range(72):
        f = resolve_frame_path(dir_path, i)
        if f is None or not f.is_file():
            has_indexed = False
            break
        img = cv2.imread(str(f))
        if img is None:
            has_indexed = False
            break
        frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    if has_indexed and len(frames) == 72:
        return np.array(frames, dtype=np.uint8)

    # Fallback to listing all images
    frames = []
    image_files = sorted([f for f in os.listdir(dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
    for f in image_files:
        img_path = dir_path / f
        img = cv2.imread(str(img_path))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frames.append(img)
    return np.array(frames, dtype=np.uint8)

def save_aligned_stack(path: str | Path, ppl, xpl, theta_deg, valid, M,
                       rot_measured_deg, rot_sign, center_xy, align_fallback, **extra):
    np.savez_compressed(
        path,
        ppl=ppl,
        xpl=xpl,
        theta_deg=theta_deg,
        valid=valid,
        M=M,
        rot_measured_deg=rot_measured_deg,
        rot_sign=rot_sign,
        center_xy=center_xy,
        align_fallback=align_fallback,
        **extra,
    )

def load_aligned_stack(path: str | Path) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}

def save_maps(path: str | Path, maps_dict: dict):
    np.savez_compressed(path, **maps_dict)

def load_maps(path: str | Path) -> dict:
    data = np.load(path)
    return {k: data[k] for k in data.files}

def save_segments(img_path: str | Path, json_path: str | Path, label_img: np.ndarray, regions: list):
    import imageio.v3 as iio
    iio.imwrite(img_path, label_img.astype(np.uint16))
    with open(json_path, 'w') as f:
        json.dump(regions, f)

def load_segments(img_path: str | Path, json_path: str | Path) -> tuple[np.ndarray, list]:
    import imageio.v3 as iio
    label_img = iio.imread(img_path).astype(np.uint16)
    with open(json_path, 'r') as f:
        regions = json.load(f)
    return label_img, regions

def load_meta(specimen_dir: str | Path) -> dict:
    with open(Path(specimen_dir) / 'meta.json', 'r') as f:
        return json.load(f)

def load_sites_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_specimens_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def load_knowledge(knowledge_dir: str | Path) -> dict:
    """Load the knowledge base from ``knowledge/``.

    Returns a dict with keys:
    - ``minerals``: DataFrame from ``minerals.csv``
    - ``aliases``: DataFrame from ``aliases.csv``
    - ``hierarchy``: dict from ``hierarchy.yaml``
    - ``likelihoods``: dict from ``likelihoods.yaml``

    Missing files produce an empty DataFrame / dict and a warning.
    """
    import logging
    _log = logging.getLogger(__name__)
    dir_path = Path(knowledge_dir)

    knowledge: dict = {
        "minerals": pd.DataFrame(),
        "aliases": pd.DataFrame(),
        "hierarchy": {},
        "likelihoods": {},
    }

    minerals_path = dir_path / "minerals.csv"
    aliases_path = dir_path / "aliases.csv"
    hierarchy_path = dir_path / "hierarchy.yaml"
    likelihoods_path = dir_path / "likelihoods.yaml"

    if minerals_path.is_file():
        knowledge["minerals"] = pd.read_csv(minerals_path)
    else:
        _log.warning("knowledge/minerals.csv not found at %s", minerals_path)

    if aliases_path.is_file():
        knowledge["aliases"] = pd.read_csv(aliases_path)
    else:
        _log.warning("knowledge/aliases.csv not found at %s", aliases_path)

    if hierarchy_path.is_file():
        with open(hierarchy_path, "r", encoding="utf-8") as f:
            knowledge["hierarchy"] = yaml.safe_load(f) or {}
    else:
        _log.warning("knowledge/hierarchy.yaml not found at %s", hierarchy_path)

    if likelihoods_path.is_file():
        with open(likelihoods_path, "r", encoding="utf-8") as f:
            knowledge["likelihoods"] = yaml.safe_load(f) or {}
    else:
        _log.warning("knowledge/likelihoods.yaml not found at %s", likelihoods_path)

    return knowledge


def resolve_v1_paths(cfg: dict) -> dict:
    """Resolve V1-specific output paths from config.
    Returns dict with keys: processed, teachers, embeddings, results.
    """
    v1 = cfg.get('v1', {})
    v1_paths = v1.get('paths', {})
    base = Path(cfg.get('paths', {}).get('outputs', 'outputs'))
    return {
        'processed': Path(v1_paths.get('processed', 'data/processed/v1')),
        'teachers': Path(v1_paths.get('processed', 'data/processed/v1')) / 'teachers',
        'embeddings': Path(v1_paths.get('embeddings', 'outputs/v1/embeddings')),
        'results': Path(v1_paths.get('results', 'outputs/v1')),
    }


def save_provenance(path: Path, cfg: dict, extra: dict | None = None):
    """Save minimal V1 provenance record."""
    import hashlib, sys, datetime
    prov = {
        'schema_version': 'v1.0',
        'timestamp': datetime.datetime.now().isoformat(),
        'python_version': sys.version,
        'v1_config': cfg.get('v1', {}),
        'scale_discrepancy_note': (
            'YAML um_per_px_rot=2.09 vs saved outputs ~4.181. '
            'V1 uses pixel-unit eligibility; physical-scale features excluded from rules.'
        ),
    }
    if extra:
        prov.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(prov, f, indent=2, default=str)


def validate_v1_config(cfg: dict):
    """Validate V1 config section. Fail on unknown/incompatible settings."""
    v1 = cfg.get('v1')
    if v1 is None:
        raise ValueError('Missing v1 section in config')
    
    required_keys = {
        'version', 'data_version', 'taxonomy', 'paths',
        'region_selection', 'training', 'rules', 'families'
    }
    missing = required_keys - set(v1.keys())
    if missing:
        raise ValueError(f'Missing required v1 config keys: {missing}')
    
    tax = v1['taxonomy']
    primary_classes = tax.get('primary_classes', [])
    if not primary_classes:
        raise ValueError('v1.taxonomy.primary_classes must be non-empty')

    # Validate v1.rules
    rules = v1.get('rules')
    if not isinstance(rules, dict):
        raise ValueError('v1.rules must be a dictionary')

    # Validate v1.families (disjoint sets + primary_classes mapping)
    families = v1.get('families')
    if not isinstance(families, dict) or not families:
        raise ValueError('v1.families must be a non-empty dictionary')

    seen_species: dict[str, str] = {}
    for family, species_list in families.items():
        if not isinstance(species_list, list):
            raise ValueError(f"v1.families['{family}'] must be a list of species")
        for sp in species_list:
            if sp in seen_species:
                raise ValueError(
                    f"Species '{sp}' listed in multiple families: "
                    f"'{seen_species[sp]}' and '{family}'"
                )
            seen_species[sp] = family

    for cls in primary_classes:
        if cls not in families:
            raise ValueError(
                f"primary_classes entry '{cls}' is not a key in v1.families"
            )
    
    rs = v1['region_selection']
    if rs.get('min_area_px', 0) < 1:
        raise ValueError('v1.region_selection.min_area_px must be >= 1')

    border_exclude_px = cfg.get('features', {}).get('border_exclude_px')
    erosion_px = rs.get('erosion_px')
    if border_exclude_px != erosion_px:
        raise ValueError(
            f"features.border_exclude_px ({border_exclude_px}) must equal "
            f"v1.region_selection.erosion_px ({erosion_px})"
        )

    # Reject dangerous legacy settings that V1 doesn't use
    train = cfg.get('train', {})
    if train.get('label_smoothing', 0) > 0:
        raise ValueError('V1 requires train.label_smoothing=0 (disabled)')
    if train.get('lambda_mil', 0) > 0:
        raise ValueError('V1 requires train.lambda_mil=0 (MIL disabled)')


def require_fresh_feasibility(cfg: dict) -> None:
    """Verify that a fresh, passing label feasibility report exists.

    CONTRACTS.md Section 5:
    1. provenance_feasibility.json exists in data/processed/v1/ and contains "overall_gate_pass": True.
    2. For every fold fold_0..fold_4, teacher_fingerprints[fold] equals teachers/{fold}/teacher_params.json["fingerprint"].
    3. provenance_feasibility.json["regions_sha256"] equals the SHA-256 of the current data/processed/v1/regions.parquet.

    If any check fails, call sys.exit(1) with a clear error indicating which check failed.
    """
    v1_paths = resolve_v1_paths(cfg)
    processed_dir = Path(v1_paths["processed"])
    teachers_dir = Path(v1_paths["teachers"])

    prov_path = processed_dir / "provenance_feasibility.json"
    if not prov_path.is_file():
        sys.stderr.write(
            f"Feasibility gate check failed: {prov_path} does not exist. Run scripts/v1_label_feasibility.py first.\n"
        )
        sys.exit(1)

    try:
        with open(prov_path, "r", encoding="utf-8") as f:
            prov = json.load(f)
    except Exception as e:
        sys.stderr.write(f"Feasibility gate check failed: could not read {prov_path}: {e}\n")
        sys.exit(1)

    # 1. overall_gate_pass == True
    if prov.get("overall_gate_pass") is not True:
        sys.stderr.write(
            f"Feasibility gate check failed: {prov_path} has overall_gate_pass={prov.get('overall_gate_pass')} (must be True).\n"
        )
        sys.exit(1)

    # 2. Check teacher fingerprints for fold_0..fold_4
    from src.rotapetro.provenance import current_teacher_fingerprint
    cur_teacher_fp = current_teacher_fingerprint(cfg)

    prov_fps = prov.get("teacher_fingerprints")
    if not isinstance(prov_fps, dict):
        sys.stderr.write(
            f"Feasibility gate check failed: 'teacher_fingerprints' missing or not a dict in {prov_path}. Rerun scripts/v1_label_feasibility.py.\n"
        )
        sys.exit(1)

    for i in range(5):
        fold = f"fold_{i}"
        if fold not in prov_fps:
            sys.stderr.write(
                f"Feasibility gate check failed: {fold} missing from teacher_fingerprints in {prov_path}. Rerun scripts/v1_label_feasibility.py.\n"
            )
            sys.exit(1)

        expected_fp = prov_fps[fold]
        t_params_path = teachers_dir / fold / "teacher_params.json"
        if not t_params_path.is_file():
            sys.stderr.write(
                f"Feasibility gate check failed: {t_params_path} does not exist for fold {fold}. Rerun scripts/06_autolabel.py.\n"
            )
            sys.exit(1)

        try:
            with open(t_params_path, "r", encoding="utf-8") as f:
                t_params = json.load(f)
        except Exception as e:
            sys.stderr.write(f"Feasibility gate check failed: could not read {t_params_path}: {e}. Rerun scripts/06_autolabel.py.\n")
            sys.exit(1)

        actual_fp = t_params.get("fingerprint")
        if actual_fp != expected_fp:
            sys.stderr.write(
                f"Feasibility gate check failed: teacher fingerprint mismatch for {fold} "
                f"(provenance={expected_fp}, teacher_params={actual_fp}). Rerun scripts/06_autolabel.py.\n"
            )
            sys.exit(1)

        if actual_fp != cur_teacher_fp:
            sys.stderr.write(
                f"Feasibility gate check failed: teacher fingerprint mismatch for {fold} "
                f"(teacher_params={actual_fp}, current={cur_teacher_fp}). Rerun scripts/06_autolabel.py.\n"
            )
            sys.exit(1)

        # Recompute targets.parquet sha256 and compare with teacher_params and feasibility provenance
        t_targets_path = teachers_dir / fold / "targets.parquet"
        if not t_targets_path.is_file():
            sys.stderr.write(
                f"Feasibility gate check failed: {t_targets_path} missing. "
                "rerun 06_autolabel.py --v1 (or --force)\n"
            )
            sys.exit(1)

        try:
            with open(t_targets_path, "rb") as f:
                actual_targets_sha = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            sys.stderr.write(
                f"Feasibility gate check failed: could not hash {t_targets_path}: {e}. "
                "rerun 06_autolabel.py --v1 (or --force)\n"
            )
            sys.exit(1)

        expected_params_t_sha = t_params.get("targets_sha256")
        if not expected_params_t_sha or actual_targets_sha != expected_params_t_sha:
            sys.stderr.write(
                f"Feasibility gate check failed: targets.parquet sha256 mismatch for {fold} against teacher_params "
                f"(actual={actual_targets_sha}, teacher_params={expected_params_t_sha}). "
                "rerun 06_autolabel.py --v1 (or --force)\n"
            )
            sys.exit(1)

        prov_targets_map = prov.get("targets_sha256", {})
        expected_prov_t_sha = prov_targets_map.get(fold) if isinstance(prov_targets_map, dict) else None
        if not expected_prov_t_sha or actual_targets_sha != expected_prov_t_sha:
            sys.stderr.write(
                f"Feasibility gate check failed: targets.parquet sha256 mismatch for {fold} against provenance_feasibility "
                f"(actual={actual_targets_sha}, provenance={expected_prov_t_sha}). "
                "rerun 06_autolabel.py --v1 (or --force)\n"
            )
            sys.exit(1)

    # 3. Check regions_sha256 matches data/processed/v1/regions.parquet
    regions_path = processed_dir / "regions.parquet"
    if not regions_path.is_file():
        sys.stderr.write(
            f"Feasibility gate check failed: {regions_path} does not exist. Rerun scripts/04_features.py.\n"
        )
        sys.exit(1)

    try:
        with open(regions_path, "rb") as f:
            current_regions_sha256 = hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        sys.stderr.write(f"Feasibility gate check failed: could not hash {regions_path}: {e}\n")
        sys.exit(1)

    prov_regions_sha256 = prov.get("regions_sha256")
    if prov_regions_sha256 != current_regions_sha256:
        sys.stderr.write(
            f"Feasibility gate check failed: regions_sha256 mismatch "
            f"(provenance={prov_regions_sha256}, current={current_regions_sha256}). Rerun scripts/06_autolabel.py / scripts/v1_label_feasibility.py.\n"
        )
        sys.exit(1)

    # 4. Check folds_sha256 matches current splits/folds.json
    folds_path = processed_dir / "splits" / "folds.json"
    if not folds_path.is_file():
        folds_path = processed_dir.parent / "splits" / "folds.json"
    if not folds_path.is_file():
        sys.stderr.write(
            f"Feasibility gate check failed: folds.json does not exist. Rerun splits generation.\n"
        )
        sys.exit(1)

    try:
        with open(folds_path, "rb") as f:
            current_folds_sha256 = hashlib.sha256(f.read()).hexdigest()
    except Exception as e:
        sys.stderr.write(f"Feasibility gate check failed: could not hash {folds_path}: {e}\n")
        sys.exit(1)

    prov_folds_sha256 = prov.get("folds_sha256")
    if prov_folds_sha256 != current_folds_sha256:
        sys.stderr.write(
            f"Feasibility gate check failed: folds_sha256 mismatch "
            f"(provenance={prov_folds_sha256}, current={current_folds_sha256}). Rerun scripts/06_autolabel.py / scripts/v1_label_feasibility.py.\n"
        )
        sys.exit(1)


