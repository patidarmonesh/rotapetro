"""Phase 7 — Training dataset assembly and splits (§13).

Module: src/rotapetro/dataset.py
Inputs:  labels.parquet, regions.parquet, classes.yaml,
         aligned stacks (*.npz), segments (*.png)
Outputs: folds.json, class weights per fold

§13.1  Sample = region with type ∈ {grain, opaque, aggregate}, small = False.
§13.2  Crop extraction on the fly from aligned stacks (never stored).
§13.3  __getitem__ returns the tensor dict specified in the spec.
§13.4  Splits: 5 folds with StratifiedGroupKFold.
§13.5  Class imbalance bookkeeping.
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

TEST_PREDICTED_STATES: tuple[str, ...] = ("singleton", "partial_set", "abstain")


# ══════════════════════════════════════════════════════════════════════════
#  Splits (§13.4)
# ══════════════════════════════════════════════════════════════════════════

def build_folds(
    specimens_df: pd.DataFrame,
    cfg: dict,
    output_path: Path | None = None,
) -> dict:
    """Build 5-fold specimen splits with StratifiedGroupKFold (§13.4).

    Parameters
    ----------
    specimens_df : DataFrame
        Must have columns: ``code``, ``rock_family``.
    cfg : dict
        Must contain ``dataset.folds``, ``dataset.val_frac``, ``dataset.seed``.
    output_path : Path, optional
        If given, write the folds to this JSON file.

    Returns
    -------
    folds : dict
        ``{fold_k: {train: [codes], val: [codes], test: [codes]}}``.
    """
    from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
    from collections import Counter

    ds_cfg = cfg.get("dataset", {})
    n_folds = int(ds_cfg.get("folds", 5))
    val_frac = float(ds_cfg.get("val_frac", 0.15))
    seed = int(ds_cfg.get("seed", 0))

    # Deduplicate to unique specimens
    specs = specimens_df.drop_duplicates(subset="code").copy()
    codes = specs["code"].values
    strata = specs["rock_family"].values.copy()
    # groups = codes for StratifiedGroupKFold (each code is its own group)
    groups = codes

    # Merge rare strata: any rock_family with < n_folds members → "_other"
    counts = Counter(strata)
    for i, s in enumerate(strata):
        if counts[s] < n_folds:
            strata[i] = "_other"

    # Reduce n_folds if we still don't have enough specimens
    min_stratum = min(Counter(strata).values()) if len(strata) > 0 else 0
    effective_folds = min(n_folds, min_stratum, len(codes))
    if effective_folds < n_folds:
        logger.warning("Reducing folds from %d to %d (smallest stratum "
                       "has %d specimens)", n_folds, effective_folds,
                       min_stratum)
        n_folds = max(2, effective_folds)

    # Try stratified split; fall back to plain GroupKFold
    try:
        splitter = StratifiedGroupKFold(
            n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(codes, strata, groups))
    except ValueError:
        logger.warning("StratifiedGroupKFold failed — falling back to "
                       "GroupKFold (no rock_family stratification)")
        splitter = GroupKFold(n_splits=n_folds)
        splits = list(splitter.split(codes, strata, groups))

    folds: dict = {}
    for k, (remain_idx, test_idx) in enumerate(splits):
        test_codes = sorted(codes[test_idx].tolist())
        remain_codes = codes[remain_idx]
        remain_strata = strata[remain_idx]

        # Split remaining into train + val (grouped, seed 0 per §13.4)
        rng = np.random.RandomState(seed)
        n_val = max(1, int(len(remain_codes) * val_frac))
        perm = rng.permutation(len(remain_codes))
        val_codes = sorted(remain_codes[perm[:n_val]].tolist())
        train_codes = sorted(remain_codes[perm[n_val:]].tolist())

        folds[f"fold_{k}"] = {
            "train": train_codes,
            "val": val_codes,
            "test": test_codes,
        }

    # Save once, never regenerate
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(folds, f, indent=2)
        logger.info("Saved %d folds to %s", n_folds, output_path)

    return folds


# ══════════════════════════════════════════════════════════════════════════
#  Class imbalance bookkeeping (§13.5)
# ══════════════════════════════════════════════════════════════════════════

def compute_class_counts(
    samples_df: pd.DataFrame,
    folds: dict,
    fine_classes: list[str],
    group_classes: list[str],
    site_to_code: dict[str, str],
) -> dict:
    """Compute per-fold training class counts for fine and group labels.

    **Important**: ``samples_df`` must be the *filtered* samples (from
    ``filter_samples``), not the raw ``labels_df``.  This ensures that
    small/excluded regions do not inflate the counts.

    Returns dict ``{fold_k: {fine_counts: {cls: n}, group_counts: {cls: n}}}``.
    """
    df = samples_df.copy()

    # Ensure code column is available
    if "code" not in df.columns and "site_id" in df.columns:
        df["code"] = df["site_id"].map(
            lambda sid: site_to_code.get(str(sid), ""))

    fold_counts: dict = {}
    for fold_name, split in folds.items():
        train_codes = set(split["train"])
        train_samples = df[df["code"].isin(train_codes)]

        fine_counter = Counter(
            train_samples["label_fine"].dropna().tolist())
        group_counter = Counter(
            train_samples["label_group"].dropna().tolist())

        fold_counts[fold_name] = {
            "fine_counts": {c: fine_counter.get(c, 0) for c in fine_classes},
            "group_counts": {c: group_counter.get(c, 0)
                             for c in group_classes},
        }

    return fold_counts


# ══════════════════════════════════════════════════════════════════════════
#  Crop extraction (§13.2)
# ══════════════════════════════════════════════════════════════════════════

def _make_square_bbox(
    x: int, y: int, w: int, h: int,
    scale: float, min_px: int,
    img_h: int, img_w: int,
) -> tuple[int, int, int, int, int, int, int, int]:
    """Compute the square crop bbox and padding.

    Returns (x0, y0, S, pad_left, pad_top, pad_right, pad_bottom, S_clamped).
    """
    S = max(max(w, h) * scale, min_px)
    S = int(math.ceil(S))

    # Centre of bbox
    cx = x + w / 2.0
    cy = y + h / 2.0

    x0 = int(round(cx - S / 2.0))
    y0 = int(round(cy - S / 2.0))

    # Clamp to image and compute padding
    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    x0 = max(0, x0)
    y0 = max(0, y0)

    x1 = min(img_w, x0 + S)
    y1 = min(img_h, y0 + S)

    pad_right = max(0, S - (x1 - x0) - pad_left)
    pad_bottom = max(0, S - (y1 - y0) - pad_top)

    return x0, y0, S, pad_left, pad_top, pad_right, pad_bottom


def _crop_and_resize(
    frame: np.ndarray,
    x0: int, y0: int, S: int,
    pad_l: int, pad_t: int, pad_r: int, pad_b: int,
    target_px: int,
    interp: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """Crop a region from a frame, pad with zeros, resize to target_px."""
    H, W = frame.shape[:2]
    x1 = min(W, x0 + S - pad_l - pad_r)
    y1 = min(H, y0 + S - pad_t - pad_b)

    crop = frame[y0:y1, x0:x1].copy()

    if crop.size == 0:
        if frame.ndim == 3:
            return np.zeros((target_px, target_px, frame.shape[2]),
                            dtype=frame.dtype)
        return np.zeros((target_px, target_px), dtype=frame.dtype)

    # Pad
    if pad_l or pad_t or pad_r or pad_b:
        if frame.ndim == 3:
            crop = np.pad(crop,
                          ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                          mode="constant")
        else:
            crop = np.pad(crop,
                          ((pad_t, pad_b), (pad_l, pad_r)),
                          mode="constant")

    # Resize
    resized = cv2.resize(crop, (target_px, target_px),
                         interpolation=interp)
    return resized


def extract_crops(
    aligned: dict,
    label_img: np.ndarray,
    label_value: int,
    bbox_xywh: tuple[int, int, int, int],
    cfg: dict,
) -> dict:
    """Extract PPL/XPL crops and mask for one region (§13.2).

    Returns dict with:
        ppl: uint8 [36, 224, 224, 3]
        xpl: uint8 [36, 224, 224, 3]
        mask: bool [224, 224]
        theta_deg: float [36]
    """
    ds_cfg = cfg.get("dataset", {})
    scale = float(ds_cfg.get("crop_scale", 1.3))
    min_px = int(ds_cfg.get("crop_min_px", 48))
    target_px = int(ds_cfg.get("input_px", 224))
    context_dim = float(ds_cfg.get("context_dim", 0.35))

    ppl_stack = aligned["ppl"]   # [36, H, W, 3] uint8
    xpl_stack = aligned["xpl"]   # [36, H, W, 3] uint8
    theta_deg = aligned["theta_deg"].astype(np.float32)  # [36]
    H, W = ppl_stack.shape[1], ppl_stack.shape[2]

    x, y, w, h = bbox_xywh
    x0, y0, S, pl, pt, pr, pb = _make_square_bbox(
        x, y, w, h, scale, min_px, H, W)

    # Crop and resize the mask
    region_mask = (label_img == label_value).astype(np.uint8) * 255
    mask_resized = _crop_and_resize(
        region_mask, x0, y0, S, pl, pt, pr, pb, target_px,
        interp=cv2.INTER_NEAREST)
    mask = mask_resized > 127  # bool [224, 224]

    # Crop PPL and XPL for all 36 angles
    ppl_crops = np.zeros((36, target_px, target_px, 3), dtype=np.uint8)
    xpl_crops = np.zeros((36, target_px, target_px, 3), dtype=np.uint8)

    for k in range(36):
        ppl_crop = _crop_and_resize(
            ppl_stack[k], x0, y0, S, pl, pt, pr, pb, target_px)
        xpl_crop = _crop_and_resize(
            xpl_stack[k], x0, y0, S, pl, pt, pr, pb, target_px)

        # Context dimming (§13.2): multiply pixels outside mask by 0.35
        if context_dim < 1.0:
            outside = ~mask
            ppl_crop[outside] = (
                ppl_crop[outside].astype(np.float32) * context_dim
            ).astype(np.uint8)
            xpl_crop[outside] = (
                xpl_crop[outside].astype(np.float32) * context_dim
            ).astype(np.uint8)

        ppl_crops[k] = ppl_crop
        xpl_crops[k] = xpl_crop

    return {
        "ppl": ppl_crops,
        "xpl": xpl_crops,
        "mask": mask,
        "theta_deg": theta_deg,
    }


# ══════════════════════════════════════════════════════════════════════════
#  RegionSequenceDataset (§13.3)
# ══════════════════════════════════════════════════════════════════════════

# Prop target encodings (consistent with model heads)
PROP_KEYS = ["optic_class", "pleo_cls", "biref_cls", "ext_type", "twin_type"]

OPTIC_CLASS_MAP = {"isotropic": 0, "anisotropic": 1, "opaque": 2}
PLEO_CLS_MAP = {"none": 0, "weak": 1, "moderate": 2, "strong": 3}
EXT_TYPE_MAP = {"parallel": 0, "inclined": 1, "symmetric": 2}
TWIN_TYPE_MAP = {"none": 0, "simple": 1, "lamellar": 2, "crosshatched": 3,
                 "sector": 4}          # Phase 4 also outputs "sector"


class RegionSequenceDataset(Dataset):
    """PyTorch dataset returning tensor dicts for one region (§13.3).

    Parameters
    ----------
    samples_df : DataFrame
        Filtered samples (type ∈ {grain, opaque, aggregate}, small=False),
        merged from regions + labels.
    classes : dict
        From classes.yaml: ``fine_classes``, ``groups``.
    aligned_dir : Path
        Directory containing ``<site_id>.npz`` aligned stacks.
    segments_dir : Path
        Directory containing ``<site_id>.png`` label images.
    cfg : dict
        Full configuration.
    mode : str
        ``'image'`` = return PPL/XPL crops.
        ``'embed'`` = return precomputed embeddings from *emb_path*.
    emb_path : Path | None
        Path to embeddings .npy memmap, shape [N, V, 36, 2, 768].
        CONVENTION (Phase 8 must follow it): for every geometric version v
        the 36 angle slots are RE-INDEXED so that emb[n, v, k] corresponds
        to theta_deg[k] = 5·k of the transformed sample.  Then the same
        theta vector is valid for all versions.
    emb_index : dict | None
        Maps region_id → index into the embeddings memmap.
    augment : bool
        If True, apply angle dropout and jitter.
    """

    def __init__(
        self,
        samples_df: pd.DataFrame,
        classes: dict | None = None,
        aligned_dir: Path | str = "",
        segments_dir: Path | str = "",
        cfg: dict | None = None,
        mode: str = "embed",
        emb_path: Path | None = None,
        emb_index: dict | None = None,
        augment: bool = True,
    ) -> None:
        self.samples = samples_df.reset_index(drop=True)
        self.aligned_dir = Path(aligned_dir)
        self.segments_dir = Path(segments_dir)
        self.cfg = cfg or {}
        self.mode = mode
        self.augment = augment

        # Classes
        classes = classes or {}
        self.fine_classes = list(classes.get("fine_classes", []))
        self.fine_to_idx = {c: i for i, c in enumerate(self.fine_classes)}
        self.n_fine = len(self.fine_classes)

        # Group vocabulary + members (written by Phase 7 to groups.yaml and
        # merged into `classes` by the caller).  group_members[g] = fine
        # classes inside group g -> used for the partial-credit loss
        # -log Σ_{c∈G} p(c|x) on group-only labels (§15.1).
        if "group_classes" in classes:
            groups_set = list(classes["group_classes"])
        else:
            groups_set = sorted(set(
                v for v in classes.get("groups", {}).values()
                if v and v != "unknown"))
        self.group_classes = groups_set
        self.group_to_idx = {g: i for i, g in enumerate(groups_set)}
        self.group_members = {
            g: [self.fine_to_idx[c] for c in members if c in self.fine_to_idx]
            for g, members in classes.get("group_members", {}).items()}
        self.return_mask = bool(cfg.get("dataset", {}).get("return_mask", False))
        self._max_cache = int(cfg.get("dataset", {}).get("site_cache", 8))

        # Embeddings (precomputed)
        self.emb_path = emb_path
        self.emb_index = emb_index or {}
        self._emb_mmap = None
        self._n_versions = int(cfg.get("model", {}).get("n_versions", 4))

        # Candidate → final class mapping (BLOCKER 1 fix)
        # Phase 6 stores relabel_map in classes.yaml; use it to map any
        # candidate mineral to its final class (e.g. plagioclase→feldspar)
        self._candidate_to_class: dict[str, str] = {}
        relabel_map = classes.get("relabel_map", {})
        for mineral, final_class in relabel_map.items():
            if final_class in self.fine_to_idx:
                self._candidate_to_class[mineral] = final_class
        # Also map each fine class to itself
        for c in self.fine_classes:
            self._candidate_to_class[c] = c

        # Angle config
        ds_cfg = cfg.get("dataset", {})
        train_cfg = cfg.get("train", {})
        self.angle_dropout = train_cfg.get("angle_dropout", {})
        self.angle_jitter_deg = float(train_cfg.get("angle_jitter_deg", 1.0))

        # Cache aligned stacks and label images (lazy)
        self._aligned_cache: dict[str, dict] = {}
        self._label_cache: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_aligned(self, site_id: str) -> dict:
        if site_id not in self._aligned_cache:
            # each stack is ~50 MB: keep only a few sites (per worker)
            while len(self._aligned_cache) >= self._max_cache:
                self._aligned_cache.pop(next(iter(self._aligned_cache)))
            path = self.aligned_dir / f"{site_id}.npz"
            data = np.load(path)
            self._aligned_cache[site_id] = {
                k: data[k] for k in data.files}
        return self._aligned_cache[site_id]

    def _load_label_img(self, site_id: str) -> np.ndarray:
        if site_id not in self._label_cache:
            import imageio.v3 as iio
            path = self.segments_dir / f"{site_id}.png"
            self._label_cache[site_id] = iio.imread(path).astype(np.uint16)
        return self._label_cache[site_id]

    def clear_cache(self) -> None:
        """Release cached aligned stacks and label images."""
        self._aligned_cache.clear()
        self._label_cache.clear()

    def __getitem__(self, idx: int) -> dict:
        row = self.samples.iloc[idx]
        region_id = str(row["region_id"])
        site_id = str(row["site_id"])
        code = str(row.get("code", ""))

        # ── Labels ──────────────────────────────────────────────────
        label_fine_str = row.get("label_fine")
        label_group_str = row.get("label_group")

        label_fine = (self.fine_to_idx.get(label_fine_str, -1)
                      if pd.notna(label_fine_str) else -1)
        label_group = (self.group_to_idx.get(label_group_str, -1)
                       if pd.notna(label_group_str) else -1)

        # Weight (§15.1): rule labels are weighted by their confidence,
        # mono labels by mono_weight (their label_conf is 1.0).
        source = row.get("label_source", "none")
        conf = row.get("label_conf")
        weight = float(conf) if conf is not None and pd.notna(conf) else 0.0
        if source == "mono":
            weight *= float(self.cfg.get("train", {}).get("mono_weight", 1.5))

        # ── Prop targets ────────────────────────────────────────────
        prop_targets = {
            "optic_class": OPTIC_CLASS_MAP.get(
                str(row.get("optic_class", "")).lower(), -1),
            "pleo_cls": PLEO_CLS_MAP.get(
                str(row.get("pleo_cls", "")).lower(), -1),
            "biref_cls": int(row.get("biref_cls", -1))
                if pd.notna(row.get("biref_cls")) else -1,
            "ext_type": EXT_TYPE_MAP.get(
                str(row.get("ext_type", "")).lower(), -1),
            "twin_type": TWIN_TYPE_MAP.get(
                str(row.get("twin_type", "")).lower(), -1),
        }

        # ── Candidates multi-hot ────────────────────────────────────
        #    Map each candidate mineral → its final class via relabel_map
        #    (e.g. plagioclase→feldspar) before setting the multi-hot bit.
        cand_list = row.get("in_candidate_set", [])
        if isinstance(cand_list, str):
            try:
                cand_list = json.loads(cand_list)
            except (json.JSONDecodeError, ValueError):
                cand_list = []
        # parquet list columns come back as numpy arrays: `arr or []`
        # raises "truth value of an array is ambiguous"
        if cand_list is None or (isinstance(cand_list, float) and pd.isna(cand_list)):
            cand_list = []
        candidates = torch.zeros(self.n_fine, dtype=torch.bool)
        for c in list(cand_list):
            final_cls = self._candidate_to_class.get(c)
            if final_cls is not None and final_cls in self.fine_to_idx:
                candidates[self.fine_to_idx[final_cls]] = True

        group_members = torch.zeros(self.n_fine, dtype=torch.bool)
        if pd.notna(label_group_str) and label_fine == -1:
            for ci in self.group_members.get(label_group_str, []):
                group_members[ci] = True

        # ── Build base result ───────────────────────────────────────
        result: dict = {
            "group_members": group_members,
            "region_id": region_id,
            "label_fine": label_fine,
            "label_group": label_group,
            "weight": weight,
            "prop_targets": prop_targets,
            "specimen": code,
            "candidates": candidates,
        }

        # ── Mode: embed (default) ──────────────────────────────────
        if self.mode == "embed" and self.emb_path is not None:
            emb = self._load_embedding(region_id)       # [V, 36, 2, 768]
            # Training: one random geometric version per sample (§15.4).
            # Evaluation: keep all V versions for test-time averaging.
            if self.augment:
                v = np.random.randint(emb.shape[0])
                emb = emb[v:v + 1]
            result["emb"] = emb
            # theta is the same for every site (0, 5, ..., 175) — do NOT
            # load the 50 MB aligned stack just to read it.
            step = float(self.cfg.get("acquisition", {}).get("step_deg", 5.0))
            theta = (np.arange(emb.shape[1], dtype=np.float32) * step)

            # Angle jitter (recommended fix)
            if self.augment and self.angle_jitter_deg > 0:
                theta += np.random.normal(
                    0, self.angle_jitter_deg, size=theta.shape
                ).astype(np.float32)

            # Angle dropout (recommended fix)
            keep = np.arange(len(theta))
            if self.augment and self.angle_dropout:
                p_drop = float(self.angle_dropout.get("p", 0.5))
                K_options = self.angle_dropout.get("K", [6, 9, 12, 18, 36])
                if np.random.rand() < p_drop and K_options:
                    K = K_options[np.random.randint(len(K_options))]
                    step = max(1, 36 // K)
                    off = np.random.randint(step)          # random offset
                    keep = np.arange(off, 36, step)[:K]
                    # Sub-select embedding angles
                    emb = emb[:, keep, :, :]  # [V, K, 2, 768]
                    result["emb"] = emb
                    theta = theta[keep]

            result["theta_deg"] = torch.from_numpy(theta)

            # Mask is already folded into the (mask-pooled) embedding; only
            # load it if explicitly requested (dataset.return_mask).
            if not self.return_mask:
                return result
            label_img = self._load_label_img(site_id)
            bbox = _parse_bbox(row.get("bbox_xywh"))
            if bbox:
                mask = self._extract_mask(label_img, int(row["label_value"]),
                                          bbox)
                result["mask"] = torch.from_numpy(mask)
            else:
                target_px = int(self.cfg.get("dataset", {}).get(
                    "input_px", 224))
                result["mask"] = torch.zeros(
                    target_px, target_px, dtype=torch.bool)
            return result

        # ── Mode: image (fine-tuning) ──────────────────────────────
        aligned = self._load_aligned(site_id)
        label_img = self._load_label_img(site_id)
        bbox = _parse_bbox(row.get("bbox_xywh"))
        if bbox is None:
            region_id = str(row.get("region_id", "unknown"))
            raise ValueError(f"{region_id}: invalid bbox — refusing full-field fallback")

        crops = extract_crops(
            aligned, label_img, int(row["label_value"]),
            bbox, self.cfg)

        ppl = crops["ppl"]          # [36, 224, 224, 3] uint8
        xpl = crops["xpl"]          # [36, 224, 224, 3] uint8
        mask = crops["mask"]        # [224, 224] bool
        theta = crops["theta_deg"].copy()  # [36] float32

        # Angle jitter
        if self.augment and self.angle_jitter_deg > 0:
            theta += np.random.normal(
                0, self.angle_jitter_deg, size=theta.shape
            ).astype(np.float32)

        # Convert to torch: [36, 3, 224, 224] (CHW)
        ppl_t = torch.from_numpy(ppl).permute(0, 3, 1, 2)  # uint8
        xpl_t = torch.from_numpy(xpl).permute(0, 3, 1, 2)  # uint8

        # ── Angle dropout (augmentation) ────────────────────────────
        if self.augment and self.angle_dropout:
            p_drop = float(self.angle_dropout.get("p", 0.5))
            K_options = self.angle_dropout.get("K", [6, 9, 12, 18, 36])
            if torch.rand(1).item() < p_drop and K_options:
                K = K_options[torch.randint(len(K_options), (1,)).item()]
                step = max(1, 36 // K)
                off = int(torch.randint(step, (1,)).item())
                indices = torch.arange(off, 36, step)[:K]
                ppl_t = ppl_t[indices]
                xpl_t = xpl_t[indices]
                theta = theta[indices.numpy()]

        result["ppl"] = ppl_t
        result["xpl"] = xpl_t
        result["mask"] = torch.from_numpy(mask)
        result["theta_deg"] = torch.from_numpy(theta)

        return result

    def _load_embedding(self, region_id: str) -> torch.Tensor:
        """Load precomputed embedding for a region.

        Expects ``emb_path`` to be a ``.npy`` file with shape
        ``[N, V, 36, 2, 768]``.
        """
        if self._emb_mmap is None and self.emb_path is not None:
            emb_p = Path(self.emb_path)
            manifest_p = emb_p.parent / "manifest.json"
            if manifest_p.is_file():
                import json
                with open(manifest_p, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                if not manifest.get("complete", False):
                    raise RuntimeError(f"Embedding manifest at {manifest_p} is not marked complete")
            self._emb_mmap = np.load(self.emb_path, mmap_mode="r")

        if self._emb_mmap is None:
            raise RuntimeError(f"Embedding mmap not loaded: {self.emb_path}")

        idx = self.emb_index.get(region_id)
        if idx is None:
            raise KeyError(f"Region {region_id} not found in embedding index")

        # arr[idx] → [V, 36, 2, 768]
        emb = self._emb_mmap[idx].copy()
        return torch.from_numpy(emb)

    def _extract_mask(
        self, label_img: np.ndarray, label_value: int,
        bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        """Extract and resize the region mask."""
        target_px = int(self.cfg.get("dataset", {}).get("input_px", 224))
        ds_cfg = self.cfg.get("dataset", {})
        scale = float(ds_cfg.get("crop_scale", 1.3))
        min_px = int(ds_cfg.get("crop_min_px", 48))
        H, W = label_img.shape[:2]

        x, y, w, h = bbox
        x0, y0, S, pl, pt, pr, pb = _make_square_bbox(
            x, y, w, h, scale, min_px, H, W)

        region_mask = (label_img == label_value).astype(np.uint8) * 255
        mask_resized = _crop_and_resize(
            region_mask, x0, y0, S, pl, pt, pr, pb, target_px,
            interp=cv2.INTER_NEAREST)
        return mask_resized > 127


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def filter_samples(
    regions_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Filter and merge to produce the samples DataFrame (§13.1).

    Returns only regions with type ∈ {grain, opaque, aggregate} and
    small = False, merged with label columns.
    """
    valid_types = {"grain", "opaque", "aggregate"}

    # Filter regions
    mask_type = regions_df["type"].isin(valid_types)
    mask_small = ~regions_df["small"].astype(bool)
    filtered = regions_df[mask_type & mask_small].copy()

    # Merge with labels (left join)
    if "region_id" in labels_df.columns:
        label_cols = ["region_id", "label_fine", "label_group",
                      "label_source", "label_conf", "in_candidate_set",
                      "rule_top1", "rule_p1", "rule_top1_unconstrained",
                      "label_fine_original", "label_group_original",
                      # evaluation set T3 (§16.1) — never used for training
                      "title_label_fine", "title_label_group"]
        existing_cols = [c for c in label_cols if c in labels_df.columns]
        merged = filtered.merge(
            labels_df[existing_cols].drop_duplicates(subset="region_id"),
            on="region_id", how="left")
    else:
        merged = filtered

    logger.info("Filtered samples: %d / %d regions",
                len(merged), len(regions_df))
    return merged


def _parse_bbox(val) -> tuple[int, int, int, int] | None:
    """Parse a bbox_xywh value from various formats.

    V1 FIX: handles NumPy arrays from Parquet list columns.
    """
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        if val.size >= 4:
            return tuple(int(v) for v in val.flat[:4])
        return None
    if isinstance(val, (list, tuple)) and len(val) >= 4:
        return tuple(int(v) for v in val[:4])
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if len(parsed) >= 4:
                return tuple(int(v) for v in parsed[:4])
        except (json.JSONDecodeError, ValueError):
            pass
    if isinstance(val, float) and math.isnan(val):
        return None
    return None


def add_segment_geometry(samples_df: pd.DataFrame, segments_dir: Path) -> pd.DataFrame:
    """Add label_value / bbox_xywh / centroid_xy from the Phase-3 JSONs
    (regions.parquet does not contain them, but crops and masks need them)."""
    rows = []
    for site_id in samples_df["site_id"].unique():
        p = Path(segments_dir) / f"{site_id}.json"
        if not p.is_file():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for r in (data["regions"] if isinstance(data, dict) else data):
            rows.append({"region_id": r["region_id"],
                         "label_value": int(r["label_value"]),
                         "bbox_xywh": list(r["bbox_xywh"]),
                         "centroid_xy": list(r["centroid_xy"])})
    if not rows:
        logger.warning("No segment JSONs found in %s", segments_dir)
        return samples_df
    keep = [c for c in samples_df.columns
            if c not in ("label_value", "bbox_xywh", "centroid_xy")]
    out = samples_df[keep].merge(pd.DataFrame(rows), on="region_id", how="left")
    n_miss = int(out["label_value"].isna().sum())
    if n_miss:
        logger.warning("%d samples have no segment geometry", n_miss)
    return out


def collate_regions(batch: list[dict]) -> dict:
    """Collate samples with DIFFERENT numbers of angles (angle dropout).

    torch's default collate fails when K differs within a batch.  Pads
    emb/ppl/xpl/theta to the batch maximum and returns ``angle_mask``
    [B, K_max] (True = real angle) for the Transformer's key-padding mask.
    """
    out: dict = {}
    K = max(int(b["theta_deg"].shape[0]) for b in batch)
    B = len(batch)
    angle_mask = torch.zeros(B, K, dtype=torch.bool)
    thetas = torch.zeros(B, K, dtype=torch.float32)
    for i, b in enumerate(batch):
        k = int(b["theta_deg"].shape[0])
        angle_mask[i, :k] = True
        thetas[i, :k] = b["theta_deg"]
    out["theta_deg"] = thetas
    out["angle_mask"] = angle_mask
    for key, dim in (("emb", 1), ("ppl", 0), ("xpl", 0)):
        if key in batch[0]:
            ts = []
            for b in batch:
                t = b[key]
                pad = K - t.shape[dim]
                if pad:
                    shape = list(t.shape)
                    shape[dim] = pad
                    t = torch.cat([t, torch.zeros(shape, dtype=t.dtype)], dim=dim)
                ts.append(t)
            out[key] = torch.stack(ts)
    for key in ("label_fine", "label_group"):
        out[key] = torch.tensor([b[key] for b in batch], dtype=torch.long)
    out["weight"] = torch.tensor([b["weight"] for b in batch], dtype=torch.float32)
    for key in ("candidates", "group_members", "mask"):
        if key in batch[0]:
            out[key] = torch.stack([b[key] for b in batch])
    out["prop_targets"] = {k: torch.tensor([b["prop_targets"][k] for b in batch],
                                           dtype=torch.long)
                           for k in batch[0]["prop_targets"]}
    out["region_id"] = [b["region_id"] for b in batch]
    out["specimen"] = [b["specimen"] for b in batch]
    return out


# ══════════════════════════════════════════════════════════════════════════
#  V1 Embedding Dataset (§7b)
# ══════════════════════════════════════════════════════════════════════════

def _hash_seed(seed: int, epoch: int, region_id: str) -> int:
    """Deterministic 32-bit hash combining seed, epoch, and region_id."""
    import hashlib
    key = f"{seed}_{epoch}_{region_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little")


class V1EmbeddingDataset(Dataset):
    """V1 PyTorch Dataset over precomputed rotation embeddings.

    Supports static_k1 (1 view), static_k12 (12 fixed views), and
    rotational arms (12 uniformly spaced views with random offset per region/epoch).
    """

    def __init__(
        self,
        emb_path: Path | str,
        index_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        families: list[str],
        split_codes: set[str] | list[str],
        phase: str = "train",
        arm: str = "rotational",
        seed: int = 0,
        primary_idx: int = 1,
        k12_idx: tuple[int, ...] = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34),
        include_abstain: bool = False,
        samples_path: Path | str | None = None,
    ) -> None:
        self.phase = phase
        self.arm = arm
        self.seed = int(seed)
        self.epoch = 0
        self.primary_idx = int(primary_idx)
        self.k12_idx = tuple(k12_idx)
        self.families = list(families)
        self.fam_idx = {f: i for i, f in enumerate(self.families)}

        emb_p = Path(emb_path)
        if not emb_p.is_file():
            raise FileNotFoundError(f"Embedding cache file not found: {emb_p}")

        # Manifest is REQUIRED — prevents training on partial/benchmark caches
        manifest_p = emb_p.parent / "manifest.json"
        if not manifest_p.is_file():
            raise FileNotFoundError(
                f"Embedding manifest not found: {manifest_p} — run 08_embed.py without --limit")
        import json
        with open(manifest_p, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        if not manifest.get("complete", False):
            raise RuntimeError(
                f"Embedding cache at {manifest_p} is not marked complete — "
                f"run 08_embed.py without --limit first")

        if "N" in manifest and manifest["N"] is not None:
            if len(index_df) != manifest["N"]:
                raise ValueError(
                    f"manifest.N ({manifest['N']}) != len(index_df) ({len(index_df)})"
                )

        # Validate samples.parquet matches embedding manifest
        if samples_path is None and len(emb_p.parents) >= 4:
            cand = emb_p.parents[3] / "data" / "processed" / "v1" / "samples.parquet"
            if cand.is_file():
                samples_path = cand

        if samples_path is not None:
            s_p = Path(samples_path)
            if not s_p.is_file():
                raise FileNotFoundError(f"V1 samples.parquet not found at {s_p}")
            if manifest.get("samples_hash"):
                import hashlib
                with open(s_p, "rb") as sf:
                    s_hash = hashlib.sha256(sf.read()).hexdigest()
                if s_hash != manifest["samples_hash"]:
                    raise ValueError(
                        f"Embedding manifest samples_hash ({manifest['samples_hash']}) does not match "
                        f"{s_p} hash ({s_hash})"
                    )
            if "N" in manifest and manifest["N"] is not None:
                s_df = pd.read_parquet(s_p)
                if manifest["N"] != len(s_df):
                    raise ValueError(
                        f"manifest.N ({manifest['N']}) != len(samples) ({len(s_df)})"
                    )

        self.emb = np.load(emb_p, mmap_mode="r")  # shape [N, 1, 36, 2, 768]

        idx_df = index_df.copy()
        if "cache_row" not in idx_df.columns:
            idx_df["cache_row"] = np.arange(len(idx_df))

        target_cols = [c for c in ["region_id", "code", "state", "allowed", "reason", "top_species", "top_species_p", "metadata_dependent"] if c in targets_df.columns]
        if "code" in idx_df.columns and "code" in target_cols:
            target_cols.remove("code")
        merged = idx_df.merge(targets_df[target_cols], on="region_id", how="inner")

        split_set = set(str(c) for c in split_codes)
        mask = merged["code"].astype(str).isin(split_set)

        if self.phase == "train" or not include_abstain:
            target_states = ("singleton", "partial_set")
        else:
            target_states = TEST_PREDICTED_STATES

        mask &= merged["state"].isin(target_states)
        self.rows = merged[mask].reset_index(drop=True)

        # Assert every required region of the split exists in the index
        expected_regions = set(
            targets_df[
                targets_df["code"].astype(str).isin(split_set)
                & targets_df["state"].isin(target_states)
            ]["region_id"]
        )
        cached_regions = set(idx_df["region_id"])
        missing = expected_regions - cached_regions
        if missing:
            state_desc = "/".join(target_states)
            raise RuntimeError(
                f"{len(missing)} {state_desc} regions for split ({self.phase}) missing from embedding cache "
                f"(have {len(cached_regions)}, need {len(expected_regions)}). "
                f"First 5: {sorted(missing)[:5]}. Re-run 08_embed.py without --limit."
            )

        logger.info(
            "V1EmbeddingDataset initialized: phase=%s, arm=%s, %d samples (from %d split codes)",
            self.phase, self.arm, len(self.rows), len(split_set),
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows.iloc[i]
        reg_id = str(row["region_id"])

        if self.phase == "train":
            rng = np.random.default_rng(_hash_seed(self.seed, self.epoch, reg_id))
            o = int(rng.integers(36))
            idx = [(o + 3 * j) % 36 for j in range(12)]
        else:
            idx = [self.primary_idx] if self.arm == "static_k1" else list(self.k12_idx)

        cache_row = int(row["cache_row"])
        x = torch.from_numpy(self.emb[cache_row, 0, idx].astype(np.float32))  # [K, 2, 768]
        emb_ppl = x[:, 0]
        emb_xpl = x[:, 1]

        theta = torch.tensor([5.0 * k for k in idx], dtype=torch.float32)

        allowed = torch.zeros(len(self.families), dtype=torch.bool)
        raw_allowed = row["allowed"]
        if isinstance(raw_allowed, (list, np.ndarray)):
            for f in raw_allowed:
                if f in self.fam_idx:
                    allowed[self.fam_idx[f]] = True

        return {
            "emb_ppl": emb_ppl,
            "emb_xpl": emb_xpl,
            "theta_deg": theta,
            "allowed": allowed,
            "state": str(row["state"]),
            "region_id": reg_id,
            "specimen": str(row["code"]),
        }

