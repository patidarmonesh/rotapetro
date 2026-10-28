#!/usr/bin/env python3
"""Phase 6 QC: V1 Label Visual Review Generator with Masks & IDs.

Script: scripts/visualize_labels_v1.py
Ownership: Agent Q (QC, Reports, and Cleanup)

Features:
- Category from state only: singleton / partial_set / abstain / excluded / mixed.
- Conflict styling only for reason in ("list_conflict", "list_conflict_optics_kept").
- Empty panel prints "0 list conflicts".
- Metadata dependence is separate marker (dashed contour or "[M]" tag);
  does NOT change category or singleton counts.
- Out-of-scope stays under abstains; hard_contradiction as sub-panel.
- Output path: outputs/v1/qc/label_visualizations/fold_{k}/...
- Angle views: --angles (default 1,10,19,28) showing PPL and XPL crops from aligned/{site}.npz.
- Uses label_value from segment JSON instead of parsing strings.
- Fails loudly: missing maps -> print warning and skip site.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.rotapetro.io import load_config, resolve_v1_paths
except ImportError:
    load_config = None
    resolve_v1_paths = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("visualize_labels_v1")

# ── Color Palette (BGR for OpenCV) ───────────────────────────────────────────
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "singleton": (46, 184, 46),         # Vibrant Green
    "partial_set": (220, 180, 0),       # Bright Cyan / Teal
    "abstain": (0, 140, 255),           # Vivid Amber / Orange
    "excluded": (90, 90, 190),          # Muted Crimson / Gray
    "mixed": (0, 215, 255),             # Golden Yellow
    "list_conflict": (200, 50, 200),    # Magenta / Purple (conflict styling only)
    "hard_contradiction": (0, 0, 220),  # Deep Red (sub-panel only)
}

CATEGORY_TITLES: dict[str, str] = {
    "singleton": "Singletons (High-Confidence Primary Classes)",
    "partial_set": "Partial Sets (Multi-Candidate Supervised Sets)",
    "abstain": "Abstains (Diffuse, No Positive Cue, or Out-Of-Scope)",
    "excluded": "Excluded (Quality Gate Rejections: Small Area / Low R2_4)",
    "mixed": "Mixed Minerals (Aggregates / Fine Aggregates)",
    "list_conflict": "List Conflicts (Candidate List vs Optical Decision)",
    "hard_contradiction": "Hard Contradictions (Optical Evidence Contradicts Metadata)",
}


# ── Pure Categorisation and Styling Functions ─────────────────────────────────

def categorize_region(state: str) -> str:
    """Categorize region strictly from its state string alone."""
    st = str(state).strip().lower()
    if st in ("singleton", "partial_set", "abstain", "excluded", "mixed"):
        return st
    return "excluded"


def is_conflict_styled(reason: str | None) -> bool:
    """Check if a region receives conflict styling (magenta / conflict tag)."""
    if not reason:
        return False
    return str(reason).strip() in ("list_conflict", "list_conflict_optics_kept")


def is_metadata_dependent(metadata_dependent: object) -> bool:
    """Check if region is metadata dependent without altering its state category."""
    if metadata_dependent is None or pd.isna(metadata_dependent):
        return False
    return bool(metadata_dependent)


def is_hard_contradiction(reason: str | None) -> bool:
    """Check if region decision reason is hard_contradiction."""
    if not reason:
        return False
    return str(reason).strip() == "hard_contradiction"


def is_out_of_scope(reason: str | None) -> bool:
    """Check if region decision reason is out-of-scope (retained under abstains)."""
    if not reason:
        return False
    return str(reason).strip() in ("out_of_scope", "out_of_scope_plausible")


# ── CLI Parsing ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix B1: Generate V1 label visual review panels with masks, contours, angles, and IDs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="0",
        help="Target fold index (0-4) or name (e.g. fold_0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of grain cards to display in each category inspection panel",
    )
    parser.add_argument(
        "--sites",
        type=str,
        default=None,
        help="Optional comma-separated list of pilot site IDs (e.g. 03agt06_r1,03agt06_r2)",
    )
    parser.add_argument(
        "--angles",
        type=str,
        default="1,10,19,28",
        help="Comma-separated frame indices to show for angle views (default: 1,10,19,28)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/v1/qc/label_visualizations",
        help="Base directory where output panels and overlays are saved",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output panels",
    )
    return parser.parse_args()


# ── Formatting Helpers ────────────────────────────────────────────────────────

def format_allowed(val: object) -> str:
    """Format allowed minerals array/list cleanly into a comma-separated string."""
    if val is None:
        return "none"
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return "none"
        return ", ".join(str(x) for x in val)
    if pd.isna(val):
        return "none"
    s = str(val).strip()
    return s if s else "none"


def wrap_text(text: str, max_chars: int = 34) -> list[str]:
    """Wrap a string into lines that do not exceed max_chars."""
    words = str(text).replace("\n", " ").split()
    if not words:
        return ["n/a"]
    lines: list[str] = []
    current = words[0]
    for w in words[1:]:
        if len(current) + 1 + len(w) <= max_chars:
            current += " " + w
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines


def format_reason(row: pd.Series, category: str) -> str:
    """Extract a concise, informative reason for the grain card."""
    raw_reason = str(row.get("reason", "")).strip()
    if raw_reason in ("", "none", "nan") and "reject_reason_v1" in row:
        raw_reason = str(row.get("reject_reason_v1", "")).strip()

    if category == "list_conflict" or is_conflict_styled(raw_reason):
        st = row.get("state", "unk")
        opt_st = row.get("optics_state", "unk")
        opt_al = format_allowed(row.get("optics_allowed"))
        dec = row.get("deciding_terms")
        dec_str = f" | {format_allowed(dec)}" if (dec is not None and len(dec) > 0) else ""
        return f"{raw_reason}: meta {st} vs optics {opt_st} ({opt_al}){dec_str}"

    if category == "singleton":
        dec = row.get("deciding_terms")
        if dec is not None and isinstance(dec, (list, np.ndarray)) and len(dec) > 0:
            return f"cues: {', '.join(str(d) for d in dec)}"
        if raw_reason:
            return raw_reason
        return "passed singleton threshold"

    if category == "partial_set":
        dec = row.get("deciding_terms")
        if dec is not None and isinstance(dec, (list, np.ndarray)) and len(dec) > 0:
            return f"cues: {', '.join(str(d) for d in dec)}"
        return "set cumulative prob"

    if category == "abstain":
        return raw_reason if raw_reason else "diffuse posteriors"

    if category == "excluded":
        return raw_reason if raw_reason else "quality gate failed"

    return raw_reason if raw_reason else "n/a"


def _add_segment_geometry(regions_df: pd.DataFrame, seg_dir: Path) -> pd.DataFrame:
    """Add label_value / bbox_xywh / centroid_xy from Phase 3 JSONs without string parsing."""
    rows = []
    for site_id in regions_df["site_id"].unique():
        p = seg_dir / f"{site_id}.json"
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            for r in (data["regions"] if isinstance(data, dict) else data):
                rows.append({
                    "region_id": r["region_id"],
                    "label_value": r["label_value"],
                    "bbox_xywh": r.get("bbox_xywh"),
                    "centroid_xy": r.get("centroid_xy"),
                })
        except Exception as e:
            logger.warning("Could not read segment JSON for %s: %s", site_id, e)
    if not rows:
        return regions_df
    geo = pd.DataFrame(rows)
    keep = [c for c in regions_df.columns if c not in ("label_value", "bbox_xywh", "centroid_xy")]
    return regions_df[keep].merge(geo, on="region_id", how="left")


# ── Asset Loading ─────────────────────────────────────────────────────────────

class AssetCache:
    """Loads and caches site images (PPL, XPL), aligned stacks, and segment JSONs."""

    def __init__(self, root: Path, interim_dir: Path):
        self.root = root
        self.interim_dir = interim_dir
        self.maps_dir = interim_dir / "maps"
        self.segments_dir = interim_dir / "segments"
        self.aligned_dir = interim_dir / "aligned"
        self._cache: dict[str, dict[str, object] | None] = {}
        self._seg_json_cache: dict[str, dict[str, int]] = {}

    def get_label_value(self, site_id: str, region_id: str) -> int | None:
        """Lookup label_value directly from segment JSON without parsing region_id."""
        if site_id not in self._seg_json_cache:
            p = self.segments_dir / f"{site_id}.json"
            mapping: dict[str, int] = {}
            if p.is_file():
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    regs = data["regions"] if isinstance(data, dict) else data
                    for r in regs:
                        mapping[str(r["region_id"])] = int(r["label_value"])
                except Exception as e:
                    logger.warning("Could not read segment JSON for %s: %s", site_id, e)
            self._seg_json_cache[site_id] = mapping
        return self._seg_json_cache[site_id].get(region_id)

    def get_site_assets(self, site_id: str) -> dict[str, object] | None:
        """Returns site assets or None if maps missing (fail loudly)."""
        if site_id in self._cache:
            return self._cache[site_id]

        maps_path = self.maps_dir / f"{site_id}.npz"
        if not maps_path.is_file():
            logger.warning("Missing maps for site %s at %s; skipping site.", site_id, maps_path)
            self._cache[site_id] = None
            return None

        ppl_img = None
        xpl_img = None
        try:
            npz = np.load(maps_path)
            if "ppl_mean_rgb" in npz:
                ppl_img = cv2.cvtColor(npz["ppl_mean_rgb"], cv2.COLOR_RGB2BGR)
            if "xpl_maxproj_rgb" in npz:
                xpl_img = cv2.cvtColor(npz["xpl_maxproj_rgb"], cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning("Error loading maps for %s: %s; skipping site.", site_id, e)
            self._cache[site_id] = None
            return None

        if ppl_img is None or xpl_img is None:
            logger.warning("Maps for %s missing ppl_mean_rgb or xpl_maxproj_rgb; skipping site.", site_id)
            self._cache[site_id] = None
            return None

        mask_path = self.segments_dir / f"{site_id}.png"
        mask = None
        if mask_path.is_file():
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        aligned_data = None
        aligned_path = self.aligned_dir / f"{site_id}.npz"
        if aligned_path.is_file():
            try:
                al_npz = np.load(aligned_path)
                aligned_data = {
                    "ppl": al_npz.get("ppl"),
                    "xpl": al_npz.get("xpl"),
                }
            except Exception as e:
                logger.warning("Could not load aligned stack for %s: %s", site_id, e)

        data = {
            "mask": mask,
            "ppl": ppl_img,
            "xpl": xpl_img,
            "aligned": aligned_data,
        }
        self._cache[site_id] = data
        return data


# ── Grain Card Generator ──────────────────────────────────────────────────────

def render_grain_card(
    row: pd.Series,
    category: str,
    assets: dict[str, object] | None,
    angles: list[int] | None = None,
    asset_cache: AssetCache | None = None,
    tile_size: int = 68,
    card_height: int = 240,
) -> np.ndarray | None:
    """Renders a single grain inspection card with multi-angle PPL/XPL crops, contour, mask, and metadata."""
    if assets is None:
        return None

    mask_full = assets.get("mask")
    ppl_full = assets.get("ppl")
    xpl_full = assets.get("xpl")
    if mask_full is None or ppl_full is None or xpl_full is None:
        return None

    site_id = str(row["site_id"])
    region_id = str(row["region_id"])

    # Retrieve label_value without parsing string
    label_value = row.get("label_value")
    if label_value is not None and not (isinstance(label_value, float) and np.isnan(label_value)):
        g_val = int(label_value)
    elif asset_cache is not None:
        g_val = asset_cache.get_label_value(site_id, region_id)
    else:
        g_val = None

    if g_val is None:
        logger.warning("Could not resolve label_value for %s", region_id)
        return None

    grain_mask = (mask_full == g_val)
    if not np.any(grain_mask):
        return None

    ys, xs = np.where(grain_mask)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())

    pad_x = max(6, int((x_max - x_min + 1) * 0.20))
    pad_y = max(6, int((y_max - y_min + 1) * 0.20))
    h_img, w_img = mask_full.shape[:2]
    bx0 = max(0, x_min - pad_x)
    by0 = max(0, y_min - pad_y)
    bx1 = min(w_img, x_max + 1 + pad_x)
    by1 = min(h_img, y_max + 1 + pad_y)

    crop_mask = grain_mask[by0:by1, bx0:bx1]
    crop_mask_u8 = (crop_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(crop_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Determine visual category and conflict styling
    region_cat = categorize_region(row.get("state", category))
    is_conflict = is_conflict_styled(row.get("reason"))
    meta_dep = is_metadata_dependent(row.get("metadata_dependent"))

    cat_color = CATEGORY_COLORS["list_conflict"] if is_conflict else CATEGORY_COLORS.get(region_cat, (100, 100, 100))

    # Build angle view tiles
    angle_tiles_ppl: list[np.ndarray] = []
    angle_tiles_xpl: list[np.ndarray] = []
    angles_to_use = angles if (angles and len(angles) > 0) else [1, 10, 19, 28]

    al_data = assets.get("aligned")
    has_aligned = (
        isinstance(al_data, dict)
        and al_data.get("ppl") is not None
        and al_data.get("xpl") is not None
    )

    if has_aligned:
        ppl_stack = al_data["ppl"]
        xpl_stack = al_data["xpl"]
        n_frames = len(ppl_stack)
        for ang_idx in angles_to_use:
            idx = int(ang_idx) % n_frames
            # Aligned stacks are in RGB format
            raw_ppl = ppl_stack[idx][by0:by1, bx0:bx1]
            raw_xpl = xpl_stack[idx][by0:by1, bx0:bx1]
            bgr_ppl = cv2.cvtColor(raw_ppl, cv2.COLOR_RGB2BGR)
            bgr_xpl = cv2.cvtColor(raw_xpl, cv2.COLOR_RGB2BGR)

            tile_ppl = bgr_ppl.copy()
            cv2.drawContours(tile_ppl, contours, -1, cat_color, 1)

            tile_xpl = bgr_xpl.copy()
            cv2.drawContours(tile_xpl, contours, -1, cat_color, 1)

            t_ppl = cv2.resize(tile_ppl, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
            t_xpl = cv2.resize(tile_xpl, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

            # Lighting + Angle badge
            cv2.rectangle(t_ppl, (2, 2), (48, 14), (20, 20, 20), -1)
            cv2.putText(t_ppl, f"PPL@{idx}", (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.rectangle(t_xpl, (2, 2), (48, 14), (20, 20, 20), -1)
            cv2.putText(t_xpl, f"XPL@{idx}", (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

            angle_tiles_ppl.append(t_ppl)
            angle_tiles_xpl.append(t_xpl)
    else:
        # Fallback to mean PPL and maxproj XPL from maps
        crop_ppl = ppl_full[by0:by1, bx0:bx1].copy()
        crop_xpl = xpl_full[by0:by1, bx0:bx1].copy()
        tile_ppl = crop_ppl.copy()
        cv2.drawContours(tile_ppl, contours, -1, cat_color, 1)
        tile_xpl = crop_xpl.copy()
        cv2.drawContours(tile_xpl, contours, -1, cat_color, 1)

        t_ppl = cv2.resize(tile_ppl, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)
        t_xpl = cv2.resize(tile_xpl, (tile_size, tile_size), interpolation=cv2.INTER_LINEAR)

        cv2.rectangle(t_ppl, (2, 2), (32, 14), (20, 20, 20), -1)
        cv2.putText(t_ppl, "PPL", (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(t_xpl, (2, 2), (32, 14), (20, 20, 20), -1)
        cv2.putText(t_xpl, "XPL", (4, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)

        angle_tiles_ppl.append(t_ppl)
        angle_tiles_xpl.append(t_xpl)

    n_cols = len(angle_tiles_ppl)
    gap = 4
    card_width = max(280, n_cols * tile_size + (n_cols + 1) * gap)

    card = np.ones((card_height, card_width, 3), dtype=np.uint8) * 248
    cv2.rectangle(card, (0, 0), (card_width - 1, card_height - 1), (205, 205, 210), 1)

    # Place PPL and XPL tile rows
    y_ppl = 5
    y_xpl = y_ppl + tile_size + gap
    for i in range(n_cols):
        x_tile = gap + i * (tile_size + gap)
        card[y_ppl:y_ppl + tile_size, x_tile:x_tile + tile_size] = angle_tiles_ppl[i]
        card[y_xpl:y_xpl + tile_size, x_tile:x_tile + tile_size] = angle_tiles_xpl[i]

    sep_y = y_xpl + tile_size + 4
    cv2.line(card, (4, sep_y), (card_width - 5, sep_y), (210, 210, 215), 1)

    # Metadata text section
    text_y = sep_y + 13
    id_txt = f"ID: {region_id}"
    if meta_dep:
        id_txt += "  [M]"
    if is_conflict:
        id_txt += "  [CONFLICT]"
    cv2.putText(card, id_txt, (6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (10, 10, 10), 1, cv2.LINE_AA)

    text_y += 14
    st_val = str(row.get("state", "n/a"))
    cv2.putText(card, f"State: {st_val}", (6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.32, cat_color, 1, cv2.LINE_AA)

    text_y += 14
    minerals_str = format_allowed(row.get("allowed"))
    cv2.putText(card, f"Mineral(s): {minerals_str}", (6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.31, (30, 30, 30), 1, cv2.LINE_AA)

    text_y += 13
    reason_str = format_reason(row, category)
    reason_lines = wrap_text(f"Reason: {reason_str}", max_chars=int(card_width / 8.5))
    for r_line in reason_lines[:2]:
        cv2.putText(card, r_line, (6, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.29, (80, 80, 85), 1, cv2.LINE_AA)
        text_y += 12

    return card


def pad_card(
    card: np.ndarray,
    target_h: int,
    target_w: int,
    bg_color: tuple[int, int, int] = (248, 248, 248),
) -> np.ndarray:
    """Pad card to target_h and target_w with background color to ensure uniform dimensions."""
    h, w = card.shape[:2]
    if h == target_h and w == target_w:
        return card

    padded = np.full((target_h, target_w, 3), bg_color, dtype=np.uint8)
    cv2.rectangle(padded, (0, 0), (target_w - 1, target_h - 1), (205, 205, 210), 1)

    copy_h = min(h, target_h)
    copy_w = min(w, target_w)
    padded[:copy_h, :copy_w] = card[:copy_h, :copy_w]
    return padded


# ── Category Panel Assembly ───────────────────────────────────────────────────

def assemble_category_panel(
    cards: list[np.ndarray],
    category: str,
    fold_name: str,
    n_cols: int = 4,
    margin: int = 8,
    fixed_width: int | None = None,
    fixed_height: int | None = None,
) -> np.ndarray:
    """Assembles grain cards into a structured inspection panel with header.
    Pads every card to a uniform fixed width and height before placement.
    """
    n_cards = len(cards)
    if n_cards == 0:
        empty = np.ones((160, 640, 3), dtype=np.uint8) * 242
        cv2.rectangle(empty, (0, 0), (640, 50), (35, 30, 25), -1)
        cv2.putText(
            empty,
            f"RotaPetro V1 QC Inspection — {category.title()} ({fold_name})",
            (20, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        msg = "0 list conflicts" if category in ("list_conflict", "list_conflicts") else f"0 {category}"
        cv2.putText(
            empty,
            msg,
            (40, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 0, 180),
            2,
            cv2.LINE_AA,
        )
        return empty

    card_h = fixed_height if fixed_height is not None else max(c.shape[0] for c in cards)
    card_w = fixed_width if fixed_width is not None else max(c.shape[1] for c in cards)
    padded_cards = [pad_card(c, card_h, card_w) for c in cards]

    cols = min(n_cols, n_cards)
    rows = (n_cards + cols - 1) // cols

    header_h = 60
    panel_w = cols * card_w + (cols + 1) * margin
    panel_h = header_h + rows * card_h + (rows + 1) * margin

    panel = np.ones((panel_h, panel_w, 3), dtype=np.uint8) * 242

    # Draw header banner
    cv2.rectangle(panel, (0, 0), (panel_w, header_h), (35, 30, 25), -1)
    cat_title = CATEGORY_TITLES.get(category, category.title())
    cat_color = CATEGORY_COLORS.get(category, (255, 255, 255))

    cv2.rectangle(panel, (15, 16), (27, 44), cat_color, -1)
    cv2.putText(
        panel,
        f"RotaPetro V1 QC Inspection — {cat_title}",
        (38, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"Fold: {fold_name} | Showing {n_cards} Grains with Multi-Angle Views, IDs, and Decisions",
        (38, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (180, 195, 205),
        1,
        cv2.LINE_AA,
    )

    for idx, card in enumerate(padded_cards):
        r_idx = idx // cols
        c_idx = idx % cols
        x0 = margin + c_idx * (card_w + margin)
        y0 = header_h + margin + r_idx * (card_h + margin)
        panel[y0:y0 + card_h, x0:x0 + card_w] = card

    return panel


# ── Pilot Site Full Slide Overlay ─────────────────────────────────────────────

def render_site_overlay(
    site_id: str,
    site_targets: pd.DataFrame,
    assets: dict[str, object] | None,
    fold_name: str,
    specimen_code: str = "",
    asset_cache: AssetCache | None = None,
) -> np.ndarray | None:
    """Generates a full-slide annotated visual review overlay with side legend for a pilot site."""
    if assets is None:
        return None

    mask_full = assets.get("mask")
    ppl_full = assets.get("ppl")
    if mask_full is None or ppl_full is None:
        return None

    scale = 2
    h_orig, w_orig = ppl_full.shape[:2]
    h_up, w_up = h_orig * scale, w_orig * scale

    ppl_up = cv2.resize(ppl_full, (w_up, h_up), interpolation=cv2.INTER_LINEAR)
    mask_up = cv2.resize(mask_full, (w_up, h_up), interpolation=cv2.INTER_NEAREST)

    overlay = ppl_up.copy()
    target_dict = {str(r["region_id"]): r for _, r in site_targets.iterrows()}

    # Resolve val to region_id mapping directly from segment JSON
    seg_json_path = (
        asset_cache.segments_dir / f"{site_id}.json"
        if asset_cache
        else _PROJECT_ROOT / "data/interim/segments" / f"{site_id}.json"
    )
    val_to_region: dict[int, str] = {}
    if seg_json_path.is_file():
        try:
            data = json.loads(seg_json_path.read_text(encoding="utf-8"))
            regs = data["regions"] if isinstance(data, dict) else data
            for r in regs:
                val_to_region[int(r["label_value"])] = str(r["region_id"])
        except Exception as e:
            logger.warning("Error reading segment JSON for %s: %s", site_id, e)

    unique_vals = np.unique(mask_up)
    category_counts: dict[str, int] = {
        "singleton": 0,
        "partial_set": 0,
        "abstain": 0,
        "excluded": 0,
        "mixed": 0,
    }
    conflict_count = 0
    meta_dep_count = 0
    mineral_singleton_counts: dict[str, int] = {}

    for val in unique_vals:
        if val == 0:
            continue
        g_id = val_to_region.get(val)
        if not g_id:
            continue
        row = target_dict.get(g_id)
        if row is None:
            continue

        # Category from state ONLY
        cat = categorize_region(row.get("state", "excluded"))
        category_counts[cat] = category_counts.get(cat, 0) + 1

        if cat == "singleton":
            min_str = format_allowed(row.get("allowed"))
            mineral_singleton_counts[min_str] = mineral_singleton_counts.get(min_str, 0) + 1

        is_conflict = is_conflict_styled(row.get("reason"))
        if is_conflict:
            conflict_count += 1

        meta_dep = is_metadata_dependent(row.get("metadata_dependent"))
        if meta_dep:
            meta_dep_count += 1

        # Conflict styling only if reason is list_conflict / list_conflict_optics_kept
        color = CATEGORY_COLORS["list_conflict"] if is_conflict else CATEGORY_COLORS.get(cat, (120, 120, 120))

        region_px = (mask_up == val)
        if not np.any(region_px):
            continue

        overlay[region_px] = color

        reg_u8 = (region_px.astype(np.uint8)) * 255
        cnts, _ = cv2.findContours(reg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cnts, -1, (255, 255, 255), 1)

        # Draw centroid label
        M = cv2.moments(reg_u8)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            short_id = f"g{val:03d}"
            m_tag = " [M]" if meta_dep else ""
            if cat == "singleton":
                min_label = format_allowed(row.get("allowed"))[:3].capitalize()
                txt = f"{short_id} {min_label}{m_tag}"
            elif cat == "partial_set":
                txt = f"{short_id} Set{m_tag}"
            elif is_conflict:
                txt = f"{short_id} !{m_tag}"
            else:
                txt = f"{short_id}{m_tag}"

            font = cv2.FONT_HERSHEY_SIMPLEX
            f_scale = 0.30
            thick = 1
            t_size, _ = cv2.getTextSize(txt, font, f_scale, thick)
            bx0 = cx - t_size[0] // 2 - 2
            by0 = cy - t_size[1] // 2 - 2
            bx1 = cx + t_size[0] // 2 + 2
            by1 = cy + t_size[1] // 2 + 2

            cv2.rectangle(overlay, (bx0, by0), (bx1, by1), (15, 15, 15), -1)
            cv2.putText(overlay, txt, (bx0 + 2, by1 - 3), font, f_scale, (255, 255, 255), thick, cv2.LINE_AA)

    alpha = 0.30
    blended_slide = cv2.addWeighted(overlay, alpha, ppl_up, 1.0 - alpha, 0)

    # Side legend panel
    legend_w = 400
    total_w = w_up + legend_w
    final_canvas = np.ones((h_up, total_w, 3), dtype=np.uint8) * 245
    final_canvas[:, :w_up] = blended_slide
    cv2.line(final_canvas, (w_up, 0), (w_up, h_up), (200, 200, 205), 2)

    lx = w_up + 20
    ly = 35

    cv2.putText(final_canvas, f"Site QC: {site_id}", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 25), 2, cv2.LINE_AA)
    ly += 25
    cv2.putText(final_canvas, f"Specimen: {specimen_code} | Fold: {fold_name}", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 95), 1, cv2.LINE_AA)
    ly += 16
    cv2.line(final_canvas, (lx, ly), (total_w - 20, ly), (210, 210, 215), 1)

    ly += 30
    cv2.putText(final_canvas, "Category Breakdown (State Only)", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 25), 1, cv2.LINE_AA)
    ly += 22

    total_regions = len(site_targets)

    for cat_key in ["singleton", "partial_set", "abstain", "excluded", "mixed"]:
        c = CATEGORY_COLORS[cat_key]
        cnt = category_counts.get(cat_key, 0)
        pct = 100.0 * cnt / max(total_regions, 1)
        name = cat_key.replace("_", " ").title()

        cv2.rectangle(final_canvas, (lx, ly - 14), (lx + 24, ly + 4), c, -1)
        cv2.rectangle(final_canvas, (lx, ly - 14), (lx + 24, ly + 4), (100, 100, 100), 1)
        cv2.putText(final_canvas, f"{name}: {cnt} ({pct:.1f}%)", (lx + 34, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (15, 15, 20), 1, cv2.LINE_AA)
        ly += 26

    ly += 6
    cv2.line(final_canvas, (lx, ly), (total_w - 20, ly), (210, 210, 215), 1)
    ly += 22
    cv2.putText(final_canvas, f"List Conflicts Styled: {conflict_count}", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, CATEGORY_COLORS["list_conflict"], 1, cv2.LINE_AA)
    ly += 20
    cv2.putText(final_canvas, f"Metadata-Dependent [M]: {meta_dep_count}", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 65), 1, cv2.LINE_AA)
    ly += 12
    cv2.line(final_canvas, (lx, ly), (total_w - 20, ly), (210, 210, 215), 1)

    ly += 28
    cv2.putText(final_canvas, "Singletons by Mineral", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (20, 20, 25), 1, cv2.LINE_AA)
    ly += 20
    if mineral_singleton_counts:
        for min_name, m_cnt in sorted(mineral_singleton_counts.items(), key=lambda x: -x[1]):
            cv2.putText(final_canvas, f"  • {min_name}: {m_cnt}", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 40, 45), 1, cv2.LINE_AA)
            ly += 20
    else:
        cv2.putText(final_canvas, "  • None in this slide", (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 125), 1, cv2.LINE_AA)
        ly += 20

    return final_canvas


# ── Main Execution Flow ───────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    config_path = _PROJECT_ROOT / args.config
    if not config_path.is_file():
        logger.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    cfg = load_config(config_path) if load_config else {}

    fold_arg = str(args.fold).strip()
    fold_name = fold_arg if fold_arg.startswith("fold_") else f"fold_{fold_arg}"

    # Target output path: outputs/v1/qc/label_visualizations/fold_{k}/...
    outdir = _PROJECT_ROOT / args.outdir / fold_name
    outdir.mkdir(parents=True, exist_ok=True)

    v1_paths = resolve_v1_paths(cfg) if resolve_v1_paths else {}
    proc_dir = _PROJECT_ROOT / v1_paths.get("processed", "data/processed/v1")
    targets_path = proc_dir / "teachers" / fold_name / "targets.parquet"
    regions_path = proc_dir / "regions.parquet"

    if not targets_path.is_file():
        logger.error("V1 fold targets not found at: %s", targets_path)
        sys.exit(1)
    if not regions_path.is_file():
        logger.error("V1 regions not found at: %s", regions_path)
        sys.exit(1)

    logger.info("Loading targets from %s", targets_path)
    df_targets = pd.read_parquet(targets_path)
    logger.info("Loaded %d targets (%s)", len(df_targets), fold_name)

    logger.info("Loading regions from %s", regions_path)
    df_regions = pd.read_parquet(regions_path)
    logger.info("Loaded %d regions", len(df_regions))

    interim_dir = _PROJECT_ROOT / cfg.get("paths", {}).get("interim", "data/interim")
    segments_dir = interim_dir / "segments"
    df_regions = _add_segment_geometry(df_regions, segments_dir)

    merged = df_targets.merge(
        df_regions[
            [c for c in ["region_id", "area_px", "area_um2", "aspect", "eligible_v1", "reject_reason_v1", "label_value"] if c in df_regions.columns]
        ].drop_duplicates("region_id"),
        on="region_id",
        how="left",
    )

    asset_cache = AssetCache(_PROJECT_ROOT, interim_dir)

    angles = [int(a.strip()) for a in args.angles.split(",") if a.strip()]

    # Select Pilot Sites
    pilot_sites: list[str] = []
    if args.sites:
        pilot_sites = [s.strip() for s in args.sites.split(",") if s.strip()]
    elif cfg.get("v1", {}).get("pilot_sites"):
        pilot_sites = list(cfg["v1"]["pilot_sites"])
    elif cfg.get("pilot_sites"):
        pilot_sites = list(cfg["pilot_sites"])
    else:
        pilot_sites = list(df_targets["site_id"].unique()[:3])

    logger.info("Selected pilot sites (%d): %s", len(pilot_sites), pilot_sites)

    manifest_records: dict[str, object] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "fold": fold_name,
        "pilot_sites": pilot_sites,
        "angles": angles,
        "category_panels": {},
        "site_overlays": {},
    }

    limit = max(1, args.limit)

    # Categories to generate panels for
    # Primary categories from state only:
    primary_categories = ["singleton", "partial_set", "abstain", "excluded", "mixed"]
    # Sub-panels for conflicts and hard contradictions:
    sub_panels = ["list_conflict", "hard_contradiction"]

    for cat in primary_categories + sub_panels:
        logger.info("Generating visual review panel for category: %s (limit: %d)", cat, limit)

        if cat in primary_categories:
            cat_df = merged[merged["state"] == cat]
        elif cat == "list_conflict":
            cat_df = merged[merged["reason"].isin(["list_conflict", "list_conflict_optics_kept"])]
        elif cat == "hard_contradiction":
            cat_df = merged[merged["reason"] == "hard_contradiction"]
        else:
            cat_df = pd.DataFrame()

        cards: list[np.ndarray] = []
        if not cat_df.empty:
            pilot_sample = cat_df[cat_df["site_id"].isin(pilot_sites)]
            other_sample = cat_df[~cat_df["site_id"].isin(pilot_sites)]

            n_from_pilot = min(len(pilot_sample), limit)
            selected_rows = [pilot_sample.iloc[i] for i in range(n_from_pilot)]

            n_needed = limit - len(selected_rows)
            if n_needed > 0 and len(other_sample) > 0:
                sample_other = other_sample.sample(
                    n=min(n_needed, len(other_sample)),
                    random_state=42,
                )
                for _, r in sample_other.iterrows():
                    selected_rows.append(r)

            for r in selected_rows:
                sid = str(r["site_id"])
                site_assets = asset_cache.get_site_assets(sid)
                if site_assets is None or site_assets.get("ppl") is None or site_assets.get("xpl") is None:
                    logger.warning("Skipping region %s at site %s: missing maps", r.get("region_id"), sid)
                    continue
                card = render_grain_card(
                    r, cat, site_assets, angles=angles, asset_cache=asset_cache
                )
                if card is not None:
                    cards.append(card)

        # Assemble panel (even if 0 cards, prints empty message, e.g. "0 list conflicts")
        panel_img = assemble_category_panel(cards, cat, fold_name, n_cols=4)
        panel_path = outdir / f"panel_{cat}.png"
        cv2.imwrite(str(panel_path), panel_img)
        logger.info("Saved %s (%dx%d, %d cards)", panel_path.name, panel_img.shape[1], panel_img.shape[0], len(cards))
        manifest_records["category_panels"][cat] = {
            "path": str(panel_path),
            "n_cards": len(cards),
            "dimensions": [panel_img.shape[1], panel_img.shape[0]],
        }

    # Generate Full-Slide Overlays for Pilot Sites
    logger.info("Generating full-slide annotated overlays for %d pilot sites...", len(pilot_sites))
    for sid in pilot_sites:
        sub_targets = merged[merged["site_id"] == sid]
        if sub_targets.empty:
            logger.warning("No targets found for pilot site: %s", sid)
            continue

        site_assets = asset_cache.get_site_assets(sid)
        if site_assets is None or site_assets.get("ppl") is None or site_assets.get("xpl") is None:
            logger.warning("Skipping pilot site overlay %s due to missing maps", sid)
            continue

        code = str(sub_targets.iloc[0].get("code", sid.split("_")[0]))
        slide_img = render_site_overlay(
            sid, sub_targets, site_assets, fold_name, specimen_code=code, asset_cache=asset_cache
        )
        if slide_img is not None:
            slide_path = outdir / f"{sid}_labeled_slide.png"
            cv2.imwrite(str(slide_path), slide_img)
            logger.info("Saved slide overlay: %s", slide_path.name)
            manifest_records["site_overlays"][sid] = {
                "path": str(slide_path),
                "n_regions": len(sub_targets),
                "dimensions": [slide_img.shape[1], slide_img.shape[0]],
            }

    # Write Manifest Metadata
    manifest_path = outdir / "visualization_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_records, f, indent=2)
    logger.info("Saved visualization manifest to %s", manifest_path)
    logger.info("V1 Label Visual Review successfully generated in %s", outdir)


if __name__ == "__main__":
    main()
