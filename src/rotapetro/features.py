"""Phase 4 — Grain property cards / feature extraction (§10).

Module: src/rotapetro/features.py
Inputs:  aligned stacks (§7.2), maps (§8.6), segments (§9.5)
Outputs: data/processed/regions.parquet  (one row per non-hole region)

Computes all per-region features from §10.1–10.8.  Robust statistics
exclude a 2-px border band and use ``aniso`` as weight where noted.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage
from skimage.filters import sato
from skimage.measure import regionprops
from skimage.morphology import skeletonize
from skimage.transform import probabilistic_hough_line

logger = logging.getLogger(__name__)

FEATURE_VERSION = "4-v1.1"   # v4-v1.1: deterministic Hough lines with rng=0, single-angle features

V1_STATUS_COLUMNS = [
    "optic_class_status",
    "pleo_status",
    "absorption_status",
    "twin_status",
    "extinction_status",
    "cleavage_status",
]

V1_SINGLE_ANGLE_COLUMNS = [
    "k1_ppl_L",
    "k1_ppl_a",
    "k1_ppl_b",
    "k1_ppl_chroma",
    "k1_ppl_hue_deg",
    "k1_ppl_colour",
    "k1_opaque_frac",
    "k1_xpl_bright_frac",
    "k1_optic_class",
    "k1_optic_class_status",
]


# ──────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────

def _angle_between_mod180(a_deg: float, b_deg: float) -> float:
    """Smallest angle between two directions mod 180°, result in [0, 90]."""
    d = abs(a_deg - b_deg) % 180.0
    return min(d, 180.0 - d)


def _angle_between_mod90(a_deg: float, b_deg: float) -> float:
    """Smallest angle between two directions mod 90°, result in [0, 45]."""
    d = abs(a_deg - b_deg) % 90.0
    return min(d, 90.0 - d)


def _erode_mask(region_mask: np.ndarray, border_px: int) -> np.ndarray:
    """Erode a boolean mask by border_px; fall back to original if empty."""
    if border_px <= 0:
        return region_mask
    eroded = scipy.ndimage.binary_erosion(region_mask, iterations=border_px)
    if not np.any(eroded):
        logger.debug("Erosion by %d px emptied mask (area=%d px); using original",
                     border_px, int(region_mask.sum()))
        return region_mask
    return eroded


def _ppl_luminance(ppl_rgb: np.ndarray) -> np.ndarray:
    """BT.601 luminance from uint8 RGB [...,3], returns float32."""
    return (0.299 * ppl_rgb[..., 0].astype(np.float32)
            + 0.587 * ppl_rgb[..., 1].astype(np.float32)
            + 0.114 * ppl_rgb[..., 2].astype(np.float32))


def _frame_white_refs(ppl_frame: np.ndarray, valid: np.ndarray) -> tuple[float, np.ndarray]:
    """Compute white level (W) and white balance gains for a single frame.

    Copies maps.py's formula applied to the selected frame only:
    - W = 99th percentile of that frame's PPL luminance over valid pixels;
    - neutral mask = (L > 0.9*W) & (max(RGB) - min(RGB) < 20) & valid;
    - gains = mean(channel means) / channel means, or ones if the mask is empty.
    """
    ppl_raw = ppl_frame.astype(np.float32)
    L_ppl = _ppl_luminance(ppl_raw)

    if np.any(valid):
        W_level = float(np.percentile(L_ppl[valid], 99))
    else:
        W_level = 255.0

    chan_spread = np.max(ppl_raw, axis=-1) - np.min(ppl_raw, axis=-1)
    neutral_mask = (L_ppl > 0.9 * W_level) & (chan_spread < 20.0) & valid

    if np.any(neutral_mask):
        channel_means = np.mean(ppl_raw[neutral_mask], axis=0)
        grey_target = float(np.mean(channel_means))
        wb_gains = np.array(
            [grey_target / max(float(c), 1.0) for c in channel_means],
            dtype=np.float32,
        )
    else:
        wb_gains = np.ones(3, dtype=np.float32)

    return W_level, wb_gains


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB [H,W,3] → float32 Lab (L 0-100, a/b centred at 0)."""
    bgr = rgb[..., ::-1].copy()
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)


def _hue_to_colour(hue_deg: float, chroma: float, L: float,
                   min_chroma: float = 2.5, light_L: float = 70.0) -> str:
    """Map CIELAB hue + chroma + L to petrographic colour name.

    OU photomicrographs are weakly saturated: on granite/11agt25 the
    non-pleochroic (colourless) regions had median chroma 0.4 (max 4.4)
    while biotite had mean-colour chroma 2.6-4.5.  A chroma threshold of 6
    called 183/198 regions "colourless", including biotite.  Here *chroma*
    should be the most saturated of the mean / dark / light PPL colours.
    """
    if L < 20:
        return "black"
    if chroma < min_chroma:
        return "colourless" if L >= light_L else "grey"
    h = hue_deg % 360
    if 20 <= h < 75:
        return "brown"
    if 75 <= h < 110:
        return "yellow"
    if 110 <= h < 200:
        return "green"
    if 200 <= h < 290:
        return "blue"
    return "pink/red"


def _find_nearest_frame(theta_deg: np.ndarray, target_deg: float) -> int:
    """Index of the frame whose angle is closest to target_deg (mod 180)."""
    diff = np.abs((theta_deg - target_deg + 90) % 180 - 90)
    return int(np.argmin(diff))


def _merge_adjacent_peaks(hist: np.ndarray, bin_centres: np.ndarray,
                          min_frac: float, total: float) -> list[float]:
    """Find histogram peaks: merge adjacent bins above threshold, handle
    the 0°/180° circular wrap, and return peaks sorted by weight
    (strongest first).

    Each returned value is the length-weighted mean angle of its merged
    peak.
    """
    threshold = min_frac * total
    n = len(hist)
    above = hist >= threshold

    # Mark visited bins so we don't double-count
    visited = [False] * n
    peaks: list[tuple[float, float]] = []  # (weight, angle)

    # Start from a bin that is *not* above threshold (to avoid splitting
    # a wrap-around peak).  If all bins are above, just start at 0.
    start = 0
    for i in range(n):
        if not above[i]:
            start = (i + 1) % n
            break

    for step in range(n):
        i = (start + step) % n
        if visited[i] or not above[i]:
            visited[i] = True
            continue

        # Accumulate a run of consecutive above-threshold bins
        w_sum = 0.0
        # Use sin/cos averaging to handle the 0/180 wrap correctly
        sin_sum = 0.0
        cos_sum = 0.0
        j = i
        while above[j] and not visited[j]:
            visited[j] = True
            rad = math.radians(2.0 * bin_centres[j])  # double to handle 180° periodicity
            sin_sum += hist[j] * math.sin(rad)
            cos_sum += hist[j] * math.cos(rad)
            w_sum += hist[j]
            j = (j + 1) % n

        angle = (math.degrees(math.atan2(sin_sum, cos_sum)) / 2.0) % 180.0
        peaks.append((w_sum, angle))

    # Sort by weight descending
    peaks.sort(key=lambda p: -p[0])
    return [angle for _, angle in peaks]


# ──────────────────────────────────────────────────────────────────────────
#  FeatureExtractor
# ──────────────────────────────────────────────────────────────────────────

class FeatureExtractor:
    """Phase 4 (§10): extract per-region grain property cards."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        feat = cfg.get("features", {})
        self.border_px = feat.get("border_exclude_px", 2)
        self.min_aspect = feat.get("min_aspect_for_axis", 1.3)

        twin = feat.get("twin", {})
        self.twin_min_frac = twin.get("min_frac", 0.10)
        self.twin_min_sep = twin.get("min_sep_deg", 8)

        cleav = feat.get("cleavage", {})
        self.sato_sigmas = cleav.get("sato_sigmas", [1, 2])
        self.ridge_pct = cleav.get("ridge_pct", 95)
        self.hough_min_len = cleav.get("hough_min_len", 12)
        self.peak_frac = cleav.get("peak_frac", 0.20)

        colour = feat.get("colour", {})
        self.min_chroma = colour.get("min_chroma", 2.5)
        self.light_L = colour.get("light_L", 70.0)
        self.min_twin_r2 = twin.get("min_r2_4", 0.5)
        self.max_cluster_std = twin.get("max_cluster_std_deg", 7.0)
        self.twin_grad_ratio_max = twin.get("max_boundary_grad_ratio", 2.0)

        acq = cfg.get("acquisition", {})
        self.rot_sign = acq.get("rot_sign", 1)
        self.p0 = acq.get("polariser_deg", 0.0)

        v1_rs = cfg.get("v1", {}).get("region_selection", {})
        self.v1_min_area_px = v1_rs.get("min_area_px", 1000)
        self.v1_min_interior_px = v1_rs.get("min_interior_px", 400)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def extract_site(
        self,
        label_img: np.ndarray,
        region_list: list[dict],
        aligned: dict,
        maps: dict,
        site_id: str,
        code: str,
        um_per_px: float | None = None,
    ) -> list[dict]:
        """Extract features for all non-hole regions in one site.

        Parameters
        ----------
        um_per_px : float or None
            Site-specific calibration (from Phase 3 JSON).  Falls back
            to the global config value if ``None``.

        Returns list of dicts (one per region), ready for DataFrame.
        """
        # §Fix 7: prefer site-specific um_per_px from Phase 3 JSON
        if um_per_px is None:
            um_per_px = self.cfg.get("acquisition", {}).get("um_per_px_rot", 4.0)

        valid = aligned["valid"].astype(bool)
        H, W = valid.shape

        # Maps (float16 → float32)
        # V1 FIX: preserve NaN in angular/phase fields — NaN means
        # unresolved, NOT measured-zero.  Only amplitude/intensity maps
        # safely default to 0.
        aniso = np.nan_to_num(maps["aniso"].astype(np.float32), nan=0.0)
        ext_deg = maps["ext_deg"].astype(np.float32)              # NaN = unresolved angle
        r2_4 = np.nan_to_num(maps["r2_4"].astype(np.float32), nan=0.0)
        ppl_max_L = np.nan_to_num(maps["ppl_max_L"].astype(np.float32), nan=0.0)
        ppl_Lab = np.nan_to_num(maps["ppl_Lab"].astype(np.float32), nan=0.0)
        pleo_dE = maps["pleo_dE"].astype(np.float32)              # NaN = unresolvable ΔE
        pleo_dark_deg = maps["pleo_dark_deg"].astype(np.float32)  # NaN = unresolved absorption
        relief_mod = np.nan_to_num(maps["relief_mod"].astype(np.float32), nan=0.0)
        ppl_tex = np.nan_to_num(maps["ppl_tex"].astype(np.float32), nan=0.0)
        gamma_est_nm = maps["gamma_est_nm"].astype(np.float32)    # NaN = unresolved retardation
        biref_cls_map = maps["biref_cls"].astype(np.int8)
        ppl_mean_rgb = maps["ppl_mean_rgb"]     # uint8
        W_level = float(maps["white_level"])

        # §Fix 3: aniso_mask — the definitive mask of "anisotropic pixels"
        # from Phase 2 (aniso >= 0.15 AND xpl_max_L >= 0.08*W)
        if "aniso_mask" in maps:
            aniso_mask = maps["aniso_mask"].astype(bool)
        else:
            # Reconstruct from thresholds
            xpl_dark_rel = self.cfg.get("maps", {}).get("xpl_dark_rel", 0.08)
            aniso_min = self.cfg.get("maps", {}).get("aniso_min", 0.15)
            xpl_max_L = np.nan_to_num(maps.get("xpl_max_L",
                np.zeros((H, W), dtype=np.float32)).astype(np.float32), nan=0.0)
            aniso_mask = (aniso >= aniso_min) & (xpl_max_L >= xpl_dark_rel * W_level)

        # xpl_rel_rgb → Lab
        xpl_rel_rgb = np.nan_to_num(maps["xpl_rel_rgb"].astype(np.float32), nan=0.0)
        xrl_uint8 = np.clip(xpl_rel_rgb * 255.0, 0, 255).astype(np.uint8)
        xpl_rel_lab = _rgb_to_lab(xrl_uint8)

        # Reconstruct opaque_px
        opaque_rel_L = self.cfg.get("maps", {}).get("opaque_rel_L", 0.15)
        opaque_px = (ppl_max_L < opaque_rel_L * W_level) & valid

        # PPL gradient for boundary contrast
        ppl_gray = _ppl_luminance(ppl_mean_rgb)
        ppl_grad = np.hypot(
            cv2.Sobel(ppl_gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(ppl_gray, cv2.CV_32F, 0, 1, ksize=3))
        valid_grad = ppl_grad[valid]
        site_median_grad = float(np.median(valid_grad)) if len(valid_grad) > 0 else 1.0

        # Site-level ridge threshold for cleavage (§10.7)
        ridge_img = sato(ppl_gray.astype(np.float64), sigmas=self.sato_sigmas,
                         black_ridges=True, mode="reflect")
        valid_ridge = ridge_img[valid]
        ridge_thresh = float(np.percentile(valid_ridge, self.ridge_pct)) \
            if len(valid_ridge) > 0 else 0.0

        rot_sign = int(aligned.get("rot_sign", self.rot_sign))
        theta_deg = aligned["theta_deg"]       # [36]
        ppl_stack = aligned["ppl"]             # [36,H,W,3] uint8

        rows: list[dict] = []
        bc_values: list[float] = []

        # Site colour cast: median a*, b* of bright, weakly pleochroic
        # pixels (mostly quartz/feldspar, which are colourless).  Removing it
        # stops a camera tint (11agt25 PPL is bluish) being read as "blue".
        L_ = ppl_Lab[..., 0]
        cand = valid & (L_ > self.light_L) & (pleo_dE < 4)
        if cand.sum() > 500:
            self.cast_ab = (float(np.median(ppl_Lab[..., 1][cand])),
                            float(np.median(ppl_Lab[..., 2][cand])))
        else:
            self.cast_ab = (0.0, 0.0)

        for region_info in region_list:
            if region_info["type"] == "hole":
                continue

            lv = region_info["label_value"]
            region_mask = label_img == lv

            row = self._extract_one(
                region_mask=region_mask,
                region_info=region_info,
                aniso=aniso, ext_deg=ext_deg, r2_4=r2_4,
                ppl_max_L=ppl_max_L, ppl_Lab=ppl_Lab,
                pleo_dE=pleo_dE, pleo_dark_deg=pleo_dark_deg,
                relief_mod=relief_mod, ppl_tex=ppl_tex,
                gamma_est_nm=gamma_est_nm, biref_cls_map=biref_cls_map,
                xpl_rel_lab=xpl_rel_lab, aniso_mask=aniso_mask,
                opaque_px=opaque_px, ppl_grad=ppl_grad,
                ppl_gray=ppl_gray, ridge_img=ridge_img,
                ridge_thresh=ridge_thresh,
                site_median_grad=site_median_grad,
                W_level=W_level, rot_sign=rot_sign,
                ppl_stack=ppl_stack, theta_deg=theta_deg,
                site_id=site_id, code=code,
                um_per_px=um_per_px,
            )
            rows.append(row)
            bc_values.append(row["boundary_contrast"])

        # §10.8 relief_cls: site-relative quartiles of boundary_contrast
        if bc_values:
            bc_arr = np.array(bc_values)
            q25, q50, q75 = np.percentile(bc_arr, [25, 50, 75])
            for row in rows:
                bc = row["boundary_contrast"]
                if bc <= q25:
                    row["relief_cls"] = "low"
                elif bc <= q50:
                    row["relief_cls"] = "moderate"
                elif bc <= q75:
                    row["relief_cls"] = "high"
                else:
                    row["relief_cls"] = "very high"

        return rows

    def extract_single_angle(
        self,
        label_img: np.ndarray,
        region_list: list[dict],
        aligned: dict,
        maps_dict: dict | None = None,
        site_id: str = "",
        frame_idx: int = 1,
    ) -> list[dict]:
        """Extract single-angle feature columns (§CONTRACTS.md Section 1, Task D1.1).

        Parameters
        ----------
        label_img : np.ndarray
            Label image of segmentation.
        region_list : list[dict]
            List of region dicts from segmentation.
        aligned : dict
            Aligned stack containing 'ppl', 'xpl', and optionally 'valid'.
        maps_dict : dict
            Maps dictionary containing 'white_level' and 'wb_gains'.
        site_id : str
            Identifier of current site.
        frame_idx : int
            Index of the frame (default 1, i.e. 5°).

        Returns
        -------
        list[dict]
            One dict per non-hole region containing region_id and k1_* features.
        """
        if aligned is None:
            return []

        ppl_stack = aligned.get("ppl")
        xpl_stack = aligned.get("xpl")
        if ppl_stack is None:
            return []

        frame_idx = int(frame_idx)
        ppl_frame = ppl_stack[frame_idx].astype(np.float32)
        xpl_frame = xpl_stack[frame_idx].astype(np.float32) if xpl_stack is not None else np.zeros_like(ppl_frame)

        if "valid" in aligned:
            valid = aligned["valid"].astype(bool)
        else:
            valid = np.ones(ppl_frame.shape[:2], dtype=bool)

        W_level, wb_gains = _frame_white_refs(ppl_frame, valid)
        ppl_frame = np.clip(ppl_frame * wb_gains, 0.0, 255.0)
        xpl_frame = np.clip(xpl_frame * wb_gains, 0.0, 255.0)

        feat = self.cfg.get("features", {}) if self.cfg else {}
        maps_cfg = self.cfg.get("maps", {}) if self.cfg else {}
        opaque_rel_L = maps_cfg.get("opaque_rel_L", 0.15)
        xpl_dark_rel = maps_cfg.get("xpl_dark_rel", 0.08)
        light_L = getattr(self, "light_L", feat.get("colour", {}).get("light_L", 70.0))
        min_chroma = getattr(self, "min_chroma", feat.get("colour", {}).get("min_chroma", 2.5))

        ppl_lum = _ppl_luminance(ppl_frame)
        xpl_lum = _ppl_luminance(xpl_frame)

        opaque_px = (ppl_lum < opaque_rel_L * W_level) & valid
        xpl_bright_px = (xpl_lum >= xpl_dark_rel * W_level) & valid

        ppl_lab = _rgb_to_lab(ppl_frame)
        cand = valid & (ppl_lab[..., 0] > light_L)
        if np.any(cand):
            ca = float(np.median(ppl_lab[..., 1][cand]))
            cb = float(np.median(ppl_lab[..., 2][cand]))
        else:
            ca, cb = 0.0, 0.0

        border_px = getattr(self, "border_px", feat.get("border_exclude_px", 2))

        k1_rows: list[dict] = []
        for region_info in (region_list or []):
            if region_info.get("type") == "hole":
                continue

            lv = region_info["label_value"]
            region_mask = label_img == lv
            eroded = _erode_mask(region_mask, border_px)

            L_int = ppl_lab[..., 0][eroded]
            a_int = ppl_lab[..., 1][eroded]
            b_int = ppl_lab[..., 2][eroded]

            if len(L_int) > 0:
                k1_ppl_L = float(np.median(L_int))
                med_a = float(np.median(a_int))
                med_b = float(np.median(b_int))
                k1_ppl_a = float(med_a - ca)
                k1_ppl_b = float(med_b - cb)
                k1_ppl_chroma = float(math.hypot(k1_ppl_a, k1_ppl_b))
                k1_ppl_hue_deg = float(math.degrees(math.atan2(k1_ppl_b, k1_ppl_a)) % 360.0)
            else:
                k1_ppl_L = 0.0
                k1_ppl_a = 0.0
                k1_ppl_b = 0.0
                k1_ppl_chroma = 0.0
                k1_ppl_hue_deg = 0.0

            k1_ppl_colour = _hue_to_colour(k1_ppl_hue_deg, k1_ppl_chroma, k1_ppl_L, min_chroma, light_L)

            op_int = opaque_px[eroded]
            xb_int = xpl_bright_px[eroded]
            n_int = max(len(op_int), 1)
            k1_opaque_frac = float(op_int.sum() / n_int)
            k1_xpl_bright_frac = float(xb_int.sum() / n_int)

            if k1_opaque_frac > 0.8:
                k1_optic_class = "opaque"
            elif k1_xpl_bright_frac >= 0.2:
                k1_optic_class = "anisotropic"
            else:
                k1_optic_class = "unresolved"

            k1_optic_class_status = "observed" if k1_optic_class in ("opaque", "anisotropic") else "unresolved"

            k1_rows.append({
                "region_id": region_info.get("region_id", f"{site_id}_{lv}"),
                "k1_ppl_L": round(k1_ppl_L, 1),
                "k1_ppl_a": round(k1_ppl_a, 1),
                "k1_ppl_b": round(k1_ppl_b, 1),
                "k1_ppl_chroma": round(k1_ppl_chroma, 1),
                "k1_ppl_hue_deg": round(k1_ppl_hue_deg, 1),
                "k1_ppl_colour": k1_ppl_colour,
                "k1_opaque_frac": round(k1_opaque_frac, 3),
                "k1_xpl_bright_frac": round(k1_xpl_bright_frac, 3),
                "k1_optic_class": k1_optic_class,
                "k1_optic_class_status": k1_optic_class_status,
            })

        return k1_rows

    # ------------------------------------------------------------------
    #  Per-region extraction
    # ------------------------------------------------------------------

    def _extract_one(
        self, *, region_mask, region_info,
        aniso, ext_deg, r2_4, ppl_max_L, ppl_Lab,
        pleo_dE, pleo_dark_deg, relief_mod, ppl_tex,
        gamma_est_nm, biref_cls_map, xpl_rel_lab, aniso_mask,
        opaque_px, ppl_grad, ppl_gray, ridge_img, ridge_thresh,
        site_median_grad, W_level, rot_sign,
        ppl_stack, theta_deg,
        site_id, code, um_per_px,
    ) -> dict:
        """Extract all §10 features for one region."""
        eroded = _erode_mask(region_mask, self.border_px)
        conf = None

        # ── §10.1 Identity & geometry ───────────────────────────────
        props_label = region_mask.astype(np.uint8)
        rps = regionprops(props_label)
        rp = rps[0] if rps else None

        area_px = int(region_mask.sum())
        area_um2 = area_px * um_per_px ** 2
        eq_diam_um = 2.0 * math.sqrt(area_px / math.pi) * um_per_px

        major_px = float(rp.axis_major_length) if rp else 0.0
        minor_px = float(rp.axis_minor_length) if rp else 1.0
        aspect = major_px / max(minor_px, 1e-6)
        solidity = float(rp.solidity) if rp else 0.0
        eccentricity = float(rp.eccentricity) if rp else 0.0

        partial = region_info.get("partial", False)

        # skimage (>=0.16) orientation is measured from the ROW axis, so
        # degrees(-orientation) is 90° off.  +90 gives the image convention
        # used everywhere else here (0° = +x, increasing towards +y, same as
        # atan2(dy, dx) of Hough lines).  Tested on synthetic ellipses.
        if rp and aspect >= self.min_aspect and not partial:
            long_axis_deg = (float(np.degrees(-rp.orientation)) + 90.0) % 180.0
        else:
            long_axis_deg = np.nan

        if partial:
            habit = np.nan
        elif aspect < 1.5:
            habit = "equant"
        elif aspect <= 4.0:
            habit = "prismatic"
        else:
            habit = "acicular"

        # ── Pixel arrays inside eroded mask ──────────────────────────
        an = aniso[eroded]
        ex = ext_deg[eroded]
        ex = ex[np.isfinite(ex)]
        r2 = r2_4[eroded]
        pl_dE = pleo_dE[eroded]
        pl_dE = pl_dE[np.isfinite(pl_dE)]
        pl_dark = pleo_dark_deg[eroded]
        pl_dark = pl_dark[np.isfinite(pl_dark)]
        rm = relief_mod[eroded]
        pt = ppl_tex[eroded]
        lab = ppl_Lab[eroded]            # [M, 3]
        gam = gamma_est_nm[eroded]
        gam = gam[np.isfinite(gam)]
        bcl = biref_cls_map[eroded]
        xrl = xpl_rel_lab[eroded]        # [M, 3]
        op = opaque_px[eroded]
        am = aniso_mask[eroded]          # bool: truly anisotropic pixels

        # ── §10.2 Optical class ─────────────────────────────────────
        # Spec: opaque if opaque_frac > 0.8; isotropic if aniso_p90 < 0.15
        # and not opaque; else anisotropic.
        opaque_frac = float(op.sum()) / max(len(op), 1)
        aniso_med = float(np.median(an)) if len(an) > 0 else 0.0
        aniso_p90 = float(np.percentile(an, 90)) if len(an) > 0 else 0.0
        r2_4_med = float(np.median(r2)) if len(r2) > 0 else 0.0

        # Opaque from PPL darkness; isotropic = hardly any pixel ever gets
        # bright in XPL (aniso_mask from Phase 2).  "aniso" alone is
        # unreliable on dark pixels (noise / small denominator).
        aniso_frac = float(am.mean()) if len(am) > 0 else 0.0
        if opaque_frac > 0.8:
            optic_class = "opaque"
        elif aniso_frac < 0.2:
            optic_class = "isotropic"
        else:
            optic_class = "anisotropic"

        # ── §10.3 Colour & pleochroism ──────────────────────────────
        ppl_L_val = float(np.median(lab[:, 0])) if len(lab) > 0 else 0.0
        ppl_a_val = float(np.median(lab[:, 1])) if len(lab) > 0 else 0.0
        ppl_b_val = float(np.median(lab[:, 2])) if len(lab) > 0 else 0.0
        ca, cb = getattr(self, "cast_ab", (0.0, 0.0))
        ppl_chroma = math.hypot(ppl_a_val - ca, ppl_b_val - cb)
        ppl_hue_deg = math.degrees(math.atan2(ppl_b_val - cb, ppl_a_val - ca)) % 360.0
        pleo_dE_med = float(np.nanmedian(pl_dE)) if len(pl_dE) > 0 else 0.0
        if pleo_dE_med < 4:
            pleo_cls = "none"
        elif pleo_dE_med < 10:
            pleo_cls = "weak"
        elif pleo_dE_med < 20:
            pleo_cls = "moderate"
        else:
            pleo_cls = "strong"

        # §Fix 2 & Task A5: Period-180° circular mean for pleochroism / absorption dark angle
        # Requires minimum support: len(pl_dark) >= max(min_px, min_frac * n_interior_px)
        n_interior_px = int(eroded.sum())
        pleo_dark_med = self._compute_absorption_dark_angle(pl_dark, n_interior_px)

        pleo_colours = self._compute_pleo_colours(
            eroded, pleo_dark_med, ppl_stack, theta_deg, pleo_cls)
        # numeric extremes (NaN if not pleochroic) + colour name from the
        # most saturated of mean / dark / light colours
        pd_L = pd_C = pl_L = pl_C = np.nan
        hue_use, chroma_use = ppl_hue_deg, ppl_chroma
        if pleo_colours:
            (dL, da, db), (lL, la, lb) = [tuple(map(float, c.split(",")))
                                          for c in pleo_colours.split(";")]
            da, db, la, lb = da - ca, db - cb, la - ca, lb - cb
            pd_L, pd_C = dL, math.hypot(da, db)
            pl_L, pl_C = lL, math.hypot(la, lb)
            for C, a_, b_ in ((pd_C, da, db), (pl_C, la, lb)):
                if C > chroma_use:
                    chroma_use, hue_use = C, math.degrees(math.atan2(b_, a_)) % 360
        ppl_colour = _hue_to_colour(hue_use, chroma_use, ppl_L_val,
                                    self.min_chroma, self.light_L)

        # §Fix 1: absorption_vs_long_deg — 0–90° (not folded to 0–45°)
        absorption_vs_long = np.nan
        if pleo_cls != "none" and not np.isnan(long_axis_deg) and not np.isnan(pleo_dark_med):
            # Stage angle -> direction in the aligned image: p0 + rot_sign*θ.
            # Verified on the two large biotites of granite_r1 (aspect 5.5
            # and 3.7): absorption ∥ cleavage 2°/11°, extinction 0.4°/9°.
            # The spec's "p0 − rot_sign·θ" gave 44°/42° (wrong).
            abs_axis = (self.p0 + rot_sign * pleo_dark_med) % 180.0
            absorption_vs_long = _angle_between_mod180(abs_axis, long_axis_deg)

        # ── §10.4 Interference / birefringence ──────────────────────
        xrl_L = float(np.median(xrl[:, 0])) if len(xrl) > 0 else 0.0
        xrl_a = float(np.median(xrl[:, 1])) if len(xrl) > 0 else 0.0
        xrl_b = float(np.median(xrl[:, 2])) if len(xrl) > 0 else 0.0

        # §Fix 3: gamma stats from truly anisotropic pixels only
        gam_aniso = gamma_est_nm[eroded][am]
        gam_aniso = gam_aniso[np.isfinite(gam_aniso)]
        gamma_med_nm = float(np.nanmedian(gam_aniso)) if len(gam_aniso) > 0 else np.nan
        gamma_p90_nm = float(np.nanpercentile(gam_aniso, 90)) if len(gam_aniso) > 0 else np.nan

        # biref_cls: mode weighted by aniso; C4 if > 30% pixels are C4
        bcl_valid = bcl[am]
        an_valid = an[am]
        if len(bcl_valid) > 0:
            c4_frac = float((bcl_valid == 4).sum()) / len(bcl_valid)
            if c4_frac > 0.3:
                biref_cls_val = 4
            else:
                weighted_counts = np.zeros(5, dtype=np.float64)
                for c in range(5):
                    mask_c = bcl_valid == c
                    if mask_c.any():
                        weighted_counts[c] = an_valid[mask_c].sum()
                biref_cls_val = int(np.argmax(weighted_counts))
        else:
            biref_cls_val = 0

        relief_mod_med = float(np.median(rm)) if len(rm) > 0 else 0.0
        colour_masking = ppl_chroma > 20

        # ── §10.5 Extinction, twinning, undulose ────────────────────
        ext_mean_deg = np.nan
        ext_R = np.nan
        ext_std_deg = np.nan
        n_ext_domains = 1
        twin_type = "none"
        undulose_score = 0.0
        ext_vs_long_deg = np.nan
        ext_type = "undefined"

        # confident extinction pixels: truly anisotropic + good sin² fit
        conf = am & (r2 >= self.min_twin_r2)
        conf_full = np.zeros_like(eroded)
        conf_full[eroded] = conf
        if optic_class == "anisotropic" and conf.sum() >= 20:
            ex_all = ext_deg[eroded]
            ex_c, an_c = ex_all[conf], an[conf]
            # Circular stats with period 90°
            z = an_c * np.exp(1j * 4.0 * np.radians(ex_c))
            mean_z = np.mean(z)
            mean_aniso = float(np.mean(an_c))

            ext_mean_deg = float(np.degrees(np.angle(mean_z)) / 4.0) % 90.0
            ext_R = float(np.abs(mean_z) / max(mean_aniso, 1e-6))
            ext_R = min(ext_R, 1.0)
            if ext_R > 1e-6:
                ext_std_deg = float(math.sqrt(-2.0 * math.log(ext_R))
                                    / 4.0 * (180.0 / math.pi))
            else:
                ext_std_deg = 45.0

            n_ext_domains = self._count_ext_domains(ex_c, an_c)

            twin_type = self._classify_twin(
                n_ext_domains, region_mask, ext_deg, aniso, conf_full,
                ppl_grad)

            # §Fix 5: undulose only when n_ext_domains == 1
            if n_ext_domains == 1:
                undulose_score = self._compute_undulose(
                    region_mask, ext_deg, aniso)

            # §10.6: ext_vs_long_deg
            if not np.isnan(long_axis_deg):
                v = (self.p0 + rot_sign * ext_mean_deg) % 90.0
                ext_vs_long_deg = _angle_between_mod90(v, long_axis_deg % 90.0)
                if ext_vs_long_deg < 5:
                    ext_type = "parallel"
                elif ext_vs_long_deg <= 45:
                    ext_type = "inclined"

        # ── §10.7 Cleavage ──────────────────────────────────────────
        cleav = self._extract_cleavage(
            region_mask, eroded, ppl_gray, ridge_img, ridge_thresh,
            ext_mean_deg, rot_sign)

        # ── §10.8 Relief & texture ──────────────────────────────────
        border_3 = region_mask & ~_erode_mask(region_mask, 3)
        border_grad = ppl_grad[border_3]
        boundary_contrast = float(np.mean(border_grad) / max(site_median_grad, 1e-6)) \
            if len(border_grad) > 0 else 0.0
        ppl_tex_med = float(np.median(pt)) if len(pt) > 0 else 0.0

        # ── V1 Eligibility (§10.9) ──────────────────────────────────
        # interior_px from true erosion, NOT the fallback mask
        true_eroded = scipy.ndimage.binary_erosion(region_mask, iterations=self.border_px)
        interior_px = int(true_eroded.sum()) if np.any(true_eroded) else 0
        rtype = str(region_info.get("type", ""))
        if area_px < self.v1_min_area_px:
            reject_reason_v1 = "small_area"
        elif interior_px < self.v1_min_interior_px:
            reject_reason_v1 = "small_interior"
        elif rtype not in ("grain", "opaque"):
            reject_reason_v1 = "not_grain"
        else:
            reject_reason_v1 = ""
        eligible_v1 = bool(reject_reason_v1 == "")

        # ── Build output row ────────────────────────────────────────
        return {
            # Identity
            "region_id": region_info["region_id"],
            "site_id": site_id,
            "code": code,
            "type": region_info["type"],
            "partial": partial,
            "small": region_info.get("small", False),
            "n_domains": region_info.get("n_domains", 1),
            # §10.1 Geometry
            "area_px": area_px,
            "area_um2": round(area_um2, 1),
            "eq_diam_um": round(eq_diam_um, 1),
            "major_px": round(major_px, 1),
            "minor_px": round(minor_px, 1),
            "aspect": round(aspect, 2),
            "long_axis_deg": round(long_axis_deg, 1) if not np.isnan(long_axis_deg) else np.nan,
            "solidity": round(solidity, 3),
            "eccentricity": round(eccentricity, 3),
            "habit": habit,
            # §10.2 Optical class
            "opaque_frac": round(opaque_frac, 3),
            "aniso_med": round(aniso_med, 3),
            "aniso_p90": round(aniso_p90, 3),
            "aniso_frac": round(aniso_frac, 3),
            "r2_4_med": round(r2_4_med, 3),
            "optic_class": optic_class,
            "optic_class_status": "observed" if len(an) >= 20 else "invalid",
            # §10.3 Colour
            "ppl_L": round(ppl_L_val, 1),
            "ppl_a": round(ppl_a_val, 1),
            "ppl_b": round(ppl_b_val, 1),
            "ppl_chroma": round(ppl_chroma, 1),
            "ppl_hue_deg": round(ppl_hue_deg, 1),
            "ppl_colour": ppl_colour,
            "pleo_dE_med": round(pleo_dE_med, 1),
            "pleo_cls": pleo_cls,
            "pleo_status": (
                "observed" if pleo_dE_med >= 4 and len(pl_dE) >= 20 else
                ("not_detected_with_bound" if len(pl_dE) >= 20 else "invalid")
            ),
            "pleo_bound_dE": (
                round(float(np.percentile(pl_dE, 95)), 1)
                if len(pl_dE) >= 20 and pleo_dE_med < 4 else None
            ),
            "pleo_colours": pleo_colours,
            "ppl_chroma_max": round(float(chroma_use), 2),
            "pleo_dark_L": round(pd_L, 1) if not np.isnan(pd_L) else np.nan,
            "pleo_dark_chroma": round(pd_C, 2) if not np.isnan(pd_C) else np.nan,
            "pleo_light_L": round(pl_L, 1) if not np.isnan(pl_L) else np.nan,
            "pleo_light_chroma": round(pl_C, 2) if not np.isnan(pl_C) else np.nan,
            "absorption_vs_long_deg": (round(absorption_vs_long, 1)
                                       if not np.isnan(absorption_vs_long) else np.nan),
            "absorption_status": (
                "not_applicable" if pleo_cls == 'none' or optic_class in ('opaque', 'isotropic') else
                "observed" if pleo_cls in ("moderate", "strong") and not np.isnan(absorption_vs_long) else
                "unresolved"
            ),
            # §10.4 Interference
            "xpl_rel_L": round(xrl_L, 1),
            "xpl_rel_a": round(xrl_a, 1),
            "xpl_rel_b": round(xrl_b, 1),
            "gamma_med_nm": round(gamma_med_nm, 0) if not np.isnan(gamma_med_nm) else np.nan,
            "gamma_p90_nm": round(gamma_p90_nm, 0) if not np.isnan(gamma_p90_nm) else np.nan,
            "biref_cls": biref_cls_val,
            "relief_mod_med": round(relief_mod_med, 3),
            "colour_masking": colour_masking,
            # §10.5 Extinction / twinning
            "ext_mean_deg": round(ext_mean_deg, 1) if not np.isnan(ext_mean_deg) else np.nan,
            "ext_R": round(ext_R, 3) if not np.isnan(ext_R) else np.nan,
            "ext_std_deg": round(ext_std_deg, 1) if not np.isnan(ext_std_deg) else np.nan,
            "n_ext_domains": n_ext_domains,
            "twin_type": twin_type,
            "twin_status": (
                "observed" if twin_type != "none" else
                ("unresolved" if optic_class == "anisotropic" else "not_applicable")
            ),
            "undulose_score": round(undulose_score, 3),
            "ext_vs_long_deg": (round(ext_vs_long_deg, 1)
                                if not np.isnan(ext_vs_long_deg) else np.nan),
            "ext_type": ext_type,
            "extinction_status": (
                "observed" if optic_class == "anisotropic" and conf.sum() >= 20 else
                "not_applicable" if optic_class in ("opaque", "isotropic") else
                "invalid"
            ) if conf is not None else "invalid",
            "extinction_support": int(conf.sum()) if conf is not None else 0,
            # §10.7 Cleavage
            **cleav,
            # §10.8 Relief / texture
            "boundary_contrast": round(boundary_contrast, 3),
            "ppl_tex_med": round(ppl_tex_med, 1),
            "relief_cls": "low",  # set by caller from site-level quartiles
            # Meta
            "feature_version": FEATURE_VERSION,
            "um_per_px": um_per_px,
            "interior_px": interior_px,
            "interior_frac": round(interior_px / max(area_px, 1), 3),
            "eligible_v1": eligible_v1,
            "reject_reason_v1": reject_reason_v1,
        }

    # ------------------------------------------------------------------
    #  Task A5: Absorption dark angle with minimum support
    # ------------------------------------------------------------------

    def _compute_absorption_dark_angle(
        self,
        pl_dark: np.ndarray,
        n_interior_px: int,
    ) -> float:
        """Compute absorption dark angle via period-180° circular mean with minimum support."""
        feat = self.cfg.get("features", {}) if self.cfg else {}
        min_px = feat.get("absorption_min_px", 20)
        min_frac = feat.get("absorption_min_frac", 0.10)
        if len(pl_dark) < max(min_px, min_frac * n_interior_px):
            return float("nan")

        rad2 = np.radians(2.0 * pl_dark)
        mean_z = np.mean(np.exp(1j * rad2))
        pleo_R = float(np.abs(mean_z))
        if pleo_R >= 0.3:
            return float((np.degrees(np.angle(mean_z)) / 2.0) % 180.0)
        return float("nan")

    # ------------------------------------------------------------------
    #  §Fix 2: pleochroism extreme colours from the aligned PPL stack
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_pleo_colours(
        eroded: np.ndarray,
        pleo_dark_med_deg: float,
        ppl_stack: np.ndarray,
        theta_deg: np.ndarray,
        pleo_cls: str,
    ) -> str:
        """Return "L,a,b;L,a,b" for the dark and light PPL colours.

        Uses the aligned PPL frame closest to the region's median
        pleo_dark_deg (dark colour) and dark+90° (light colour).
        """
        if pleo_cls == "none":
            return ""
        if ppl_stack is None or theta_deg is None:
            return ""
        if np.isnan(pleo_dark_med_deg):
            return ""

        # Frame closest to dark angle
        dark_idx = _find_nearest_frame(theta_deg, pleo_dark_med_deg)
        light_idx = _find_nearest_frame(theta_deg, pleo_dark_med_deg + 90.0)

        # Extract Lab from those frames for the region pixels
        dark_rgb = ppl_stack[dark_idx]    # [H, W, 3] uint8
        light_rgb = ppl_stack[light_idx]

        dark_lab = _rgb_to_lab(dark_rgb)
        light_lab = _rgb_to_lab(light_rgb)

        # Median Lab over region pixels
        d_L = float(np.median(dark_lab[eroded, 0]))
        d_a = float(np.median(dark_lab[eroded, 1]))
        d_b = float(np.median(dark_lab[eroded, 2]))
        l_L = float(np.median(light_lab[eroded, 0]))
        l_a = float(np.median(light_lab[eroded, 1]))
        l_b = float(np.median(light_lab[eroded, 2]))

        return (f"{d_L:.1f},{d_a:.1f},{d_b:.1f};"
                f"{l_L:.1f},{l_a:.1f},{l_b:.1f}")

    # ------------------------------------------------------------------
    #  §10.5: extinction domain counting
    # ------------------------------------------------------------------

    def _count_ext_domains(self, ext_px: np.ndarray, an_px: np.ndarray) -> int:
        """Circular k-means (k=1..3) on (cos4ext, sin4ext).

        A cluster counts only if ≥ 10% of pixels and centres differ ≥ 8°.
        Additionally, k is only accepted if within-cluster circular
        variance drops by ≥ 30% relative to k=1 (prevents over-splitting
        broad/noisy distributions).
        """
        if len(ext_px) < 20:
            return 1
        rad4 = 4.0 * np.radians(ext_px)
        X = np.stack([np.cos(rad4), np.sin(rad4)], axis=1)  # [M, 2]

        # Baseline: k=1 circular variance
        mean_vec = X.mean(axis=0)
        R_bar_1 = float(np.linalg.norm(mean_vec))
        var_1 = 1.0 - R_bar_1   # circular variance for k=1

        best_k = 1
        best_var = var_1

        for k in (2, 3):
            centres, assigns = self._circular_kmeans(X, k, n_iter=20)
            counts = np.bincount(assigns, minlength=k)
            fracs = counts / max(counts.sum(), 1)

            # Filter: ≥ 10% of pixels
            valid_ids = [c for c in range(k)
                         if fracs[c] >= self.twin_min_frac]
            if len(valid_ids) <= best_k:
                continue

            valid_centres = [centres[c] for c in valid_ids]

            # Check separation ≥ 8° (→ ≥ 32° in 4× angle space)
            min_sep_rad4 = 4.0 * np.radians(self.twin_min_sep)
            sep_ok = True
            for i in range(len(valid_centres)):
                for j in range(i + 1, len(valid_centres)):
                    ai = math.atan2(valid_centres[i][1], valid_centres[i][0])
                    aj = math.atan2(valid_centres[j][1], valid_centres[j][0])
                    sep = abs(ai - aj)
                    sep = min(sep, 2 * math.pi - sep)
                    if sep < min_sep_rad4:
                        sep_ok = False
                        break
                if not sep_ok:
                    break
            if not sep_ok:
                continue

            # Weighted mean within-cluster circular variance; every domain
            # must itself be tight (a real twin domain extinguishes at one
            # angle), otherwise a broad/noisy distribution gets split.
            wcv = 0.0
            tight = True
            for c in valid_ids:
                members = X[assigns == c]
                if len(members) > 0:
                    R_c = float(np.linalg.norm(members.mean(axis=0)))
                    wcv += fracs[c] * (1.0 - R_c)
                    std_deg = math.degrees(math.sqrt(-2.0 * math.log(max(R_c, 1e-9)))) / 4.0
                    tight &= std_deg <= self.max_cluster_std
            if not tight:
                continue

            # Accept only if variance drops by ≥ 50% versus k=1
            if var_1 > 1e-6 and wcv < best_var * 0.50:
                best_k = len(valid_ids)
                best_var = wcv

        return best_k

    @staticmethod
    def _circular_kmeans(X: np.ndarray, k: int, n_iter: int = 20) -> tuple:
        """Simple circular k-means on unit-vector data [M, 2]."""
        M = len(X)
        angles = np.linspace(0, 2 * np.pi, k, endpoint=False)
        centres = np.stack([np.cos(angles), np.sin(angles)], axis=1)

        assigns = np.zeros(M, dtype=np.int32)
        for _ in range(n_iter):
            dots = X @ centres.T
            assigns = np.argmax(dots, axis=1)
            new_centres = np.zeros_like(centres)
            for c in range(k):
                members = X[assigns == c]
                if len(members) > 0:
                    s = members.sum(axis=0)
                    norm = np.linalg.norm(s)
                    new_centres[c] = s / max(norm, 1e-12)
                else:
                    new_centres[c] = centres[c]
            centres = new_centres

        return centres, assigns

    # ------------------------------------------------------------------
    #  §10.5: twin type classification (§Fix 4: fuller implementation)
    # ------------------------------------------------------------------

    def _classify_twin(
        self, n_domains: int,
        region_mask: np.ndarray,
        ext_deg_img: np.ndarray,
        aniso_img: np.ndarray,
        conf_mask: np.ndarray | None = None,
        ppl_grad: np.ndarray | None = None,
    ) -> str:
        """Classify twin_type from domain count and spatial analysis.

        - none: 1 domain
        - simple: 2 domains, 1 boundary line
        - lamellar: ≥3 alternating stripes, one dominant boundary orientation
        - crosshatched: two boundary orientations 70–110° apart
        - sector: ≥3 domains meeting near one point
        """
        if n_domains <= 1:
            return "none"

        eroded = _erode_mask(region_mask, 2)
        if conf_mask is not None and conf_mask.sum() >= 20:
            eroded = eroded & conf_mask
        if not eroded.any():
            return "none"

        # Get extinction-domain labels inside the region
        ext_px = ext_deg_img[eroded]
        rad4 = 4.0 * np.radians(ext_px)
        X = np.stack([np.cos(rad4), np.sin(rad4)], axis=1)
        centres, assigns = self._circular_kmeans(X, n_domains, n_iter=20)

        # Build spatial domain map (cropped to bbox)
        ys, xs = np.where(region_mask)
        y0, x0 = ys.min(), xs.min()
        y1, x1 = ys.max() + 1, xs.max() + 1
        crop_eroded = eroded[y0:y1, x0:x1]

        domain_map = np.full(crop_eroded.shape, -1, dtype=np.int32)
        domain_map[crop_eroded] = assigns

        # Find inter-domain boundary pixels
        domain_bnd = np.zeros(crop_eroded.shape, dtype=bool)
        for a, b in ((domain_map[:, :-1], domain_map[:, 1:]),
                      (domain_map[:-1, :], domain_map[1:, :])):
            m = (a != b) & (a >= 0) & (b >= 0)
            domain_bnd[:m.shape[0], :m.shape[1]] |= m

        if not domain_bnd.any():
            return "none"

        # Twin / sector boundaries are (nearly) invisible in PPL; a boundary
        # between two separate grains shows a dark line or relief step.
        # If the domain boundary is much sharper in PPL than the grain
        # interior, the "domains" are two grains merged by segmentation.
        if ppl_grad is not None:
            g_crop = ppl_grad[y0:y1, x0:x1]
            g_bnd = float(np.median(g_crop[domain_bnd]))
            g_int = float(np.median(g_crop[crop_eroded & ~domain_bnd]))
            self._last_twin_grad_ratio = g_bnd / max(g_int, 1e-6)
            if g_bnd > self.twin_grad_ratio_max * max(g_int, 1e-6):
                return "none"

        # Hough on the domain-boundary skeleton
        skel = skeletonize(domain_bnd)
        lines = probabilistic_hough_line(skel, line_length=8, line_gap=3, rng=0)

        if n_domains == 2:
            # "simple" = 2 domains, 1 boundary line.
            # Check that the boundary skeleton is roughly one connected
            # linear feature by running Hough and verifying most boundary
            # pixels concentrate around a single orientation.
            if lines is None or len(lines) == 0:
                # Two domains exist but boundary has no detected straight
                # line; must NOT be classified as simple.
                return "none"
            ors = []
            lns = []
            for (x1l, y1l), (x2l, y2l) in lines:
                dx, dy = x2l - x1l, y2l - y1l
                lns.append(math.sqrt(dx * dx + dy * dy))
                ors.append(math.degrees(math.atan2(dy, dx)) % 180.0)
            ors = np.array(ors)
            lns = np.array(lns)
            # Circular concentration of boundary orientation
            rad2 = np.radians(2.0 * ors)
            R_bnd = float(np.abs(np.sum(lns * np.exp(1j * rad2))) / max(lns.sum(), 1e-12))
            # A simple (e.g. Carlsbad) twin has ONE straight composition
            # plane crossing the grain.  Two domains separated by a curved /
            # fragmented boundary are more likely two grains merged by the
            # segmentation or patchy undulose extinction -> not a twin
            # (n_ext_domains still records 2).
            span = float(lns.sum())
            size = math.sqrt(float(crop_eroded.sum()))
            if R_bnd >= 0.85 and span >= 0.5 * size:
                return "simple"
            return "none"

        # n_domains >= 3
        if len(lines) < 3:
            return "sector"  # ≥3 domains but no linear boundaries

        # Orientation histogram of domain boundary lines
        orientations = []
        lengths = []
        for (x1l, y1l), (x2l, y2l) in lines:
            dx, dy = x2l - x1l, y2l - y1l
            length = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            orientations.append(angle)
            lengths.append(length)

        orientations = np.array(orientations)
        lengths = np.array(lengths)
        total_length = float(lengths.sum())

        # Find peaks using merged-adjacent-bin approach
        n_bins = 18
        hist, bin_edges = np.histogram(orientations, bins=n_bins,
                                       range=(0, 180), weights=lengths)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        peaks = _merge_adjacent_peaks(hist, bin_centres,
                                      self.peak_frac, total_length)
        n_peaks = len(peaks)

        if n_peaks >= 2:
            sep = _angle_between_mod180(peaks[0], peaks[1])
            if 70 <= sep <= 110:
                return "crosshatched"

        # Check for lamellar: one dominant orientation + alternating stripes
        if n_peaks >= 1:
            # Count profile alternations perpendicular to main direction
            main_angle_rad = math.radians(peaks[0])
            # Project eroded domain labels onto perpendicular axis
            ey, ex_ = np.where(crop_eroded)
            perp = -np.sin(main_angle_rad) * ex_ + np.cos(main_angle_rad) * ey
            dom_vals = domain_map[crop_eroded]
            order = np.argsort(perp)
            dom_sorted = dom_vals[order]
            # Count transitions
            transitions = int(np.sum(dom_sorted[1:] != dom_sorted[:-1]))
            if transitions >= 3:
                return "lamellar"

        return "sector"

    # ------------------------------------------------------------------
    #  §10.5: undulose extinction (§Fix 5: single domain only)
    # ------------------------------------------------------------------

    def _compute_undulose(
        self,
        region_mask: np.ndarray,
        ext_deg_img: np.ndarray,
        aniso_img: np.ndarray,
    ) -> float:
        """Median spatial gradient of extinction angle (deg per 10 px).

        Only called when n_ext_domains == 1, so twin boundaries do not
        contaminate the gradient.  Low-aniso pixels are excluded because
        their extinction estimates are unreliable.
        """
        aniso_min = self.cfg.get("maps", {}).get("aniso_min", 0.15)
        eroded = _erode_mask(region_mask, 3)  # extra erosion to avoid edges
        # Restrict to pixels with reliable extinction
        confident = eroded & (aniso_img >= aniso_min)
        if confident.sum() < 20:
            return 0.0

        # Phase-gradient via cos/sin of 4·ext to avoid wrapping
        rad4 = 4.0 * np.radians(ext_deg_img)
        cos4 = np.where(confident, np.cos(rad4), 0.0).astype(np.float32)
        sin4 = np.where(confident, np.sin(rad4), 0.0).astype(np.float32)

        gx_c = cv2.Sobel(cos4, cv2.CV_32F, 1, 0, ksize=3)
        gy_c = cv2.Sobel(cos4, cv2.CV_32F, 0, 1, ksize=3)
        gx_s = cv2.Sobel(sin4, cv2.CV_32F, 1, 0, ksize=3)
        gy_s = cv2.Sobel(sin4, cv2.CV_32F, 0, 1, ksize=3)

        # d(angle4)/dx ≈ −sin4·dcos4/dx + cos4·dsin4/dx
        c4 = np.cos(rad4).astype(np.float32)
        s4 = np.sin(rad4).astype(np.float32)
        dang_dx = -s4 * gx_c + c4 * gx_s
        dang_dy = -s4 * gy_c + c4 * gy_s
        grad_mag = np.hypot(dang_dx, dang_dy)

        # Only read gradient at confident pixels (not at boundaries of
        # the zeroed-out region, which create artificial edges)
        interior = scipy.ndimage.binary_erosion(confident, iterations=1)
        if not interior.any():
            interior = confident

        # Convert: radians-per-px in 4× space → degrees per 10 px
        grad_deg = np.degrees(grad_mag[interior]) / 4.0 * 10.0
        return float(np.median(grad_deg)) if len(grad_deg) > 0 else 0.0

    # ------------------------------------------------------------------
    #  §10.7: cleavage (§Fix 8: proper peak merging + line-pixel count)
    # ------------------------------------------------------------------

    def _extract_cleavage(
        self,
        region_mask: np.ndarray,
        eroded: np.ndarray,
        ppl_gray: np.ndarray,
        ridge_img: np.ndarray,
        ridge_thresh: float,
        ext_mean_deg: float,
        rot_sign: int,
    ) -> dict:
        """Extract cleavage features (§10.7)."""
        result: dict = {
            "cleavage_strength": 0.0,
            "n_cleavage_dirs": 0,
            "cleavage_status": "unresolved",
            "cleavage_angle_deg": np.nan,
            "ext_vs_cleavage_deg": np.nan,
            "ext_bisects_cleavage": False,
        }

        if not eroded.any():
            return result

        # Crop to bbox
        ys, xs = np.where(region_mask)
        y0, x0 = ys.min(), xs.min()
        y1, x1 = ys.max() + 1, xs.max() + 1

        crop_eroded = eroded[y0:y1, x0:x1]
        crop_ridge = ridge_img[y0:y1, x0:x1].copy()
        crop_ridge[~crop_eroded] = 0

        binary = crop_ridge > ridge_thresh
        if not binary.any():
            return result

        skel = skeletonize(binary)
        lines = probabilistic_hough_line(
            skel, line_length=self.hough_min_len, line_gap=3, rng=0)
        if not lines:
            return result

        # §Fix 8: count only skeleton pixels that lie on detected lines
        line_mask = np.zeros_like(skel, dtype=bool)
        for (x1l, y1l), (x2l, y2l) in lines:
            cv2.line(line_mask.view(np.uint8), (x1l, y1l), (x2l, y2l), 1, 1)
        line_on_skel = skel & line_mask
        ridge_on_lines = int(line_on_skel.sum())
        area = int(region_mask.sum())
        result["cleavage_strength"] = round(ridge_on_lines / max(area, 1), 4)

        # Orientation histogram weighted by line length
        orientations = []
        lengths = []
        for (x1l, y1l), (x2l, y2l) in lines:
            dx, dy = x2l - x1l, y2l - y1l
            length = math.sqrt(dx * dx + dy * dy)
            angle = math.degrees(math.atan2(dy, dx)) % 180.0
            orientations.append(angle)
            lengths.append(length)

        orientations = np.array(orientations)
        lengths = np.array(lengths)
        total_length = float(lengths.sum())

        # §Fix 8: merge adjacent bins to avoid splitting one direction
        n_bins = 18
        hist, bin_edges = np.histogram(orientations, bins=n_bins,
                                       range=(0, 180), weights=lengths)
        bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        peaks = _merge_adjacent_peaks(hist, bin_centres,
                                      self.peak_frac, total_length)
        n_dirs = min(len(peaks), 2)
        result["n_cleavage_dirs"] = n_dirs
        result["cleavage_status"] = "observed" if n_dirs > 0 else "unresolved"

        if n_dirs >= 1:
            main_angle = peaks[0]
            if not np.isnan(ext_mean_deg):
                v = (self.p0 + rot_sign * ext_mean_deg) % 180.0
                # two vibration directions (v, v+90) -> fold to 0–45°
                result["ext_vs_cleavage_deg"] = round(
                    _angle_between_mod90(v, main_angle), 1)

            if n_dirs >= 2:
                second_angle = peaks[1]
                cleav_angle = _angle_between_mod180(main_angle, second_angle)
                result["cleavage_angle_deg"] = round(min(cleav_angle, 90.0), 1)

                # ext_bisects_cleavage: within 5°
                if not np.isnan(ext_mean_deg):
                    v = (self.p0 + rot_sign * ext_mean_deg) % 180.0
                    bisector = ((main_angle + second_angle) / 2.0) % 180.0
                    bisector_alt = (bisector + 90.0) % 180.0
                    d1 = _angle_between_mod180(v, bisector)
                    d2 = _angle_between_mod180(v, bisector_alt)
                    result["ext_bisects_cleavage"] = min(d1, d2) < 5.0

        return result
