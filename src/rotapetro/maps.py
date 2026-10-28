"""Phase 2 — Optical property maps (§8).

Module: src/rotapetro/maps.py
Inputs:  aligned stacks (§7.2 NPZ)
Outputs: data/interim/maps/<site_id>.npz

Computes ALL per-pixel optical property maps from §8.2–8.6 using
vectorised numpy.  Background (invalid) pixels are NaN for floats,
0 for ints/bools.
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────

def _deltaE76(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIE ΔE76 colour difference (Euclidean in Lab)."""
    return np.sqrt(np.sum((lab1 - lab2) ** 2, axis=-1))


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """BT.601 luminance from RGB (last axis = 3).  0–255 float."""
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def _rgb_to_lab(rgb_float: np.ndarray) -> np.ndarray:
    """RGB float32 [0-255] → CIELAB float32 (L 0-100, a,b ≈ -128..+127).

    Uses cv2 on float32 input scaled to [0,1], which yields proper Lab
    ranges directly (L 0–100, a/b centred at 0 ±127).
    """
    bgr = np.clip(rgb_float, 0, 255).astype(np.float32) / 255.0
    bgr = bgr[..., ::-1]  # RGB → BGR
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)


def _masked_gaussian(data: np.ndarray, valid: np.ndarray,
                     sigma: float) -> np.ndarray:
    """Gaussian blur that respects *valid* via normalised convolution.

    Invalid pixels do not contribute to the blur of valid pixels.
    """
    weight = valid.astype(np.float32)
    data_m = np.where(valid, data.astype(np.float32), 0.0)
    num = cv2.GaussianBlur(data_m, (0, 0), sigma)
    den = cv2.GaussianBlur(weight, (0, 0), sigma)
    return np.where(den > 1e-6, num / den, 0.0)


def _masked_blur(data: np.ndarray, valid: np.ndarray,
                 ksize: tuple[int, int]) -> np.ndarray:
    """Box blur that respects *valid* via normalised convolution."""
    weight = valid.astype(np.float32)
    data_m = np.where(valid, data.astype(np.float32), 0.0)
    num = cv2.blur(data_m, ksize)
    den = cv2.blur(weight, ksize)
    return np.where(den > 1e-6, num / den, 0.0)


# ──────────────────────────────────────────────────────────────────────────
#  PropertyMapper
# ──────────────────────────────────────────────────────────────────────────

class PropertyMapper:
    """Phase 2 (§8): compute per-pixel optical property maps.

    Parameters
    ----------
    cfg : dict
        Loaded from ``configs/default.yaml``.
    lut_path : str | Path | None
        Path to ``knowledge/michel_levy_lut.npz``.  If None, the
        Michel-Lévy lookup is skipped (gamma arrays will be zero).
    """

    def __init__(self, cfg: dict, lut_path: str | Path | None = None) -> None:
        self.cfg = cfg
        maps_cfg = cfg.get("maps", {})

        # Thresholds from the spec / config
        self.aniso_min = maps_cfg.get("aniso_min", 0.15)
        self.high_order_white = maps_cfg.get("high_order_white", {})
        self.biref_class_nm = maps_cfg.get(
            "biref_class_nm", [100, 350, 1100, 2000])
        self.opaque_rel_L = maps_cfg.get("opaque_rel_L", 0.15)
        # A pixel that never gets brighter than this fraction of the white
        # level in XPL is isotropic or opaque, whatever its "aniso" value
        # (aniso = A4/(m0+1) is dominated by noise when m0 is only a few
        # grey levels: true opaque pixels in granite_r1 had median aniso 0.76).
        self.xpl_dark_rel = maps_cfg.get("xpl_dark_rel", 0.08)

        # Load Michel-Lévy LUT
        self.lut_gamma_nm: np.ndarray | None = None
        self.lut_lab: np.ndarray | None = None
        if lut_path is not None and Path(lut_path).is_file():
            lut = np.load(lut_path)
            self.lut_gamma_nm = lut["gamma_nm"].astype(np.float32)
            self.lut_lab = lut["lab"].astype(np.float32)
            logger.info("  Michel-Lévy LUT loaded (%d entries)", len(self.lut_gamma_nm))

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def compute_maps(self, aligned_dict: dict) -> dict:
        """Compute ALL optical property maps (§8.2–8.6).

        Parameters
        ----------
        aligned_dict : dict
            Output of Phase 1 (NPZ keys per §7.2).

        Returns
        -------
        dict
            Keys matching §8.6.
        """
        ppl_raw = aligned_dict["ppl"].astype(np.float32)   # [N,H,W,3] RGB
        xpl_raw = aligned_dict["xpl"].astype(np.float32)
        theta_deg = aligned_dict["theta_deg"].astype(np.float32)
        valid = aligned_dict["valid"].astype(bool)

        N, H, W, _ = ppl_raw.shape
        theta_rad = np.radians(theta_deg)  # [N]

        # ── §8.4  White level and white balance (do first) ──────────
        L_ppl_pre = _luminance(ppl_raw)                        # [N,H,W]
        ppl_max_L_pre = np.max(L_ppl_pre, axis=0)             # [H,W]

        W_level: float
        if np.any(valid):
            W_level = float(np.percentile(ppl_max_L_pre[valid], 99))
        else:
            W_level = 255.0

        # Neutral-white reference pixels for white balance.
        # §8.4 says "apply gains … before any Lab conversion", so we detect
        # neutral pixels in RGB space: a pixel with max(R,G,B)−min(R,G,B) < 20
        # is approximately Lab chroma < 10 for bright pixels.
        ppl_mean_pre = np.mean(ppl_raw, axis=0)                # [H,W,3]
        chan_spread = (np.max(ppl_mean_pre, axis=-1)
                       - np.min(ppl_mean_pre, axis=-1))
        wb_mask = ((ppl_max_L_pre > 0.9 * W_level)
                   & (chan_spread < 20.0)
                   & valid)

        if np.any(wb_mask):
            channel_means = np.mean(ppl_mean_pre[wb_mask], axis=0)  # [3]
            grey_target = float(np.mean(channel_means))
            wb_gains = np.array(
                [grey_target / max(c, 1.0) for c in channel_means],
                dtype=np.float32)
        else:
            wb_gains = np.ones(3, dtype=np.float32)

        # Apply gains to all PPL and XPL colours before any Lab conversion
        ppl = ppl_raw * wb_gains[np.newaxis, np.newaxis, np.newaxis, :]
        xpl = xpl_raw * wb_gains[np.newaxis, np.newaxis, np.newaxis, :]

        # ── §8.2  Luminance and harmonics ───────────────────────────
        L_xpl = _luminance(xpl)                                # [N,H,W]
        L_ppl = _luminance(ppl)                                # [N,H,W]

        # --- 4th harmonic of XPL luminance ---
        cos4 = np.cos(4 * theta_rad)[:, np.newaxis, np.newaxis]  # [N,1,1]
        sin4 = np.sin(4 * theta_rad)[:, np.newaxis, np.newaxis]

        m0 = np.mean(L_xpl, axis=0)                           # [H,W]
        c4 = (2.0 / N) * np.sum(L_xpl * cos4, axis=0)
        s4 = (2.0 / N) * np.sum(L_xpl * sin4, axis=0)
        A4 = np.hypot(c4, s4)
        phi4 = np.arctan2(s4, c4)

        # Extinction position: §8.2
        ext_deg = np.degrees(((phi4 + np.pi) / 4.0) % (np.pi / 2.0))

        # Anisotropy index: §8.2
        aniso = np.clip(A4 / (m0 + 1.0), 0.0, 1.0)

        # Brightest XPL luminance over the rotation (isotropic/opaque test)
        xpl_max_L = np.max(L_xpl, axis=0)

        # Fit quality r²₄ = (A4²/2) / Var(L_xpl)
        var_xpl = np.var(L_xpl, axis=0)
        r2_4 = np.where(var_xpl > 0, (A4 ** 2 / 2.0) / var_xpl, 0.0)
        r2_4 = np.clip(r2_4, 0.0, 1.0)

        # --- xpl_max_rgb: mean of the two frames at ext+45° and ext+135° ---
        ext_45 = (ext_deg + 45.0) % 180.0
        ext_135 = (ext_deg + 135.0) % 180.0

        def _nearest_frame_idx(target_deg: np.ndarray) -> np.ndarray:
            """Index of the frame nearest to target_deg (mod 180°)."""
            diff = np.abs(theta_deg[:, None, None] - target_deg[None, :, :])
            diff = np.minimum(diff % 180.0, (-diff) % 180.0)
            return np.argmin(diff, axis=0)

        idx_45 = _nearest_frame_idx(ext_45)
        idx_135 = _nearest_frame_idx(ext_135)
        rr, cc = np.indices((H, W))
        xpl_max_rgb = (xpl[idx_45, rr, cc] + xpl[idx_135, rr, cc]) / 2.0

        # xpl_maxproj_rgb: per-pixel max over all 36 XPL frames
        xpl_maxproj_rgb = np.max(xpl, axis=0)

        # ── §8.3  PPL colour, pleochroism, opacity, relief ──────────
        ppl_mean_rgb = np.mean(ppl, axis=0)                    # [H,W,3]
        ppl_min_L = np.min(L_ppl, axis=0)
        ppl_max_L = np.max(L_ppl, axis=0)

        # 2nd harmonics of PPL luminance
        cos2 = np.cos(2 * theta_rad)[:, np.newaxis, np.newaxis]
        sin2 = np.sin(2 * theta_rad)[:, np.newaxis, np.newaxis]

        c2_L = (2.0 / N) * np.sum(L_ppl * cos2, axis=0)
        s2_L = (2.0 / N) * np.sum(L_ppl * sin2, axis=0)

        # Absorption angle (darkest PPL direction)
        pleo_dark_deg = np.degrees(
            ((np.arctan2(s2_L, c2_L) + np.pi) / 2.0) % np.pi)

        # 2nd harmonics per PPL colour channel
        ppl_c2 = np.zeros((H, W, 3), dtype=np.float32)
        ppl_s2 = np.zeros((H, W, 3), dtype=np.float32)
        ppl_a0 = np.zeros((H, W, 3), dtype=np.float32)
        for ch in range(3):
            Ic = ppl[..., ch]  # [N,H,W]
            ppl_a0[..., ch] = np.mean(Ic, axis=0)
            ppl_c2[..., ch] = (2.0 / N) * np.sum(Ic * cos2, axis=0)
            ppl_s2[..., ch] = (2.0 / N) * np.sum(Ic * sin2, axis=0)

        # Evaluate fitted curve at θ_dark and θ_dark + 90°
        dark_rad = np.radians(pleo_dark_deg)
        light_rad = dark_rad + np.pi / 2.0

        col_dark_rgb = (ppl_a0
                        + ppl_c2 * np.cos(2 * dark_rad)[..., np.newaxis]
                        + ppl_s2 * np.sin(2 * dark_rad)[..., np.newaxis])
        col_light_rgb = (ppl_a0
                         + ppl_c2 * np.cos(2 * light_rad)[..., np.newaxis]
                         + ppl_s2 * np.sin(2 * light_rad)[..., np.newaxis])

        col_dark_lab = _rgb_to_lab(col_dark_rgb)
        col_light_lab = _rgb_to_lab(col_light_rgb)
        pleo_dE = _deltaE76(col_dark_lab, col_light_lab)

        # Opacity: §8.3
        opaque_px = (ppl_max_L < self.opaque_rel_L * W_level) & valid

        # Relief change ("twinkling"): §8.3
        # Gradient magnitude G[k] of L_ppl[k], then 2nd-harmonic amplitude.
        # Use normalised convolution so invalid pixels don't leak into the
        # Gaussian or the final 7×7 smoothing.
        G = np.zeros_like(L_ppl)  # [N,H,W]
        for k in range(N):
            blurred = _masked_gaussian(L_ppl[k], valid, sigma=1.0)
            gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            G[k] = np.where(valid, np.hypot(gx, gy), 0.0)

        mean_G = np.mean(G, axis=0)
        c2_G = (2.0 / N) * np.sum(G * cos2, axis=0)
        s2_G = (2.0 / N) * np.sum(G * sin2, axis=0)
        A2_G = np.hypot(c2_G, s2_G)
        relief_raw = A2_G / (mean_G + 1.0)
        relief_mod = _masked_gaussian(relief_raw, valid, sigma=2.5)

        # PPL texture: local std (7×7) of high-pass mean L_ppl.
        # All spatial operations use normalised convolution.
        mean_L_ppl = np.mean(L_ppl, axis=0)
        mean_L_smooth = _masked_blur(mean_L_ppl, valid, (7, 7))
        high_pass = np.where(valid, mean_L_ppl - mean_L_smooth, 0.0)
        hp_sq_blur = _masked_blur(high_pass ** 2, valid, (7, 7))
        hp_mean_blur = _masked_blur(high_pass, valid, (7, 7))
        ppl_tex = np.sqrt(np.maximum(hp_sq_blur - hp_mean_blur ** 2, 0.0))

        # PPL Lab (white-balanced mean PPL)
        ppl_Lab = _rgb_to_lab(ppl_mean_rgb)

        # ── §8.4  Relative XPL colour ───────────────────────────────
        xpl_rel_rgb = np.clip(
            xpl_max_rgb / np.maximum(ppl_mean_rgb, 1.0), 0.0, 1.5)

        # ── §8.5  Michel-Lévy lookup ────────────────────────────────
        gamma_est_nm = np.full((H, W), np.nan, dtype=np.float32)
        gamma_dE_arr = np.full((H, W), np.nan, dtype=np.float32)
        biref_cls = np.zeros((H, W), dtype=np.int8)

        aniso_mask = ((aniso >= self.aniso_min)
                      & (xpl_max_L >= self.xpl_dark_rel * W_level)
                      & valid)

        if self.lut_lab is not None and np.any(aniso_mask):
            # Convert xpl_rel_rgb to Lab for the lookup
            xpl_rel_lab = _rgb_to_lab(xpl_rel_rgb * 255.0)

            # Restrict LUT to Γ ≤ 2000 nm for the argmin search
            lut_mask = self.lut_gamma_nm <= 2000
            lut_gamma_sub = self.lut_gamma_nm[lut_mask]
            lut_lab_sub = self.lut_lab[lut_mask]

            # Extract aniso-pixel Lab values
            yx = np.where(aniso_mask)
            pix_lab = xpl_rel_lab[yx]  # [M, 3]

            # Chunked argmin to limit memory: [M, len(lut)] can be huge
            chunk = 5000   # 5000 x 1001 x 3 float32 ≈ 60 MB per chunk
            best_nm = np.zeros(len(pix_lab), dtype=np.float32)
            best_dE = np.zeros(len(pix_lab), dtype=np.float32)

            for i0 in range(0, len(pix_lab), chunk):
                i1 = min(i0 + chunk, len(pix_lab))
                diff = pix_lab[i0:i1, np.newaxis, :] - lut_lab_sub[np.newaxis, :, :]
                dE = np.sqrt(np.sum(diff ** 2, axis=2))  # [chunk, lut]
                idx = np.argmin(dE, axis=1)
                best_nm[i0:i1] = lut_gamma_sub[idx]
                best_dE[i0:i1] = dE[np.arange(len(idx)), idx]

            gamma_est_nm[yx] = best_nm
            gamma_dE_arr[yx] = best_dE

            # High-order white rule (§8.5)
            how = self.high_order_white
            dE_thr = how.get("dE", 25)
            L_thr = how.get("L", 75)
            chroma_thr = how.get("chroma", 15)
            relief_thr = how.get("relief_mod", 0.3)

            # L* and chroma of the observed XPL colour (not PPL)
            L_star_xrl = xpl_rel_lab[..., 0]
            chroma_xrl = np.hypot(xpl_rel_lab[..., 1], xpl_rel_lab[..., 2])

            hw_mask = (aniso_mask
                       & (gamma_dE_arr > dE_thr)
                       & (L_star_xrl > L_thr)
                       & (chroma_xrl < chroma_thr)
                       & (relief_mod > relief_thr))
            gamma_est_nm[hw_mask] = 2500.0

        # Birefringence classes (§8.5)
        # C3 = 1100–2000 (inclusive), C4 = >2000
        c0, c1, c2, c3 = self.biref_class_nm  # [100, 350, 1100, 2000]
        biref_cls[aniso_mask & (gamma_est_nm < c0)] = 0
        biref_cls[aniso_mask & (gamma_est_nm >= c0) & (gamma_est_nm < c1)] = 1
        biref_cls[aniso_mask & (gamma_est_nm >= c1) & (gamma_est_nm < c2)] = 2
        biref_cls[aniso_mask & (gamma_est_nm >= c2) & (gamma_est_nm <= c3)] = 3
        biref_cls[aniso_mask & (gamma_est_nm > c3)] = 4

        # ── Mask invalid pixels ──
        nan_float = [ext_deg, aniso, r2_4, ppl_min_L, ppl_max_L, xpl_max_L,
                     pleo_dark_deg, pleo_dE, relief_mod, ppl_tex,
                     gamma_est_nm, gamma_dE_arr]
        for arr in nan_float:
            arr[~valid] = np.nan

        nan_float_3ch = [xpl_rel_rgb, ppl_Lab]
        for arr in nan_float_3ch:
            arr[~valid] = np.nan

        zero_uint8_3ch = [xpl_max_rgb, xpl_maxproj_rgb, ppl_mean_rgb]
        for arr in zero_uint8_3ch:
            arr[~valid] = 0

        opaque_px[~valid] = False
        biref_cls[~valid] = -1          # -1 = invalid, 0 = C0 (incl. isotropic)

        # ── §8.6  Output format ──
        return {
            "ext_deg":          ext_deg.astype(np.float16),
            "aniso":            aniso.astype(np.float16),
            "r2_4":             r2_4.astype(np.float16),
            "xpl_max_L":        xpl_max_L.astype(np.float16),
            "aniso_mask":       aniso_mask,
            "xpl_max_rgb":      np.clip(xpl_max_rgb, 0, 255).astype(np.uint8),
            "xpl_rel_rgb":      xpl_rel_rgb.astype(np.float16),
            "xpl_maxproj_rgb":  np.clip(xpl_maxproj_rgb, 0, 255).astype(np.uint8),
            "gamma_est_nm":     gamma_est_nm.astype(np.float16),
            "gamma_dE":         gamma_dE_arr.astype(np.float16),
            "biref_cls":        biref_cls,
            "ppl_mean_rgb":     np.clip(ppl_mean_rgb, 0, 255).astype(np.uint8),
            "ppl_Lab":          ppl_Lab.astype(np.float16),
            "pleo_dE":          pleo_dE.astype(np.float16),
            "pleo_dark_deg":    pleo_dark_deg.astype(np.float16),
            "ppl_min_L":        ppl_min_L.astype(np.float16),
            "ppl_max_L":        ppl_max_L.astype(np.float16),
            "relief_mod":       relief_mod.astype(np.float16),
            "ppl_tex":          ppl_tex.astype(np.float16),
            "valid":            valid,
            "white_level":      np.float32(W_level),
            "wb_gains":         wb_gains,
        }
