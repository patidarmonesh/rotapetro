"""Phase 1 — Align the rotation frames (§7).

Module: src/rotapetro/align.py
Inputs:  sites.csv (usable sites) + raw frames
Outputs: data/interim/aligned/<site_id>.npz
         data/interim/qc/align/<site_id>.png
"""
from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.rotapetro.io import resolve_pol_dir, resolve_frame_path

logger = logging.getLogger(__name__)


class FrameAligner:
    """Phase 1 (§7): align rotation frames to the reference (frame 000).

    Parameters
    ----------
    cfg : dict
        Loaded from ``configs/default.yaml``.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        acq = cfg.get("acquisition", {})
        align_cfg = cfg.get("align", {})

        self.n_raw_frames = acq.get("n_raw_frames", 72)
        self.frame_size = acq.get("frame_size", 490)
        self.valid_radius = acq.get("valid_radius_px", 232)
        self.step_deg = acq.get("step_deg", 5.0)

        self.orb_features = align_cfg.get("orb_features", 3000)
        self.n_matches = align_cfg.get("n_matches", 600)
        self.ransac_px = align_cfg.get("ransac_px", 2.0)
        self.max_angle_err = align_cfg.get("max_angle_err_deg", 1.5)
        self.max_scale_err = align_cfg.get("max_scale_err", 0.01)

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    def align_site(self, raw_folder: str | Path) -> dict:
        """Align all 72 PPL+XPL frames of one site to frame 000.

        Implements the full §7.1 algorithm: direct ORB registration →
        chain fallback → nominal fallback.  Then fold 360° → 180° and
        build the valid mask.

        Parameters
        ----------
        raw_folder : path
            ``data/raw/ukvm_dataset/specimens/<code>/rot/r<k>``

        Returns
        -------
        dict
            Keys matching §7.2: ppl, xpl, theta_deg, valid, M,
            rot_measured_deg, rot_sign, center_xy, align_fallback.
        """
        raw_folder = Path(raw_folder)
        N = self.n_raw_frames  # 72
        H = W = self.frame_size  # 490

        # ── Step 1: read all 72 PPL + 72 XPL frames ──
        ppl_dir = resolve_pol_dir(raw_folder, "ppl")
        xpl_dir = resolve_pol_dir(raw_folder, "xpl")
        if ppl_dir is None:
            raise FileNotFoundError(f"Missing ppl directory in {raw_folder}")
        if xpl_dir is None:
            raise FileNotFoundError(f"Missing xpl directory in {raw_folder}")
        ppl_raw = self._load_all_frames(ppl_dir, N)  # [72,H,W,3] RGB uint8
        xpl_raw = self._load_all_frames(xpl_dir, N)

        # ── Step 2: reference = PPL frame 000, grayscale ──
        ref_gray = cv2.cvtColor(
            cv2.cvtColor(ppl_raw[0], cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
        orb_mask = _circle_mask_uint8(H, W, radius=230, cx=W // 2, cy=H // 2)
        kp_ref, des_ref = self._detect_orb(ref_gray, orb_mask)

        # Arrays to fill
        M_all = np.zeros((N, 2, 3), dtype=np.float32)
        M_all[0] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)  # identity
        rot_measured = np.zeros(N, dtype=np.float32)
        fallback = np.zeros(N, dtype=bool)
        ecc_cc = np.zeros(N, dtype=np.float32)
        ecc_ok = np.zeros(N, dtype=bool)
        ecc_cc[0], ecc_ok[0] = 1.0, True
        ecc_mask = _circle_mask_uint8(H, W, radius=215, cx=W // 2, cy=H // 2)

        # For chain registration: keep previous-frame keypoints
        prev_gray = ref_gray
        kp_prev, des_prev = kp_ref, des_ref

        # ── Determine rotation sign from frame 1 (§7.1 step 4) ──
        rot_sign: int = self._determine_sign(
            ppl_raw, orb_mask, kp_ref, des_ref)

        # ── Step 3–4: register each frame i=1..71 ──
        for i in range(1, N):
            frame_gray = cv2.cvtColor(
                cv2.cvtColor(ppl_raw[i], cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
            kp_i, des_i = self._detect_orb(frame_gray, orb_mask)

            M, rot_i, s_i, ok = self._try_direct_registration(
                kp_i, des_i, kp_ref, des_ref, i, rot_sign)

            # ── Fallback 1: chain registration (frame i to i-1) ──
            if not ok:
                M_chain, rot_chain, s_chain, ok_chain = \
                    self._try_direct_registration(
                        kp_i, des_i, kp_prev, des_prev, 1, rot_sign)
                if ok_chain and M_chain is not None:
                    # Compose: frame i → frame i-1 → frame 000
                    # M_chain: frame i → frame i-1
                    # M_prev:  frame i-1 → frame 000
                    # Correct order: M_prev @ M_chain
                    M_prev_3x3 = np.vstack([M_all[i - 1], [0, 0, 1]])
                    M_chain_3x3 = np.vstack([M_chain, [0, 0, 1]])
                    M_composed_3x3 = M_prev_3x3 @ M_chain_3x3
                    M = M_composed_3x3[:2, :]

                    rot_i = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
                    s_i = np.hypot(M[0, 0], M[1, 0])
                    ok = self._check_sanity(rot_i, s_i, i, rot_sign)

            # ── Fallback 2: nominal rotation ──
            if not ok or M is None:
                fallback[i] = True
                nominal_deg = rot_sign * self.step_deg * i
                # Use median rotation centre from successful frames
                cx, cy = self._get_rotation_center(M_all, i)
                M = cv2.getRotationMatrix2D(
                    (float(cx), float(cy)), -nominal_deg, 1.0)
                M = M.astype(np.float32)
                rot_i = nominal_deg
                logger.debug("  frame %d: nominal fallback (%.1f°)", i, nominal_deg)

            # ── Step 4b: sub-pixel refinement (residual ECC on the coarsely
            #    aligned frame; avoids OpenCV's Euclidean-ECC failure near 180°)
            M, ecc_cc[i], ecc_ok[i] = self._refine_ecc(ref_gray, frame_gray, M,
                                                       ecc_mask)
            rot_i = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))

            M_all[i] = M
            rot_measured[i] = rot_i

            # Update chain-registration reference
            prev_gray = frame_gray
            kp_prev, des_prev = kp_i, des_i

        # rot_sign is always determined before the loop

        # ── Step 5: warp all PPL and XPL frames ──
        ppl_warped = np.zeros((N, H, W, 3), dtype=np.uint8)
        xpl_warped = np.zeros((N, H, W, 3), dtype=np.uint8)
        warp_valid = np.ones((N, H, W), dtype=bool)
        # Validity is GEOMETRIC (where the warped frame has real data), not
        # "pixel > 0": opaque grains and extinct grains are legitimately black
        # in XPL and must not be removed from the valid mask.
        ones = np.full((H, W), 255, dtype=np.uint8)

        ppl_warped[0] = ppl_raw[0]
        xpl_warped[0] = xpl_raw[0]

        for i in range(1, N):
            ppl_bgr = cv2.cvtColor(ppl_raw[i], cv2.COLOR_RGB2BGR)
            xpl_bgr = cv2.cvtColor(xpl_raw[i], cv2.COLOR_RGB2BGR)

            w_ppl = cv2.warpAffine(
                ppl_bgr, M_all[i], (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            w_xpl = cv2.warpAffine(
                xpl_bgr, M_all[i], (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

            ppl_warped[i] = cv2.cvtColor(w_ppl, cv2.COLOR_BGR2RGB)
            xpl_warped[i] = cv2.cvtColor(w_xpl, cv2.COLOR_BGR2RGB)

            w_ones = cv2.warpAffine(ones, M_all[i], (W, H),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            warp_valid[i] = w_ones > 0

        # ── Step 6: valid mask ──
        # Circle of radius 232 px around rotation centre AND non-zero in ALL frames
        cx_center, cy_center = self._get_rotation_center(M_all, N)
        circle_valid = _circle_mask_bool(H, W, self.valid_radius,
                                         int(round(cx_center)),
                                         int(round(cy_center)))
        all_nonzero = np.all(warp_valid, axis=0)
        valid = circle_valid & all_nonzero

        # ── Step 7: fold 360° → 180° ──
        # Average frame i with frame i+36 → 36 frames at 0, 5, …, 175°
        n_folded = N // 2  # 36
        ppl_folded = np.zeros((n_folded, H, W, 3), dtype=np.uint8)
        xpl_folded = np.zeros((n_folded, H, W, 3), dtype=np.uint8)
        theta_deg = np.arange(n_folded, dtype=np.float32) * self.step_deg

        for k in range(n_folded):
            # Float mean of frame k and frame k+36, rounded back to uint8
            ppl_folded[k] = np.round(
                (ppl_warped[k].astype(np.float32)
                 + ppl_warped[k + n_folded].astype(np.float32)) / 2.0
            ).astype(np.uint8)
            xpl_folded[k] = np.round(
                (xpl_warped[k].astype(np.float32)
                 + xpl_warped[k + n_folded].astype(np.float32)) / 2.0
            ).astype(np.uint8)

        return {
            "ppl": ppl_folded,                              # uint8 [36,490,490,3]
            "xpl": xpl_folded,                              # uint8 [36,490,490,3]
            "theta_deg": theta_deg,                         # float32 [36]
            "valid": valid,                                 # bool [490,490]
            "M": M_all,                                     # float32 [72,2,3]
            "rot_measured_deg": rot_measured,                # float32 [72]
            "rot_sign": np.int8(rot_sign),                  # int8 scalar
            "center_xy": np.array([cx_center, cy_center],
                                  dtype=np.float32),        # float32 [2]
            "align_fallback": fallback,                     # bool [72]
            "ecc_cc": ecc_cc,                               # float32 [72]
            "ecc_ok": ecc_ok,                               # bool [72]
        }

    def compute_qc(self, result: dict, ppl_warped_000: np.ndarray,
                   ppl_warped_036: np.ndarray) -> dict:
        """Compute QC metrics from §7.3.

        Parameters
        ----------
        result : dict
            Output of ``align_site()``.
        ppl_warped_000, ppl_warped_036 : ndarray [H,W,3] uint8
            Aligned (but not yet folded) PPL frames 000 and 036.

        Returns
        -------
        dict with keys: median_angle_err, n_fallback, fold_diff_mean, passed.
        """
        rot_sign = int(result["rot_sign"])
        rot_measured = result["rot_measured_deg"]
        fallback = result["align_fallback"]
        valid = result["valid"]

        # Median angle error (degrees)
        expected = np.array([rot_sign * self.step_deg * i
                             for i in range(self.n_raw_frames)],
                            dtype=np.float32)
        # Wrap the difference into [-180, 180) before taking |.|
        errors = np.abs((rot_measured - expected + 180.0) % 360.0 - 180.0)
        non_fb = ~fallback
        median_err = float(np.median(errors[non_fb])) if non_fb.any() else 999.0

        # Number of fallback frames
        n_fallback = int(fallback.sum())

        # Mean absolute difference between frames 000 and 036 (before fold)
        gray_000 = cv2.cvtColor(
            cv2.cvtColor(ppl_warped_000, cv2.COLOR_RGB2BGR),
            cv2.COLOR_BGR2GRAY).astype(np.float32)
        gray_036 = cv2.cvtColor(
            cv2.cvtColor(ppl_warped_036, cv2.COLOR_RGB2BGR),
            cv2.COLOR_BGR2GRAY).astype(np.float32)
        # Compare after a light blur (sigma 1.5 px): bilinear resampling of
        # frame 036 blurs it, so an unblurred MAD mostly measures interpolation
        # and JPEG noise, not misalignment.
        b000 = cv2.GaussianBlur(gray_000, (0, 0), 1.5)
        b036 = cv2.GaussianBlur(gray_036, (0, 0), 1.5)
        fold_diff = float(np.mean(np.abs(b000[valid] - b036[valid])))

        # Residual translation between 000 and aligned 036 (gradient phase
        # correlation) — a direct measure of misalignment in pixels.
        def _grad(g):
            gx = cv2.Sobel(g, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
            return (np.hypot(gx, gy) * valid).astype(np.float32)
        (dx, dy), _ = cv2.phaseCorrelate(_grad(gray_000), _grad(gray_036))
        residual_shift = float(np.hypot(dx, dy))

        passed = (median_err < 0.5
                  and n_fallback < 3
                  and residual_shift < 0.3
                  and fold_diff < 10.0)   # soft: JPEG/defocus also contribute

        return {
            "median_angle_err": median_err,
            "n_fallback": n_fallback,
            "fold_diff_mean": fold_diff,
            "residual_shift_px": residual_shift,
            "passed": passed,
        }

    # ──────────────────────────────────────────────────────────────────────
    #  Private: ORB detection and matching
    # ──────────────────────────────────────────────────────────────────────

    def _detect_orb(self, gray: np.ndarray,
                    mask: np.ndarray) -> tuple[list, np.ndarray | None]:
        """Detect ORB features inside the given mask."""
        orb = cv2.ORB_create(nfeatures=self.orb_features)
        kp, des = orb.detectAndCompute(gray, mask)
        return kp, des

    def _match_features(self, des_query: np.ndarray,
                        des_ref: np.ndarray) -> list:
        """Hamming brute-force match with cross-check; keep top N."""
        if des_query is None or des_ref is None:
            return []
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des_query, des_ref)
        matches = sorted(matches, key=lambda m: m.distance)
        return matches[:self.n_matches]

    def _determine_sign(self, ppl_raw: np.ndarray,
                        orb_mask: np.ndarray,
                        kp_ref, des_ref) -> int:
        """Determine rotation sign from early frames (preferably frame 1).

        Tries frames 1 through 5.  For each, registers to the reference
        and reads the sign of the measured rotation.  Returns +1 or -1.
        Falls back to +1 if all attempts fail.
        """
        for probe_i in range(1, min(6, len(ppl_raw))):
            gray = cv2.cvtColor(
                cv2.cvtColor(ppl_raw[probe_i], cv2.COLOR_RGB2BGR),
                cv2.COLOR_BGR2GRAY)
            kp_i, des_i = self._detect_orb(gray, orb_mask)
            matches = self._match_features(des_i, des_ref)
            if len(matches) < 10:
                continue

            pts_frame = np.float32(
                [kp_i[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts_ref = np.float32(
                [kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, _ = cv2.estimateAffinePartial2D(
                pts_frame, pts_ref,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_px)
            if M is None:
                continue

            rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
            scale = float(np.hypot(M[0, 0], M[1, 0]))

            if abs(scale - 1.0) > 0.05:
                continue  # not a plausible registration

            sign = 1 if rot_deg > 0 else -1

            # Verify: expected = sign * 5 * probe_i
            expected = sign * self.step_deg * probe_i
            diff = abs(rot_deg - expected) % 360
            diff = min(diff, 360 - diff)
            if diff <= 5.0:  # generous tolerance for sign detection
                logger.info("  Rotation sign = %+d (from frame %d, "
                            "rot=%.2f°)", sign, probe_i, rot_deg)
                return sign

        logger.warning("  Could not determine rotation sign from frames "
                       "1-5, defaulting to +1")
        return 1

    def _try_direct_registration(
        self,
        kp_frame, des_frame,
        kp_ref, des_ref,
        frame_step: int,
        rot_sign: int,
    ) -> tuple[np.ndarray | None, float, float, bool]:
        """Try ORB → RANSAC registration.  Returns (M, rot_deg, scale, ok).

        ``frame_step`` is the expected step count (i for direct, 1 for
        chain).
        """
        matches = self._match_features(des_frame, des_ref)
        if len(matches) < 10:
            return None, 0.0, 0.0, False

        pts_frame = np.float32(
            [kp_frame[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        pts_ref = np.float32(
            [kp_ref[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        M, inliers = cv2.estimateAffinePartial2D(
            pts_frame, pts_ref,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_px)

        if M is None:
            return None, 0.0, 0.0, False

        rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        ok = self._check_sanity(rot_deg, scale, frame_step, rot_sign)

        return M, rot_deg, scale, ok

    def _check_sanity(self, rot_deg: float, scale: float,
                      frame_step: int, rot_sign: int) -> bool:
        """§7.1 step 4: |rot_i − (±5·i)| ≤ 1.5° (mod 360) and |s−1| ≤ 0.01."""
        expected = rot_sign * self.step_deg * frame_step
        diff = abs(rot_deg - expected) % 360
        diff = min(diff, 360 - diff)
        return diff <= self.max_angle_err and abs(scale - 1.0) <= self.max_scale_err

    def _refine_ecc(self, ref_gray: np.ndarray, frame_gray: np.ndarray,
                    M: np.ndarray, mask: np.ndarray
                    ) -> tuple[np.ndarray, float, bool]:
        """Sub-pixel refinement of a frame→reference transform.

        ORB+RANSAC leaves ~0.5–1.5 px residual error.  We warp the frame with
        the coarse M, then run ECC (Euclidean) between the reference and the
        warped frame starting from identity, and compose the small correction.
        Running ECC on the coarsely aligned image matters: OpenCV's Euclidean
        ECC parametrises the angle with asin(), so it diverges for rotations
        near 180° if started from the full transform.

        Returns (M_refined, correlation, ok).  Keeps M if ECC fails or the
        correction is implausibly large.
        """
        H, W = ref_gray.shape
        warped = cv2.warpAffine(frame_gray, M, (W, H), flags=cv2.INTER_LINEAR)
        # Match GRADIENT-magnitude images, not intensities: grain boundaries
        # stay put under rotation while PPL colours change (pleochroism),
        # which lowers intensity correlation (e.g. biotite-rich fields).
        ref = _grad_mag(ref_gray)
        mov = _grad_mag(warped)
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5)
        try:
            cc, warp = cv2.findTransformECC(ref, mov, warp,
                                            cv2.MOTION_EUCLIDEAN, crit, mask, 1)
        except cv2.error:
            return M, 0.0, False
        # warp maps ref coords -> warped coords; correction maps warped -> ref
        corr = cv2.invertAffineTransform(warp)
        d_theta = abs(np.degrees(np.arctan2(corr[1, 0], corr[0, 0])))
        d_t = float(np.hypot(corr[0, 2], corr[1, 2]))
        if cc < 0.7 or d_theta > 1.0 or d_t > 3.0:
            return M, float(cc), False
        M3 = np.vstack([M, [0, 0, 1]])
        C3 = np.vstack([corr, [0, 0, 1]])
        return (C3 @ M3)[:2, :].astype(np.float32), float(cc), True

    def _get_rotation_center(self, M_all: np.ndarray,
                             n_valid: int) -> tuple[float, float]:
        """Estimate the rotation centre as the median fixed point of
        the successfully estimated affine transforms.

        For a similarity transform M that represents a rotation about
        (cx, cy), the fixed point satisfies ``M @ [cx, cy, 1]^T = [cx, cy]^T``,
        which gives the linear system ``(I - A) @ c = t`` where
        ``A = M[:, :2]`` and ``t = M[:, 2]``.

        Falls back to (frame_size/2, frame_size/2) if nothing works.
        """
        centres = []
        half = self.frame_size / 2.0
        for i in range(1, min(n_valid, len(M_all))):
            M = M_all[i]
            A = M[:, :2]           # 2×2
            t = M[:, 2]            # 2
            I_minus_A = np.eye(2) - A
            det = np.linalg.det(I_minus_A)
            if abs(det) < 1e-8:
                continue  # near-identity transform, can't solve
            c = np.linalg.solve(I_minus_A, t)
            cx, cy = float(c[0]), float(c[1])
            # Sanity: should be near image centre
            if abs(cx - half) < 50 and abs(cy - half) < 50:
                centres.append((cx, cy))

        if centres:
            arr = np.array(centres)
            return float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))
        return half, half

    # ──────────────────────────────────────────────────────────────────────
    #  Private: frame I/O
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_all_frames(folder: Path, n: int) -> np.ndarray:
        """Load n frames as uint8 RGB [n, H, W, 3].
        
        Uses resolve_frame_path for case-insensitive file extensions.
        """
        frames = []
        for i in range(n):
            path = resolve_frame_path(folder, i)
            if path is None:
                raise FileNotFoundError(f"Cannot find frame {i:03d} in {folder}")
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"Cannot read {path}")
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return np.array(frames, dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ──────────────────────────────────────────────────────────────────────────

def _grad_mag(gray: np.ndarray) -> np.ndarray:
    """Normalised gradient magnitude (float32, max 1) for ECC."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 1.0)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0),
                      cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m / max(float(m.max()), 1e-6)


def _circle_mask_uint8(h: int, w: int, radius: int,
                       cx: int, cy: int) -> np.ndarray:
    """uint8 255/0 mask for ORB (cv2 wants uint8 mask)."""
    y, x = np.ogrid[:h, :w]
    return (((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2).astype(np.uint8) * 255


def _circle_mask_bool(h: int, w: int, radius: int,
                      cx: int, cy: int) -> np.ndarray:
    """Boolean mask for valid-pixel computation."""
    y, x = np.ogrid[:h, :w]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2


def save_qc_png(ppl_folded: np.ndarray, valid: np.ndarray,
                site_id: str, save_path: str | Path) -> None:
    """§7.3 QC image: PPL at 0°/45°/90°/135° side-by-side + flicker diff.

    Parameters
    ----------
    ppl_folded : ndarray [36, H, W, 3] uint8 RGB
    valid : ndarray [H, W] bool
    site_id : str
    save_path : path
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Frame indices for 0°, 45°, 90°, 135° (step=5° → idx 0, 9, 18, 27)
    indices = [0, 9, 18, 27]
    labels = ["PPL 0°", "PPL 45°", "PPL 90°", "PPL 135°"]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for ax, idx, label in zip(axes[:4], indices, labels):
        ax.imshow(ppl_folded[idx])
        ax.set_title(label)
        ax.axis("off")

    # Edge overlay of 0° (red) and 90° (green): grain boundaries do not move
    # with rotation, so a good alignment shows yellow (coincident) edges;
    # double red/green edges mean misalignment.  (A 0°−45° difference mostly
    # shows pleochroism, not alignment errors.)
    def _edges(img):
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        e = cv2.Canny(cv2.GaussianBlur(g, (0, 0), 1.0), 40, 100)
        return (e > 0) & valid
    overlay = np.zeros(ppl_folded[0].shape, dtype=np.uint8)
    overlay[_edges(ppl_folded[0]), 0] = 255
    overlay[_edges(ppl_folded[18]), 1] = 255
    axes[4].imshow(overlay)
    axes[4].set_title("Edges 0° (red) / 90° (green)")
    axes[4].axis("off")

    fig.suptitle(site_id, fontsize=14)
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
