"""Phase 3 — Automatic grain segmentation (§9).

Module: src/rotapetro/segment.py
Inputs:  aligned stacks (§7.2) + maps (§8.6)
Outputs: segments/<site_id>.png  (uint16 label image)
         segments/<site_id>.json (region list)
         qc/segment/<site_id>.png

Implements §9.1–9.5: feature image, SLIC, RAG merge, twin merge,
small-region merge, region-type classification.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import regionprops
from skimage.segmentation import slic

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 RGB [H,W,3] → float32 Lab (L 0-100, a/b centred at 0)."""
    bgr = rgb[..., ::-1].copy()
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)
    return lab


def _lab_deltaE(lab1: np.ndarray, lab2: np.ndarray) -> float:
    """ΔE76 between two Lab vectors (1-D or scalar per-channel)."""
    return float(np.sqrt(np.sum((lab1 - lab2) ** 2)))


def _region_means(F: np.ndarray, labels: np.ndarray, n: int) -> np.ndarray:
    """Mean feature vector per label 0..n (vectorised with bincount)."""
    flat = labels.ravel()
    cnt = np.bincount(flat, minlength=n + 1).astype(np.float64)
    means = np.zeros((n + 1, F.shape[-1]), dtype=np.float64)
    for ch in range(F.shape[-1]):
        means[:, ch] = np.bincount(flat, weights=F[..., ch].ravel().astype(np.float64),
                                   minlength=n + 1)
    return means / np.maximum(cnt, 1)[:, None]


def _adjacent_pairs(labels: np.ndarray) -> np.ndarray:
    """Unique (a, b) pairs, a < b, of 4-connected neighbouring labels > 0."""
    pairs = []
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.append(np.stack([np.minimum(a[m], b[m]), np.maximum(a[m], b[m])], 1))
    p = np.concatenate(pairs, 0)
    return np.unique(p, axis=0) if len(p) else p.reshape(0, 2)


def _threshold_merge(F: np.ndarray, labels: np.ndarray, valid: np.ndarray,
                     thresh: float) -> np.ndarray:
    """Merge neighbouring superpixels whose mean features differ < thresh.

    Same rule as skimage's ``cut_threshold`` (drop edges ≥ thresh, take
    connected components), but for any number of channels and ignoring the
    background (label 0).
    """
    labels = np.where(valid, labels, 0).astype(np.int32)
    n = int(labels.max())
    if n == 0:
        return labels
    means = _region_means(F, labels, n)
    pairs = _adjacent_pairs(labels)
    if len(pairs) == 0:
        return labels
    d = np.linalg.norm(means[pairs[:, 0]] - means[pairs[:, 1]], axis=1)
    keep = pairs[d < thresh]
    graph = coo_matrix((np.ones(len(keep)), (keep[:, 0], keep[:, 1])),
                       shape=(n + 1, n + 1))
    _, comp = connected_components(graph, directed=False)
    out = comp[labels] + 1          # component ids start at 0
    out[labels == 0] = 0
    return out.astype(np.int32)


def _hierarchical_merge(F: np.ndarray, labels: np.ndarray, valid: np.ndarray,
                        thresh: float) -> np.ndarray:
    """Greedy region merging with mean updates (no chaining).

    Repeatedly merges the most similar pair of NEIGHBOURING regions,
    recomputing the merged region's mean, until the closest pair differs by
    >= thresh.  Unlike single-linkage (cut_threshold), a chain of small
    colour steps across a grain boundary cannot link two different grains.
    """
    import heapq
    labels = np.where(valid, labels, 0).astype(np.int32)
    n = int(labels.max())
    if n == 0:
        return labels
    flat = labels.ravel()
    cnt = np.bincount(flat, minlength=n + 1).astype(np.float64)
    sums = np.stack([np.bincount(flat, weights=F[..., c].ravel().astype(np.float64),
                                 minlength=n + 1) for c in range(F.shape[-1])], 1)
    nbrs: dict[int, set[int]] = {i: set() for i in range(1, n + 1)}
    for a, b in _adjacent_pairs(labels):
        nbrs[int(a)].add(int(b)); nbrs[int(b)].add(int(a))
    parent = np.arange(n + 1)
    alive = cnt > 0
    version = np.zeros(n + 1, dtype=np.int64)

    def dist(a: int, b: int) -> float:
        return float(np.linalg.norm(sums[a] / cnt[a] - sums[b] / cnt[b]))

    heap = [(dist(a, b), a, b, 0, 0) for a in nbrs for b in nbrs[a] if a < b]
    heapq.heapify(heap)
    while heap:
        d, a, b, va, vb = heapq.heappop(heap)
        if d >= thresh:
            break
        if not (alive[a] and alive[b]) or va != version[a] or vb != version[b]:
            continue
        # merge b into a
        sums[a] += sums[b]; cnt[a] += cnt[b]; alive[b] = False; parent[b] = a
        version[a] += 1
        nbrs[a] |= nbrs[b]; nbrs[a].discard(a); nbrs[a].discard(b)
        for c in nbrs[b]:
            if c != a:
                nbrs[c].discard(b); nbrs[c].add(a)
        del nbrs[b]
        for c in nbrs[a]:
            lo, hi = (a, c) if a < c else (c, a)
            heapq.heappush(heap, (dist(a, c), lo, hi, version[lo], version[hi]))
    # resolve parents
    root = np.arange(n + 1)
    for i in range(1, n + 1):
        r = i
        while parent[r] != r:
            r = parent[r]
        root[i] = r
    out = root[labels]
    out[labels == 0] = 0
    return out.astype(np.int32)


def _circular_variance(angles_rad: np.ndarray) -> float:
    """Circular variance of angles (period already folded into input).

    Returns 1 − R̄ where R̄ = |mean(exp(i·angle))|.
    A uniform distribution gives ~1; a single direction gives ~0.
    """
    z = np.exp(1j * angles_rad)
    R = np.abs(np.mean(z))
    return float(1.0 - R)


# ──────────────────────────────────────────────────────────────────────────
#  GrainSegmenter
# ──────────────────────────────────────────────────────────────────────────

class GrainSegmenter:
    """Phase 3 (§9): segment aligned rotation stacks into grain regions.

    Parameters
    ----------
    cfg : dict
        Loaded from ``configs/default.yaml``.
    """

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        seg = cfg.get("segment", {})

        # Tested on granite_r1 / 11agt25_r1 (≈2.1 µm/px): 60 px superpixels
        # with low compactness follow thin laths and grain boundaries much
        # better than 150 / 10.
        self.px_per_segment = seg.get("slic_px_per_segment", 60)
        self.compactness = seg.get("slic_compactness", 5)
        self.T_merge = seg.get("T_merge", 18)
        # "hierarchical" (default): merge most-similar pair first, update
        # means; "single_linkage": skimage cut_threshold behaviour.
        self.merge_mode = seg.get("merge_mode", "hierarchical")
        self.min_region_px = seg.get("min_region_px", 150)
        self.min_sample_px = seg.get("min_sample_px", 400)
        self.hole_rel_L = seg.get("hole_rel_L", 0.92)

        twin = seg.get("twin_merge", {})
        self.twin_dE_ppl = twin.get("dE_ppl", 6)
        self.twin_max_biref_diff = twin.get("max_biref_diff", 1)
        self.twin_grad_ratio = twin.get("boundary_grad_ratio", 0.5)
        self.twin_min_shared = twin.get("min_shared_px", 15)

        agg = seg.get("aggregate", {})
        self.agg_circvar = agg.get("circvar", 0.6)
        self.agg_r2_4 = agg.get("r2_4", 0.4)

        acq = cfg.get("acquisition", {})
        self.um_per_px = acq.get("um_per_px_rot", 4.0)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def segment_site(self, aligned: dict, maps: dict,
                     site_id: str) -> tuple[np.ndarray, list[dict]]:
        """Segment one site into grain regions.

        Parameters
        ----------
        aligned : dict
            Phase 1 output (NPZ keys per §7.2).
        maps : dict
            Phase 2 output (NPZ keys per §8.6).
        site_id : str

        Returns
        -------
        label_img : ndarray [H,W] uint16
            0 = background/invalid, 1..n = region labels.
        region_list : list[dict]
            Per-region metadata per §9.5.
        """
        valid = aligned["valid"].astype(bool)
        H, W = valid.shape
        valid_area = int(valid.sum())

        if valid_area == 0:
            return np.zeros((H, W), dtype=np.uint16), []

        # Retrieve maps (cast float16 → float32 where needed)
        aniso = np.nan_to_num(maps["aniso"].astype(np.float32), nan=0.0)
        ext_deg = np.nan_to_num(maps["ext_deg"].astype(np.float32), nan=0.0)
        r2_4 = np.nan_to_num(maps["r2_4"].astype(np.float32), nan=0.0)
        ppl_max_L = np.nan_to_num(maps["ppl_max_L"].astype(np.float32), nan=0.0)
        ppl_tex = np.nan_to_num(maps["ppl_tex"].astype(np.float32), nan=0.0)
        # Brightest XPL value over the rotation (Phase 2 fix).  Isotropic,
        # opaque and hole pixels stay dark; "aniso" is unreliable there.
        if "xpl_max_L" in maps:
            xpl_max_L = np.nan_to_num(maps["xpl_max_L"].astype(np.float32), nan=0.0)
        else:
            raise KeyError("maps have no 'xpl_max_L' — re-run Phase 2 with the "
                           "fixed maps.py (--force)")
        biref_cls = maps["biref_cls"].astype(np.int8)
        xpl_maxproj_rgb = maps["xpl_maxproj_rgb"]   # uint8
        ppl_mean_rgb = maps["ppl_mean_rgb"]          # uint8
        W_level = float(maps["white_level"])

        # opaque_px: reconstruct from ppl_max_L (not stored separately)
        opaque_rel_L = self.cfg.get("maps", {}).get("opaque_rel_L", 0.15)
        opaque_px = (ppl_max_L < opaque_rel_L * W_level) & valid

        # ── §9.1  Feature image ─────────────────────────────────────
        F = self._build_feature_image(
            xpl_maxproj_rgb, ppl_mean_rgb, aniso, ext_deg, valid)

        # ── §9.2  SLIC superpixels ──────────────────────────────────
        n_seg = max(valid_area // self.px_per_segment, 10)
        labels = slic(
            F, n_segments=n_seg, compactness=self.compactness,
            mask=valid, channel_axis=-1, start_label=1)
        labels = labels.astype(np.int32)
        logger.info("    SLIC: %d superpixels", len(np.unique(labels)) - 1)

        # ── §9.3  RAG merge ─────────────────────────────────────────
        # First pass: merge on full 8-channel feature distance.
        # (skimage.graph.rag_mean_color only supports 3-channel images —
        # it crashes on the 8-channel F — and it also treats label 0 /
        # the invalid background as a region that can be merged.)
        if self.merge_mode == "single_linkage":
            labels = _threshold_merge(F, labels, valid, self.T_merge)
        else:
            labels = _hierarchical_merge(F, labels, valid, self.T_merge)
        n_after_rag = len(np.unique(labels)) - 1
        logger.info("    RAG merge (T=%d): %d regions", self.T_merge, n_after_rag)

        # PPL Lab image for twin merge
        ppl_lab = _rgb_to_lab(ppl_mean_rgb)  # [H,W,3] float32

        # PPL gradient magnitude for boundary scoring
        ppl_gray = cv2.cvtColor(
            ppl_mean_rgb[..., ::-1].copy(), cv2.COLOR_BGR2GRAY).astype(np.float32)
        ppl_grad = np.hypot(
            cv2.Sobel(ppl_gray, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(ppl_gray, cv2.CV_32F, 0, 1, ksize=3))

        # Second pass: twin/sector merge (§9.3)
        labels, n_domains_map = self._twin_merge(
            labels, ppl_lab, biref_cls, ppl_grad, valid)
        n_after_twin = len(np.unique(labels)) - 1
        logger.info("    Twin merge: %d regions", n_after_twin)

        # Remove small regions (< min_region_px) by merging to most similar
        # n_domains should count twin/sector pieces only — absorbing tiny
        # boundary slivers must not inflate it (it did: 16 "domains").
        labels = self._remove_small_regions(labels, F, valid, None)
        n_final = len(np.unique(labels)) - 1
        logger.info("    After small-region removal: %d regions", n_final)

        # ── Renumber labels 1..n ─────────────────────────────────────
        labels, old_to_new = self._renumber(labels)

        # ── §9.4  Classify region types ──────────────────────────────
        region_list = self._classify_regions(
            labels, valid, aniso, ext_deg, r2_4,
            ppl_max_L, ppl_tex, opaque_px, W_level,
            n_domains_map, old_to_new, site_id, xpl_max_L)

        return labels.astype(np.uint16), region_list

    # ------------------------------------------------------------------
    #  §9.1: feature image
    # ------------------------------------------------------------------

    def _build_feature_image(
        self,
        xpl_maxproj_rgb: np.ndarray,
        ppl_mean_rgb: np.ndarray,
        aniso: np.ndarray,
        ext_deg: np.ndarray,
        valid: np.ndarray,
    ) -> np.ndarray:
        """Build 8-channel feature image F (float32).

        Ch 1-3: Lab(xpl_maxproj) × 1.0
        Ch 4-6: Lab(ppl_mean) × 0.7
        Ch 7-8: 40·aniso·cos(4·ext), 40·aniso·sin(4·ext)
        Each channel smoothed with 3×3 median.
        """
        H, W = valid.shape
        xpl_lab = _rgb_to_lab(xpl_maxproj_rgb)
        ppl_lab = _rgb_to_lab(ppl_mean_rgb)

        ext_rad = np.radians(ext_deg)

        F = np.zeros((H, W, 8), dtype=np.float32)
        F[..., 0:3] = xpl_lab * 1.0
        F[..., 3:6] = ppl_lab * 0.7
        F[..., 6] = 40.0 * aniso * np.cos(4 * ext_rad)
        F[..., 7] = 40.0 * aniso * np.sin(4 * ext_rad)

        # Zero out invalid pixels before median filtering
        for ch in range(8):
            F[~valid, ch] = 0.0

        # Smooth each channel with 3×3 median
        for ch in range(8):
            F[..., ch] = scipy.ndimage.median_filter(F[..., ch], size=3)

        return F

    # ------------------------------------------------------------------
    #  §9.3: twin/sector merge
    # ------------------------------------------------------------------

    def _twin_merge(
        self,
        labels: np.ndarray,
        ppl_lab: np.ndarray,
        biref_cls: np.ndarray,
        ppl_grad: np.ndarray,
        valid: np.ndarray,
    ) -> tuple[np.ndarray, dict[int, int]]:
        """Twin and sector merge pass (§9.3).

        Merge adjacent regions A, B if:
        1. ΔE(PPL Lab mean) < dE_ppl
        2. |biref_cls_A − biref_cls_B| ≤ max_biref_diff
        3. mean PPL gradient along shared boundary < ratio × median of all
        4. shared boundary ≥ min_shared_px

        Returns (labels, n_domains_map) where n_domains_map tracks how many
        original regions were merged into each surviving label.
        """
        labels = labels.copy()
        unique = np.unique(labels)
        unique = unique[unique != 0]

        # Track how many pieces were merged into each label
        n_domains: dict[int, int] = {int(lb): 1 for lb in unique}

        # Precompute per-region PPL Lab means and biref mode
        region_lab: dict[int, np.ndarray] = {}
        region_biref: dict[int, int] = {}
        for lb in unique:
            mask = labels == lb
            region_lab[int(lb)] = np.mean(ppl_lab[mask], axis=0)
            biref_vals = biref_cls[mask]
            biref_vals = biref_vals[biref_vals >= 0]  # exclude invalid (-1)
            if len(biref_vals) > 0:
                region_biref[int(lb)] = int(np.bincount(biref_vals).argmax())
            else:
                region_biref[int(lb)] = 0

        # Find all boundary pixel pairs: dilate each label and find adjacency
        # Build adjacency dict with shared boundary info
        adjacency = self._compute_adjacency(labels, ppl_grad, valid)

        # Median PPL gradient across ALL boundaries in the site
        all_grad = []
        for (a, b), info in adjacency.items():
            all_grad.extend(info["grads"])
        if len(all_grad) == 0:
            return labels, n_domains
        median_grad = float(np.median(all_grad))

        # Iterate merge passes until no more merges
        changed = True
        while changed:
            changed = False
            # Rebuild adjacency after each pass
            adjacency = self._compute_adjacency(labels, ppl_grad, valid)
            unique = np.unique(labels)
            unique = unique[unique != 0]

            for (a, b), info in sorted(adjacency.items()):
                if a not in region_lab or b not in region_lab:
                    continue

                # Check all 4 conditions
                dE = _lab_deltaE(region_lab[a], region_lab[b])
                if dE >= self.twin_dE_ppl:
                    continue

                biref_diff = abs(region_biref.get(a, 0) - region_biref.get(b, 0))
                if biref_diff > self.twin_max_biref_diff:
                    continue

                shared_len = info["length"]
                if shared_len < self.twin_min_shared:
                    continue

                mean_boundary_grad = float(np.mean(info["grads"])) if info["grads"] else 999.0
                if mean_boundary_grad >= self.twin_grad_ratio * median_grad:
                    continue

                # All conditions met → merge b into a
                labels[labels == b] = a
                n_domains[a] = n_domains.get(a, 1) + n_domains.get(b, 1)

                # Update region stats
                mask_a = labels == a
                region_lab[a] = np.mean(ppl_lab[mask_a], axis=0)
                biref_vals = biref_cls[mask_a]
                biref_vals = biref_vals[biref_vals >= 0]
                if len(biref_vals) > 0:
                    region_biref[a] = int(np.bincount(biref_vals).argmax())

                if b in region_lab:
                    del region_lab[b]
                if b in region_biref:
                    del region_biref[b]
                if b in n_domains:
                    del n_domains[b]

                changed = True
                break  # restart after merge (adjacency changed)

        return labels, n_domains

    def _compute_adjacency(
        self,
        labels: np.ndarray,
        ppl_grad: np.ndarray,
        valid: np.ndarray,
    ) -> dict[tuple[int, int], dict]:
        """Compute adjacency: for each pair of neighbouring regions,
        record boundary length and the PPL gradient values along it.

        Uses non-wrapping sliced comparisons (right and bottom neighbours
        only) so each boundary edge is visited exactly once.
        """
        adjacency: dict[tuple[int, int], dict] = {}

        # Right neighbour: compare column j with column j+1
        a_r = labels[:, :-1]
        b_r = labels[:, 1:]
        v_r = valid[:, :-1] & valid[:, 1:]
        diff_r = (a_r != b_r) & (a_r > 0) & (b_r > 0) & v_r
        ys, xs = np.where(diff_r)
        for y, x in zip(ys, xs):
            la, lb = int(a_r[y, x]), int(b_r[y, x])
            key = (min(la, lb), max(la, lb))
            if key not in adjacency:
                adjacency[key] = {"length": 0, "grads": []}
            adjacency[key]["length"] += 1
            g = 0.5 * (ppl_grad[y, x] + ppl_grad[y, x + 1])
            adjacency[key]["grads"].append(float(g))

        # Bottom neighbour: compare row i with row i+1
        a_b = labels[:-1, :]
        b_b = labels[1:, :]
        v_b = valid[:-1, :] & valid[1:, :]
        diff_b = (a_b != b_b) & (a_b > 0) & (b_b > 0) & v_b
        ys, xs = np.where(diff_b)
        for y, x in zip(ys, xs):
            la, lb = int(a_b[y, x]), int(b_b[y, x])
            key = (min(la, lb), max(la, lb))
            if key not in adjacency:
                adjacency[key] = {"length": 0, "grads": []}
            adjacency[key]["length"] += 1
            g = 0.5 * (ppl_grad[y, x] + ppl_grad[y + 1, x])
            adjacency[key]["grads"].append(float(g))

        return adjacency

    # ------------------------------------------------------------------
    #  Small region removal
    # ------------------------------------------------------------------

    def _remove_small_regions(
        self,
        labels: np.ndarray,
        F: np.ndarray,
        valid: np.ndarray,
        n_domains_map: dict[int, int] | None = None,
    ) -> np.ndarray:
        """Remove regions < min_region_px by merging into most similar
        neighbour (by feature-image mean distance).

        If *n_domains_map* is provided, domain counts are accumulated
        when a small region is absorbed into its neighbour.
        """
        labels = labels.copy()
        changed = True
        while changed:
            changed = False
            unique, counts = np.unique(labels, return_counts=True)
            for lb, cnt in zip(unique, counts):
                if lb == 0 or cnt >= self.min_region_px:
                    continue

                mask = labels == lb
                # Find neighbours
                dilated = scipy.ndimage.binary_dilation(mask, iterations=1)
                neighbour_mask = dilated & ~mask & (labels > 0) & valid
                if not np.any(neighbour_mask):
                    # No valid neighbour — keep the region as-is rather
                    # than silently deleting valid pixels to background.
                    continue

                # Mean feature vector of this region
                mean_f = np.mean(F[mask], axis=0)

                # Find the most similar neighbour
                neighbour_labels = np.unique(labels[neighbour_mask])
                best_lab = 0
                best_dist = np.inf
                for nlb in neighbour_labels:
                    if nlb == 0 or nlb == lb:
                        continue
                    mean_n = np.mean(F[labels == nlb], axis=0)
                    dist = float(np.sqrt(np.sum((mean_f - mean_n) ** 2)))
                    if dist < best_dist:
                        best_dist = dist
                        best_lab = nlb

                if best_lab > 0:
                    labels[mask] = best_lab
                    # Propagate domain count
                    if n_domains_map is not None:
                        absorbed = n_domains_map.pop(int(lb), 1)
                        n_domains_map[int(best_lab)] = (
                            n_domains_map.get(int(best_lab), 1) + absorbed)
                    changed = True

        return labels

    # ------------------------------------------------------------------
    #  Renumber labels
    # ------------------------------------------------------------------

    @staticmethod
    def _renumber(labels: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
        """Renumber labels to 1..n, preserving 0 = background."""
        unique = np.unique(labels)
        unique = unique[unique != 0]
        old_to_new: dict[int, int] = {}
        new_labels = np.zeros_like(labels, dtype=np.int32)
        for i, old in enumerate(sorted(unique), start=1):
            old_to_new[int(old)] = i
            new_labels[labels == old] = i
        return new_labels, old_to_new

    # ------------------------------------------------------------------
    #  §9.4: region type classification
    # ------------------------------------------------------------------

    def _classify_regions(
        self,
        labels: np.ndarray,
        valid: np.ndarray,
        aniso: np.ndarray,
        ext_deg: np.ndarray,
        r2_4: np.ndarray,
        ppl_max_L: np.ndarray,
        ppl_tex: np.ndarray,
        opaque_px: np.ndarray,
        W_level: float,
        n_domains_map: dict[int, int],
        old_to_new: dict[int, int],
        site_id: str,
        xpl_max_L: np.ndarray,
    ) -> list[dict]:
        """Classify each region into hole / opaque / aggregate / grain."""
        H, W = labels.shape
        xpl_dark = self.cfg.get("maps", {}).get("xpl_dark_rel", 0.08) * W_level

        # Site-level 10th percentile of ppl_tex for hole detection
        valid_tex = ppl_tex[valid]
        if len(valid_tex) > 0:
            tex_p10 = float(np.percentile(valid_tex, 10))
        else:
            tex_p10 = 0.0

        # Circular variance of 4·ext (precomputed per 5×5 neighbourhood)
        ext_4_rad = np.radians(4.0 * ext_deg)
        # Local circular variance via cos/sin means over 5×5
        cos_4ext = np.cos(ext_4_rad)
        sin_4ext = np.sin(ext_4_rad)
        cos_mean = scipy.ndimage.uniform_filter(
            np.where(valid, cos_4ext, 0.0).astype(np.float64), size=5)
        sin_mean = scipy.ndimage.uniform_filter(
            np.where(valid, sin_4ext, 0.0).astype(np.float64), size=5)
        weight = scipy.ndimage.uniform_filter(
            valid.astype(np.float64), size=5)
        # Normalised means
        cos_mean = np.where(weight > 0.01, cos_mean / weight, 0.0)
        sin_mean = np.where(weight > 0.01, sin_mean / weight, 0.0)
        R_local = np.sqrt(cos_mean ** 2 + sin_mean ** 2)
        circvar_local = 1.0 - R_local   # 0 = uniform orientation, 1 = random

        # Valid mask boundary for partial detection
        # Erode valid by 1 px; pixels in valid but not in eroded = edge
        valid_eroded = scipy.ndimage.binary_erosion(valid, iterations=1)
        valid_edge = valid & ~valid_eroded

        unique = np.unique(labels)
        unique = unique[unique != 0]

        region_list: list[dict] = []

        for lb in sorted(unique):
            mask = labels == lb
            area = int(mask.sum())

            # Bounding box
            ys, xs = np.where(mask)
            x0, y0 = int(xs.min()), int(ys.min())
            bw, bh = int(xs.max()) - x0 + 1, int(ys.max()) - y0 + 1
            cx = float(xs.mean())
            cy = float(ys.mean())

            # Region type (§9.4)
            # --- hole ---
            med_ppl_max = float(np.median(ppl_max_L[mask]))
            med_xpl_max = float(np.median(xpl_max_L[mask]))
            med_tex = float(np.median(ppl_tex[mask]))
            # hole/epoxy: bright in PPL, never bright in XPL, featureless
            is_hole = (med_ppl_max > self.hole_rel_L * W_level
                       and med_xpl_max < xpl_dark
                       and med_tex <= tex_p10)

            # --- opaque ---
            opaque_frac = float(opaque_px[mask].sum()) / max(area, 1)
            is_opaque = opaque_frac > 0.8

            # --- aggregate ---
            med_circvar = float(np.median(circvar_local[mask]))
            med_r2_4 = float(np.median(r2_4[mask]))
            # Isotropic grains (garnet, glass) also have random "extinction"
            # angles and low r2_4 — require the region to be anisotropic.
            is_aggregate = (area >= self.min_sample_px
                            and med_xpl_max >= xpl_dark
                            and med_circvar > self.agg_circvar
                            and med_r2_4 < self.agg_r2_4)

            if is_hole:
                rtype = "hole"
            elif is_opaque:
                rtype = "opaque"
            elif is_aggregate:
                rtype = "aggregate"
            else:
                rtype = "grain"

            # Flags
            partial = bool(np.any(mask & valid_edge))
            small = area < self.min_sample_px

            # n_domains from twin merge tracking
            # Map old label → new label to look up domains
            nd = 1
            for old_lb, new_lb in old_to_new.items():
                if new_lb == lb:
                    nd = max(nd, n_domains_map.get(old_lb, 1))

            region_list.append({
                "region_id": f"{site_id}_g{lb:03d}",
                "label_value": int(lb),
                "type": rtype,
                "area_px": area,
                "bbox_xywh": [x0, y0, bw, bh],
                "centroid_xy": [round(cx, 1), round(cy, 1)],
                "partial": partial,
                "small": small,
                "n_domains": nd,
                "ppl_max_rel": round(med_ppl_max / max(W_level, 1.0), 3),
                "xpl_max_rel": round(med_xpl_max / max(W_level, 1.0), 3),
            })

        return region_list


# ──────────────────────────────────────────────────────────────────────────
#  QC overlay
# ──────────────────────────────────────────────────────────────────────────

def save_segment_qc_png(
    ppl_0: np.ndarray,
    xpl_maxproj: np.ndarray,
    label_img: np.ndarray,
    region_list: list[dict],
    site_id: str,
    save_path: str | Path,
) -> None:
    """§9.7 QC: draw region boundaries on PPL 0° and XPL max projection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Create boundary mask using non-wrapping neighbour comparisons
    boundaries = np.zeros(label_img.shape, dtype=bool)
    # Right neighbour
    boundaries[:, :-1] |= ((label_img[:, :-1] != label_img[:, 1:])
                           & (label_img[:, :-1] > 0))
    boundaries[:, 1:]  |= ((label_img[:, 1:] != label_img[:, :-1])
                           & (label_img[:, 1:] > 0))
    # Bottom neighbour
    boundaries[:-1, :] |= ((label_img[:-1, :] != label_img[1:, :])
                           & (label_img[:-1, :] > 0))
    boundaries[1:, :]  |= ((label_img[1:, :] != label_img[:-1, :])
                           & (label_img[1:, :] > 0))

    # Colour boundaries by type
    type_colors = {
        "grain": [1.0, 1.0, 0.0],       # yellow
        "opaque": [1.0, 0.0, 0.0],       # red
        "aggregate": [0.0, 1.0, 1.0],    # cyan
        "hole": [0.5, 0.5, 0.5],         # grey
    }
    # Build label → type lookup
    label_type: dict[int, str] = {}
    for r in region_list:
        label_type[r["label_value"]] = r["type"]

    # Overlay function
    def _overlay(ax, img, title):
        ax.imshow(img)
        # Create coloured boundary overlay
        overlay = np.zeros((*label_img.shape, 4), dtype=np.float32)
        by, bx = np.where(boundaries)
        for y, x in zip(by, bx):
            lv = int(label_img[y, x])
            rt = label_type.get(lv, "grain")
            c = type_colors.get(rt, [1, 1, 0])
            overlay[y, x] = [c[0], c[1], c[2], 0.8]
        ax.imshow(overlay)
        ax.set_title(title)
        ax.axis("off")

    _overlay(axes[0], ppl_0, f"{site_id} — PPL 0°")
    _overlay(axes[1], xpl_maxproj, f"{site_id} — XPL max proj")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(save_path), dpi=120)
    plt.close(fig)
