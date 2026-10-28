"""Phase 0 — Clean and audit the download (§6).

Module: src/rotapetro/clean.py
Inputs:  data/raw/ukvm_dataset/
Outputs: data/interim/sites.csv
         data/interim/specimens_clean.csv
         data/interim/qc/phase0_report.md
         data/interim/qc/phase0_brightness/<site_id>.csv
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from src.rotapetro.io import resolve_pol_dir, resolve_frame_path

logger = logging.getLogger(__name__)

# ── §6.4: monomineralic rock names (exact match after .lower().strip()) ──
_MONO_ROCK_NAMES = frozenset([
    "quartzite", "marble", "dolostone", "dunite",
    "anorthosite", "chert", "pure limestone",
])
# Word-boundary pattern used on the rock-name part of the title
# ("Quartzite - Cape Wrath" -> "quartzite") and on the category.
_MONO_RE = re.compile(r"\b(" + "|".join(sorted(_MONO_ROCK_NAMES)) + r")\b")
_VIEWER_BASE = "https://virtualmicroscope.org/sites/default/files/html5Assets"

# ── §6.3: rock-family mapping from title / category keywords ──
# Ordered so more-specific patterns are tried first.
_ROCK_FAMILY_RULES: list[tuple[list[str], str]] = [
    (["quartzite"],                                      "quartzite"),
    (["marble"],                                         "marble"),
    (["dolostone", "dolomite limestone"],                 "dolostone"),
    (["slate"],                                          "slate"),
    (["phyllite"],                                       "phyllite"),
    (["schist"],                                         "schist"),
    (["gneiss", "migmatite"],                            "gneiss"),
    (["eclogite"],                                       "eclogite"),
    (["amphibolite"],                                    "amphibolite"),
    (["serpentinite"],                                   "serpentinite"),
    (["hornfels"],                                       "hornfels"),
    (["skarn", "calc-silicate"],                         "skarn"),
    (["mylonite", "cataclasite"],                        "mylonite"),
    (["granulite", "charnockite"],                       "granulite"),
    (["granite", "granodiorite", "tonalite", "pegmatite"], "granitoid"),
    (["syenite", "monzonite"],                           "syenite"),
    (["diorite"],                                        "diorite"),
    (["gabbro", "norite", "troctolite"],                 "gabbro"),
    (["peridotite", "dunite", "lherzolite", "harzburgite"], "peridotite"),
    (["basalt", "dolerite", "diabase"],                  "basalt"),
    (["andesite", "dacite"],                             "andesite"),
    (["rhyolite", "obsidian", "pumice", "ignimbrite"],   "rhyolite"),
    (["trachyte", "phonolite"],                          "trachyte"),
    (["limestone", "chalk", "oolite", "oolitic"],       "limestone"),
    (["tuff", "agglomerate", "ignimbrite"],             "pyroclastic"),
    (["ironstone"],                                     "ironstone"),
    (["sandstone", "arenite", "greywacke", "arkose"],    "sandstone"),
    (["mudstone", "shale", "siltstone", "claystone"],    "mudstone"),
    (["conglomerate", "breccia"],                        "conglomerate"),
    (["chert", "flint"],                                 "chert"),
    (["coal"],                                           "coal"),
    (["meteorite"],                                      "meteorite"),
]


class DataCleaner:
    """Phase 0 (§6): clean and audit the UKVM download.

    Parameters
    ----------
    cfg : dict
        Loaded from ``configs/default.yaml``.
    """

    def __init__(self, cfg: dict, config_path: str | Path | None = None) -> None:
        self.cfg = cfg
        self.config_path = Path(config_path) if config_path else None
        raw_root = Path(cfg["paths"]["raw"])
        # Spec §22.A sets paths.raw = data/raw/ukvm_dataset.  Accept both that
        # and the parent folder (data/raw) so either config works.
        if (raw_root / "specimens").is_dir():
            self.raw_dir = raw_root
        else:
            self.raw_dir = raw_root / "ukvm_dataset"
        if not (self.raw_dir / "specimens").is_dir():
            raise FileNotFoundError(
                f"No specimens/ folder under {raw_root} or {raw_root / 'ukvm_dataset'}"
                " — check paths.raw in the config")
        self.interim_dir = Path(cfg["paths"]["interim"])
        self.qc_dir = self.interim_dir / "qc"
        self.knowledge_dir = Path(cfg["paths"].get("knowledge", "knowledge"))

        # Acquisition constants from config
        acq = cfg.get("acquisition", {})
        self.n_raw_frames = acq.get("n_raw_frames", 72)
        self.frame_size = acq.get("frame_size", 490)
        self.valid_radius_px = acq.get("valid_radius_px", 232)
        self.exposure_jump_pct = 0.15  # §6.2: 15 %
        calib = cfg.get("calibration", {})
        self.max_calib_specimens = int(calib.get("max_specimens", 10))
        self.request_delay_s = float(calib.get("request_delay_s", 0.5))
        self._data_js_cache: dict[str, dict | None] = {}
        self.calibration: dict[str, Any] = {}

        # Will be populated during run
        self.sites_df: pd.DataFrame = pd.DataFrame()
        self.specimens_df: pd.DataFrame = pd.DataFrame()
        self._alias_map: dict[str, str] | None = None

    # ──────────────────────────────────────────────────────────────────────
    #  Public API
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Execute the full Phase-0 pipeline (§6.1 – §6.7)."""
        logger.info("Phase 0: starting clean and audit")
        self._load_aliases()
        self.deduplicate_sites()        # §6.1
        self.check_completeness()       # §6.2
        self.rebuild_specimen_table()   # §6.3
        self.flag_monomineralic()       # §6.4
        self.extract_rotation_titles()  # §6.5
        self.calibrate_pixel_size()     # §6.6
        self.write_report()             # §6.7
        logger.info("Phase 0 complete.  Review data/interim/qc/phase0_report.md "
                     "before continuing.")

    # ──────────────────────────────────────────────────────────────────────
    #  §6.1  Remove duplicate rotation sites
    # ──────────────────────────────────────────────────────────────────────

    def deduplicate_sites(self) -> None:
        """For every specimen, fingerprint each rot/r* and keep unique ones.

        Site IDs follow the spec format: ``<code>_r<k>`` where k is
        renumbered 1..n per specimen after duplicate removal.

        Hash comparison is per-specimen (§6.1: "For every specimen, …").
        Raw folders are sorted deterministically by natural order.
        Sites with missing hash files are recorded as unusable but not dropped.
        """
        logger.info("§6.1  Deduplicating rotation sites")
        rows: list[dict[str, Any]] = []
        specimens_dir = self.raw_dir / "specimens"
        if not specimens_dir.is_dir():
            logger.warning("specimens/ directory not found at %s", specimens_dir)
            self.sites_df = pd.DataFrame(columns=[
                "site_id", "code", "raw_folder", "dup_of",
                "usable", "exposure_jump", "title", "title_minerals",
            ])
            return

        for spec_dir in sorted(specimens_dir.iterdir()):
            if not spec_dir.is_dir():
                continue
            code = spec_dir.name
            rot_dir = spec_dir / "rot"
            if not rot_dir.is_dir():
                continue

            # Sort raw folders deterministically (r1, r2, …)
            raw_folders = sorted(
                [d for d in rot_dir.iterdir() if d.is_dir()],
                key=lambda p: _natural_sort_key(p.name),
            )

            # Per-specimen hash → first site_id  (spec says "for every specimen")
            seen_hashes: dict[str, str] = {}
            kept_counter = 0

            for raw_folder in raw_folders:
                h = self._site_hash(raw_folder)

                if h is None:
                    # Cannot compute fingerprint — record but mark unusable
                    logger.warning("  %s: unhashable rotation folder (cannot find ppl/000, xpl/000, or ppl/037): %s",
                                   code, raw_folder)
                    rows.append({
                        "site_id": None,
                        "code": code,
                        "raw_folder": raw_folder.as_posix(),
                        "dup_of": None,
                        "usable": False,
                        "exposure_jump": False,
                        "title": "",
                        "title_minerals": "",
                        "_hash_ok": False,
                    })
                    continue

                if h in seen_hashes:
                    # Duplicate of an earlier site within same specimen
                    rows.append({
                        "site_id": None,
                        "code": code,
                        "raw_folder": raw_folder.as_posix(),
                        "dup_of": seen_hashes[h],
                        "usable": False,
                        "exposure_jump": False,
                        "title": "",
                        "title_minerals": "",
                        "_hash_ok": True,
                    })
                else:
                    # Unique — renumber per-specimen starting at 1
                    kept_counter += 1
                    sid = f"{code}_r{kept_counter}"
                    seen_hashes[h] = sid
                    rows.append({
                        "site_id": sid,
                        "code": code,
                        "raw_folder": raw_folder.as_posix(),
                        "dup_of": None,
                        "usable": True,      # tentative; §6.2 may flip
                        "exposure_jump": False,
                        "title": "",
                        "title_minerals": "",
                        "_hash_ok": True,
                    })

        self.sites_df = pd.DataFrame(rows)
        self._save_sites_csv()
        n_dup = int(self.sites_df["dup_of"].notna().sum())
        n_uni = int(self.sites_df["site_id"].notna().sum())
        n_unhashable = int((~self.sites_df["_hash_ok"].astype(bool)).sum()) \
            if "_hash_ok" in self.sites_df.columns else 0
        logger.info("  %d raw folders → %d unique sites, %d duplicates, "
                     "%d unhashable", len(rows), n_uni, n_dup, n_unhashable)

    # ──────────────────────────────────────────────────────────────────────
    #  §6.2  Completeness checks
    # ──────────────────────────────────────────────────────────────────────

    def check_completeness(self) -> None:
        """Verify frames exist, decode, are 490×490; record brightness.

        ``exposure_jump`` is set as a FLAG (does NOT make usable=False).
        Per-frame brightness is saved to QC CSVs.
        """
        logger.info("§6.2  Completeness checks")
        cx = self.frame_size // 2
        cy = self.frame_size // 2
        circle_mask = _circle_mask(self.frame_size, self.frame_size,
                                   radius=235, cx=cx, cy=cy)

        brightness_dir = self.qc_dir / "phase0_brightness"
        brightness_dir.mkdir(parents=True, exist_ok=True)

        for idx, row in self.sites_df.iterrows():
            if not row["usable"]:
                continue

            site_dir = Path(row["raw_folder"])
            site_id = row["site_id"]
            usable = True
            exposure_jump = False
            brightness_records: list[dict] = []

            for pol in ("ppl", "xpl"):
                pol_dir = resolve_pol_dir(site_dir, pol)
                if pol_dir is None:
                    logger.warning("  %s: missing %s/ directory in %s",
                                   site_id, pol, site_dir)
                    usable = False
                    break

                prev_mean: float | None = None
                for i in range(self.n_raw_frames):
                    fpath = resolve_frame_path(pol_dir, i)

                    # File exists?
                    if fpath is None or not fpath.is_file():
                        logger.warning("  %s: missing %s/%03d.jpg in %s",
                                       site_id, pol, i, pol_dir)
                        usable = False
                        break

                    # Decodes?
                    img = cv2.imread(str(fpath), cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        logger.warning("  %s: cannot decode %s", site_id, fpath)
                        usable = False
                        break

                    # Correct shape?
                    if img.shape != (self.frame_size, self.frame_size):
                        logger.warning("  %s: wrong shape %s in %s",
                                       site_id, img.shape, fpath)
                        usable = False
                        break

                    # Record brightness inside the circle
                    mean_val = float(np.mean(img[circle_mask]))
                    brightness_records.append({
                        "frame": i, "pol": pol, "mean_brightness": mean_val,
                    })

                    # §6.2: exposure jump detection (PPL only)
                    if pol == "ppl" and prev_mean is not None:
                        change = abs(mean_val - prev_mean) / max(prev_mean, 1e-6)
                        if change > self.exposure_jump_pct:
                            exposure_jump = True
                            logger.warning(
                                "  %s: exposure jump %.1f%% at PPL "
                                "frame %03d→%03d", site_id,
                                change * 100, i - 1, i)
                    if pol == "ppl":
                        prev_mean = mean_val

                if not usable:
                    break   # no need to check XPL if PPL already failed

            # exposure_jump is a FLAG — does NOT set usable=False
            self.sites_df.at[idx, "usable"] = usable
            self.sites_df.at[idx, "exposure_jump"] = exposure_jump

            # Persist the brightness audit
            if brightness_records:
                bdf = pd.DataFrame(brightness_records)
                bdf.to_csv(brightness_dir / f"{site_id}.csv", index=False)

        self._save_sites_csv()
        n_usable = int(self.sites_df["usable"].sum())
        n_jump = int(self.sites_df["exposure_jump"].sum())
        logger.info("  %d usable sites, %d with exposure-jump flag",
                     n_usable, n_jump)

    # ──────────────────────────────────────────────────────────────────────
    #  §6.3  Rebuild the specimen table
    # ──────────────────────────────────────────────────────────────────────

    def rebuild_specimen_table(self) -> None:
        """Create specimens_clean.csv from all meta.json files.

        Uses correct meta.json keys (``rock_forming_mineral``,
        ``accessory_minerals``).  Falls back to ``title + type`` when
        ``category`` is missing.  Derives ``rock_family`` from keywords.
        Counts ``n_sites_unique`` as non-duplicate sites (regardless of
        usable flag).
        """
        logger.info("§6.3  Rebuilding specimen table")
        rows: list[dict[str, Any]] = []
        specimens_dir = self.raw_dir / "specimens"
        if not specimens_dir.is_dir():
            self.specimens_df = pd.DataFrame()
            return

        for spec_dir in sorted(specimens_dir.iterdir()):
            if not spec_dir.is_dir():
                continue
            meta_path = spec_dir / "meta.json"
            if not meta_path.is_file():
                continue

            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)

            code = spec_dir.name

            # §5 bug #3: correct keys are rock_forming_mineral / accessory_minerals
            rock_forming_raw: list[str] = meta.get(
                "rock_forming_mineral",
                meta.get("rock_forming", []))
            accessory_raw: list[str] = meta.get(
                "accessory_minerals",
                meta.get("accessory_mineral", []))
            if rock_forming_raw is None:
                rock_forming_raw = []
            if accessory_raw is None:
                accessory_raw = []
            if isinstance(rock_forming_raw, str):
                rock_forming_raw = [m.strip() for m in rock_forming_raw.split(";")
                                    if m.strip()]
            if isinstance(accessory_raw, str):
                accessory_raw = [m.strip() for m in accessory_raw.split(";")
                                 if m.strip()]

            # n_sites_unique = unique (non-dup) sites regardless of usable
            n_unique = int(
                self.sites_df.loc[
                    (self.sites_df["code"] == code)
                    & (self.sites_df["site_id"].notna()),
                    "site_id",
                ].nunique()
            )

            # meta.json stores most fields as LISTS (e.g. "type": ["igneous"])
            title = _as_text(meta.get("title"))
            type_ = _as_text(meta.get("type"))
            category = _as_text(meta.get("category"))   # may be empty (§5 #4)

            # §5 bug #4: missing category → use title + type for the family
            rock_family = _map_rock_family(title, f"{category} {type_}")

            rows.append({
                "code": code,
                "node": meta.get("node", ""),
                "title": title,
                "type": type_,
                "category": category,
                "rock_forming": ";".join(rock_forming_raw),
                "accessory": ";".join(accessory_raw),
                "n_sites_unique": n_unique,
                "locality": _as_text(meta.get("locality")),
                "lat": meta.get("lat", ""),
                "lon": meta.get("lon", ""),
                "owner": _as_text(meta.get("owner")),
                "is_monomineralic": False,   # set in §6.4
                "rock_family": rock_family,
            })

        self.specimens_df = pd.DataFrame(rows)
        logger.info("  %d specimens", len(self.specimens_df))

    # ──────────────────────────────────────────────────────────────────────
    #  §6.4  Flag monomineralic specimens
    # ──────────────────────────────────────────────────────────────────────

    def flag_monomineralic(self) -> None:
        """Set is_monomineralic using alias-resolved mineral list or rock names.

        Condition 1: rock_forming contains exactly one *canonical* mineral
        after alias mapping (§11.4).
        Condition 2: title or category exactly equals one of the mono rock
        names (not substring match).
        """
        logger.info("§6.4  Flagging monomineralic specimens")

        for idx, row in self.specimens_df.iterrows():
            is_mono = False

            # Condition 1: exactly one canonical mineral in rock_forming
            rf_raw = str(row["rock_forming"]).split(";")
            canonical_minerals = set()
            for name in rf_raw:
                name = name.strip().lower()
                if not name:
                    continue
                canonical = self._resolve_alias(name)
                canonical_minerals.add(canonical)

            # Only a single MINERAL counts (a single group such as "mica" or
            # "feldspar" is not a one-mineral rock).
            if (len(canonical_minerals) == 1
                    and self._alias_level(next(iter(canonical_minerals)))
                    != "group"):
                is_mono = True

            # Condition 2: rock name in the title's rock-name part or category
            # (word match: "Quartzite - Cape Wrath" -> quartzite)
            if not is_mono:
                rock_part = str(row["title"]).split(" - ")[0].lower()
                cat_lower = str(row["category"]).lower()
                if _MONO_RE.search(rock_part) or _MONO_RE.search(cat_lower):
                    is_mono = True

            self.specimens_df.at[idx, "is_monomineralic"] = is_mono

        self.specimens_df.to_csv(
            self.interim_dir / "specimens_clean.csv", index=False)

        n_mono = int(self.specimens_df["is_monomineralic"].sum())
        logger.info("  %d monomineralic specimens", n_mono)

    # ──────────────────────────────────────────────────────────────────────
    #  §6.5  Rotation-site titles
    # ──────────────────────────────────────────────────────────────────────

    def extract_rotation_titles(self) -> None:
        """Parse OU viewer ``data/data.js`` to extract rotation-site titles.

        Verified against actual UKVM HTML structure (tested on 11agt25,
        11agt24).  The viewer loads ``data/data.js`` which defines a JS
        object::

            data={"h5aa":{...}, "images": [
                {"label":"PPL", "pixelsPerMM": 545.0,
                 "rotations": [
                     {"x":..., "y":...,
                      "pplData":"<code>_r1_ppl/<code>_r1_ppl0",
                      "xplData":"<code>_r1_xpl/<code>_r1_xpl0",
                      "title":"Rotation 1"},
                     ...
                 ]}, ...
            ]}

        We parse the PPL image entry's ``rotations`` array.  ``pplData``
        contains the raw folder name (e.g. ``11agt25_r1_ppl/...``), which
        maps ``r1`` → the raw ``rot/r1`` folder.

        Stores both ``title`` (the raw title string) and ``title_minerals``
        (alias-resolved mineral names found in the title) in sites_df.
        """
        logger.info("§6.5  Extracting rotation-site titles")

        codes = self.sites_df.loc[
            self.sites_df["site_id"].notna(), "code"
        ].unique()

        for code in sorted(codes):
            try:
                data_obj = self._fetch_data_js(code)
            except Exception as exc:
                logger.warning("  %s: could not fetch data.js: %s", code, exc)
                continue

            if data_obj is None:
                continue

            # Extract rotations from the PPL image entry
            rotations = self._extract_rotations(data_obj)
            if not rotations:
                logger.info("  %s: no rotations found in data.js", code)
                continue

            # Deduplicate by pplData prefix (§6.5 step 3)
            seen_ppl: dict[str, dict] = {}
            for rot in rotations:
                ppl_key = rot.get("pplData", "")
                if ppl_key and ppl_key not in seen_ppl:
                    seen_ppl[ppl_key] = rot

            # Map pplData → raw folder → site_id (§6.5 step 4)
            code_mask = (
                (self.sites_df["code"] == code)
                & (self.sites_df["site_id"].notna())
            )

            # The downloader recorded, per raw folder, the URL of frame 1
            # (meta.json rotation_fields[].ppl_source).  Matching that URL to
            # pplData is exact; folder-name matching is only a fallback.
            point_to_ppl = self._point_to_ppldata(code)

            for idx in self.sites_df[code_mask].index:
                raw_folder = Path(self.sites_df.at[idx, "raw_folder"])
                raw_name = raw_folder.name  # e.g. "r1"

                matched_rot = None
                src = point_to_ppl.get(raw_name)
                if src is not None:
                    for ppl_data, rot in seen_ppl.items():
                        if src.startswith(ppl_data):
                            matched_rot = rot
                            break
                for ppl_data, rot in ([] if matched_rot else seen_ppl.items()):
                    # pplData is like "11agt25_r1_ppl/11agt25_r1_ppl0"
                    # Extract the rK part
                    ppl_rk = self._extract_rk_from_ppldata(ppl_data, code)
                    if ppl_rk and ppl_rk == raw_name:
                        matched_rot = rot
                        break

                if matched_rot is None:
                    continue

                title = matched_rot.get("title", "")
                self.sites_df.at[idx, "title"] = title

                # §6.5 step 5: extract minerals from title
                title_minerals = self._extract_minerals_from_title(title)
                if title_minerals:
                    self.sites_df.at[idx, "title_minerals"] = \
                        ";".join(title_minerals)

            logger.info("  %s: matched %d rotations from data.js",
                        code, len(seen_ppl))

        self._save_sites_csv()

    # ──────────────────────────────────────────────────────────────────────
    #  §6.6  Pixel-size calibration
    # ──────────────────────────────────────────────────────────────────────

    def calibrate_pixel_size(self) -> None:
        """§6.6: measure ``um_per_px_rot`` (rotation-frame pixel size).

        For each specimen (up to ``calibration.max_specimens``):

        1. ``pixelsPerMM`` of the FULL-resolution whole slide comes from the
           viewer's ``data/data.js``  →  ``um_per_px_full = 1000/pixelsPerMM``.
        2. The saved ``wsi/ppl.png`` is a DOWNSCALED zoom level.  The factor
           is read from ``wsi/ppl_info.json`` (``downscale_from_full`` or
           ``full_width/saved_width``) — it is 2 only for some slides.
           ``um_per_px_wsi = um_per_px_full * downscale``.
        3. SIFT-register rotation frame 000 onto the saved WSI → scale ``s``
           (WSI pixels per frame pixel).  ``um_per_px_rot = um_per_px_wsi * s``.

        Results go to ``data/interim/calibration.json`` and to the specimen
        table; the median is written into the config file that was passed
        on the command line (only the ``um_per_px_rot:`` line is edited, so
        comments are kept).
        """
        logger.info("§6.6  Pixel-size calibration")
        per_spec: dict[str, dict[str, Any]] = {}
        usable = self.sites_df[self.sites_df["usable"] == True]

        for code in sorted(usable["code"].unique()):
            if len(per_spec) >= self.max_calib_specimens:
                break
            spec_dir = self.raw_dir / "specimens" / code
            data_obj = self._fetch_data_js(code)
            if data_obj is None:
                continue
            ppmm = self._extract_pixels_per_mm(data_obj)
            if not ppmm or ppmm <= 0:
                continue
            wsi_png, info = self._find_wsi_ppl(spec_dir / "wsi")
            if wsi_png is None or info is None:
                continue
            full_w = info.get("full_width") or data_obj.get("width")
            saved_w = info.get("saved_width")
            downscale = info.get("downscale_from_full") or (
                (full_w / saved_w) if (full_w and saved_w) else None)
            if not downscale:
                continue
            um_full = 1000.0 / ppmm
            um_wsi = um_full * float(downscale)

            site_vals = []
            for _, srow in usable[usable["code"] == code].iterrows():
                ppl_dir = resolve_pol_dir(srow["raw_folder"], "ppl")
                frame = resolve_frame_path(ppl_dir, 0) if ppl_dir else None
                if frame is None or not frame.is_file():
                    continue
                s_ = self._register_frame_to_wsi(frame, wsi_png)
                if s_ is not None:
                    site_vals.append(um_wsi * s_)
            if not site_vals:
                logger.warning("  %s: registration failed for all sites", code)
                continue
            val = float(np.median(site_vals))
            slide_w_mm = (full_w / ppmm) if full_w else float("nan")
            per_spec[code] = {
                "pixelsPerMM": ppmm, "um_per_px_full": um_full,
                "downscale": float(downscale), "um_per_px_wsi": um_wsi,
                "um_per_px_rot": val, "n_sites": len(site_vals),
                "slide_width_mm": slide_w_mm,
                "fov_mm": val * self.frame_size / 1000.0,
            }
            logger.info("  %s: pixelsPerMM=%.1f  slide=%.1f mm  WSI %.3f um/px  "
                        "→ um_per_px_rot=%.3f (FOV %.2f mm, %d sites)",
                        code, ppmm, slide_w_mm, um_wsi, val,
                        per_spec[code]["fov_mm"], len(site_vals))
            if not (15.0 <= slide_w_mm <= 80.0):
                logger.warning("  %s: slide width %.1f mm looks wrong — check "
                               "pixelsPerMM", code, slide_w_mm)

        self.calibration = {"per_specimen": per_spec}
        if per_spec:
            vals = np.array([v["um_per_px_rot"] for v in per_spec.values()])
            med = float(np.median(vals))
            cv = float(vals.std() / vals.mean()) if len(vals) > 1 else 0.0
            self.calibration.update({"median_um_per_px_rot": med,
                                     "cv": cv, "n_specimens": len(vals)})
            logger.info("  Median um_per_px_rot = %.3f (n=%d, CV=%.1f%%)",
                        med, len(vals), cv * 100)
            if cv > 0.05:
                logger.warning("  Pixel size varies >5%% between specimens — "
                               "use the per-specimen values, not one number")
            self._update_config_pixel_size(med)
        else:
            current = self.cfg.get("acquisition", {}).get("um_per_px_rot", 4.0)
            logger.warning("  Could not measure — keeping um_per_px_rot=%.2f "
                           "(UNVERIFIED)", current)

        # per-specimen value into the specimen table
        if not self.specimens_df.empty:
            self.specimens_df["um_per_px_rot"] = self.specimens_df["code"].map(
                lambda c: per_spec.get(c, {}).get("um_per_px_rot", np.nan))
            self.specimens_df.to_csv(
                self.interim_dir / "specimens_clean.csv", index=False)
        (self.interim_dir / "calibration.json").write_text(
            json.dumps(self.calibration, indent=2), encoding="utf-8")

    # ──────────────────────────────────────────────────────────────────────
    #  §6.7  Phase-0 report
    # ──────────────────────────────────────────────────────────────────────

    def write_report(self) -> None:
        """Write the full Phase-0 audit report to
        ``data/interim/qc/phase0_report.md``."""
        logger.info("§6.7  Writing Phase-0 report")
        self.qc_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.qc_dir / "phase0_report.md"

        n_specimens = len(self.specimens_df)
        n_raw = len(self.sites_df)
        n_unique = int(self.sites_df["site_id"].notna().sum())
        n_dup = int(self.sites_df["dup_of"].notna().sum())
        n_unhashable = n_raw - n_unique - n_dup
        n_usable = int(self.sites_df["usable"].sum())
        n_exposure_jump = int(self.sites_df["exposure_jump"].sum())
        n_mono = int(self.specimens_df["is_monomineralic"].sum()) \
            if "is_monomineralic" in self.specimens_df.columns else 0

        # Mineral name frequencies from OU lists (§6.7)
        mineral_counter: Counter[str] = Counter()
        for _, row in self.specimens_df.iterrows():
            for col in ("rock_forming", "accessory"):
                val = str(row.get(col, ""))
                for m in val.split(";"):
                    m = m.strip().lower()
                    if m:
                        mineral_counter[m] += 1

        # ALL titles found (including generic), not just mineral-bearing
        has_title = self.sites_df["title"].astype(str).str.strip().str.len() > 0
        titles_df = self.sites_df.loc[
            has_title, ["site_id", "title", "title_minerals"]
        ].copy()

        # Rock families and counts
        family_counts = (
            self.specimens_df["rock_family"]
            .value_counts()
            .to_dict()
        ) if "rock_family" in self.specimens_df.columns else {}

        # ── Build report ──
        lines: list[str] = []
        lines.append("# Phase 0 — Data Cleaning Report\n")

        lines.append("## Dataset overview\n")
        lines.append(f"- Specimens: **{n_specimens}**")
        lines.append(f"- Raw rotation folders: {n_raw}")
        lines.append(f"- Unique sites (after dedup): **{n_unique}**")
        lines.append(f"- Duplicates removed: {n_dup}")
        lines.append(f"- Unclassifiable (missing hash files): {n_unhashable}")
        lines.append(f"- Usable sites: **{n_usable}**")
        lines.append(f"- Sites with exposure-jump flag: {n_exposure_jump}")
        lines.append(f"- Monomineralic specimens: **{n_mono}**\n")

        lines.append("## Rotation-site titles found\n")
        if len(titles_df) > 0:
            lines.append("| site_id | title | title_minerals |")
            lines.append("|---------|-------|----------------|")
            for _, tr in titles_df.iterrows():
                sid = tr["site_id"] if pd.notna(tr["site_id"]) else ""
                ttl = tr["title"] if pd.notna(tr["title"]) else ""
                tm = tr["title_minerals"] if pd.notna(tr["title_minerals"]) else ""
                lines.append(f"| {sid} | {ttl} | {tm} |")
        else:
            lines.append("No titles were found (HTML parsing was skipped "
                         "or no rotations matched).")
        lines.append("")

        lines.append("## Mineral name frequencies (from OU lists)\n")
        lines.append("These are the raw OU mineral names before alias "
                      "mapping.  Use them to populate `knowledge/aliases.csv`.\n")
        lines.append("| mineral | count |")
        lines.append("|---------|-------|")
        for name, count in mineral_counter.most_common():
            lines.append(f"| {name} | {count} |")
        if not mineral_counter:
            lines.append("| _(none)_ | — |")
        lines.append("")

        lines.append("## Rock families and counts\n")
        lines.append("| rock_family | count |")
        lines.append("|-------------|-------|")
        for fam, count in sorted(family_counts.items(), key=lambda x: -x[1]):
            lines.append(f"| {fam} | {count} |")
        if not family_counts:
            lines.append("| _(none mapped)_ | — |")
        lines.append("")

        lines.append("## Pixel-size calibration (§6.6)\n")
        cal = self.calibration or {}
        if cal.get("per_specimen"):
            lines.append(f"- Median um_per_px_rot: **{cal['median_um_per_px_rot']:.3f}** "
                         f"(n={cal['n_specimens']}, CV={cal['cv']*100:.1f}%)\n")
            lines.append("| code | pixelsPerMM | slide width mm | WSI um/px | rot um/px | FOV mm | sites |")
            lines.append("|---|---|---|---|---|---|---|")
            for c, v in cal["per_specimen"].items():
                lines.append(f"| {c} | {v['pixelsPerMM']:.1f} | {v['slide_width_mm']:.1f} | "
                             f"{v['um_per_px_wsi']:.3f} | {v['um_per_px_rot']:.3f} | "
                             f"{v['fov_mm']:.2f} | {v['n_sites']} |")
        else:
            lines.append("**Not measured** — um_per_px_rot is unverified.")
        lines.append("")

        lines.append("## Sites needing attention\n")
        bad = self.sites_df[(self.sites_df["site_id"].notna())
                            & (~self.sites_df["usable"].astype(bool))]
        jump = self.sites_df[self.sites_df["exposure_jump"].astype(bool)]
        lines.append(f"- Unusable unique sites: {', '.join(bad['site_id']) or 'none'}")
        lines.append(f"- Exposure-jump sites: {', '.join(jump['site_id'].dropna()) or 'none'}")
        if "is_monomineralic" in self.specimens_df.columns:
            mono = self.specimens_df[self.specimens_df["is_monomineralic"].astype(bool)]
            lines.append(f"- Monomineralic specimens: "
                         f"{', '.join(mono['code'] + ' (' + mono['title'] + ')') or 'none'}")
        if "rock_family" in self.specimens_df.columns:
            oth = self.specimens_df[self.specimens_df["rock_family"] == "other"]
            lines.append(f"- Titles mapped to family 'other' (extend the rules): "
                         f"{'; '.join(oth['title']) or 'none'}")
        lines.append("")

        report_text = "\n".join(lines)
        report_path.write_text(report_text, encoding="utf-8")
        logger.info("  Report written to %s", report_path)

    # ──────────────────────────────────────────────────────────────────────
    #  Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _save_sites_csv(self) -> None:
        """Persist sites.csv to data/interim/."""
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        cols = ["site_id", "code", "raw_folder", "dup_of",
                "usable", "exposure_jump", "title", "title_minerals"]
        self.sites_df.to_csv(
            self.interim_dir / "sites.csv",
            columns=[c for c in cols if c in self.sites_df.columns],
            index=False,
        )

    @staticmethod
    def _md5_file(path: Path) -> str | None:
        """Return hex MD5 digest, or None if the file does not exist."""
        if not path.is_file():
            return None
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def _site_hash(self, raw_folder: Path) -> str | None:
        """§6.1: h = md5(ppl/000) + md5(xpl/000) + md5(ppl/037). Case-insensitive."""
        ppl_dir = resolve_pol_dir(raw_folder, "ppl")
        xpl_dir = resolve_pol_dir(raw_folder, "xpl")
        if ppl_dir is None or xpl_dir is None:
            return None
        f_ppl0 = resolve_frame_path(ppl_dir, 0)
        f_xpl0 = resolve_frame_path(xpl_dir, 0)
        f_ppl37 = resolve_frame_path(ppl_dir, 37)
        if f_ppl0 is None or f_xpl0 is None or f_ppl37 is None:
            return None
        h1 = self._md5_file(f_ppl0)
        h2 = self._md5_file(f_xpl0)
        h3 = self._md5_file(f_ppl37)
        if h1 is None or h2 is None or h3 is None:
            return None
        return h1 + h2 + h3

    # ── Alias resolution ──

    def _load_aliases(self) -> None:
        """Load aliases.csv into a dict for name resolution."""
        alias_path = self.knowledge_dir / "aliases.csv"
        self._alias_map = {}
        self._alias_levels: dict[str, str] = {}
        if alias_path.is_file():
            df = pd.read_csv(alias_path)
            for _, row in df.iterrows():
                canon = str(row["canonical"]).strip().lower()
                self._alias_map[str(row["ou_name"]).strip().lower()] = canon
                if "level" in df.columns:
                    self._alias_levels[canon] = str(row["level"]).strip().lower()
        else:
            logger.warning("  %s not found — mineral names are used as-is. "
                           "Create it from the Phase-0 frequency list and "
                           "re-run Phase 0.", alias_path)
        logger.info("  Loaded %d aliases from %s",
                     len(self._alias_map), alias_path)

    def _alias_level(self, canonical: str) -> str:
        """'mineral' or 'group' (unknown names count as 'mineral')."""
        return getattr(self, "_alias_levels", {}).get(canonical, "mineral")

    def _resolve_alias(self, name: str) -> str:
        """Map an OU mineral name to its canonical name via aliases.csv."""
        name = name.strip().lower()
        if self._alias_map and name in self._alias_map:
            return self._alias_map[name]
        return name   # identity if unknown

    # ── §6.5 helpers ──

    def _fetch_data_js(self, code: str) -> dict | None:
        """Fetch and parse ``data/data.js`` for one specimen.

        The file defines a JS variable: ``data={...}``.  We strip the
        ``data=`` prefix and parse the rest as JSON (it is valid JSON in
        all tested specimens).
        """
        if code in self._data_js_cache:
            return self._data_js_cache[code]

        # Cache next to the raw data so Phase 0 is reproducible offline and
        # the site is contacted only once per specimen.
        cache = self.raw_dir / "specimens" / code / "viewer_data.js"
        if cache.is_file():
            text = cache.read_text(encoding="utf-8")
        else:
            try:
                import requests
            except ImportError:
                logger.warning("  requests not installed — skipping data.js fetch")
                return None
            url = f"{_VIEWER_BASE}/{code}/data/data.js"
            try:
                time.sleep(self.request_delay_s)
                resp = requests.get(url, timeout=15, headers={
                    "User-Agent": "IITK-petrography-research (phase0)"})
                resp.raise_for_status()
            except Exception as exc:
                logger.warning("  Could not fetch %s: %s", url, exc)
                self._data_js_cache[code] = None
                return None
            text = resp.text
            cache.write_text(text, encoding="utf-8")

        text = text.strip()
        # Strip everything before the first "{" ("data=", "var data =", ...)
        brace = text.find("{")
        if brace < 0:
            self._data_js_cache[code] = None
            return None
        text = text[brace:].rstrip().rstrip(";")

        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            logger.warning("  %s: could not parse data.js as JSON: %s",
                           code, exc)
            obj = None
        self._data_js_cache[code] = obj
        return obj

    @staticmethod
    def _extract_rotations(data_obj: dict) -> list[dict]:
        """Extract the rotations array from the PPL image entry."""
        images = data_obj.get("images", [])
        for img in images:
            if img.get("label", "").upper() == "PPL":
                return img.get("rotations", [])
        # Fallback: return rotations from first image with rotations
        for img in images:
            rots = img.get("rotations", [])
            if rots:
                return rots
        return []

    @staticmethod
    def _extract_pixels_per_mm(data_obj: dict) -> float | None:
        """Extract ``pixelsPerMM`` from the PPL image entry."""
        images = data_obj.get("images", [])
        for img in images:
            if img.get("label", "").upper() == "PPL":
                val = img.get("pixelsPerMM")
                if val is not None:
                    return float(val)
        return None

    def _point_to_ppldata(self, code: str) -> dict[str, str]:
        """raw folder name (r1, r2 …) → frame-URL path after '/data/'."""
        meta_path = self.raw_dir / "specimens" / code / "meta.json"
        out: dict[str, str] = {}
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return out
        for rf in meta.get("rotation_fields", []) or []:
            src = str(rf.get("ppl_source", ""))
            if "/data/" in src:
                out[str(rf.get("point"))] = src.split("/data/", 1)[1]
        return out

    @staticmethod
    def _extract_rk_from_ppldata(ppl_data: str, code: str) -> str | None:
        """Extract the rK folder name from a pplData string.

        Example: ``"11agt25_r1_ppl/11agt25_r1_ppl0"`` → ``"r1"``
        """
        # Pattern: <code>_r<k>_ppl/...
        pattern = re.compile(re.escape(code) + r"_(r\d+)_ppl")
        m = pattern.search(ppl_data)
        if m:
            return m.group(1)
        return None

    def _extract_minerals_from_title(self, title: str) -> list[str]:
        """Check if a rotation title names a mineral.

        Returns list of canonical mineral names found, or empty list if
        the title is generic (e.g. "Rotation 1").
        """
        if not title or not self._alias_map:
            return []

        title_lower = title.strip().lower()
        # Skip obviously generic titles
        if re.match(r"^rotation\s+\d+$", title_lower):
            return []

        found: list[str] = []
        # Check every known alias/mineral name against the title text
        for ou_name, canonical in self._alias_map.items():
            # word match: "mica" must not match "micaceous"
            if re.search(r"\b" + re.escape(ou_name) + r"s?\b", title_lower):
                if canonical not in found:
                    found.append(canonical)

        return found

    # ── §6.6 helpers ──

    @staticmethod
    def _find_wsi_ppl(wsi_dir: Path) -> tuple[Path | None, dict | None]:
        """Whole-slide PPL image + its *_info.json (case-insensitive)."""
        if not wsi_dir.is_dir():
            return None, None
        png = info = None
        for p in wsi_dir.iterdir():
            n = p.name.lower()
            if n == "ppl.png":
                png = p
            elif n == "ppl_info.json":
                try:
                    info = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    info = None
        return png, info

    @staticmethod
    def _register_frame_to_wsi(frame_path: Path, wsi_path: Path,
                               work_width: int = 3800) -> float | None:
        """Scale (saved-WSI pixels per rotation-frame pixel), or None.

        SIFT + Lowe ratio test on a downscaled copy of the WSI.  ORB with
        cross-check was tested on 11agt25 and failed (1.5–3.5 % inliers);
        SIFT on a ~3800-px-wide WSI gives ~70 % inliers.  The measured scale
        is converted back to full saved-WSI pixels.
        """
        try:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
            wsi = cv2.imread(str(wsi_path), cv2.IMREAD_GRAYSCALE)
            if frame is None or wsi is None:
                return None
            f = min(1.0, work_width / wsi.shape[1])
            wsi_s = cv2.resize(wsi, None, fx=f, fy=f,
                               interpolation=cv2.INTER_AREA) if f < 1 else wsi
            h, w = frame.shape
            yy, xx = np.ogrid[:h, :w]
            fmask = (((xx - w / 2) ** 2 + (yy - h / 2) ** 2)
                     <= (0.46 * w) ** 2).astype(np.uint8) * 255
            sift = cv2.SIFT_create(nfeatures=60000)
            kf, df = sift.detectAndCompute(frame, fmask)
            kw, dw = sift.detectAndCompute(wsi_s, None)
            if df is None or dw is None or len(kf) < 20:
                return None
            knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(df, dw, k=2)
            good = [m for m, n in (p for p in knn if len(p) == 2)
                    if m.distance < 0.8 * n.distance]
            if len(good) < 12:
                logger.warning("  %s: only %d SIFT matches", frame_path, len(good))
                return None
            pf = np.float32([kf[m.queryIdx].pt for m in good])
            pw = np.float32([kw[m.trainIdx].pt for m in good])
            M, inl = cv2.estimateAffinePartial2D(pf, pw, method=cv2.RANSAC,
                                                 ransacReprojThreshold=4.0)
            if M is None or inl is None:
                return None
            n_in = int(inl.sum())
            if n_in < 12 or n_in / len(good) < 0.3:
                logger.warning("  %s: weak registration (%d/%d inliers)",
                               frame_path, n_in, len(good))
                return None
            scale = float(np.hypot(M[0, 0], M[1, 0])) / f
            if not (0.05 <= scale <= 5.0):
                return None
            logger.info("  %s → WSI: scale=%.4f  inliers=%d/%d",
                        Path(frame_path).parent.parent.name, scale, n_in, len(good))
            return scale
        except Exception as exc:
            logger.warning("  Registration failed: %s", exc)
            return None

    def _update_config_pixel_size(self, um_per_px_rot: float) -> None:
        """Edit only the ``um_per_px_rot:`` line of the config that was
        passed on the command line (keeps comments and order)."""
        path = self.config_path or Path("configs/default.yaml")
        if not path.is_file():
            logger.warning("  %s not found — set um_per_px_rot=%.3f by hand",
                           path, um_per_px_rot)
            return
        text = path.read_text(encoding="utf-8")
        new, n = re.subn(r"(^\s*um_per_px_rot:\s*)[^#\n]*",
                         lambda m: f"{m.group(1)}{um_per_px_rot:.3f}  ",
                         text, count=1, flags=re.M)
        if n == 0:
            logger.warning("  um_per_px_rot line not found in %s — set it by "
                           "hand to %.3f", path, um_per_px_rot)
            return
        path.write_text(new, encoding="utf-8")
        logger.info("  Updated %s: um_per_px_rot = %.3f", path, um_per_px_rot)


# ──────────────────────────────────────────────────────────────────────────
#  Module-level helpers
# ──────────────────────────────────────────────────────────────────────────

def _as_text(v: Any) -> str:
    """meta.json stores many fields as lists; join them into one string."""
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return "; ".join(str(x).strip() for x in v if str(x).strip())
    return str(v).strip()


def _circle_mask(h: int, w: int, radius: int, cx: int, cy: int) -> np.ndarray:
    """Boolean mask for a circle."""
    y, x = np.ogrid[:h, :w]
    return ((x - cx) ** 2 + (y - cy) ** 2) <= radius ** 2


def _natural_sort_key(s: str):
    """Sort key that handles embedded numbers (r1 < r2 < r10)."""
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r"(\d+)", s)]


def _map_rock_family(title: str, category: str) -> str:
    """§6.3: derive rock_family from title and category by keyword matching."""
    text = f"{title} {category}".lower()
    for keywords, family in _ROCK_FAMILY_RULES:
        for kw in keywords:
            if kw in text:
                return family
    return "other"
