"""Phase 6 — Automatic labels / Rule engine (§12).

Module: src/rotapetro/rules.py
Inputs:  regions.parquet, specimens_clean.csv, sites.csv, knowledge/*
Outputs: labels.parquet, classes.yaml

The rule engine scores each region against every candidate mineral using
13 observation terms derived from the Phase 4 feature card and the Phase 5
knowledge base.  Four independent label sources are supported:

    S1. Rule engine (training labels)
    S2. Monomineralic specimens (authoritative — overrides S1)
    S3. Rotation titles (evaluation only — stored in separate columns)
    S4. OU mineral list (bag-level MIL signal)
"""
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_EPS = 1e-30   # prevent log(0)

INFORMATIVE = {
    "optic_class", "pleochroism", "ppl_colour", "twinning", "undulose",
    "twinkling", "absorption_axis", "cleavage",
}

V1_STATUS_COLUMNS = [
    "optic_class_status",
    "pleo_status",
    "absorption_status",
    "twin_status",
    "extinction_status",
    "cleavage_status",
]


# ══════════════════════════════════════════════════════════════════════════
#  Rule Engine
# ══════════════════════════════════════════════════════════════════════════

class RuleEngine:
    """Score regions against minerals using 13 optical-property terms (§12.2).

    Parameters
    ----------
    kb : KnowledgeBase
        The loaded knowledge base (Phase 5).
    cfg : dict
        Full project configuration.
    """

    def __init__(self, kb, cfg: dict) -> None:
        from src.rotapetro.knowledge import KnowledgeBase
        self.kb: KnowledgeBase = kb
        self.cfg = cfg
        self.rules_cfg = cfg.get("rules", {})

        # V1 rules settings (§4a)
        v1r = cfg.get("v1", {}).get("rules", {})
        self.v1 = bool(v1r)
        self.disabled_terms = set(v1r.get("disabled_terms", []))
        if "absorption" in self.disabled_terms:
            self.disabled_terms.add("absorption_axis")
        elif "absorption_axis" in self.disabled_terms:
            self.disabled_terms.add("absorption")
        self.status_default = v1r.get("status_default", "observed")   # legacy default
        self.require_status_columns = bool(v1r.get("require_status_columns", False))
        self.disable_mono = bool(v1r.get("disable_mono", False))
        self.families = cfg.get("v1", {}).get("families", {})
        self.species_to_family = {s: f for f, lst in self.families.items() for s in lst}
        self.mixed_minerals = set(cfg.get("v1", {}).get("mixed_minerals", []))
        self.list_conflict_policy = str(v1r.get("list_conflict_policy", "abstain"))
        self.hard_contradiction_terms = list(v1r.get("hard_contradiction_terms", ["optic_class", "pleochroism"]))

        # Load configurable likelihoods from knowledge base (Fix #4)
        self.weights = self.kb.get_observation_weights()
        self.likelihoods = self.kb.get_likelihoods()

        # Acceptance thresholds
        self.accept_p = float(self.rules_cfg.get("accept_p", 0.80))
        self.accept_margin = float(self.rules_cfg.get("accept_margin", 0.40))
        self.group_p = float(self.rules_cfg.get("group_p", 0.85))
        self.min_r2_4 = float(self.rules_cfg.get("min_r2_4", 0.4))
        self.min_features = int(self.rules_cfg.get("min_features", 4))
        self.always_allowed = set(
            self.rules_cfg.get("always_allowed", ["opaque", "fine_aggregate"]))

        # Data-dependent thresholds (set by calibrate_thresholds()).  The
        # hard-coded 2.0 (undulose) / 0.3 (twinkling) did not match the
        # Phase-4 scales: on granite+11agt25 undulose_score had median 19 and
        # relief_mod_med median 0.20, so almost every grain looked "undulose"
        # (-> quartz) and half looked "twinkling".
        self.undulose_high = self.rules_cfg.get("undulose_high")
        self.twinkle_high = self.rules_cfg.get("twinkle_high")
        self.undulose_q = float(self.rules_cfg.get("undulose_high_quantile", 0.80))
        self.twinkle_q = float(self.rules_cfg.get("twinkle_high_quantile", 0.90))

        self.p_unlisted = float(self.rules_cfg.get("p_unlisted", 0.25))
        self.non_label_groups = set(self.rules_cfg.get(
            "non_label_groups", ["accessory", "isotropic_other", "sheet_other"]))

        # Pleochroism levels (ordered)
        self._pleo_levels = ["none", "weak", "moderate", "strong"]

        # Relief levels (ordered)
        self._relief_levels = ["low", "moderate", "high", "very_high"]

    def calibrate_thresholds(
        self, regions_df: pd.DataFrame,
        train_codes: list[str] | set[str] | None = None,
    ) -> None:
        """Set undulose / twinkling thresholds from the data (quantiles over
        non-small grain regions) unless fixed values are given in
        ``rules.undulose_high`` / ``rules.twinkle_high``.

        Parameters
        ----------
        train_codes : set of specimen codes, optional
            V1 FIX: if given, calibrate only from training-fold specimens
            to prevent test-fold data leakage.  When None, calibrates
            from the full dataset (legacy behaviour for Phase 6).
        """
        g = regions_df[(regions_df["type"] == "grain") & (~regions_df["small"])]
        if train_codes is not None:
            train_codes = set(train_codes)
            if "code" in g.columns:
                g = g[g["code"].isin(train_codes)]
                logger.info("Calibrating thresholds from %d training specimens "
                            "(%d grain regions)", len(train_codes), len(g))
        if self.undulose_high is None and "undulose_score" in g:
            self.undulose_high = float(g["undulose_score"].quantile(self.undulose_q))
        if self.twinkle_high is None and "relief_mod_med" in g:
            # pleochroic grains change contrast on rotation -> exclude them
            np_ = g[g["pleo_cls"].isin(["none", "weak"])]
            self.twinkle_high = float(np_["relief_mod_med"].quantile(self.twinkle_q))
        logger.info("Thresholds: undulose_high=%.2f  twinkle_high=%.3f",
                    self.undulose_high or float("nan"),
                    self.twinkle_high or float("nan"))

    # ──────────────────────────────────────────────────────────────────
    #  Fix #4: configurable likelihood helper
    # ──────────────────────────────────────────────────────────────────

    def _status(self, r: dict, key: str) -> str:
        """Get measurement status for key, falling back to status_default."""
        v = r.get(key)
        if _is_nan(v):
            return self.status_default
        return str(v)

    def _p(self, feature: str, key: str, default: float) -> float:
        """Get P(obs|m) from likelihoods.yaml, falling back to *default*.

        Usage::

            p = self._p("optic_class", "match", 0.90)
        """
        return float(
            self.likelihoods.get(feature, {}).get(key, default))

    def _w(self, feature: str, default: float) -> float:
        """Get observation weight from likelihoods.yaml."""
        return float(self.weights.get(feature, default))

    def _is_match_cue(self, term: str, level: str, w: float | None = None) -> bool:
        """Return True if evidence level represents a positive match cue with positive weight."""
        if w is None:
            w = self._w(term, 1.0)
        return level == "match" and w > 0

    # ──────────────────────────────────────────────────────────────────
    #  §12.2: Score a region against one mineral
    # ──────────────────────────────────────────────────────────────────

    def _score_mineral(
        self, r: dict, m_name: str, m_info: dict,
    ) -> tuple[float, dict[str, float], dict[str, str]]:
        """Compute score(m | r), per-feature evidence dict, and per-feature evidence_level dict.

        Uses ``kb.expected_obs(m_name)`` to translate textbook vocabulary
        into the Phase-4 feature vocabulary before comparison.

        Returns (total_score, evidence_dict, evidence_level_dict).
        """
        score = 0.0
        evidence: dict[str, float] = {}
        evidence_level: dict[str, str] = {}
        n_features = 0

        # Pre-translate textbook → Phase-4 vocabulary (BLOCKER #2 fix)
        exp = self.kb.expected_obs(m_name)
        if not exp:
            return score, evidence, evidence_level

        # Helper: add a term (§4c: switch off quarantined terms in one place)
        def _add(name: str, p: float, w: float, level: str) -> None:
            if name in self.disabled_terms or (name == "absorption_axis" and "absorption" in self.disabled_terms):
                return
            nonlocal score, n_features
            contrib = w * math.log(max(p, _EPS))
            score += contrib
            evidence[name] = contrib
            evidence_level[name] = level
            n_features += 1

        # ── 1. Optic class (w=2.0) ──────────────────────────────────
        obs_oc = r.get("optic_class")
        exp_oc = exp.get("optic_class", "")
        oc_status = self._status(r, "optic_class_status")
        if (obs_oc and not _is_nan(obs_oc)
                and str(obs_oc).strip().lower() != "unresolved"
                and exp_oc and oc_status not in ("invalid", "unresolved")):
            w = self._w("optic_class", 2.0)
            obs_oc = str(obs_oc).strip().lower()
            # expected_obs already normalises uniaxial/biaxial → anisotropic
            obs_norm = ("anisotropic"
                        if obs_oc in ("uniaxial", "biaxial", "anisotropic")
                        else obs_oc)
            if obs_norm == exp_oc:
                p = self._p("optic_class", "match", 0.90)
                lvl = "match"
            else:
                p = self._p("optic_class", "mismatch", 0.03)
                lvl = "mismatch"
            _add("optic_class", p, w, lvl)

        # ── 2. Birefringence (w=1.5, modified by colour_masks) ──────
        obs_biref = r.get("biref_cls")
        if obs_biref is not None and not _is_nan(obs_biref):
            obs_biref = int(obs_biref)
            expected_classes = exp.get("biref_cls", set())
            w = self._w("birefringence", 1.5)

            # Colour-masks modifier (§12.2: yes→×0.3, partial→×0.6)
            cmb = exp.get("colour_masks_biref", "no")
            biref_cfg = self.likelihoods.get("birefringence", {})
            if cmb == "yes":
                w *= float(biref_cfg.get("colour_masks_yes", 0.3))
            elif cmb == "partial":
                w *= float(biref_cfg.get("colour_masks_partial", 0.6))

            if obs_biref in expected_classes:
                p = self._p("birefringence", "match", 0.70)
                lvl = "match"
            elif any(abs(obs_biref - ec) <= 1 for ec in expected_classes):
                p = self._p("birefringence", "adjacent", 0.20)
                lvl = "partial"
            else:
                p = self._p("birefringence", "mismatch", 0.03)
                lvl = "mismatch"
            _add("birefringence", p, w, lvl)

        # ── 3. Pleochroism (w=1.2) (§4d) ───────────────────────────
        obs_pleo = r.get("pleo_cls")
        pleo_status = self._status(r, "pleo_status")
        if obs_pleo and not _is_nan(obs_pleo):
            obs_pleo = str(obs_pleo).strip().lower()
            valid_pleo = (
                (obs_pleo == "none" and pleo_status == "not_detected_with_bound")
                or (obs_pleo in ("weak", "moderate", "strong") and pleo_status == "observed")
            )
            if valid_pleo:
                w = self._w("pleochroism", 1.2)
                exp_pleo = exp.get("pleochroism", set())
                if obs_pleo in exp_pleo:
                    p = self._p("pleochroism", "match", 0.75)
                    lvl = "match"
                elif self._pleo_adjacent(obs_pleo, list(exp_pleo)):
                    p = self._p("pleochroism", "partial", 0.20)
                    lvl = "partial"
                else:
                    p = self._p("pleochroism", "mismatch", 0.05)
                    lvl = "mismatch"
                _add("pleochroism", p, w, lvl)

        # ── 4. PPL colour (w=1.0) ───────────────────────────────────
        obs_colour = r.get("ppl_colour")
        if obs_colour and not _is_nan(obs_colour):
            obs_colour = str(obs_colour).strip().lower()
            w = self._w("ppl_colour", 1.0)
            exp_colours = exp.get("ppl_colour", set())
            if obs_colour in exp_colours:
                p = self._p("ppl_colour", "match", 0.70)
                lvl = "match"
            elif ("colourless" in exp_colours
                  and r.get("ppl_chroma", 999) < 10):
                p = self._p("ppl_colour", "partial", 0.15)
                lvl = "partial"
            else:
                p = self._p("ppl_colour", "mismatch", 0.05)
                lvl = "mismatch"
            _add("ppl_colour", p, w, lvl)

        # ── 5. Twinning (w=1.2) ─────────────────────────────────────
        obs_twin = r.get("twin_type")
        twin_status = self._status(r, "twin_status")
        # V1 FIX: only use twinning evidence when status is 'observed'
        # (twins actually detected). 'unresolved' means the detector
        # couldn't determine — this is NOT evidence of absence.
        if (obs_twin and not _is_nan(obs_twin)
                and twin_status == "observed"):
            obs_twin = str(obs_twin).strip().lower()
            w = self._w("twinning", 1.2)
            exp_twins = exp.get("twinning", set())
            if obs_twin in exp_twins:
                p = self._p("twinning", "match", 0.70)
                lvl = "match"
            elif (obs_twin == "none" and
                  any(t in ("lamellar", "crosshatched")
                      for t in exp_twins)):
                # Twins not always visible
                p = self._p("twinning", "partial", 0.35)
                lvl = "partial"
            else:
                p = self._p("twinning", "mismatch", 0.05)
                lvl = "mismatch"
            _add("twinning", p, w, lvl)

        # ── 6. Extinction type (w=1.0, conditional) ─────────────────
        obs_ext = r.get("ext_type")
        aspect = r.get("aspect", 0)
        n_cleav = r.get("n_cleavage_dirs", 0)
        ext_status = self._status(r, "extinction_status")
        exp_ext_set = exp.get("extinction", set())
        # carbonates: "symmetric" = extinction bisects the two cleavages
        if (("symmetric" in exp_ext_set) and r.get("ext_bisects_cleavage") is True):
            _add("extinction_type", self._p("extinction_type", "match", 0.75),
                 self._w("extinction_type", 1.0), "match")
            obs_ext = None
        # "undefined" (no long axis) carries no information -> skip the term
        # V1 FIX: also skip when extinction_status is invalid/not_applicable
        if (obs_ext and not _is_nan(obs_ext)
                and str(obs_ext).strip().lower() != "undefined"
                and ext_status not in ("invalid", "not_applicable")
                and (_safe_float(aspect, 0.0) >= 1.5
                     or _safe_float(n_cleav, 0.0) >= 1)):
            obs_ext = str(obs_ext).strip().lower()
            w = self._w("extinction_type", 1.0)
            exp_ext = exp.get("extinction", set())
            if obs_ext in exp_ext:
                p = self._p("extinction_type", "match", 0.75)
                lvl = "match"
                # For inclined: also check angle range
                if obs_ext == "inclined":
                    ext_vs_long = _safe_float(r.get("ext_vs_long_deg", 0))
                    max_angle = exp.get("ext_angle_max_deg", float("nan"))
                    if not math.isnan(max_angle) and ext_vs_long > max_angle + 5:
                        p = self._p("extinction_type", "mismatch", 0.15)
                        lvl = "mismatch"
            else:
                p = self._p("extinction_type", "mismatch", 0.15)
                lvl = "mismatch"
            _add("extinction_type", p, w, lvl)

        # ── 7. Undulose extinction (w=0.5) ──────────────────────────
        obs_undulose = r.get("undulose_score")
        if obs_undulose is not None and not _is_nan(obs_undulose):
            w = self._w("undulose", 0.5)
            thr = self.undulose_high if self.undulose_high is not None else 2.0
            high = _safe_float(obs_undulose, 0.0) > thr
            if m_name == "quartz" and high:
                p = self._p("undulose", "quartz", 0.60)
                lvl = "match"
            elif high:
                p = self._p("undulose", "other", 0.15)
                lvl = "mismatch"
            else:
                p = 0.50  # neither helps nor hurts
                lvl = "neutral"
            _add("undulose", p, w, lvl)

        # ── 8. Cleavage (w=0.6) ─────────────────────────────────────
        # V1 FIX: only use cleavage evidence when cleavage_status is
        # 'observed'. Default zero when detector fails is NOT absence.
        obs_n_cleav = r.get("n_cleavage_dirs")
        cleav_status = self._status(r, "cleavage_status")
        if ("cleavage" not in self.disabled_terms
                and obs_n_cleav is not None and not _is_nan(obs_n_cleav)
                and cleav_status == "observed"):
            obs_n_cleav = int(obs_n_cleav)
            w = self._w("cleavage", 0.6)
            exp_dirs = exp.get("cleavage_dirs", 0)
            if obs_n_cleav == exp_dirs:
                p = self._p("cleavage", "match", 0.60)
                lvl = "match"
                # Bonus for correct angle (uses pre-computed acute ranges)
                obs_angle = r.get("cleavage_angle_deg")
                if obs_angle is not None and not _is_nan(obs_angle):
                    obs_angle = _safe_float(obs_angle)
                    for lo, hi in exp.get("cleavage_acute_deg", []):
                        if lo <= obs_angle <= hi:
                            p += self._p("cleavage", "partial", 0.20)
                            break
            else:
                p = self._p("cleavage", "mismatch", 0.20)
                lvl = "mismatch"
            _add("cleavage", p, w, lvl)

        # ── 9. Absorption axis (w=1.0, conditional) ─────────────────
        # V1 FIX: only use absorption when status is 'observed'
        # (NaN absorption ≠ 0°; it means unavailable)
        obs_pleo_cls = r.get("pleo_cls")
        absorption_status = self._status(r, "absorption_status")
        if ("absorption_axis" not in self.disabled_terms
                and "absorption" not in self.disabled_terms
                and obs_pleo_cls and str(obs_pleo_cls).lower() in ("moderate", "strong")
                and _safe_float(r.get("aspect", 0), 0.0) >= 1.5
                and absorption_status == "observed"):
            w = self._w("absorption_axis", 1.0)
            obs_abs = _safe_float(r.get("absorption_vs_long_deg"))
            exp_abs = exp.get("absorption_vs_long", "n/a")
            if exp_abs == "parallel":
                if obs_abs < 20:
                    p = (self._p("absorption_axis", "match", 0.80))
                    lvl = "match"
                else:
                    p = (self._p("absorption_axis", "mismatch", 0.20))
                    lvl = "mismatch"
            elif exp_abs == "perpendicular":
                if obs_abs > 70:
                    p = (self._p("absorption_axis", "match", 0.80))
                    lvl = "match"
                else:
                    p = (self._p("absorption_axis", "mismatch", 0.20))
                    lvl = "mismatch"
            else:
                p = 0.50  # n/a — neutral
                lvl = "neutral"
            _add("absorption_axis", p, w, lvl)

        # ── 10. Twinkling (w=0.8) ───────────────────────────────────
        obs_relief_mod = r.get("relief_mod_med")
        pleo_now = str(r.get("pleo_cls", "")).lower()
        if (obs_relief_mod is not None and not _is_nan(obs_relief_mod)
                and pleo_now not in ("moderate", "strong")):
            w = self._w("twinkling", 0.8)
            exp_twinkle = exp.get("twinkling", False)
            thr = self.twinkle_high if self.twinkle_high is not None else 0.3
            high_relief_mod = _safe_float(obs_relief_mod, 0.0) > thr
            if exp_twinkle:
                if high_relief_mod:
                    p = self._p("twinkling", "match", 0.80)
                    lvl = "match"
                else:
                    p = self._p("twinkling", "mismatch", 0.20)
                    lvl = "mismatch"
            else:
                if not high_relief_mod:
                    p = self._p("twinkling", "match", 0.80)
                    lvl = "match"
                else:
                    p = self._p("twinkling", "mismatch", 0.20)
                    lvl = "mismatch"
            _add("twinkling", p, w, lvl)

        # ── 11. Habit (w=0.4) ───────────────────────────────────────
        obs_habit = r.get("habit")
        if obs_habit and not _is_nan(obs_habit):
            obs_habit = str(obs_habit).strip().lower()
            w = self._w("habit", 0.4)
            exp_habits = exp.get("habit", set())
            if obs_habit in exp_habits:
                p = self._p("habit", "match", 0.60)
                lvl = "match"
            else:
                p = self._p("habit", "mismatch", 0.30)
                lvl = "mismatch"
            _add("habit", p, w, lvl)

        # ── 13. Region type (w=2.0) ─────────────────────────────────
        obs_type = r.get("type")
        if obs_type and not _is_nan(obs_type):
            obs_type = str(obs_type).strip().lower()
            w = self._w("region_type", 2.0)
            if obs_type == "aggregate":
                if m_name == "fine_aggregate":
                    p = self._p("region_type", "aggregate_match", 0.90)
                    lvl = "match"
                else:
                    p = self._p("region_type", "aggregate_mismatch", 0.05)
                    lvl = "mismatch"
            elif m_name == "fine_aggregate":
                # non-aggregate region for fine_aggregate mineral (Fix #8)
                p = self._p("region_type", "aggregate_mismatch", 0.05)
                lvl = "mismatch"
            else:
                p = 0.50  # neutral for grain/opaque types
                lvl = "neutral"
            _add("region_type", p, w, lvl)

        return score, evidence, evidence_level

    # ──────────────────────────────────────────────────────────────────
    #  §12.2: Score a region against all candidates
    # ──────────────────────────────────────────────────────────────────

    def score_region(
        self, region: dict, candidates: set[str],
    ) -> dict:
        """Score region against all candidate minerals.

        Returns dict with keys:
            scores, posteriors, top1, p1, top2, p2, evidence, evidence_level, n_features.
        """
        scores: dict[str, float] = {}
        evidence_all: dict[str, dict[str, float]] = {}
        evidence_level_all: dict[str, dict[str, str]] = {}
        n_features = 0

        for m_name in sorted(candidates):
            m_info = self.kb.get_mineral(m_name)
            if not m_info:
                continue
            s, ev, ev_lvl = self._score_mineral(region, m_name, m_info)
            scores[m_name] = s
            evidence_all[m_name] = ev
            evidence_level_all[m_name] = ev_lvl
            n_features = max(n_features, len(ev))

        # "Unlisted" pseudo-candidate: fits every observed term with the
        # same modest probability.  Softmax over a small candidate list
        # otherwise picks the "least bad" mineral with high confidence even
        # when every candidate contradicts the observations (OU lists are
        # incomplete).  If it wins, the region stays unlabelled.
        if scores and self.p_unlisted > 0:
            ref = max(evidence_all.values(), key=len)
            w_sum = 0.0
            for term in ref:
                # recover the weight used for this term from its contribution
                w_sum += self._w({"optic_class": "optic_class"}.get(term, term),
                                 1.0)
            scores["__unlisted__"] = w_sum * math.log(self.p_unlisted)
            evidence_all["__unlisted__"] = {}
            evidence_level_all["__unlisted__"] = {}

        if not scores:
            return {
                "scores": {}, "posteriors": {},
                "top1": None, "p1": 0.0,
                "top2": None, "p2": 0.0,
                "evidence": {}, "evidence_level": {}, "n_features": 0,
            }

        # Softmax → posterior
        max_s = max(scores.values())
        exp_s = {m: math.exp(s - max_s) for m, s in scores.items()}
        z = sum(exp_s.values())
        posteriors = {m: e / z for m, e in exp_s.items()}

        ranked = sorted(posteriors.items(), key=lambda x: (-x[1], x[0]))
        top1, p1 = ranked[0]
        top2, p2 = ranked[1] if len(ranked) > 1 else (None, 0.0)

        return {
            "scores": scores,
            "posteriors": posteriors,
            "top1": top1, "p1": p1,
            "top2": top2, "p2": p2,
            "evidence": evidence_all,
            "evidence_level": evidence_level_all,
            "n_features": n_features,
        }

    # ──────────────────────────────────────────────────────────────────
    #  §12.3: Acceptance rules
    # ──────────────────────────────────────────────────────────────────

    def label_region(
        self, region: dict, candidates: set[str],
    ) -> dict:
        """Score and apply acceptance rules to produce a label.

        Returns dict with rule_top1/p1/top2/p2, rule_evidence_json,
        label_fine, label_group, label_source, label_conf.
        """
        result = self.score_region(region, candidates)

        top1 = result["top1"]
        p1 = result["p1"]
        top2 = result["top2"]
        p2 = result["p2"]
        n_feat = result["n_features"]
        posteriors = result["posteriors"]

        # Evidence JSON for top-2
        ev_json = {}
        if top1 and top1 in result["evidence"]:
            ev_json[top1] = result["evidence"][top1]
        if top2 and top2 in result["evidence"]:
            ev_json[top2] = result["evidence"][top2]

        # V1: identify the deciding evidence — the optical term that
        # contributes most to the score gap between top1 and top2
        deciding_evidence = None
        if top1 and top2 and top1 in result["evidence"] and top2 in result["evidence"]:
            ev1 = result["evidence"][top1]
            ev2 = result["evidence"][top2]
            all_terms = sorted(set(ev1.keys()) | set(ev2.keys()))
            best_gap = 0.0
            for term in all_terms:
                gap = ev1.get(term, 0.0) - ev2.get(term, 0.0)
                if gap > best_gap:
                    best_gap = gap
                    deciding_evidence = term

        base = {
            "rule_top1": top1,
            "rule_p1": round(p1, 4),
            "rule_top2": top2,
            "rule_p2": round(p2, 4) if p2 else 0.0,
            "rule_evidence_json": json.dumps(ev_json),
            "deciding_evidence": deciding_evidence,
            "n_optical_features": n_feat,
        }

        if top1 == "__unlisted__":
            base.update({"label_fine": None, "label_group": None,
                         "label_source": "none", "label_conf": 0.0})
            return base

        # ── fine_aggregate special case ─────────────────────────────
        rtype = str(region.get("type", "")).lower()
        if rtype == "aggregate" and "fine_aggregate" in candidates:
            base.update({
                "label_fine": "fine_aggregate",
                "label_group": "aggregate",
                "label_source": "rule",
                "label_conf": 0.95,
            })
            return base

        # ── Fine label acceptance ───────────────────────────────────
        margin = p1 - p2
        is_grain_or_opaque = rtype in ("grain", "opaque")
        not_small = not region.get("small", False)

        # r2_4 check — waive for isotropic/opaque
        obs_oc = str(region.get("optic_class", "")).lower()
        r2_4_ok = True
        if obs_oc not in ("isotropic", "opaque"):
            r2_4_ok = _safe_float(region.get("r2_4_med", 0), 0.0) >= self.min_r2_4

        quality_ok = (is_grain_or_opaque and not_small and r2_4_ok)
        if (p1 >= self.accept_p
                and margin >= self.accept_margin
                and quality_ok):
            base.update({
                "label_fine": top1,
                "label_group": self.kb.get_group(top1),
                "label_source": "rule",
                "label_conf": round(p1, 4),
            })
            return base

        # (opaque goes through normal acceptance rules; fine_aggregate
        # is handled by the aggregate special case above)

        # ── Group label ─────────────────────────────────────────────
        # Sum posteriors at EVERY level above each mineral (k_feldspar,
        # feldspar, felsic ...) and take the most specific group that
        # reaches group_p.  Summing only the immediate CSV group could never
        # give "feldspar" for a plagioclase-vs-orthoclase region, or
        # "felsic" for a quartz-vs-feldspar one.
        group_sums: dict[str, float] = {}
        group_size: dict[str, int] = {}
        for m, p in posteriors.items():
            if m == "__unlisted__":
                continue
            # set(): get_group() (CSV column) and get_ancestors() (hierarchy)
            # usually name the same group — counting it twice doubled the
            # sum (label_conf 1.5 seen in testing)
            for g in sorted(set([self.kb.get_group(m)] + self.kb.get_ancestors(m))):
                if g == m or g == "unknown":
                    continue
                group_sums[g] = group_sums.get(g, 0.0) + p
        for g in group_sums:
            group_size[g] = len(self.kb.expand_group(g)) or 999
        # catch-all groups ("accessory", ...) are not meaningful labels:
        # tourmaline and cassiterite have nothing optical in common
        ok = [g for g, v in group_sums.items()
              if v >= self.group_p and g not in self.non_label_groups]
        best_group = min(ok, key=lambda g: (group_size[g], -group_sums[g], g),
                         default=None)
        # same quality gate as fine labels (otherwise a single-member
        # group such as "silica" re-admits regions rejected above)
        if best_group and quality_ok:
            base.update({
                "label_fine": None,
                "label_group": best_group,
                "label_source": "rule",
                "label_conf": round(group_sums[best_group], 4),
            })
            return base

        # ── Unlabelled ──────────────────────────────────────────────
        base.update({
            "label_fine": None,
            "label_group": None,
            "label_source": "none",
            "label_conf": 0.0,
        })
        return base

    # ──────────────────────────────────────────────────────────────────
    #  §4g: Family-level decision (V1 taxonomy)
    # ──────────────────────────────────────────────────────────────────

    def family_decision(
        self,
        region: dict,
        candidates: set[str],
        res: dict | None = None,
    ) -> dict:
        """Family-level decision with 4 states (§4g).

        States:
        - mixed: aggregate regions or mixed minerals
        - singleton: winning family reaches singleton_p threshold
        - partial_set: cumulative probability reaches set_cum_p within max_set_size
        - abstain: out-of-scope plausible, diffuse, no positive cue, or hard contradiction
        """
        v1r = self.cfg.get("v1", {}).get("rules", {})
        if res is None:
            res = self.score_region(region, candidates)
        post = res.get("posteriors", {})
        top1 = res.get("top1")
        p1 = float(res.get("p1", 0.0))
        if not post:
            return {
                "state": "abstain", "allowed": [], "reason": "no_candidates",
                "top1": top1, "p1": p1, "deciding_evidence": None,
            }

        # 1. mixed objects
        if str(region.get("type", "")).lower() == "aggregate" or \
           (top1 in self.mixed_minerals):
            return {
                "state": "mixed", "allowed": [], "reason": "aggregate_or_fine",
                "top1": top1, "p1": p1, "deciding_evidence": None,
            }

        # 2. sum posteriors per family; everything else (incl. __unlisted__) = OUT
        fam_p = {f: 0.0 for f in self.families}
        p_out = 0.0
        for m, p in post.items():
            f = self.species_to_family.get(m)
            if f is None:
                p_out += p
            else:
                fam_p[f] += p
        max_out_of_scope_p = float(v1r.get("max_out_of_scope_p", 0.20))
        if p_out >= max_out_of_scope_p:
            return {
                "state": "abstain", "allowed": [], "reason": "out_of_scope_plausible",
                "fam_p": fam_p, "p_out": p_out,
                "top1": top1, "p1": p1, "deciding_evidence": None,
            }

        ranked = sorted(fam_p.items(), key=lambda x: (-x[1], x[0]))
        top_f, top_p = ranked[0]

        # 3. require >=1 real optical cue for the winning family
        in_species = [m for m in post if self.species_to_family.get(m) == top_f]
        if not in_species:
            return {
                "state": "abstain", "allowed": [], "reason": "no_positive_cue",
                "fam_p": fam_p, "p_out": p_out,
                "top1": top1, "p1": p1, "deciding_evidence": None,
            }
        best_in = max(in_species, key=lambda m: (post[m], m))
        out_species = [
            m for m in post
            if m != "__unlisted__" and self.species_to_family.get(m) != top_f
        ]
        best_out = max(out_species, key=lambda m: (post[m], m)) if out_species else None

        ev = res.get("evidence", {})
        ev_lvl = res.get("evidence_level", {})

        def _get_cue_level(spec: str, term: str, val: float) -> str:
            if spec in ev_lvl and term in ev_lvl[spec]:
                return ev_lvl[spec][term]
            if val >= 0.0:
                return "match"
            w_val = self._w(term, 1.0)
            if w_val > 0.0 and (val / w_val) >= -10.0:
                try:
                    p_val = math.exp(val / w_val)
                    match_thresh = (
                        self._p("undulose", "quartz", 0.60)
                        if term == "undulose"
                        else self._p(term, "match", 0.60)
                    )
                    if p_val >= (match_thresh - 1e-4):
                        return "match"
                except OverflowError:
                    pass
            return "mismatch"

        info_terms = [
            t for t, v in ev.get(best_in, {}).items()
            if t in INFORMATIVE and t not in self.disabled_terms
            and not (t == "absorption_axis" and "absorption" in self.disabled_terms)
            and self._is_match_cue(t, _get_cue_level(best_in, t, v), self._w(t, 1.0))
            and (best_out is None or v > ev.get(best_out, {}).get(t, -1e9))
        ]
        n_info = len(info_terms)
        min_informative_terms = int(v1r.get("min_informative_terms", 1))
        if n_info < min_informative_terms:
            return {
                "state": "abstain", "allowed": [], "reason": "no_positive_cue",
                "fam_p": fam_p, "p_out": p_out, "n_info": n_info, "info_terms": info_terms,
                "top1": top1, "p1": p1, "deciding_evidence": None,
            }

        # Deciding evidence: the optical term that contributes most to the
        # score gap between best_in and best_out, or the top informative cue
        deciding_evidence = None
        if best_in and best_out and best_in in ev and best_out in ev:
            ev1 = ev[best_in]
            ev2 = ev[best_out]
            all_terms = sorted(set(ev1.keys()) | set(ev2.keys()))
            best_gap = 0.0
            for term in all_terms:
                gap = ev1.get(term, 0.0) - ev2.get(term, 0.0)
                if gap > best_gap:
                    best_gap = gap
                    deciding_evidence = term
        if deciding_evidence is None and info_terms:
            deciding_evidence = info_terms[0]

        # 4. singleton or partial set
        singleton_p = float(v1r.get("singleton_p", 0.80))
        set_cum_p = float(v1r.get("set_cum_p", 0.90))
        max_set_size = int(v1r.get("max_set_size", 2))

        decision_state = "abstain"
        allowed = []
        reason = ""

        if top_p >= singleton_p:
            decision_state = "singleton"
            allowed = [top_f]
            reason = ""
        else:
            cum, S = 0.0, []
            for f, p in ranked:
                S.append(f)
                cum += p
                if cum >= set_cum_p:
                    break
            if cum >= set_cum_p and len(S) <= max_set_size:
                decision_state = "partial_set"
                allowed = sorted(S)
                reason = ""
            else:
                decision_state = "abstain"
                allowed = []
                reason = "diffuse"

        # Hard contradiction check: check EVERY family in the final allowed set
        if decision_state in ("singleton", "partial_set"):
            sorted_cands = sorted(candidates)
            for fam in allowed:
                fam_spp = [m for m in sorted_cands if self.species_to_family.get(m) == fam]
                if not fam_spp:
                    fam_spp = [m for m in self.families.get(fam, []) if m in candidates]
                if not fam_spp:
                    fam_spp = list(self.families.get(fam, []))

                cand_spp = [m for m in fam_spp if post.get(m, 0.0) >= 0.05]
                if not cand_spp:
                    cand_spp = fam_spp

                for term in self.hard_contradiction_terms:
                    if term in self.disabled_terms:
                        continue
                    if cand_spp and all(ev_lvl.get(m, {}).get(term) == "mismatch" for m in cand_spp):
                        return {
                            "state": "abstain",
                            "allowed": [],
                            "reason": "hard_contradiction",
                            "fam_p": fam_p,
                            "p_out": p_out,
                            "n_info": n_info,
                            "info_terms": info_terms,
                            "top1": top1,
                            "p1": p1,
                            "deciding_evidence": term,
                        }

        return {
            "state": decision_state, "allowed": allowed, "reason": reason,
            "fam_p": fam_p, "p_out": p_out, "n_info": n_info, "info_terms": info_terms,
            "top1": top1, "p1": p1,
            "deciding_evidence": deciding_evidence,
        }

    @staticmethod
    def single_angle_view(region: dict) -> dict:
        """Create a single-angle view of a region dict by substituting k1_* fields
        and setting rotation-only features to NaN / unresolved status.
        """
        r = dict(region)

        # Single-angle replacements
        r["optic_class"] = region.get("k1_optic_class")
        r["optic_class_status"] = region.get("k1_optic_class_status")
        r["ppl_colour"] = region.get("k1_ppl_colour")
        r["ppl_chroma"] = region.get("k1_ppl_chroma")
        r["ppl_L"] = region.get("k1_ppl_L")
        r["ppl_a"] = region.get("k1_ppl_a")
        r["ppl_b"] = region.get("k1_ppl_b")
        r["ppl_hue_deg"] = region.get("k1_ppl_hue_deg")

        # Rotation-only fields to float("nan") with "unresolved" status
        nan = float("nan")
        r["pleo_cls"] = nan
        r["pleo_status"] = "unresolved"
        r["pleo_delta_e"] = nan
        r["pleo_dE_med"] = nan
        r["twin_type"] = nan
        r["twin_status"] = "unresolved"
        r["ext_type"] = nan
        r["extinction_status"] = "unresolved"
        r["ext_vs_long_deg"] = nan
        r["ext_bisects_cleavage"] = nan
        r["undulose_score"] = nan
        r["undulose_range_deg"] = nan
        r["absorption_status"] = "unresolved"
        r["absorption_vs_long_deg"] = nan
        r["relief_mod_med"] = nan
        r["biref_cls"] = nan

        return r

    @staticmethod
    def compute_static_category(
        full_state: str,
        full_allowed: list[str] | set[str] | None,
        k1_state: str,
        k1_allowed: list[str] | set[str] | None,
    ) -> str:
        """Compute static comparison category between full-rotation and single-angle targets
        per CONTRACTS.md Section 3.
        """
        st = str(full_state).strip().lower()
        k1_st = str(k1_state).strip().lower()
        if st in ("excluded", "mixed"):
            return "n/a"
        full_labelled = st in ("singleton", "partial_set")
        if full_labelled and k1_st == "mixed":
            return "conflict"
        k1_labelled = k1_st in ("singleton", "partial_set")
        s_full = set(full_allowed or [])
        s_k1 = set(k1_allowed or [])

        if full_labelled and k1_labelled:
            if st == k1_st and s_full == s_k1:
                return "reproduced"
            elif s_full < s_k1:
                return "k1_broader"
            elif s_k1 < s_full:
                return "k1_narrower"
            elif s_full.isdisjoint(s_k1):
                return "conflict"
            else:
                return "conflict"
        elif full_labelled and k1_st == "abstain":
            return "full_only"
        elif st == "abstain" and k1_labelled:
            return "k1_only"
        elif st == "abstain" and k1_st == "abstain":
            return "unresolved_both"
        return "n/a"

    # Note on low_r2_4 quality gate:
    # low_r2_4 grains are excluded as a declared population restriction; results are conditional on this population.
    def check_quality_gate(self, region: dict) -> tuple[bool, str]:
        """Check V1 quality gate (§4g).

        Requires region['eligible_v1'], plus r2_4_med >= min_r2_4 for anisotropic regions.
        low_r2_4 grains are excluded as a declared population restriction; results are conditional on this population.
        """
        if not region.get("eligible_v1", False):
            reason = region.get("reject_reason_v1", "not_eligible") or "not_eligible"
            return False, reason

        obs_oc = str(region.get("optic_class", "")).lower()
        if obs_oc not in ("isotropic", "opaque"):
            r2_4 = _safe_float(region.get("r2_4_med", 0), 0.0)
            if r2_4 < self.min_r2_4:
                return False, "low_r2_4"

        return True, ""

    def evaluate_region_v1(
        self,
        region: dict,
        candidates: set[str],
        skip_gate: bool = False,
    ) -> dict:
        """Run V1 central decision function (§4g, CONTRACTS.md Section 2)."""
        rtype = str(region.get("type", "")).lower()
        if not skip_gate and rtype != "aggregate":
            gate_ok, gate_reason = self.check_quality_gate(region)
            if not gate_ok:
                reason = gate_reason or "quality_gate"
                return {
                    "state": "excluded",
                    "allowed": [],
                    "reason": reason,
                    "optics_state": "excluded",
                    "optics_allowed": [],
                    "optics_reason": reason,
                    "optics_list_conflict": False,
                    "metadata_dependent": False,
                    "top1": None,
                    "p1": 0.0,
                    "fam_p": {f: 0.0 for f in self.families},
                    "p_out": 0.0,
                    "n_info": 0,
                    "info_terms": [],
                    "deciding_evidence": None,
                }

        all_minerals = set(self.kb.all_minerals())
        cand_set = set(candidates)
        optics = self.family_decision(region, all_minerals)

        if cand_set == all_minerals:
            meta = optics
        else:
            meta = self.family_decision(region, cand_set)

        optics_state = optics.get("state", "abstain")
        optics_allowed = optics.get("allowed", [])
        optics_reason = optics.get("reason", "")

        meta_state = meta.get("state", "abstain")
        meta_allowed = meta.get("allowed", [])
        meta_reason = meta.get("reason", "")

        # Check optics list conflict: True iff optics_state == "singleton" and optics_allowed[0] not in meta_allowed
        optics_list_conflict = bool(
            optics_state == "singleton"
            and optics_allowed
            and optics_allowed[0] not in meta_allowed
        )

        branch_used = meta
        # Apply hard contradiction or conflict policy
        if meta_reason == "hard_contradiction":
            state = "abstain"
            allowed = []
            reason = "hard_contradiction"
            deciding_evidence = meta.get("deciding_evidence")
        elif optics_list_conflict:
            policy = self.list_conflict_policy
            if policy == "keep_optics":
                state = optics_state
                allowed = list(optics_allowed)
                reason = "list_conflict_optics_kept"
                deciding_evidence = optics.get("deciding_evidence")
                branch_used = optics
            else:  # "abstain"
                state = "abstain"
                allowed = []
                if meta_state in ("singleton", "partial_set"):
                    reason = "list_conflict"
                    deciding_evidence = optics.get("deciding_evidence") or meta.get("deciding_evidence")
                else:
                    reason = meta_reason
                    deciding_evidence = meta.get("deciding_evidence")
        else:
            state = meta_state
            allowed = list(meta_allowed)
            reason = meta_reason
            deciding_evidence = meta.get("deciding_evidence")

        metadata_dependent = bool((state, allowed) != (optics_state, optics_allowed))

        res_dict = {
            "state": state,
            "allowed": allowed,
            "reason": reason,
            "optics_state": optics_state,
            "optics_allowed": optics_allowed,
            "optics_reason": optics_reason,
            "optics_list_conflict": optics_list_conflict,
            "metadata_dependent": metadata_dependent,
            "top1": branch_used.get("top1"),
            "p1": float(branch_used.get("p1", 0.0)),
            "fam_p": branch_used.get("fam_p", {f: 0.0 for f in self.families}),
            "p_out": float(branch_used.get("p_out", 0.0)),
            "n_info": int(branch_used.get("n_info", 0)),
            "info_terms": branch_used.get("info_terms", []),
            "deciding_evidence": deciding_evidence,
        }

        if optics_list_conflict and self.list_conflict_policy == "keep_optics":
            res_dict["meta_top1"] = meta.get("top1")
            res_dict["meta_p1"] = float(meta.get("p1", 0.0))
            res_dict["meta_fam_p"] = meta.get("fam_p", {f: 0.0 for f in self.families})
            res_dict["meta_info_terms"] = meta.get("info_terms", [])
            res_dict["meta_reason"] = meta_reason
            res_dict["meta_p_out"] = float(meta.get("p_out", 0.0))
            res_dict["meta_n_info"] = int(meta.get("n_info", 0))

        return res_dict

    # ──────────────────────────────────────────────────────────────────
    #  §12.4: Monomineralic specimens (S2)
    #  Fix #7: mono is authoritative — overrides S1 for grain regions
    # ──────────────────────────────────────────────────────────────────

    def label_monomineralic(
        self,
        regions_df: pd.DataFrame,
        mineral: str,
        site_ids: set[str],
    ) -> tuple[pd.DataFrame, bool]:
        """Label all grain regions in monomineralic specimen sites.

        Sanity check: >=70% of grain area must be consistent with the
        mineral's optic class and birefringence class.

        Returns (labels_df, sanity_passed).
        """
        m_info = self.kb.get_mineral(mineral)
        if not m_info:
            logger.warning("Monomineralic mineral '%s' not in knowledge "
                           "base", mineral)
            return pd.DataFrame(), False

        exp_oc = str(m_info.get("optic_class", "")).lower()
        exp_biref = set(self.kb.get_biref_classes(mineral))
        exp_aniso = exp_oc in ("uniaxial", "biaxial", "anisotropic")

        site_regions = regions_df[regions_df["site_id"].isin(site_ids)]
        grains = site_regions[site_regions["type"] == "grain"]

        if grains.empty:
            return pd.DataFrame(), True

        # Sanity check: >=70% of grain area consistent
        total_area = grains["area_px"].sum()
        consistent_area = 0
        exp_obs = self.kb.expected_obs(mineral)
        for _, r in grains.iterrows():
            if self._consistent_with(r, exp_obs):
                consistent_area += r["area_px"]

        frac = consistent_area / max(total_area, 1)
        if frac < 0.70:
            logger.warning("Monomineralic '%s' failed sanity check "
                           "(%.0f%% consistent) — removing flag",
                           mineral, frac * 100)
            return pd.DataFrame(), False

        # Label all regions in these sites
        labels = []
        for _, r in site_regions.iterrows():
            rtype = str(r.get("type", "")).lower()
            obs_oc = str(r.get("optic_class", "")).lower()

            if rtype == "opaque" and exp_oc != "opaque":
                labels.append({
                    "region_id": r["region_id"],
                    "label_fine": "opaque",
                    "label_group": "opaque",
                    "label_source": "mono",
                    "label_conf": 1.0,
                })
            elif rtype == "grain":
                # per-region check: a mica flake in a quartzite must not be
                # labelled quartz with confidence 1.0
                if not self._consistent_with(r, exp_obs):
                    continue
                labels.append({
                    "region_id": r["region_id"],
                    "label_fine": mineral,
                    "label_group": self.kb.get_group(mineral),
                    "label_source": "mono",
                    "label_conf": 1.0,
                })
            # aggregate/hole: skip

        return pd.DataFrame(labels), True

    def _consistent_with(self, r, exp: dict) -> bool:
        """Region is physically compatible with a mineral: optic class,
        birefringence class (±1 only if colour can mask it), pleochroism
        (±1 level) and PPL colour ("grey" is accepted as uninformative)."""
        oc = str(r.get("optic_class", "")).lower()
        if oc != exp.get("optic_class"):
            return False
        b = r.get("biref_cls")
        if b is not None and not _is_nan(b):
            eb = exp.get("biref_cls", set())
            tol = 0 if exp.get("colour_masks_biref", "no") == "no" else 1
            if not any(abs(int(b) - e) <= tol for e in eb):
                return False
        pl = str(r.get("pleo_cls", "")).lower()
        if pl and pl != "nan":
            ep = exp.get("pleochroism", set())
            if pl not in ep and not self._pleo_adjacent(pl, list(ep)):
                return False
        col = str(r.get("ppl_colour", "")).lower()
        if col and col not in ("nan", "grey") and col not in exp.get("ppl_colour", set()):
            return False
        return True

    # ──────────────────────────────────────────────────────────────────
    #  §12.5: Rotation titles (S3) — evaluation only
    #  Fix #1: stored in separate title_* columns, never contaminates S1/S2
    #  Fix #2: uses segment mask pixel lookup, not centroid distance
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def find_title_region(
        regions_df: pd.DataFrame,
        site_id: str,
        label_img: np.ndarray | None,
        centre_xy: tuple[int, int] = (245, 245),
        radius_px: int = 40,
    ) -> str | None:
        """Find the region at the image centre for title labelling (§12.5).

        Fix #2: Uses the actual segmentation mask to find the region
        containing the centre pixel, falling back to largest region
        within *radius_px*.

        Returns region_id or None.
        """
        cx, cy = centre_xy
        site_regs = regions_df[regions_df["site_id"] == site_id]
        grains = site_regs[site_regs["type"].isin(["grain", "opaque"])]
        if grains.empty:
            return None

        # ── Primary: exact pixel lookup from segmentation mask ──────
        if label_img is not None:
            H, W = label_img.shape[:2]
            if 0 <= cy < H and 0 <= cx < W:
                pixel_label = int(label_img[cy, cx])
                if pixel_label > 0:
                    # Check if this label belongs to a grain/opaque region
                    lv = (grains["label_value"] if "label_value" in grains
                          else grains["region_id"].str.rsplit("_g", n=1).str[1].astype(int))
                    match = grains[lv == pixel_label]
                    if not match.empty:
                        return str(match.iloc[0]["region_id"])

        # ── Fallback: largest region with centroid within radius ────
        best_id = None
        best_area = 0
        for _, r in grains.iterrows():
            centroid = r.get("centroid_xy")
            if centroid is None:
                continue
            if isinstance(centroid, str):
                try:
                    centroid = json.loads(centroid)
                except (json.JSONDecodeError, ValueError):
                    continue
            if not centroid or len(centroid) < 2:
                continue
            rx, ry = float(centroid[0]), float(centroid[1])
            dist = math.sqrt((rx - cx) ** 2 + (ry - cy) ** 2)
            area = _safe_float(r.get("area_px", 0), 0.0)
            if dist <= radius_px and area > best_area:
                best_area = area
                best_id = str(r["region_id"])

        return best_id

    # ──────────────────────────────────────────────────────────────────
    #  §12.6: Final class list + relabelling
    #  Fix #3: returns relabel mapping and applies it to labels_df
    # ──────────────────────────────────────────────────────────────────

    def build_class_list(
        self,
        labels_df: pd.DataFrame,
        regions_df: pd.DataFrame,
        cfg: dict,
    ) -> tuple[dict, dict[str, str]]:
        """Build the final class list and the relabel mapping.

        Returns
        -------
        class_info : dict
            ``fine_classes``, ``groups``, ``mineral_counts``.
        relabel_map : dict[str, str]
            Maps each label_fine value to its final class name.
            Qualified minerals map to themselves; non-qualified minerals
            map to their parent group (if it qualifies) or ``"other"``.
        """
        cls_cfg = cfg.get("classes", {})
        min_regions = int(cls_cfg.get("min_regions", 30))
        min_specimens = int(cls_cfg.get("min_specimens", 3))

        # Merge labels with regions for specimen info
        merged = labels_df.merge(
            regions_df[["region_id", "site_id", "code"]].drop_duplicates(
                subset="region_id"),
            on="region_id", how="left",
        )
        fine_labelled = merged[merged["label_fine"].notna()].copy()

        # Count per mineral
        mineral_counts: dict[str, dict] = {}
        for mineral, grp in fine_labelled.groupby("label_fine"):
            mineral_counts[mineral] = {
                "n_regions": len(grp),
                "n_specimens": grp["code"].nunique(),
            }

        # Determine which minerals qualify on their own
        qualified_minerals: set[str] = set()
        non_qualified: dict[str, str] = {}  # mineral → its group

        for mineral, counts in mineral_counts.items():
            group = self.kb.get_group(mineral)
            if (counts["n_regions"] >= min_regions
                    and counts["n_specimens"] >= min_specimens):
                qualified_minerals.add(mineral)
            else:
                non_qualified[mineral] = group

        # Aggregate non-qualified minerals by group
        group_pool: dict[str, list[str]] = {}
        for mineral, group in non_qualified.items():
            group_pool.setdefault(group, []).append(mineral)

        qualified_groups: set[str] = set()
        for group, minerals in group_pool.items():
            total_regions = sum(
                mineral_counts[m]["n_regions"] for m in minerals)
            total_specimens = len(set().union(*(
                set(fine_labelled[fine_labelled["label_fine"] == m]["code"])
                for m in minerals
            )))
            if (total_regions >= min_regions
                    and total_specimens >= min_specimens):
                qualified_groups.add(group)

        # Build relabel mapping (Fix #3 + Phase 7 completeness fix)
        # Cover ALL canonical minerals, not just those in mineral_counts,
        # so Phase 7 can map every OU candidate to a final class.
        relabel_map: dict[str, str] = {}
        for mineral in self.kb.all_minerals():
            if mineral in qualified_minerals:
                relabel_map[mineral] = mineral
            else:
                group = self.kb.get_group(mineral)
                if group in qualified_groups:
                    relabel_map[mineral] = group
                else:
                    relabel_map[mineral] = "other"
        # Also include minerals from mineral_counts that may not be in KB
        # (defensive — shouldn't happen, but safe)
        for mineral in mineral_counts:
            if mineral not in relabel_map:
                if mineral in qualified_minerals:
                    relabel_map[mineral] = mineral
                else:
                    group = non_qualified.get(mineral, "other")
                    if group in qualified_groups:
                        relabel_map[mineral] = group
                    else:
                        relabel_map[mineral] = "other"

        # opaque / fine_aggregate are always kept as classes (below), so
        # they must map to themselves — otherwise their labels would be
        # rewritten to "other" while the class still exists.
        for special in ("opaque", "fine_aggregate"):
            relabel_map[special] = special

        # Build fine_classes list
        fine_classes = sorted(qualified_minerals | qualified_groups)
        # Always keep opaque and fine_aggregate
        for special in ["opaque", "fine_aggregate"]:
            if special not in fine_classes:
                fine_classes.append(special)
        if "other" not in fine_classes:
            fine_classes.append("other")
        fine_classes = sorted(set(fine_classes))

        # Groups mapping
        groups_map: dict[str, str] = {}
        for mineral in mineral_counts:
            groups_map[mineral] = self.kb.get_group(mineral)

        class_info = {
            "fine_classes": fine_classes,
            "groups": groups_map,
            "mineral_counts": mineral_counts,
            "relabel_map": relabel_map,  # Phase 7 needs this for MIL candidates
        }

        return class_info, relabel_map

    @staticmethod
    def apply_relabel(
        labels_df: pd.DataFrame, relabel_map: dict[str, str],
    ) -> pd.DataFrame:
        """Apply the relabel mapping to labels_df (Fix #3).

        Rewrites ``label_fine`` so every value is in ``classes.yaml``.
        Also updates ``label_group`` to stay consistent with the new
        ``label_fine``.  Stores originals in ``_original`` columns.
        """
        df = labels_df.copy()
        df["label_fine_original"] = df["label_fine"]
        df["label_group_original"] = df["label_group"]
        df["label_fine"] = df["label_fine"].map(
            lambda x: relabel_map.get(x, x) if pd.notna(x) else x)
        # When label_fine was remapped to a group name or "other",
        # update label_group to match
        def _update_group(row):
            lf = row["label_fine"]
            if pd.isna(lf):
                return row["label_group"]
            mapped = relabel_map.get(row.get("label_fine_original"), None)
            if mapped and mapped != row.get("label_fine_original"):
                # The mineral was relabelled → group is the new class
                return mapped
            return row["label_group"]
        df["label_group"] = df.apply(_update_group, axis=1)
        return df

    # ──────────────────────────────────────────────────────────────────
    #  §12.7: Label all regions
    # ──────────────────────────────────────────────────────────────────

    def label_all(
        self,
        regions_df: pd.DataFrame,
        specimens_df: pd.DataFrame,
        sites_df: pd.DataFrame,
        segments_dir: Path | None = None,
    ) -> pd.DataFrame:
        """Run the full labelling pipeline (S1–S4).

        Parameters
        ----------
        segments_dir : Path, optional
            Directory containing ``<site_id>_labels.png`` segmentation
            masks.  Used for exact centre-pixel lookup in S3.

        Returns a DataFrame matching §12.7 schema.
        """
        if self.require_status_columns:
            from src.rotapetro.features import V1_STATUS_COLUMNS
            missing = [c for c in V1_STATUS_COLUMNS if c not in regions_df.columns]
            if missing:
                raise ValueError(f"Legacy features missing required V1 status columns: {missing}")

        # Build specimen → mineral lists mapping
        spec_minerals: dict[str, tuple[list[str], list[str]]] = {}
        for _, row in specimens_df.iterrows():
            code = str(row["code"]).strip()
            rf = _parse_semicolon(row.get("rock_forming", ""))
            ac = _parse_semicolon(row.get("accessory", ""))
            spec_minerals[code] = (rf, ac)

        # Build site → code mapping
        site_to_code: dict[str, str] = {}
        for _, row in sites_df.iterrows():
            site_to_code[str(row["site_id"])] = str(row["code"])

        all_minerals = self.kb.all_minerals()

        # ── S1: Rule engine labels ──────────────────────────────────
        all_labels: dict[str, dict] = {}

        for _, r in regions_df.iterrows():
            rid = r["region_id"]
            site_id = r["site_id"]
            code = site_to_code.get(str(site_id), r.get("code", ""))
            rdict = r.to_dict()

            # Candidate set from specimen
            rf, ac = spec_minerals.get(code, ([], []))
            candidates = self.kb.get_candidate_set(rf, ac)
            candidates_unconstrained = set(all_minerals)

            # Score with constraints (S1)
            label = self.label_region(rdict, candidates)

            # Score unconstrained (for baselines)
            result_unc = self.score_region(rdict, candidates_unconstrained)

            label["rule_top1_unconstrained"] = result_unc["top1"]
            label["rule_p1_unconstrained"] = round(result_unc["p1"], 4)
            # Store candidate set as list for parquet (Fix #8)
            label["in_candidate_set"] = sorted(candidates)
            label["region_id"] = rid

            all_labels[rid] = label

        # ── S2: Monomineralic specimens (Fix #7: authoritative) ─────
        if not self.disable_mono:
            mono_specs = specimens_df[
                specimens_df.get("is_monomineralic", pd.Series(dtype=bool))
                == True
            ]
            failed_mono_codes: set[str] = set()

            for _, spec_row in mono_specs.iterrows():
                code = str(spec_row["code"])
                rf = _parse_semicolon(spec_row.get("rock_forming", ""))
                canon = {self.kb.resolve_alias(m) for m in rf}
                canon = {c for c, lvl in canon if lvl != "group"}
                canonical = None
                if len(canon) == 1:
                    canonical = next(iter(canon))
                else:
                    # flagged by rock name (Phase 0 cond. 2): quartzite -> quartz ...
                    name = (str(spec_row.get("title", "")).split(" - ")[0] + " "
                            + str(spec_row.get("category", ""))).lower()
                    for rock, min_ in _MONO_ROCK_MINERAL.items():
                        if rock in name:
                            canonical = min_
                            break
                if canonical is None:
                    continue

                spec_sites = {
                    sid for sid, c in site_to_code.items() if c == code}

                mono_labels, sanity_ok = self.label_monomineralic(
                    regions_df, canonical, spec_sites)

                if not sanity_ok:
                    failed_mono_codes.add(code)
                    continue

                # Fix #7: mono is authoritative — always overrides S1
                for _, ml in mono_labels.iterrows():
                    rid = ml["region_id"]
                    if rid in all_labels:
                        all_labels[rid].update({
                            "label_fine": ml["label_fine"],
                            "label_group": ml["label_group"],
                            "label_source": ml["label_source"],
                            "label_conf": ml["label_conf"],
                        })

            # Record failed mono validations (Fix #7 auditability)
            if failed_mono_codes:
                logger.info("Monomineralic sanity check failed for %d "
                            "specimens: %s", len(failed_mono_codes),
                            sorted(failed_mono_codes))

        # ── S3: Rotation titles (Fix #1: separate columns) ─────────
        for _, site_row in sites_df.iterrows():
            title_minerals = site_row.get("title_minerals")
            if not title_minerals or _is_nan(title_minerals):
                continue
            tm = _parse_semicolon(title_minerals)
            if len(tm) != 1:
                continue

            mineral_raw = tm[0]
            canonical, level = self.kb.resolve_alias(mineral_raw)
            if level == "group":
                continue

            site_id = str(site_row["site_id"])

            # Fix #2: load segmentation mask for exact pixel lookup
            label_img = None
            if segments_dir is not None:
                seg_path = segments_dir / f"{site_id}.png"
                if seg_path.is_file():
                    try:
                        import imageio.v3 as iio
                        label_img = iio.imread(seg_path).astype(np.uint16)
                    except Exception:
                        pass

            title_rid = self.find_title_region(
                regions_df, site_id, label_img)

            if title_rid and title_rid in all_labels:
                all_labels[title_rid]["title_label_fine"] = canonical
                all_labels[title_rid]["title_label_group"] = (
                    self.kb.get_group(canonical))
                all_labels[title_rid]["title_label_source"] = "title"

        # ── Convert to DataFrame ────────────────────────────────────
        labels_list = list(all_labels.values())
        labels_df = pd.DataFrame(labels_list)

        # Ensure all §12.7 columns exist (V1: add deciding_evidence, n_optical_features)
        for col in ("rule_top1", "rule_p1", "rule_top2", "rule_p2",
                     "rule_top1_unconstrained", "rule_p1_unconstrained",
                     "rule_evidence_json", "deciding_evidence",
                     "n_optical_features",
                     "label_fine", "label_group", "label_source",
                     "label_conf", "in_candidate_set",
                     "title_label_fine", "title_label_group",
                     "title_label_source"):
            if col not in labels_df.columns:
                labels_df[col] = None

        return labels_df

    # ──────────────────────────────────────────────────────────────────
    #  Helpers
    # ──────────────────────────────────────────────────────────────────

    def _pleo_adjacent(
        self, observed: str, expected: list[str],
    ) -> bool:
        """Check if observed pleochroism is one level away from expected."""
        try:
            obs_idx = self._pleo_levels.index(observed)
        except ValueError:
            return False
        for e in expected:
            try:
                exp_idx = self._pleo_levels.index(e)
                if abs(obs_idx - exp_idx) == 1:
                    return True
            except ValueError:
                continue
        return False


# ScoreEngine alias for RuleEngine
ScoreEngine = RuleEngine


# ══════════════════════════════════════════════════════════════════════════
#  Module-level helpers
# ══════════════════════════════════════════════════════════════════════════

_MONO_ROCK_MINERAL = {"quartzite": "quartz", "marble": "calcite",
                      "dolostone": "dolomite", "dunite": "olivine",
                      "anorthosite": "plagioclase", "chert": "fine_aggregate",
                      "limestone": "calcite"}


def _is_nan(val) -> bool:
    """Check if a value is NaN, None, empty, or 'nan'.

    V1 FIX: 'none' is NOT treated as NaN — it is a valid value
    (e.g. twin_type='none' means 'no twins detected').
    """
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "nan"):
        return True
    return False


def _safe_float(val, default: float = float('nan')) -> float:
    """Convert value to float, returning default on failure.

    V1 FIX: default changed from 0.0 to NaN so that missing angular
    values are not silently treated as measured-zero.
    """
    if _is_nan(val):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _parse_semicolon(val) -> list[str]:
    """Parse a semicolon-separated string to a list of stripped strings."""
    if _is_nan(val):
        return []
    return [v.strip().lower() for v in str(val).split(";") if v.strip()]
