"""Unit tests for V1 rule engine repair (Step 4 / CHECK 4).

Tests:
1. A region with twin_type="none" and no twin_status column gets no twinning term.
2. With disabled_terms set, birefringence, extinction_type and relief never appear in evidence.
3. Orthoclase 0.45 + sanidine 0.45 (fake posteriors) gives singleton feldspar.
4. Quartz 0.5 + plagioclase 0.45 gives partial set [feldspar, quartz].
5. Tourmaline 0.3 + biotite 0.7 gives abstain, out_of_scope_plausible.
6. type == "aggregate" gives mixed.
7. Running twice gives identical outputs.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config
from src.rotapetro.knowledge import KnowledgeBase
from src.rotapetro.rules import RuleEngine, V1_STATUS_COLUMNS
import pandas as pd


@pytest.fixture(scope="module")
def engine() -> RuleEngine:
    cfg = load_config("configs/default.yaml")
    kb = KnowledgeBase("knowledge", cfg)
    return RuleEngine(kb, cfg)


def test_twin_type_none_without_twin_status(engine: RuleEngine):
    """A region with twin_type='none' and no twin_status column gets no twinning term."""
    r = {
        "optic_class": "anisotropic",
        "twin_type": "none",
        "area_px": 2000,
    }
    res = engine.score_region(r, {"plagioclase"})
    plag_ev = res["evidence"].get("plagioclase", {})
    assert "twinning" not in plag_ev, (
        f"Expected no twinning term without twin_status, got: {plag_ev}"
    )


def test_disabled_terms_never_appear_in_evidence(engine: RuleEngine):
    """With disabled_terms set, birefringence, extinction_type and relief never appear in evidence."""
    r = {
        "optic_class": "anisotropic",
        "optic_class_status": "observed",
        "biref_cls": 1,
        "ext_type": "parallel",
        "extinction_status": "observed",
        "relief_cls": "low",
        "aspect": 2.0,
        "n_cleavage_dirs": 1,
        "area_px": 2000,
    }
    res = engine.score_region(r, {"quartz", "plagioclase", "biotite"})
    for mineral, ev in res["evidence"].items():
        assert "birefringence" not in ev, f"birefringence found in evidence for {mineral}"
        assert "extinction_type" not in ev, f"extinction_type found in evidence for {mineral}"
        assert "relief" not in ev, f"relief found in evidence for {mineral}"


def test_orthoclase_sanidine_singleton_feldspar(engine: RuleEngine):
    """Orthoclase 0.45 + sanidine 0.45 (fake posteriors) gives singleton feldspar."""
    region = {"type": "grain"}
    candidates = {"orthoclase", "sanidine"}
    fake_res = {
        "posteriors": {"orthoclase": 0.45, "sanidine": 0.45},
        "top1": "orthoclase",
        "evidence": {
            "orthoclase": {"ppl_colour": -0.1},
            "sanidine": {"ppl_colour": -0.1},
        },
    }
    decision = engine.family_decision(region, candidates, res=fake_res)
    assert decision["state"] == "singleton", f"Expected singleton, got {decision['state']}"
    assert decision["allowed"] == ["feldspar"], f"Expected ['feldspar'], got {decision['allowed']}"


def test_quartz_plagioclase_partial_set(engine: RuleEngine):
    """Quartz 0.5 + plagioclase 0.45 gives partial set [feldspar, quartz]."""
    region = {"type": "grain"}
    candidates = {"quartz", "plagioclase"}
    fake_res = {
        "posteriors": {"quartz": 0.5, "plagioclase": 0.45},
        "top1": "quartz",
        "evidence": {
            "quartz": {"undulose": -0.2},
            "plagioclase": {"twinning": -0.3},
        },
    }
    decision = engine.family_decision(region, candidates, res=fake_res)
    assert decision["state"] == "partial_set", f"Expected partial_set, got {decision['state']}"
    assert decision["allowed"] == ["feldspar", "quartz"], (
        f"Expected ['feldspar', 'quartz'], got {decision['allowed']}"
    )


def test_tourmaline_biotite_out_of_scope_abstain(engine: RuleEngine):
    """Tourmaline 0.3 + biotite 0.7 gives abstain, out_of_scope_plausible."""
    region = {"type": "grain"}
    candidates = {"tourmaline", "biotite"}
    fake_res = {
        "posteriors": {"tourmaline": 0.3, "biotite": 0.7},
        "top1": "biotite",
        "evidence": {
            "biotite": {"pleochroism": -0.1},
            "tourmaline": {"pleochroism": -0.1},
        },
    }
    decision = engine.family_decision(region, candidates, res=fake_res)
    assert decision["state"] == "abstain", f"Expected abstain, got {decision['state']}"
    assert decision["reason"] == "out_of_scope_plausible", (
        f"Expected out_of_scope_plausible, got {decision.get('reason')}"
    )


def test_type_aggregate_gives_mixed(engine: RuleEngine):
    """type == 'aggregate' gives mixed."""
    region = {"type": "aggregate"}
    candidates = {"quartz", "plagioclase"}
    decision = engine.family_decision(region, candidates)
    assert decision["state"] == "mixed", f"Expected mixed, got {decision['state']}"
    assert decision["reason"] == "aggregate_or_fine"


def test_running_twice_gives_identical_outputs(engine: RuleEngine):
    """Running twice gives identical outputs."""
    region = {
        "region_id": "test_r1",
        "type": "grain",
        "optic_class": "anisotropic",
        "optic_class_status": "observed",
        "pleo_cls": "none",
        "pleo_status": "not_detected_with_bound",
        "ppl_colour": "colourless",
        "twin_type": "none",
        "twin_status": "unresolved",
        "undulose_score": 1.0,
        "aspect": 1.2,
        "area_px": 1500,
        "eligible_v1": True,
        "r2_4_med": 0.8,
    }
    candidates = {"quartz", "plagioclase", "biotite", "orthoclase"}

    res1 = engine.score_region(region, candidates)
    res2 = engine.score_region(region, candidates)
    assert res1 == res2, "score_region is not deterministic"

    dec1 = engine.family_decision(region, candidates)
    dec2 = engine.family_decision(region, candidates)
    assert dec1 == dec2, "family_decision is not deterministic"

    eval1 = engine.evaluate_region_v1(region, candidates)
    eval2 = engine.evaluate_region_v1(region, candidates)
    assert eval1 == eval2, "evaluate_region_v1 is not deterministic"


def test_missing_status_columns_raise(engine: RuleEngine):
    """Test E17: Verify RuleEngine.label_all raises ValueError when require_status_columns is True and status columns are missing."""
    import pandas as pd
    from src.rotapetro.features import V1_STATUS_COLUMNS

    assert engine.require_status_columns is True

    # Minimal legacy regions dataframe missing V1 status columns
    legacy_regions = pd.DataFrame([
        {
            "region_id": "test_r1",
            "type": "grain",
            "code": "test_spec",
            "site_id": "test_site",
            "area_px": 1000,
            "optic_class": "anisotropic",
        }
    ])
    specimens_df = pd.DataFrame([{"code": "test_spec", "rock_forming": "quartz"}])
    sites_df = pd.DataFrame([{"site_id": "test_site", "code": "test_spec"}])

    with pytest.raises(ValueError, match="missing required V1 status columns"):
        engine.label_all(legacy_regions, specimens_df, sites_df)

    # Adding all required status columns allows label_all to proceed
    valid_regions = legacy_regions.copy()
    for col in V1_STATUS_COLUMNS:
        valid_regions[col] = "observed"
    valid_regions["eligible_v1"] = True

    labels_df = engine.label_all(valid_regions, specimens_df, sites_df)
    assert len(labels_df) == 1



def test_score_region_deterministic_candidate_sorting(engine: RuleEngine):
    """Candidate set iteration in score_region is deterministic regardless of set hashing order (E10)."""
    region = {
        "region_id": "test_r1",
        "type": "grain",
        "optic_class": "anisotropic",
        "optic_class_status": "observed",
        "ppl_colour": "colourless",
        "area_px": 1500,
    }
    cand_list = ["quartz", "plagioclase", "biotite", "muscovite", "augite", "orthoclase"]
    set_a = set(cand_list)
    set_b = set(reversed(cand_list))

    res_a = engine.score_region(region, set_a)
    res_b = engine.score_region(region, set_b)

    assert res_a["scores"] == res_b["scores"], "Scores dictionary differs between candidate sets"
    assert res_a["posteriors"] == res_b["posteriors"], "Posteriors dictionary differs between candidate sets"
    assert res_a["top1"] == res_b["top1"], "Top1 differs between candidate sets"


def test_cleavage_and_absorption_axis_disabled(engine: RuleEngine):
    """Cleavage and absorption_axis are in disabled_terms and do not contribute to scores or info_terms."""
    assert "cleavage" in engine.disabled_terms, "cleavage should be in disabled_terms"
    assert "absorption_axis" in engine.disabled_terms or "absorption" in engine.disabled_terms, (
        "absorption_axis should be in disabled_terms"
    )

    region = {
        "region_id": "test_r_disabled",
        "type": "grain",
        "optic_class": "anisotropic",
        "optic_class_status": "observed",
        "n_cleavage_dirs": 1,
        "cleavage_status": "observed",
        "cleavage_angle_deg": 60.0,
        "pleo_cls": "strong",
        "pleo_status": "observed",
        "aspect": 2.5,
        "absorption_status": "observed",
        "absorption_vs_long_deg": 15.0,
        "area_px": 2000,
        "eligible_v1": True,
        "r2_4_med": 0.8,
    }
    candidates = {"biotite", "augite", "muscovite", "quartz"}
    res = engine.score_region(region, candidates)

    for mineral, ev in res["evidence"].items():
        assert "cleavage" not in ev, f"cleavage found in evidence for {mineral}: {ev}"
        assert "absorption_axis" not in ev, f"absorption_axis found in evidence for {mineral}: {ev}"
        assert "absorption" not in ev, f"absorption found in evidence for {mineral}: {ev}"

    decision = engine.family_decision(region, candidates, res=res)
    info_terms = decision.get("info_terms", [])
    assert "cleavage" not in info_terms, f"cleavage found in info_terms: {info_terms}"
    assert "absorption_axis" not in info_terms, f"absorption_axis found in info_terms: {info_terms}"
    assert "absorption" not in info_terms, f"absorption found in info_terms: {info_terms}"


def test_unlisted_ignored_in_competitive_species(engine: RuleEngine):
    """__unlisted__ is ignored when determining competitive species (best_out) for positive cues."""
    region = {"type": "grain"}
    candidates = {"quartz", "biotite"}
    # Construct fake posteriors where __unlisted__ has higher posterior than out-family biotite:
    # quartz: 0.85 (family quartz), __unlisted__: 0.10, biotite: 0.05 (family mica)
    fake_res = {
        "posteriors": {"quartz": 0.85, "__unlisted__": 0.10, "biotite": 0.05},
        "top1": "quartz",
        "evidence": {
            "quartz": {"ppl_colour": 0.0, "undulose": 0.0},
            "biotite": {"ppl_colour": 0.2, "undulose": -2.0},
            "__unlisted__": {},
        },
    }
    decision = engine.family_decision(region, candidates, res=fake_res)

    # If __unlisted__ were selected as competitor (0.10 > 0.05), its evidence would be empty (-1e9),
    # so quartz's ppl_colour (0.0) would beat -1e9 and be counted as an informative cue.
    # But when __unlisted__ is correctly ignored, best_out is biotite (0.2 > 0.0), so ppl_colour is NOT informative.
    # undulose (quartz 0.0 > biotite -2.0) remains an informative cue.
    assert decision["state"] == "singleton"
    assert "ppl_colour" not in decision["info_terms"], (
        f"ppl_colour should be beaten by biotite (0.2 > 0.0): {decision['info_terms']}"
    )
    assert "undulose" in decision["info_terms"], (
        f"undulose should be included in info_terms: {decision['info_terms']}"
    )
    assert decision["n_info"] == 1


def test_is_match_cue(engine: RuleEngine):
    """_is_match_cue returns True iff level == 'match' and w > 0."""
    assert engine._is_match_cue("pleochroism", "match", 1.2) is True
    assert engine._is_match_cue("pleochroism", "match") is True
    assert engine._is_match_cue("pleochroism", "partial", 1.2) is False
    assert engine._is_match_cue("pleochroism", "neutral", 1.2) is False
    assert engine._is_match_cue("pleochroism", "mismatch", 1.2) is False
    assert engine._is_match_cue("pleochroism", "match", 0.0) is False
    assert engine._is_match_cue("pleochroism", "match", -0.5) is False


def test_list_conflict_policies(engine: RuleEngine, monkeypatch):
    """evaluate_region_v1 handles both list_conflict_policy 'abstain' and 'keep_optics'."""
    region = {
        "region_id": "test_r_conflict",
        "type": "grain",
        "optic_class": "anisotropic",
        "eligible_v1": True,
        "r2_4_med": 0.8,
    }
    candidates = {"biotite", "muscovite"}
    all_minerals = set(engine.kb.all_minerals())

    def mock_family_decision(r, cands, res=None):
        if set(cands) == all_minerals:
            return {
                "state": "singleton",
                "allowed": ["quartz"],
                "reason": "",
                "fam_p": {"quartz": 0.9, "mica": 0.05, "feldspar": 0.05},
                "p_out": 0.0,
                "n_info": 1,
                "info_terms": ["undulose"],
                "top1": "quartz",
                "p1": 0.9,
                "deciding_evidence": "undulose",
            }
        return {
            "state": "singleton",
            "allowed": ["mica"],
            "reason": "",
            "fam_p": {"quartz": 0.0, "mica": 0.95, "feldspar": 0.05},
            "p_out": 0.0,
            "n_info": 1,
            "info_terms": ["pleochroism"],
            "top1": "biotite",
            "p1": 0.95,
            "deciding_evidence": "pleochroism",
        }

    monkeypatch.setattr(engine, "family_decision", mock_family_decision)

    # 1. Policy: abstain (default)
    engine.list_conflict_policy = "abstain"
    res_abstain = engine.evaluate_region_v1(region, candidates)
    assert res_abstain["optics_list_conflict"] is True
    assert res_abstain["state"] == "abstain"
    assert res_abstain["allowed"] == []
    assert res_abstain["reason"] == "list_conflict"
    assert res_abstain["optics_state"] == "singleton"
    assert res_abstain["optics_allowed"] == ["quartz"]
    assert res_abstain["metadata_dependent"] is True

    # 2. Policy: keep_optics
    engine.list_conflict_policy = "keep_optics"
    res_keep = engine.evaluate_region_v1(region, candidates)
    assert res_keep["optics_list_conflict"] is True
    assert res_keep["state"] == "singleton"
    assert res_keep["allowed"] == ["quartz"]
    assert res_keep["reason"] == "list_conflict_optics_kept"
    assert res_keep["optics_state"] == "singleton"
    assert res_keep["optics_allowed"] == ["quartz"]
    assert res_keep["metadata_dependent"] is False

    # Restore default
    engine.list_conflict_policy = "abstain"


def test_out_of_scope_conflict(engine: RuleEngine, monkeypatch):
    """Optics singleton conflicting with an out-of-scope metadata abstain."""
    region = {
        "region_id": "test_r_oos",
        "type": "grain",
        "optic_class": "anisotropic",
        "eligible_v1": True,
        "r2_4_med": 0.8,
    }
    candidates = {"tourmaline"}
    all_minerals = set(engine.kb.all_minerals())

    def mock_family_decision(r, cands, res=None):
        if set(cands) == all_minerals:
            return {
                "state": "singleton",
                "allowed": ["quartz"],
                "reason": "",
                "fam_p": {"quartz": 0.85},
                "p_out": 0.05,
                "n_info": 1,
                "info_terms": ["undulose"],
                "top1": "quartz",
                "p1": 0.85,
                "deciding_evidence": "undulose",
            }
        return {
            "state": "abstain",
            "allowed": [],
            "reason": "out_of_scope_plausible",
            "fam_p": {},
            "p_out": 0.95,
            "n_info": 0,
            "info_terms": [],
            "top1": "tourmaline",
            "p1": 0.95,
            "deciding_evidence": None,
        }

    monkeypatch.setattr(engine, "family_decision", mock_family_decision)

    engine.list_conflict_policy = "abstain"
    res = engine.evaluate_region_v1(region, candidates)
    assert res["optics_list_conflict"] is True
    assert res["state"] == "abstain"
    # When meta is already abstain, list_conflict_policy="abstain" preserves meta's reason
    assert res["reason"] == "out_of_scope_plausible"
    assert res["optics_state"] == "singleton"
    assert res["optics_allowed"] == ["quartz"]

    engine.list_conflict_policy = "keep_optics"
    res_keep = engine.evaluate_region_v1(region, candidates)
    assert res_keep["optics_list_conflict"] is True
    assert res_keep["state"] == "singleton"
    assert res_keep["allowed"] == ["quartz"]
    assert res_keep["reason"] == "list_conflict_optics_kept"
    assert res_keep["top1"] == "quartz"
    assert res_keep["p1"] == 0.85
    assert res_keep["info_terms"] == ["undulose"]
    assert res_keep["fam_p"] == {"quartz": 0.85}
    assert res_keep["meta_top1"] == "tourmaline"
    assert res_keep["meta_p1"] == 0.95
    assert res_keep["meta_reason"] == "out_of_scope_plausible"
    assert res_keep["meta_info_terms"] == []

    engine.list_conflict_policy = "abstain"


def test_list_conflict_reason_conditioned_on_meta_state(engine: RuleEngine, monkeypatch):
    """Under list_conflict_policy='abstain', reason='list_conflict' only if meta is singleton or partial_set."""
    region = {
        "region_id": "test_r_cond",
        "type": "grain",
        "optic_class": "anisotropic",
        "eligible_v1": True,
        "r2_4_med": 0.8,
    }
    candidates = {"feldspar"}
    all_minerals = set(engine.kb.all_minerals())

    optics_res = {
        "state": "singleton",
        "allowed": ["quartz"],
        "reason": "",
        "top1": "quartz",
        "p1": 0.9,
        "deciding_evidence": "undulose",
    }

    # Case 1: meta is partial_set -> reason is list_conflict
    def mock_partial(r, cands, res=None):
        if set(cands) == all_minerals:
            return optics_res
        return {
            "state": "partial_set",
            "allowed": ["feldspar", "amphibole"],
            "reason": "multimodal",
            "top1": "feldspar",
            "p1": 0.5,
            "deciding_evidence": None,
        }

    monkeypatch.setattr(engine, "family_decision", mock_partial)
    res_partial = engine.evaluate_region_v1(region, candidates)
    assert res_partial["optics_list_conflict"] is True
    assert res_partial["state"] == "abstain"
    assert res_partial["reason"] == "list_conflict"

    # Case 2: meta is abstain (e.g. diffuse) -> keep meta's reason 'diffuse'
    def mock_abstain(r, cands, res=None):
        if set(cands) == all_minerals:
            return optics_res
        return {
            "state": "abstain",
            "allowed": [],
            "reason": "diffuse",
            "top1": "feldspar",
            "p1": 0.3,
            "deciding_evidence": None,
        }

    monkeypatch.setattr(engine, "family_decision", mock_abstain)
    res_diffuse = engine.evaluate_region_v1(region, candidates)
    assert res_diffuse["optics_list_conflict"] is True
    assert res_diffuse["state"] == "abstain"
    assert res_diffuse["reason"] == "diffuse"


def test_hard_contradiction_observed_vs_unresolved(engine: RuleEngine):
    """Observed pleochroism on quartz triggers hard_contradiction; unresolved pleochroism does not."""
    # 1. Observed pleochroism on quartz grain -> hard_contradiction
    region_observed_pleo = {
        "region_id": "r_pleo_quartz",
        "type": "grain",
        "optic_class": "anisotropic",
        "optic_class_status": "observed",
        "pleo_cls": "strong",
        "pleo_status": "observed",
        "ppl_colour": "colourless",
        "aspect": 1.2,
        "area_px": 2000,
        "eligible_v1": True,
        "r2_4_med": 0.8,
    }
    candidates = {"quartz"}
    res_obs = engine.evaluate_region_v1(region_observed_pleo, candidates)
    assert res_obs["state"] == "abstain", f"Expected abstain, got {res_obs['state']}"
    assert res_obs["reason"] == "hard_contradiction", f"Expected hard_contradiction, got {res_obs['reason']}"
    assert res_obs["deciding_evidence"] == "pleochroism"

    # 2. Unresolved pleochroism on quartz grain -> no hard_contradiction
    region_unresolved_pleo = {
        "region_id": "r_unres_quartz",
        "type": "grain",
        "optic_class": "anisotropic",
        "optic_class_status": "observed",
        "pleo_cls": "strong",
        "pleo_status": "unresolved",
        "ppl_colour": "colourless",
        "aspect": 1.2,
        "area_px": 2000,
        "eligible_v1": True,
        "r2_4_med": 0.8,
    }
    res_unres = engine.evaluate_region_v1(region_unresolved_pleo, candidates)
    assert res_unres["reason"] != "hard_contradiction", (
        f"Unresolved pleochroism should not cause hard_contradiction, got {res_unres['reason']}"
    )


def test_hard_contradiction_multi_family_allowed_set(engine: RuleEngine):
    """Hard-contradiction check tests across every family in final allowed set:
    - {feldspar, mica} with strong pleochroism observed -> abstain;
    - the same with pleo_status unresolved -> partial_set kept;
    - a family with one compatible species -> kept.
    """
    candidates = {"plagioclase", "orthoclase", "biotite", "muscovite"}

    r_observed = {
        "region_id": "r_obs",
        "type": "grain",
        "optic_class": "anisotropic",
        "optic_class_status": "observed",
        "ppl_colour": "brown",
        "pleo_cls": "strong",
        "pleo_status": "observed",
        "aspect": 1.2,
        "area_px": 2000,
        "eligible_v1": True,
        "r2_4_med": 0.8,
    }

    # 1. {feldspar, mica} with strong pleochroism observed -> abstain (hard_contradiction)
    # Feldspar has no pleochroic species; all its candidate species mismatch pleochroism
    fake_res_obs = {
        "posteriors": {"biotite": 0.50, "plagioclase": 0.45},
        "top1": "biotite",
        "p1": 0.50,
        "evidence": {
            "biotite": {"pleochroism": 0.5, "ppl_colour": 0.4},
            "plagioclase": {"twinning": 0.3},
        },
        "evidence_level": {
            "biotite": {"pleochroism": "match", "ppl_colour": "match"},
            "plagioclase": {"pleochroism": "mismatch", "ppl_colour": "mismatch"},
        },
    }
    res_obs = engine.family_decision(r_observed, candidates, res=fake_res_obs)
    assert res_obs["state"] == "abstain", f"Expected abstain, got {res_obs['state']}"
    assert res_obs["reason"] == "hard_contradiction"
    assert res_obs["deciding_evidence"] == "pleochroism"

    # 2. Same with pleo_status unresolved -> partial_set kept
    r_unresolved = dict(r_observed)
    r_unresolved["pleo_status"] = "unresolved"
    fake_res_unres = {
        "posteriors": {"biotite": 0.50, "plagioclase": 0.45},
        "top1": "biotite",
        "p1": 0.50,
        "evidence": {
            "biotite": {"ppl_colour": 0.4},
            "plagioclase": {"twinning": 0.3},
        },
        "evidence_level": {
            "biotite": {"ppl_colour": "match"},
            "plagioclase": {"ppl_colour": "neutral"},
        },
    }
    res_unres = engine.family_decision(r_unresolved, candidates, res=fake_res_unres)
    assert res_unres["state"] == "partial_set", f"Expected partial_set, got {res_unres['state']}"
    assert set(res_unres["allowed"]) == {"feldspar", "mica"}

    # 3. A family with one compatible species is kept (not vetoed)
    # For mica candidates: biotite (matches pleo) + muscovite (mismatches)
    # Since biotite is compatible, mica is kept as singleton
    fake_res_mica = {
        "posteriors": {"biotite": 0.85, "muscovite": 0.15},
        "top1": "biotite",
        "p1": 0.85,
        "evidence": {
            "biotite": {"pleochroism": 0.5, "ppl_colour": 0.4},
            "muscovite": {},
        },
        "evidence_level": {
            "biotite": {"pleochroism": "match", "ppl_colour": "match"},
            "muscovite": {"pleochroism": "mismatch", "ppl_colour": "mismatch"},
        },
    }
    res_mica = engine.family_decision(r_observed, {"biotite", "muscovite"}, res=fake_res_mica)
    assert res_mica["state"] == "singleton", f"Expected singleton for mica, got {res_mica['state']}"
    assert res_mica["allowed"] == ["mica"]


# ── Checklist Aliases (E-tests in V1_CODE_CHECKLIST.md) ─────────────────
test_no_twin_absence_without_status = test_twin_type_none_without_twin_status
test_disabled_terms_never_scored = test_disabled_terms_never_appear_in_evidence
test_family_singleton_from_tie = test_orthoclase_sanidine_singleton_feldspar
test_partial_set = test_quartz_plagioclase_partial_set
test_out_of_scope_abstain = test_tourmaline_biotite_out_of_scope_abstain
test_aggregate_is_mixed = test_type_aggregate_gives_mixed
test_deterministic = test_running_twice_gives_identical_outputs
test_list_conflict = test_list_conflict_policies


def test_single_angle_view(engine: RuleEngine):
    """Test single_angle_view creates correct single-angle dict without mutating original."""
    import math

    orig = {
        "region_id": "r100",
        "type": "grain",
        "area_px": 2500,
        "eligible_v1": True,
        "optic_class": "anisotropic",
        "optic_class_status": "observed",
        "k1_optic_class": "opaque",
        "k1_optic_class_status": "observed",
        "ppl_colour": "brown",
        "k1_ppl_colour": "colourless",
        "ppl_chroma": 15.0,
        "k1_ppl_chroma": 2.1,
        "ppl_L": 65.0,
        "k1_ppl_L": 82.0,
        "ppl_a": 4.0,
        "k1_ppl_a": 0.5,
        "ppl_b": 12.0,
        "k1_ppl_b": 1.2,
        "ppl_hue_deg": 45.0,
        "k1_ppl_hue_deg": 12.0,
        # rotation fields
        "pleo_cls": "moderate",
        "pleo_status": "observed",
        "pleo_delta_e": 12.4,
        "twin_type": "lamellar",
        "twin_status": "observed",
        "ext_type": "inclined",
        "extinction_status": "observed",
        "ext_vs_long_deg": 25.0,
        "ext_bisects_cleavage": False,
        "undulose_score": 5.4,
        "undulose_range_deg": 18.0,
        "absorption_status": "observed",
        "absorption_vs_long_deg": 15.0,
        "relief_mod_med": 0.35,
        "biref_cls": 2,
    }

    orig_copy = dict(orig)
    view = engine.single_angle_view(orig)

    # 1. Original not mutated
    assert orig == orig_copy

    # 2. k1_* replacements
    assert view["optic_class"] == "opaque"
    assert view["optic_class_status"] == "observed"
    assert view["ppl_colour"] == "colourless"
    assert view["ppl_chroma"] == 2.1
    assert view["ppl_L"] == 82.0
    assert view["ppl_a"] == 0.5
    assert view["ppl_b"] == 1.2
    assert view["ppl_hue_deg"] == 12.0

    # 3. Preserved attributes
    assert view["region_id"] == "r100"
    assert view["type"] == "grain"
    assert view["area_px"] == 2500
    assert view["eligible_v1"] is True

    # 4. Rotation-only fields are NaN / unresolved
    assert math.isnan(view["pleo_cls"])
    assert view["pleo_status"] == "unresolved"
    assert math.isnan(view["pleo_delta_e"])
    assert math.isnan(view["twin_type"])
    assert view["twin_status"] == "unresolved"
    assert math.isnan(view["ext_type"])
    assert view["extinction_status"] == "unresolved"
    assert math.isnan(view["ext_vs_long_deg"])
    assert math.isnan(view["ext_bisects_cleavage"])
    assert math.isnan(view["undulose_score"])
    assert math.isnan(view["undulose_range_deg"])
    assert view["absorption_status"] == "unresolved"
    assert math.isnan(view["absorption_vs_long_deg"])
    assert math.isnan(view["relief_mod_med"])
    assert math.isnan(view["biref_cls"])

    # 5. RuleEngine class method call works
    view_cls = RuleEngine.single_angle_view(orig)
    assert view_cls["optic_class"] == "opaque"
    assert view_cls["pleo_status"] == "unresolved"


def test_compute_static_category(engine: RuleEngine):
    """Test compute_static_category per CONTRACTS.md Section 3 across all categories."""
    fn = engine.compute_static_category

    # 1. reproduced
    assert fn("singleton", ["quartz"], "singleton", ["quartz"]) == "reproduced"
    assert fn("partial_set", ["feldspar", "quartz"], "partial_set", ["quartz", "feldspar"]) == "reproduced"

    # 2. full_only
    assert fn("singleton", ["quartz"], "abstain", []) == "full_only"
    assert fn("partial_set", ["feldspar", "quartz"], "abstain", []) == "full_only"

    # 3. k1_broader (full set ⊂ k1 set)
    assert fn("singleton", ["quartz"], "partial_set", ["quartz", "feldspar"]) == "k1_broader"

    # 4. k1_narrower (k1 set ⊂ full set)
    assert fn("partial_set", ["quartz", "feldspar"], "singleton", ["quartz"]) == "k1_narrower"

    # 5. conflict (disjoint sets, or full is labelled and k1 is mixed)
    assert fn("singleton", ["quartz"], "singleton", ["feldspar"]) == "conflict"
    assert fn("partial_set", ["feldspar", "quartz"], "partial_set", ["mica", "pyroxene"]) == "conflict"
    assert fn("singleton", ["quartz"], "mixed", []) == "conflict"
    assert fn("partial_set", ["quartz", "feldspar"], "mixed", []) == "conflict"

    # 6. k1_only
    assert fn("abstain", [], "singleton", ["quartz"]) == "k1_only"
    assert fn("abstain", [], "partial_set", ["feldspar", "quartz"]) == "k1_only"

    # 7. unresolved_both
    assert fn("abstain", [], "abstain", []) == "unresolved_both"

    # 8. n/a
    assert fn("excluded", [], "singleton", ["quartz"]) == "n/a"
    assert fn("excluded", [], "abstain", []) == "n/a"

def test_low_r2_4_population_restriction_comment_and_behavior(engine: RuleEngine):
    """Check that low_r2_4 exclusion is documented as a population restriction and functions properly."""
    doc = engine.check_quality_gate.__doc__ or ""
    assert "declared population restriction" in doc
    assert "results are conditional on this population" in doc

    # Anisotropic with low r2_4 is gated
    aniso_low = {"eligible_v1": True, "optic_class": "anisotropic", "r2_4_med": 0.2}
    ok, reason = engine.check_quality_gate(aniso_low)
    assert ok is False
    assert reason == "low_r2_4"

    # Anisotropic with high r2_4 passes
    aniso_high = {"eligible_v1": True, "optic_class": "anisotropic", "r2_4_med": 0.75}
    ok, reason = engine.check_quality_gate(aniso_high)
    assert ok is True
    assert reason == ""

    # Isotropic with low r2_4 passes (extinction fit not required for isotropic)
    iso_low = {"eligible_v1": True, "optic_class": "isotropic", "r2_4_med": 0.1}
    ok, reason = engine.check_quality_gate(iso_low)
    assert ok is True






