"""Unit tests for V1 configuration validation (§1, §2).

Tests:
- Test B2: Check that v1.families in configs/default.yaml has no overlapping species
  across families, and that set(families.keys()) == set(primary_classes).
- Test B5: Test that validate_v1_config raises ValueError when a species appears in
  more than one family.
- Test that validate_v1_config succeeds on the valid default config.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config, validate_v1_config


def test_default_config_validates():
    """Verify that validate_v1_config succeeds on the valid default config."""
    cfg = load_config("configs/default.yaml")
    validate_v1_config(cfg)


def test_b2_families_disjoint_and_match_primary_classes():
    """Test B2: v1.families has no overlapping species across families and matches primary_classes."""
    cfg = load_config("configs/default.yaml")
    v1 = cfg["v1"]
    families = v1["families"]
    primary_classes = v1["taxonomy"]["primary_classes"]

    # Check set of family keys equals primary_classes
    assert set(families.keys()) == set(primary_classes), (
        f"Families keys {set(families.keys())} do not match primary classes {set(primary_classes)}"
    )

    # Check no overlapping species across families
    all_species = []
    for fam, sp_list in families.items():
        assert isinstance(sp_list, list), f"Family {fam} must map to a list of species"
        assert len(sp_list) > 0, f"Family {fam} has empty species list"
        all_species.extend(sp_list)

    assert len(all_species) == len(set(all_species)), (
        f"Overlapping species found in v1.families: duplicates = {[s for s in all_species if all_species.count(s) > 1]}"
    )


def test_b5_validate_v1_config_raises_on_overlapping_species():
    """Test B5: validate_v1_config raises ValueError when a species appears in more than one family."""
    cfg = load_config("configs/default.yaml")
    bad_cfg = copy.deepcopy(cfg)

    # Inject duplicate species 'plagioclase' into 'quartz' family
    bad_cfg["v1"]["families"]["quartz"].append("plagioclase")

    with pytest.raises(ValueError, match="multiple families|Species 'plagioclase'"):
        validate_v1_config(bad_cfg)


def test_gitattributes_line_endings():
    """Verify that .gitattributes enforces LF for .sh and .py and scripts/run_pipeline.sh has LF."""
    gitattr = _PROJECT_ROOT / ".gitattributes"
    assert gitattr.is_file(), ".gitattributes not found"
    content = gitattr.read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in content, ".gitattributes missing *.sh text eol=lf"
    assert "*.py text eol=lf" in content, ".gitattributes missing *.py text eol=lf"

    pipeline_sh = _PROJECT_ROOT / "scripts" / "run_pipeline.sh"
    raw = pipeline_sh.read_bytes()
    assert b"\r\n" not in raw, "scripts/run_pipeline.sh contains CRLF line endings"

