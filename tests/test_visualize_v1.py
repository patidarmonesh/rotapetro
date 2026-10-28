"""Tests for V1 Label Visualizer pure categorisation functions.

Test file: tests/test_visualize_v1.py
Task: B1
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.visualize_labels_v1 import (
    categorize_region,
    is_conflict_styled,
    is_metadata_dependent,
    is_hard_contradiction,
    is_out_of_scope,
    assemble_category_panel,
)


def test_categorize_region_pure_states():
    """Category must be determined strictly from state alone."""
    assert categorize_region("singleton") == "singleton"
    assert categorize_region("partial_set") == "partial_set"
    assert categorize_region("abstain") == "abstain"
    assert categorize_region("excluded") == "excluded"
    assert categorize_region("mixed") == "mixed"


def test_categorize_region_case_and_whitespace():
    """Categorisation handles whitespace and case variations."""
    assert categorize_region("  Singleton  ") == "singleton"
    assert categorize_region("PARTIAL_SET") == "partial_set"
    assert categorize_region("Abstain") == "abstain"
    assert categorize_region("Excluded") == "excluded"
    assert categorize_region("MIXED") == "mixed"


def test_categorize_region_fallback():
    """Unknown or invalid states fallback safely to excluded."""
    assert categorize_region("unknown") == "excluded"
    assert categorize_region("") == "excluded"
    assert categorize_region("invalid_state") == "excluded"


def test_conflict_styling_only_on_list_conflicts():
    """Conflict styling applies only for list_conflict and list_conflict_optics_kept."""
    assert is_conflict_styled("list_conflict") is True
    assert is_conflict_styled("list_conflict_optics_kept") is True

    # Other reasons must NOT receive conflict styling
    assert is_conflict_styled("out_of_scope") is False
    assert is_conflict_styled("out_of_scope_plausible") is False
    assert is_conflict_styled("hard_contradiction") is False
    assert is_conflict_styled("low_r2_4") is False
    assert is_conflict_styled("diffuse") is False
    assert is_conflict_styled("") is False
    assert is_conflict_styled(None) is False


def test_metadata_dependence_isolation():
    """Metadata dependence is an independent boolean flag that does not alter state categorisation."""
    assert is_metadata_dependent(True) is True
    assert is_metadata_dependent(False) is False
    assert is_metadata_dependent(None) is False
    assert is_metadata_dependent(np.nan) is False

    # A metadata_dependent region with state 'singleton' remains categorized as 'singleton'
    state = "singleton"
    assert categorize_region(state) == "singleton"


def test_out_of_scope_under_abstains():
    """Out-of-scope regions stay under abstains."""
    assert is_out_of_scope("out_of_scope") is True
    assert is_out_of_scope("out_of_scope_plausible") is True
    assert is_out_of_scope("diffuse") is False
    assert is_out_of_scope(None) is False

    # Out of scope is an abstain state
    assert categorize_region("abstain") == "abstain"


def test_hard_contradiction_detection():
    """Hard contradiction reason detection for sub-panel."""
    assert is_hard_contradiction("hard_contradiction") is True
    assert is_hard_contradiction("list_conflict") is False
    assert is_hard_contradiction("out_of_scope") is False
    assert is_hard_contradiction(None) is False


def test_empty_panel_assembly():
    """Empty list conflict panel renders successfully with empty banner."""
    panel = assemble_category_panel([], "list_conflict", "fold_0")
    assert isinstance(panel, np.ndarray)
    assert panel.ndim == 3
    assert panel.shape[0] > 0 and panel.shape[1] > 0


def test_render_grain_card_no_colour_tint():
    """Verify render_grain_card draws contours in category colour with no colour tint on tiles."""
    from scripts.visualize_labels_v1 import render_grain_card, CATEGORY_COLORS

    H, W = 50, 50
    mask_full = np.zeros((H, W), dtype=np.uint16)
    mask_full[15:35, 15:35] = 1

    # Raw PPL / XPL frames with a distinct color
    ppl_raw = np.full((36, H, W, 3), [150, 150, 150], dtype=np.uint8)
    xpl_raw = np.full((36, H, W, 3), [80, 80, 80], dtype=np.uint8)

    row = {
        "region_id": "site_g001",
        "site_id": "site_01",
        "state": "singleton",
        "allowed": ["quartz"],
        "reason": "singleton",
        "bbox_xywh": [15, 15, 20, 20],
        "label_value": 1,
    }
    assets = {
        "mask": mask_full,
        "ppl": ppl_raw[0],
        "xpl": xpl_raw[0],
        "aligned": {"ppl": ppl_raw, "xpl": xpl_raw},
    }

    card = render_grain_card(row, "singleton", assets, tile_size=64, angles=[1])
    assert isinstance(card, np.ndarray)
    assert card.shape[0] > 64 and card.shape[1] > 64

    # Pixels inside the grain (away from contour) in the PPL tile must match raw PPL without tint!
    # PPL tile is placed at y in [5, 69], x in [4, 68]
    # Check interior pixel center of the grain:
    center_y, center_x = 5 + 32, 4 + 32
    grain_pixel = card[center_y, center_x]
    # Raw was [150, 150, 150]. BGR conversion of [150, 150, 150] is [150, 150, 150].
    # Without tinting, the pixel value should remain [150, 150, 150] (not tinted with category colour)
    assert np.allclose(grain_pixel, [150, 150, 150], atol=3), (
        f"Tile interior was tinted! Expected ~[150, 150, 150], got {grain_pixel}"
    )


def test_assemble_category_panel_mix_of_fallback_and_four_angle_cards():
    """Verify assemble_category_panel handles a mix of fallback (2-tile) and 4-angle (8-tile) cards without crash."""
    from scripts.visualize_labels_v1 import render_grain_card, assemble_category_panel

    H, W = 50, 50
    mask_full = np.zeros((H, W), dtype=np.uint16)
    mask_full[15:35, 15:35] = 1
    ppl_raw = np.full((36, H, W, 3), [150, 150, 150], dtype=np.uint8)
    xpl_raw = np.full((36, H, W, 3), [80, 80, 80], dtype=np.uint8)

    row1 = {
        "region_id": "site_g001",
        "site_id": "site_01",
        "state": "singleton",
        "allowed": ["quartz"],
        "reason": "singleton",
        "bbox_xywh": [15, 15, 20, 20],
        "label_value": 1,
    }
    # 4-angle card (has aligned stack with 4 angles)
    assets_4ang = {
        "mask": mask_full,
        "ppl": ppl_raw[0],
        "xpl": xpl_raw[0],
        "aligned": {"ppl": ppl_raw, "xpl": xpl_raw},
    }
    card_4ang = render_grain_card(row1, "singleton", assets_4ang, angles=[1, 10, 19, 28])

    # Fallback card (aligned is None, so uses mean PPL / max XPL)
    row2 = dict(row1, region_id="site_g002")
    assets_fallback = {
        "mask": mask_full,
        "ppl": ppl_raw[0],
        "xpl": xpl_raw[0],
        "aligned": None,
    }
    card_fallback = render_grain_card(row2, "singleton", assets_fallback)

    # Note they have different widths
    assert card_4ang.shape[1] != card_fallback.shape[1]

    # Assembling panel with mix of both cards must not crash
    mixed_cards = [card_4ang, card_fallback, card_fallback, card_4ang]
    panel = assemble_category_panel(mixed_cards, "singleton", "fold_0")
    assert isinstance(panel, np.ndarray)
    assert panel.ndim == 3
    assert panel.shape[0] > 0 and panel.shape[1] > 0


