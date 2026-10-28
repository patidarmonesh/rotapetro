"""Unit tests for V1 dataset, view selection, and embedding memory-mapping (§7, §8).

Tests:
- Test H7: _parse_bbox handles np.array([1,2,3,4]), [1,2,3,4], "[1,2,3,4]", (1,2,3,4)
  returning (1,2,3,4), and returns None for None or invalid shapes.
- Test H9: V1EmbeddingDataset training view indices for a region: 12 indices (o+3j)%36,
  distinct, spaced 15°, identical across all 3 arms for same epoch and seed.
- Test H10: Training views change across epochs when set_epoch(1) is called vs set_epoch(0).
- Test H11: Test views: static_k1 gives [5.0] (index 1), k12 gives 12 angles [5, 20, ..., 170].
- Test H13: theta_deg == 5.0 * idx.
- Test H14: allowed boolean vector: singleton has 1 True, partial has 2 True, matches order of families.
- Test I9: Mask patch pooling weights: all-ones 224x224 mask yields equal weights (1/256 each);
  mask with only top-left 14x14 active yields 1.0 on patch 0.
- Test I10: Open_memmap NPY header read/write integrity.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.dataset import _parse_bbox, V1EmbeddingDataset


# ── Test H7 ─────────────────────────────────────────────────────────────

def test_h7_parse_bbox():
    """Test H7: _parse_bbox handles arrays, lists, JSON strings, tuples, and returns None on invalid."""
    expected = (1, 2, 3, 4)

    assert _parse_bbox(np.array([1, 2, 3, 4])) == expected
    assert _parse_bbox([1, 2, 3, 4]) == expected
    assert _parse_bbox("[1, 2, 3, 4]") == expected
    assert _parse_bbox((1, 2, 3, 4)) == expected

    # Also handle longer arrays/lists by taking first 4 elements
    assert _parse_bbox(np.array([1, 2, 3, 4, 5])) == expected
    assert _parse_bbox([1, 2, 3, 4, 5]) == expected

    # Invalid cases must return None
    assert _parse_bbox(None) is None
    assert _parse_bbox(float("nan")) is None
    assert _parse_bbox(np.array([1, 2, 3])) is None
    assert _parse_bbox([1, 2]) is None
    assert _parse_bbox("not_json") is None
    assert _parse_bbox("[1, 2]") is None
    assert _parse_bbox(123) is None


# ── Test H9, H10, H11, H13, H14 ─────────────────────────────────────────

@pytest.fixture
def dummy_dataset_frames(tmp_path_factory):
    """Create minimal index_df, targets_df, and valid embedding file for testing V1EmbeddingDataset."""
    tmp_dir = tmp_path_factory.mktemp("v1_dataset")
    emb_path = tmp_dir / "embeddings.npy"
    mmap = np.lib.format.open_memmap(str(emb_path), mode="w+", dtype=np.float16, shape=(2, 1, 36, 2, 768))
    mmap.flush()
    del mmap

    import json
    manifest = {
        "N": 2,
        "complete": True,
        "shape": [2, 1, 36, 2, 768],
        "dtype": "float16",
    }
    with open(tmp_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    index_df = pd.DataFrame([
        {"region_id": "r_single", "code": "specA", "cache_row": 0},
        {"region_id": "r_partial", "code": "specA", "cache_row": 1},
    ])
    targets_df = pd.DataFrame([
        {
            "region_id": "r_single",
            "code": "specA",
            "state": "singleton",
            "allowed": ["quartz"],
        },
        {
            "region_id": "r_partial",
            "code": "specA",
            "state": "partial_set",
            "allowed": ["quartz", "feldspar"],
        },
    ])
    families = ["quartz", "feldspar", "mica", "pyroxene", "opaque_material"]
    return emb_path, index_df, targets_df, families


def test_h9_training_view_indices_formula_and_arm_identity(dummy_dataset_frames):
    """Test H9: 12 training indices (o+3j)%36, distinct, spaced 15°, identical across all 3 arms."""
    emb_path, index_df, targets_df, families = dummy_dataset_frames

    arms = ["static_k1", "unordered_k12", "angle_aware_k12"]
    thetas_by_arm = {}

    for arm in arms:
        ds = V1EmbeddingDataset(
            emb_path=emb_path,
            index_df=index_df,
            targets_df=targets_df,
            families=families,
            split_codes={"specA"},
            phase="train",
            arm=arm,
            seed=42,
        )
        ds.set_epoch(0)
        item = ds[0]
        th = item["theta_deg"]
        thetas_by_arm[arm] = th

        # Compute view indices: theta = 5.0 * idx
        idx = [int(round(t.item() / 5.0)) for t in th]

        # 12 indices
        assert len(idx) == 12

        # Distinct
        assert len(set(idx)) == 12

        # Spaced 15° (= 3 steps of 5°)
        o = idx[0]
        expected_idx = [(o + 3 * j) % 36 for j in range(12)]
        assert idx == expected_idx

        for j in range(11):
            assert (idx[j + 1] - idx[j]) % 36 == 3

    # Identical across all 3 arms for same epoch and seed
    for arm in arms[1:]:
        assert torch.equal(thetas_by_arm[arm], thetas_by_arm[arms[0]]), (
            f"Training views differed between {arm} and {arms[0]}"
        )


def test_h10_training_views_change_across_epochs(dummy_dataset_frames):
    """Test H10: Training views change across epochs when set_epoch(1) is called vs set_epoch(0)."""
    emb_path, index_df, targets_df, families = dummy_dataset_frames

    ds = V1EmbeddingDataset(
        emb_path=emb_path,
        index_df=index_df,
        targets_df=targets_df,
        families=families,
        split_codes={"specA"},
        phase="train",
        seed=42,
    )

    ds.set_epoch(0)
    th_epoch0 = ds[0]["theta_deg"]

    ds.set_epoch(1)
    th_epoch1 = ds[0]["theta_deg"]

    assert not torch.equal(th_epoch0, th_epoch1), "Expected different view angles between epoch 0 and epoch 1"


def test_h11_and_h13_test_views_static_vs_k12(dummy_dataset_frames):
    """Test H11 & H13: Test views: static_k1 gives [5.0] (index 1), k12 gives 12 angles [5, 20, ..., 170]."""
    emb_path, index_df, targets_df, families = dummy_dataset_frames
    k12_idx = (1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34)

    # 1. static_k1 test views
    ds_static = V1EmbeddingDataset(
        emb_path=emb_path,
        index_df=index_df,
        targets_df=targets_df,
        families=families,
        split_codes={"specA"},
        phase="test",
        arm="static_k1",
        primary_idx=1,
    )
    th_static = ds_static[0]["theta_deg"]
    assert th_static.tolist() == [5.0], f"Expected [5.0], got {th_static.tolist()}"

    # 2. k12 test views
    ds_k12 = V1EmbeddingDataset(
        emb_path=emb_path,
        index_df=index_df,
        targets_df=targets_df,
        families=families,
        split_codes={"specA"},
        phase="test",
        arm="unordered_k12",
        k12_idx=k12_idx,
    )
    th_k12 = ds_k12[0]["theta_deg"]
    expected_k12 = [float(5.0 * idx) for idx in k12_idx]
    assert th_k12.tolist() == expected_k12, f"Expected {expected_k12}, got {th_k12.tolist()}"

    # Test H13: theta_deg == 5.0 * idx
    for t, idx in zip(th_k12.tolist(), k12_idx):
        assert t == pytest.approx(5.0 * idx)


def test_h14_allowed_boolean_vector(dummy_dataset_frames):
    """Test H14: allowed boolean vector: singleton has 1 True, partial has 2 True, matches order of families."""
    emb_path, index_df, targets_df, families = dummy_dataset_frames

    ds = V1EmbeddingDataset(
        emb_path=emb_path,
        index_df=index_df,
        targets_df=targets_df,
        families=families,
        split_codes={"specA"},
        phase="train",
    )

    # Row 0: singleton 'quartz' (index 0 in families)
    item_single = ds[0]
    allowed_single = item_single["allowed"]
    assert allowed_single.dtype == torch.bool
    assert allowed_single.sum().item() == 1
    assert allowed_single[families.index("quartz")].item() is True

    # Row 1: partial set ['quartz', 'feldspar'] (indices 0 and 1)
    item_partial = ds[1]
    allowed_partial = item_partial["allowed"]
    assert allowed_partial.dtype == torch.bool
    assert allowed_partial.sum().item() == 2
    assert allowed_partial[families.index("quartz")].item() is True
    assert allowed_partial[families.index("feldspar")].item() is True
    assert allowed_partial[families.index("mica")].item() is False


# ── Test I9: Mask patch pooling weights ─────────────────────────────────

def test_i9_mask_patch_pooling_weights():
    """Test I9: all-ones 224x224 mask yields equal weights (1/256 each); top-left 14x14 yields 1.0 on patch 0."""
    def compute_weights(mask_np: np.ndarray) -> torch.Tensor:
        m = torch.from_numpy(mask_np.astype(np.float32))[None, None]  # [1, 1, 224, 224]
        w = F.avg_pool2d(m, 14).flatten()  # [256]
        w = w / w.sum().clamp(min=1e-6)
        return w

    # Case 1: all-ones 224x224 mask -> equal weights (1/256 each)
    all_ones_mask = np.ones((224, 224), dtype=bool)
    w_ones = compute_weights(all_ones_mask)
    assert w_ones.shape == (256,)
    expected_uniform = torch.full((256,), 1.0 / 256.0, dtype=torch.float32)
    assert torch.allclose(w_ones, expected_uniform, atol=1e-6)

    # Case 2: only top-left 14x14 active -> 1.0 on patch 0, 0.0 on all others
    top_left_mask = np.zeros((224, 224), dtype=bool)
    top_left_mask[0:14, 0:14] = True
    w_top_left = compute_weights(top_left_mask)
    assert w_top_left[0].item() == pytest.approx(1.0, abs=1e-6)
    assert w_top_left[1:].sum().item() == pytest.approx(0.0, abs=1e-6)


# ── Test I10: Open_memmap NPY header integrity ─────────────────────────

def test_i10_open_memmap_npy_header_integrity(tmp_path: Path):
    """Test I10: Open_memmap NPY header read/write integrity for float16 embeddings."""
    npy_path = tmp_path / "embeddings.npy"
    shape = (4, 1, 36, 2, 768)
    dtype = np.float16

    # 1. Create and write with open_memmap mode='w+'
    mmap_write = np.lib.format.open_memmap(
        str(npy_path),
        mode="w+",
        dtype=dtype,
        shape=shape,
    )
    rng = np.random.default_rng(999)
    synth_data_0 = rng.standard_normal((1, 36, 2, 768)).astype(dtype)
    synth_data_3 = rng.standard_normal((1, 36, 2, 768)).astype(dtype)

    mmap_write[0] = synth_data_0
    mmap_write[3] = synth_data_3
    mmap_write.flush()
    del mmap_write

    # 2. Read back with np.load mmap_mode='r'
    mmap_read = np.load(str(npy_path), mmap_mode="r")

    assert mmap_read.shape == shape
    assert mmap_read.dtype == dtype
    assert np.array_equal(mmap_read[0], synth_data_0)
    assert np.array_equal(mmap_read[3], synth_data_3)
    assert np.all(mmap_read[1] == 0)  # Unwritten rows remain zeros


def test_samples_path_validation_raises(dummy_dataset_frames, tmp_path: Path):
    """Test that V1EmbeddingDataset raises on missing or mismatched samples_path."""
    emb_path, index_df, targets_df, families = dummy_dataset_frames

    # 1. Non-existent samples_path raises FileNotFoundError
    with pytest.raises(FileNotFoundError, match="V1 samples.parquet not found"):
        V1EmbeddingDataset(
            emb_path=emb_path,
            index_df=index_df,
            targets_df=targets_df,
            families=families,
            split_codes={"specA"},
            samples_path=tmp_path / "nonexistent_samples.parquet",
        )

    # 2. Existing samples_path with mismatched hash raises ValueError
    dummy_samples = tmp_path / "samples.parquet"
    pd.DataFrame([{"dummy": 1}]).to_parquet(dummy_samples)

    manifest_p = emb_path.parent / "manifest.json"
    import json
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump({"complete": True, "N": 2, "samples_hash": "deadbeef1234"}, f)

    with pytest.raises(ValueError, match="Embedding manifest samples_hash.*does not match"):
        V1EmbeddingDataset(
            emb_path=emb_path,
            index_df=index_df,
            targets_df=targets_df,
            families=families,
            split_codes={"specA"},
            samples_path=dummy_samples,
        )
