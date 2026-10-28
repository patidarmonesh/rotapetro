"""Unit tests for embedding extraction safety and cache invalidation (§8, §13).

Covers:
- Starting embedding extraction invalidates any existing manifest.json immediately.
- A killed/interrupted process cannot leave a stale complete cache.
- An interrupted cache (complete: False) is strictly rejected by V1EmbeddingDataset.
- A fully completed extraction marks complete: True in manifest.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.dataset import V1EmbeddingDataset
from src.rotapetro.embed import EmbeddingExtractor


class DummyFailingDataset:
    """Dataset that simulates process death / SIGINT / SIGKILL during extraction."""

    def __init__(self, n: int = 5):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        raise KeyboardInterrupt("Simulated killed process / SIGINT during extraction")


class DummySuccessfulDataset:
    """Minimal dataset for testing successful embedding extraction."""

    def __init__(self, n: int = 2):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "region_id": f"reg_{idx}",
            "ppl": np.zeros((36, 3, 224, 224), dtype=np.uint8),
            "xpl": np.zeros((36, 3, 224, 224), dtype=np.uint8),
            "mask": np.ones((224, 224), dtype=bool),
        }


def _create_mock_extractor(cfg: dict | None = None) -> EmbeddingExtractor:
    """Create an EmbeddingExtractor with mocked PyTorch Hub model to avoid downloading weights."""
    if cfg is None:
        cfg = {"model": {"n_versions": 1}}
    mock_model = MagicMock()
    mock_model.parameters.return_value = []
    mock_model.forward_features.return_value = {
        "x_norm_clstoken": torch.zeros((1, 384)),
        "x_norm_patchtokens": torch.zeros((1, 256, 384)),
    }
    with patch("torch.hub.load", return_value=mock_model):
        extractor = EmbeddingExtractor(cfg=cfg, device="cpu")
    return extractor


def test_extraction_invalidates_existing_manifest_immediately(tmp_path: Path):
    """Test that starting extraction immediately invalidates existing manifest.json.
    
    If a previous run left a complete manifest, starting a new extraction MUST
    inactivate it (complete: False, status: 'in_progress') BEFORE creating or
    overwriting embeddings.npy, so an interrupted/killed process cannot leave
    a stale complete cache on disk.
    """
    save_dir = tmp_path / "embeddings"
    save_dir.mkdir(parents=True)
    manifest_p = save_dir / "manifest.json"

    # 1. Simulate an existing COMPLETE manifest from a previous extraction
    stale_manifest = {
        "N": 5000,
        "complete": True,
        "status": "complete",
        "shape": [5000, 1, 36, 2, 768],
        "dtype": "float16",
        "samples_hash": "deadbeef1234",
    }
    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(stale_manifest, f, indent=2)

    # Verify preexisting manifest is complete
    with open(manifest_p, "r", encoding="utf-8") as f:
        pre_state = json.load(f)
    assert pre_state["complete"] is True

    # 2. Start extraction with a dataset that dies immediately (simulating killed process)
    extractor = _create_mock_extractor()
    failing_ds = DummyFailingDataset(n=10)

    with pytest.raises(KeyboardInterrupt, match="Simulated killed process"):
        extractor.extract_all(failing_ds, save_dir=save_dir)

    # 3. Verify manifest was immediately invalidated on disk
    assert manifest_p.is_file(), "manifest.json must exist"
    with open(manifest_p, "r", encoding="utf-8") as f:
        post_state = json.load(f)

    assert post_state["complete"] is False, (
        f"Expected manifest['complete'] to be False after interrupted rebuild, but got {post_state['complete']}"
    )
    assert post_state.get("status") == "in_progress"


def test_interrupted_cache_rejected_by_v1_embedding_dataset(tmp_path: Path):
    """Test that V1EmbeddingDataset strictly rejects a cache whose extraction was interrupted."""
    save_dir = tmp_path / "embeddings"
    save_dir.mkdir(parents=True)
    emb_file = save_dir / "embeddings.npy"

    # Pre-populate an invalid/in-progress manifest
    with open(save_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"complete": False, "status": "in_progress", "N": 2}, f)

    # Create dummy embeddings file and index/targets
    mmap = np.lib.format.open_memmap(str(emb_file), mode="w+", dtype=np.float16, shape=(2, 1, 36, 2, 768))
    mmap.flush()
    del mmap

    index_df = pd.DataFrame([{"region_id": "r0", "code": "specA", "cache_row": 0}])
    targets_df = pd.DataFrame([{"region_id": "r0", "code": "specA", "state": "singleton", "allowed": ["quartz"]}])
    families = ["quartz", "feldspar", "mica", "pyroxene", "opaque_material"]

    with pytest.raises(RuntimeError, match="is not marked complete"):
        V1EmbeddingDataset(
            emb_path=emb_file,
            index_df=index_df,
            targets_df=targets_df,
            families=families,
            split_codes={"specA"},
        )


def test_successful_extraction_marks_complete_true(tmp_path: Path):
    """Test that a fully completed extraction records complete: True in manifest.json."""
    save_dir = tmp_path / "embeddings"
    extractor = _create_mock_extractor()

    # Mock extract_region to return valid shaped dummy embeddings quickly
    dummy_out = np.zeros((1, 36, 2, 768), dtype=np.float16)
    extractor.extract_region = MagicMock(return_value=dummy_out)

    ds = DummySuccessfulDataset(n=2)
    extractor.extract_all(ds, save_dir=save_dir, mark_complete=True)

    manifest_p = save_dir / "manifest.json"
    assert manifest_p.is_file()
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["complete"] is True
    assert manifest["N"] == 2
    assert manifest["status"] == "complete"
    assert (save_dir / "embeddings.npy").is_file()
    assert (save_dir / "index.parquet").is_file()
