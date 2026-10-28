#!/usr/bin/env python3
"""Phase 8 — Embedding extraction (V1).

Script: scripts/08_embed.py
Inputs:  data/processed/samples.parquet, data/processed/classes.yaml,
         data/interim/aligned/*, data/interim/segments/*
Outputs: outputs/v1/embeddings/embeddings.npy, outputs/v1/embeddings/index.parquet

V1 changes:
- Outputs to V1 versioned path (outputs/v1/embeddings/)
- Uses n_versions=1 (no geometric augmentation)
- Saves provenance record
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import (
    load_config,
    require_fresh_feasibility,
    resolve_v1_paths,
    save_provenance,
    validate_v1_config,
)
from src.rotapetro.embed import EmbeddingExtractor
from src.rotapetro.dataset import RegionSequenceDataset

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8: V1 embeddings")
    parser.add_argument("--config", type=str,
                        default=str(_PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of regions to embed (speed benchmark)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    validate_v1_config(cfg)
    v1_paths = resolve_v1_paths(cfg)

    interim_dir = _PROJECT_ROOT / cfg.get("paths", {}).get("interim", "data/interim")

    # V1: output to versioned path
    emb_dir = _PROJECT_ROOT / v1_paths["embeddings"]

    # Benchmark mode: write to a separate folder to never contaminate the real cache
    is_benchmark = args.limit is not None
    if not is_benchmark:
        require_fresh_feasibility(cfg)

    if is_benchmark:
        emb_dir = emb_dir.parent / "embeddings_benchmark"
        logger.warning("BENCHMARK MODE: writing to %s — NOT the real training cache", emb_dir)

    # ── Load V1 dataset ─────────────────────────────────────────────
    # Read data/processed/v1/samples.parquet (no classes.yaml needed in V1)
    samples_path = _PROJECT_ROOT / v1_paths["processed"] / "samples.parquet"
    if not samples_path.is_file():
        logger.error("V1 samples.parquet not found at %s — run Phase 7 first", samples_path)
        sys.exit(1)
    samples_df = pd.read_parquet(samples_path)
    total_samples = len(samples_df)
    logger.info("Loaded %d V1 samples from %s", total_samples, samples_path)

    if is_benchmark:
        samples_df = samples_df.iloc[:args.limit].copy()
        logger.info("Limiting to first %d samples for benchmark (--limit %d)", len(samples_df), args.limit)

    dataset = RegionSequenceDataset(
        samples_df=samples_df,
        classes=None,  # V1 does not need classes.yaml
        aligned_dir=interim_dir / "aligned",
        segments_dir=interim_dir / "segments",
        cfg=cfg,
        mode="image",
        augment=False,
    )

    # ── Extract & Benchmark ─────────────────────────────────────────
    extractor = EmbeddingExtractor(cfg, device=args.device)

    # Interrupted embedding rebuild safety check & logging
    existing_manifest = emb_dir / "manifest.json"
    if existing_manifest.exists():
        logger.warning(
            "Existing manifest detected at %s. extract_all will invalidate it "
            "immediately (complete=False, status='in_progress') before touching embeddings.npy "
            "to guarantee an interrupted run can never leave a stale complete=True cache on disk.",
            existing_manifest,
        )
    else:
        logger.info(
            "Embedding destination: %s (clean cache). extractor.extract_all will initialize "
            "manifest with complete=False, status='in_progress' during generation.",
            emb_dir,
        )

    t0 = time.time()
    extractor.extract_all(
        dataset=dataset,
        save_dir=emb_dir,
        samples_path=samples_path,
        limit=args.limit,
        mark_complete=not is_benchmark,
    )
    logger.info(
        "Embedding extraction complete. Cache manifest at %s finalized (complete=%s).",
        emb_dir / "manifest.json",
        not is_benchmark,
    )
    elapsed = time.time() - t0
    n_done = len(dataset)
    sec_per_region = elapsed / max(n_done, 1)
    est_full_hours = (sec_per_region * total_samples) / 3600.0

    logger.info(
        "SPEED BENCHMARK: %d regions embedded in %.2f s (%.3f s/region). Estimated full run (%d regions): %.2f hours (%.1f min)",
        n_done, elapsed, sec_per_region, total_samples, est_full_hours, est_full_hours * 60,
    )

    # ── Provenance ──────────────────────────────────────────────────
    save_provenance(
        emb_dir / "provenance_embed.json", cfg,
        extra={
            "n_regions": n_done,
            "total_samples": total_samples,
            "limit": args.limit,
            "n_versions": extractor.n_versions,
            "device": str(extractor.device),
            "elapsed_sec": round(elapsed, 2),
            "sec_per_region": round(sec_per_region, 3),
            "est_full_hours": round(est_full_hours, 2),
        },
    )

    logger.info("Phase 8 (V1) complete. Embeddings at %s", emb_dir)


if __name__ == "__main__":
    main()
