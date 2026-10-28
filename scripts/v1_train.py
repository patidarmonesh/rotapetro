#!/usr/bin/env python3
"""V1 Three-Arm Training Experiment (§9).

Script: scripts/v1_train.py

Runs the matched three-arm experiment:
  Arm A: static_k1      — single angle, no angle embedding
  Arm B: unordered_k12   — 12 angles, no angle embedding
  Arm C: angle_aware_k12 — 12 angles, with angle embedding

For each arm × fold × seed:
  1. Load frozen fold splits
  2. Load precomputed embeddings via V1EmbeddingDataset
  3. Train RotaPetroNet with V1 set_loss and validation set-loss selection
  4. Evaluate on held-out test fold (or validation if test is empty)
  5. Save metrics, predictions, checkpoint, and static sensitivity
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

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
from src.rotapetro.model import RotaPetroNet
from src.rotapetro.train import Trainer, set_loss
from src.rotapetro.dataset import V1EmbeddingDataset

logger = logging.getLogger(__name__)

ARMS = ["static_k1", "unordered_k12", "angle_aware_k12"]


def seed_everything(seed: int) -> None:
    """Seed all pseudo-random number generators (§9c)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_predictions(
    model: RotaPetroNet,
    dataset: V1EmbeddingDataset,
    families: list[str],
    device: torch.device,
    test_angle_idx: int,
    batch_size: int = 32,
    fold: str | None = None,
    seed: int | None = None,
    arm: str | None = None,
) -> pd.DataFrame:
    """Generate probability predictions and metadata for a dataset (§9c)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    rows = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x_p = batch["emb_ppl"].to(device).float()
            x_x = batch["emb_xpl"].to(device).float()
            th = batch["theta_deg"].to(device).float()
            allowed_mask = batch["allowed"].cpu().numpy()
            reg_ids = batch["region_id"]
            specs = batch["specimen"]
            states = batch["state"]

            logits = model(x_p, x_x, th)["mineral"]
            probs = F.softmax(logits, dim=-1).cpu().numpy()

            for i in range(len(reg_ids)):
                p = probs[i]
                pred_idx = int(np.argmax(p))
                pred_fam = families[pred_idx]
                row = {
                    "region_id": reg_ids[i],
                    "specimen": specs[i],
                    "state": states[i],
                    "allowed": [families[j] for j, is_ok in enumerate(allowed_mask[i]) if is_ok],
                    "y_pred": pred_fam,
                    "test_angle_idx": test_angle_idx,
                    "fold": fold,
                    "seed": seed,
                    "arm": arm,
                }
                for j, fam in enumerate(families):
                    row[f"probs_{fam}"] = float(p[j])
                rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V1 three-arm training experiment (§9)")
    parser.add_argument("--config", type=str,
                        default=str(_PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--arm", type=str, default=None,
                        choices=ARMS + ["all"],
                        help="Which arm to run (default: all)")
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only this fold (0-4)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Run only this seed")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (for smoke test)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run smoke test mode (1 fold, 1 seed, train and val)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config)
    validate_v1_config(cfg)
    require_fresh_feasibility(cfg)
    v1_paths = resolve_v1_paths(cfg)
    v1_cfg = cfg.get("v1", {})
    v1_train = v1_cfg.get("training", {})
    primary_classes = v1_cfg.get("taxonomy", {}).get("primary_classes", [])
    n_classes = len(primary_classes)

    seeds = [args.seed] if args.seed is not None else v1_train.get("seeds", [0, 1, 2])
    arms = [args.arm] if (args.arm and args.arm != "all") else ARMS

    primary_angle_idx = int(v1_train.get("primary_test_angle_index", 1))
    k12_indices = tuple(v1_train.get("k12_angle_indices", [1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34]))
    batch_size = int(v1_train.get("batch_size", 32))

    # ── Load frozen folds ───────────────────────────────────────────
    folds_path = _PROJECT_ROOT / v1_paths["processed"] / "splits" / "folds.json"
    if not folds_path.is_file():
        logger.error("Frozen folds not found at %s", folds_path)
        sys.exit(1)
    with open(folds_path, "r", encoding="utf-8") as f:
        all_folds = json.load(f)

    fold_names = list(all_folds.keys())
    if args.fold is not None:
        fold_names = [f"fold_{args.fold}"]

    # ── Check embeddings ────────────────────────────────────────────
    emb_dir = _PROJECT_ROOT / v1_paths["embeddings"]
    emb_path = emb_dir / "embeddings.npy"
    index_path = emb_dir / "index.parquet"

    if not emb_path.is_file() or not index_path.is_file():
        logger.error("Embeddings not found at %s — run Phase 8 first", emb_dir)
        sys.exit(1)

    emb_index = pd.read_parquet(index_path)
    logger.info("Loaded embedding index (%d rows) from %s", len(emb_index), index_path)

    results_dir = _PROJECT_ROOT / v1_paths["results"]
    if args.smoke:
        results_dir = results_dir / "_smoke"
        logger.info("SMOKE MODE: outputs go to %s (will NOT be picked up by evaluate)", results_dir)
    all_results = []

    device_str = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    device = torch.device(device_str)
    logger.info("Using device: %s", device)

    for arm in arms:
        for fold_name in fold_names:
            fold = all_folds[fold_name]
            train_codes = set(fold["train"])
            val_codes = set(fold["val"])
            test_codes = set(fold["test"])

            targets_path = _PROJECT_ROOT / v1_paths["teachers"] / fold_name / "targets.parquet"
            if not targets_path.is_file():
                logger.error("Teacher targets not found at %s", targets_path)
                sys.exit(1)
            targets_df = pd.read_parquet(targets_path)

            samples_path = _PROJECT_ROOT / v1_paths["processed"] / "samples.parquet"

            if args.smoke:
                # In smoke test mode: use up to 200 available embedded samples for fast validation
                rids_list = list(emb_index["region_id"])[:200]
                smoke_targets = targets_df.copy()
                # Exclude regions not in the cache so validation passes
                smoke_targets.loc[~smoke_targets["region_id"].isin(rids_list), "state"] = "excluded"

                n_total = len(rids_list)
                n_smoke_test = max(1, int(n_total * 0.2))
                n_smoke_val = max(1, int(n_total * 0.2))
                n_smoke_train = max(1, n_total - n_smoke_val - n_smoke_test)

                train_rids = set(rids_list[:n_smoke_train])
                val_rids = set(rids_list[n_smoke_train:n_smoke_train + n_smoke_val])
                test_rids = set(rids_list[n_smoke_train + n_smoke_val:])

                smoke_train_code = next(iter(train_codes))
                smoke_val_code = next(iter(val_codes))
                smoke_test_code = next(iter(test_codes))

                for idx, rid in enumerate(rids_list):
                    row_mask = smoke_targets["region_id"] == rid
                    c = primary_classes[idx % len(primary_classes)]
                    if rid in train_rids:
                        smoke_targets.loc[row_mask, "code"] = smoke_train_code
                    elif rid in val_rids:
                        smoke_targets.loc[row_mask, "code"] = smoke_val_code
                    else:
                        smoke_targets.loc[row_mask, "code"] = smoke_test_code
                    is_singleton = (idx % 3 != 0)
                    smoke_targets.loc[row_mask, "state"] = "singleton" if is_singleton else "partial_set"
                    allowed_set = [c] if is_singleton else [c, primary_classes[(idx + 1) % len(primary_classes)]]
                    smoke_targets.at[smoke_targets[row_mask].index[0], "allowed"] = allowed_set

                targets_df = smoke_targets

            for seed in seeds:
                logger.info("=" * 60)
                logger.info("EXPERIMENT: ARM=%s FOLD=%s SEED=%d", arm, fold_name, seed)
                logger.info("=" * 60)

                # Seed everything
                seed_everything(seed)

                # Deep copy config (§9c)
                arm_cfg = copy.deepcopy(cfg)
                arm_cfg.setdefault("v1", {}).setdefault("training", {})
                arm_cfg["v1"]["training"]["mode"] = arm
                if args.epochs is not None:
                    arm_cfg["v1"]["training"]["epochs"] = args.epochs

                out_dir = results_dir / arm / fold_name / f"seed_{seed}"
                out_dir.mkdir(parents=True, exist_ok=True)

                # Build datasets using V1EmbeddingDataset (§9c)
                train_ds = V1EmbeddingDataset(
                    emb_path=emb_path,
                    index_df=emb_index,
                    targets_df=targets_df,
                    families=primary_classes,
                    split_codes=train_codes,
                    phase="train",
                    arm=arm,
                    seed=seed,
                    primary_idx=primary_angle_idx,
                    k12_idx=k12_indices,
                    samples_path=samples_path,
                )

                val_ds = V1EmbeddingDataset(
                    emb_path=emb_path,
                    index_df=emb_index,
                    targets_df=targets_df,
                    families=primary_classes,
                    split_codes=val_codes,
                    phase="val",
                    arm=arm,
                    seed=seed,
                    primary_idx=primary_angle_idx,
                    k12_idx=k12_indices,
                    samples_path=samples_path,
                )

                test_ds = V1EmbeddingDataset(
                    emb_path=emb_path,
                    index_df=emb_index,
                    targets_df=targets_df,
                    families=primary_classes,
                    split_codes=test_codes,
                    phase="test",
                    arm=arm,
                    seed=seed,
                    primary_idx=primary_angle_idx,
                    k12_idx=k12_indices,
                    include_abstain=True,
                    samples_path=samples_path,
                )

                logger.info(
                    "Split sizes [%s/%s/seed_%d]: train=%d, val=%d, test=%d",
                    arm, fold_name, seed, len(train_ds), len(val_ds), len(test_ds),
                )

                if len(train_ds) == 0:
                    logger.warning("No training samples available for %s/%s. Skipping.", arm, fold_name)
                    continue

                train_loader = DataLoader(
                    train_ds,
                    batch_size=min(batch_size, len(train_ds)),
                    shuffle=True,
                    generator=torch.Generator().manual_seed(seed),
                )
                if len(val_ds) == 0:
                    raise ValueError(
                        f"No validation samples for {arm}/{fold_name}/seed_{seed}. "
                        f"Cannot select model without validation data.")
                val_loader = DataLoader(
                    val_ds,
                    batch_size=min(batch_size, len(val_ds)),
                    shuffle=False,
                )

                # Build model
                model = RotaPetroNet(
                    n_fine_classes=n_classes,
                    cfg=arm_cfg,
                    arm=arm,
                )

                # Train
                trainer = Trainer(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    cfg=arm_cfg,
                    device=device_str,
                    output_dir=out_dir,
                    arm=arm,
                )
                train_summary = trainer.train()

                # Reload best checkpoint (§9c)
                best_ckpt_path = out_dir / "best.pt"
                if best_ckpt_path.is_file():
                    ckpt = torch.load(best_ckpt_path, map_location=device)
                    model.load_state_dict(ckpt["model_state"])
                    logger.info("Loaded best checkpoint from %s (best_val_loss=%.4f)",
                                best_ckpt_path, ckpt.get("best_val_loss", float("nan")))

                # Generate primary predictions (at primary_test_angle_index)
                if len(test_ds) == 0:
                    raise ValueError(
                        f"No test samples for {arm}/{fold_name}/seed_{seed}. "
                        f"Cannot write predictions without test data. "
                        f"(test_codes={len(test_codes)}, matched={len(test_ds)})")
                eval_ds = test_ds
                preds_df = generate_predictions(
                    model=model,
                    dataset=eval_ds,
                    families=primary_classes,
                    device=device,
                    test_angle_idx=primary_angle_idx,
                    batch_size=batch_size,
                    fold=fold_name,
                    seed=seed,
                    arm=arm,
                )
                preds_path = out_dir / "predictions.parquet"
                preds_df.to_parquet(preds_path, index=False)
                logger.info("Saved primary predictions (%d rows) to %s", len(preds_df), preds_path)

                # Static sensitivity (§9c)
                # For static model: predict at each of the 12 K12 angles separately, plus at index 0
                if arm == "static_k1":
                    sensitivity_angles = [0] + list(k12_indices)
                    logger.info("Running static angle sensitivity at %d angles: %s",
                                len(sensitivity_angles), sensitivity_angles)
                    for ang_idx in sensitivity_angles:
                        sens_ds = V1EmbeddingDataset(
                            emb_path=emb_path,
                            index_df=emb_index,
                            targets_df=targets_df,
                            families=primary_classes,
                            split_codes=test_codes,
                            phase="test",
                            arm=arm,
                            seed=seed,
                            primary_idx=ang_idx,
                            k12_idx=k12_indices,
                            include_abstain=True,
                            samples_path=samples_path,
                        )
                        sens_df = generate_predictions(
                            model=model,
                            dataset=sens_ds,
                            families=primary_classes,
                            device=device,
                            test_angle_idx=ang_idx,
                            batch_size=batch_size,
                            fold=fold_name,
                            seed=seed,
                            arm=arm,
                        )
                        sens_path = out_dir / f"predictions_angle_{ang_idx}.parquet"
                        sens_df.to_parquet(sens_path, index=False)
                    logger.info("Saved %d static sensitivity prediction files in %s",
                                len(sensitivity_angles), out_dir)

                # Save provenance
                save_provenance(
                    out_dir / "provenance.json", arm_cfg,
                    extra={
                        "arm": arm,
                        "fold": fold_name,
                        "seed": seed,
                        "n_train": len(train_ds),
                        "n_val": len(val_ds),
                        "n_test": len(test_ds),
                        "best_val_loss": train_summary.get("best_val_loss"),
                        "total_epochs": train_summary.get("total_epochs"),
                    },
                )

                all_results.append({
                    "arm": arm,
                    "fold": fold_name,
                    "seed": seed,
                    "best_val_loss": train_summary.get("best_val_loss"),
                    "total_epochs": train_summary.get("total_epochs"),
                    "status": "completed",
                })

    # Save summary
    summary_path = results_dir / "experiment_index.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Experiment summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
