# DEPRECATED: This is the legacy baseline training script. For V1 5-class mineral classification, use scripts/v1_train.py.

import sys
import warnings
from pathlib import Path
import argparse
import logging

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rotapetro.io import load_config
from src.rotapetro.dataset import RegionSequenceDataset
from src.rotapetro.model import RotaPetroNet
from src.rotapetro.train import Trainer
from torch.utils.data import DataLoader
import json
import yaml
import pandas as pd
import torch
import numpy as np
import random

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    print(
        "WARNING: DEPRECATED: This is the legacy baseline training script. "
        "For V1 5-class mineral classification, use scripts/v1_train.py.",
        file=sys.stderr,
    )
    warnings.warn(
        "This is the legacy baseline training script. For V1 5-class mineral classification, use scripts/v1_train.py.",
        DeprecationWarning,
        stacklevel=2,
    )

    parser = argparse.ArgumentParser(description="Phase 9: training")
    parser.add_argument("--legacy", action="store_true", help="Allow running legacy training script despite deprecation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to train")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device for training")
    args = parser.parse_args()

    if not args.legacy:
        sys.exit(2)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.warning(
        "DEPRECATED: This is the legacy baseline training script. "
        "For V1 5-class mineral classification, use scripts/v1_train.py."
    )
    logger.info(f"Starting Phase 9: Training (Fold {args.fold})")
    
    seed_everything(args.seed)
    cfg = load_config(args.config)
    
    samples_df = pd.read_parquet("data/processed/samples.parquet")
    with open("data/processed/classes.yaml", "r") as f:
        classes = yaml.safe_load(f)
    with open("data/processed/splits/folds.json", "r") as f:
        folds = json.load(f)
        
    fold_key = f"fold_{args.fold}"
    train_specs = folds[fold_key]["train"]
    val_specs = folds[fold_key]["val"]
    
    train_df = samples_df[samples_df['code'].isin(train_specs)]
    val_df = samples_df[samples_df['code'].isin(val_specs)]
    
    # Check for embeddings index
    idx_path = Path("data/processed/embeddings/index.parquet")
    if idx_path.exists():
        emb_index = pd.read_parquet(idx_path).set_index('region_id')['idx'].to_dict()
    else:
        logger.warning("No embeddings index found. Assuming mode='image' or about to fail.")
        emb_index = None

    train_ds = RegionSequenceDataset(
        samples_df=train_df, classes=classes,
        aligned_dir=Path("data/interim/aligned"),
        segments_dir=Path("data/interim/segments"),
        cfg=cfg, mode="embed" if emb_index else "image",
        emb_path=Path("data/processed/embeddings/embeddings.npy") if emb_index else None,
        emb_index=emb_index,
        augment=True
    )
    val_ds = RegionSequenceDataset(
        samples_df=val_df, classes=classes,
        aligned_dir=Path("data/interim/aligned"),
        segments_dir=Path("data/interim/segments"),
        cfg=cfg, mode="embed" if emb_index else "image",
        emb_path=Path("data/processed/embeddings/embeddings.npy") if emb_index else None,
        emb_index=emb_index,
        augment=False
    )
    
    model = RotaPetroNet(n_fine_classes=len(classes["fine_classes"]), n_group_classes=len(set(classes["groups"].values())), cfg=cfg)
    
    bs = cfg.get("train", {}).get("batch_size", 32)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    
    trainer = Trainer(model, train_loader, val_loader, cfg, device=args.device)
    trainer.train()
    
    logger.info("Finished Phase 9: Training")

if __name__ == "__main__":
    main()
