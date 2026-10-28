#!/usr/bin/env python3
"""Phase 0 — Clean and audit the download (§6).

Script: scripts/00_clean.py
Usage:  python scripts/00_clean.py --config configs/default.yaml

Reads  : data/raw/ukvm_dataset/
Writes : data/interim/sites.csv
         data/interim/specimens_clean.csv
         data/interim/qc/phase0_report.md
         data/interim/qc/phase0_brightness/<site_id>.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path so `src.rotapetro` is importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config     # noqa: E402
from src.rotapetro.clean import DataCleaner   # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 0: clean and audit the UKVM download (§6)")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="Path to the YAML config file (default: configs/default.yaml)")
    args = parser.parse_args()

    # Logging: console (INFO) + file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("phase0")

    logger.info("─── Phase 0: Clean and Audit ───")
    logger.info("Config: %s", args.config)

    cfg = load_config(args.config)
    cleaner = DataCleaner(cfg, config_path=args.config)
    cleaner.run()

    logger.info("─── Phase 0 finished.  "
                "Read data/interim/qc/phase0_report.md before continuing. ───")


if __name__ == "__main__":
    main()
