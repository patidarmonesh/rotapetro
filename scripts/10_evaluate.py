#!/usr/bin/env python3
"""Phase 10: Evaluation (§10).

Script: scripts/10_evaluate.py

Usage:
  python scripts/10_evaluate.py [--config configs/default.yaml] [--dummy]
"""
import sys
from pathlib import Path
import argparse
import logging
import json

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.rotapetro.io import load_config, validate_v1_config, resolve_v1_paths
from src.rotapetro.evaluate import Evaluator
import pandas as pd

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Phase 10: V1 Evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to base config file")
    parser.add_argument("--results_dir", type=str, default=None, help="Directory containing arm results (default: outputs/v1)")
    parser.add_argument("--teachers_dir", type=str, default=None, help="Directory containing teacher targets")
    parser.add_argument("--n_bootstrap", type=int, default=5000, help="Number of bootstrap replicates (default: 5000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--dummy", action="store_true", help="Run dummy majority-class power check on targets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    logger.info("Starting Phase 10: Evaluation")
    cfg = load_config(args.config)
    validate_v1_config(cfg)
    v1_paths = resolve_v1_paths(cfg)

    results_dir = Path(args.results_dir) if args.results_dir else (_PROJECT_ROOT / v1_paths["results"])
    teachers_dir = Path(args.teachers_dir) if args.teachers_dir else (_PROJECT_ROOT / v1_paths["teachers"])

    evaluator = Evaluator(
        cfg=cfg,
        output_dir=results_dir,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
    )

    if args.dummy:
        logger.info("Running dummy power check on fold targets...")
        target_dfs = []
        for f_dir in teachers_dir.iterdir():
            if f_dir.is_dir() and f_dir.name.startswith("fold_"):
                tp = f_dir / "targets.parquet"
                if tp.is_file():
                    tdf = pd.read_parquet(tp)
                    if "partition" in tdf.columns:
                        tdf = tdf[tdf["partition"] == "test"]
                    target_dfs.append(tdf)
        if not target_dfs:
            logger.error("No targets found in %s", teachers_dir)
            sys.exit(1)
        all_targets = pd.concat(target_dfs, ignore_index=True)
        res = evaluator.run_dummy_power_check(all_targets)
        print("\n" + "=" * 50)
        print("DUMMY PREDICTOR POWER CHECK RESULT")
        print("=" * 50)
        print(json.dumps(res, indent=2))
        return

    # Expected folds from v1 splits and seeds from config
    folds_path = _PROJECT_ROOT / v1_paths["processed"] / "splits" / "folds.json"
    if not folds_path.is_file():
        folds_path = _PROJECT_ROOT / cfg.get("paths", {}).get("processed", "data/processed") / "splits" / "folds.json"
    if not folds_path.is_file():
        logger.error("splits/folds.json not found at %s", folds_path)
        sys.exit(1)
    with open(folds_path, "r", encoding="utf-8") as f:
        folds_data = json.load(f)
    expected_folds = list(folds_data.keys())

    expected_seeds = cfg.get("v1", {}).get("training", {}).get("seeds", [0, 1, 2])
    expected_arms = cfg.get("v1", {}).get("training", {}).get("arms", ["static_k1", "unordered_k12", "angle_aware_k12"])

    summary = evaluator.evaluate_experiment(
        results_dir=results_dir,
        teachers_dir=teachers_dir,
        folds=expected_folds,
        seeds=expected_seeds,
        arms=expected_arms,
    )

    logger.info("Evaluation complete. Results saved in %s", results_dir)
    print("\n" + evaluator.generate_markdown_report(summary))


if __name__ == "__main__":
    main()
