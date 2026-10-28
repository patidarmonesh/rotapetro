import sys
from pathlib import Path
import argparse
import logging

project_root = str(Path(__file__).resolve().parents[1])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.rotapetro.config import load_config
from src.rotapetro.inference.video import VideoInference

def main():
    parser = argparse.ArgumentParser(description="Phase 11: inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--ppl", type=str, required=True, help="Path to PPL video")
    parser.add_argument("--xpl", type=str, required=True, help="Path to XPL video")
    parser.add_argument("--meta", type=str, default=None, help="Optional path to metadata file")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    parser.add_argument("--model", type=str, default=None, help="Optional path to specific model weights")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)

    logger.info("Starting Phase 11: Inference")
    cfg = load_config(args.config)
    
    inference = VideoInference(
        cfg=cfg,
        model_path=args.model,
        device=args.device
    )
    
    inference.process(
        ppl_path=args.ppl,
        xpl_path=args.xpl,
        out_dir=args.out,
        meta_path=args.meta
    )
    
    logger.info("Finished Phase 11: Inference")

if __name__ == "__main__":
    main()
