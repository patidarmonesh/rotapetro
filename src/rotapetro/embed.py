"""V1 Embedding extraction — DINO v2 embeddings for all regions.

V1 changes:
- n_versions=1 (no geometric augmentation versions)
- Single forward pass per angle (36 PPL + 36 XPL)
- CLS + mean-patch concatenation → 768-dim per frame
- Output shape: [N, 1, 36, 2, 768]  (V=1 for V1)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """Extract DINOv2 embeddings for all regions.

    Parameters
    ----------
    cfg : dict
        Project config.
    device : str
        'cuda' or 'cpu'.
    """

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = "cpu"
        self.device = torch.device(device)
        # V1: n_versions=1 (no geometric augmentation)
        self.n_versions = int(cfg.get("model", {}).get("n_versions", 1))

        # Load DINOv2 ViT-S/14
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14")
        if self.device.type == "cuda":
            self.model = self.model.to(self.device).half().eval()
            self.is_half = True
        else:
            self.model = self.model.to(self.device).eval()
            self.is_half = False

        for p in self.model.parameters():
            p.requires_grad = False

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        if self.is_half:
            self.mean = self.mean.half()
            self.std = self.std.half()

    def extract_region(
        self,
        ppl_crops: np.ndarray | torch.Tensor,
        xpl_crops: np.ndarray | torch.Tensor,
        mask: np.ndarray | torch.Tensor,
    ) -> np.ndarray:
        """Extract embeddings for one region using mask-weighted patch pooling (§8).

        Parameters
        ----------
        ppl_crops : [36, 3, 224, 224] uint8 or float
        xpl_crops : [36, 3, 224, 224] uint8 or float
        mask : [224, 224] bool or float region mask

        Returns
        -------
        emb : [V, 36, 2, 768] float16  (V=1 for V1)
        """
        # Mask-weighted patch pooling (§8)
        # DINOv2 ViT-S/14 at 224 px gives a 16x16 grid of patches (patch size 14)
        if isinstance(mask, torch.Tensor):
            m = mask.float().cpu()
        else:
            m = torch.from_numpy(mask.astype(np.float32))

        if m.ndim == 2:
            m = m[None, None]  # [1, 1, 224, 224]
        elif m.ndim == 3:
            m = m[None]

        w = torch.nn.functional.avg_pool2d(m, 14).flatten()  # [256] occupancy 0..1
        w = w.to(self.device) / w.sum().clamp(min=1e-6)
        if self.is_half:
            w = w.half()

        v_out = np.zeros(
            (self.n_versions, 36, 2, 768), dtype=np.float16)

        with torch.no_grad():
            if isinstance(ppl_crops, np.ndarray):
                ppl_t = torch.from_numpy(ppl_crops).float()
            else:
                ppl_t = ppl_crops.float()

            if isinstance(xpl_crops, np.ndarray):
                xpl_t = torch.from_numpy(xpl_crops).float()
            else:
                xpl_t = xpl_crops.float()

            if ppl_t.ndim == 4 and ppl_t.shape[-1] == 3:
                ppl_t = ppl_t.permute(0, 3, 1, 2)
            if xpl_t.ndim == 4 and xpl_t.shape[-1] == 3:
                xpl_t = xpl_t.permute(0, 3, 1, 2)

            if self.is_half:
                ppl_t = ppl_t.half()
                xpl_t = xpl_t.half()

            ppl_t = (ppl_t.to(self.device) / 255.0 - self.mean) / self.std
            xpl_t = (xpl_t.to(self.device) / 255.0 - self.mean) / self.std

            # Encode in batches of 12 to avoid OOM
            batch_size = 12
            ppl_embs, xpl_embs = [], []

            for start in range(0, 36, batch_size):
                end = min(start + batch_size, 36)

                ppl_out = self.model.forward_features(ppl_t[start:end])
                ppl_patch = ppl_out["x_norm_patchtokens"]
                ppl_pooled = (ppl_patch * w[None, :, None]).sum(dim=1)
                ppl_feat = torch.cat([ppl_out["x_norm_clstoken"], ppl_pooled], dim=-1)
                ppl_embs.append(ppl_feat.cpu().to(torch.float16).numpy())

                xpl_out = self.model.forward_features(xpl_t[start:end])
                xpl_patch = xpl_out["x_norm_patchtokens"]
                xpl_pooled = (xpl_patch * w[None, :, None]).sum(dim=1)
                xpl_feat = torch.cat([xpl_out["x_norm_clstoken"], xpl_pooled], dim=-1)
                xpl_embs.append(xpl_feat.cpu().to(torch.float16).numpy())

            v_out[0, :, 0, :] = np.concatenate(ppl_embs, axis=0)  # [36, 768]
            v_out[0, :, 1, :] = np.concatenate(xpl_embs, axis=0)  # [36, 768]

        return v_out

    def extract_all(
        self,
        dataset,
        save_dir: str | Path,
        samples_path: Path | None = None,
        limit: int | None = None,
        mark_complete: bool = True,
    ) -> Path:
        """Extract embeddings for all regions in the dataset.

        Parameters
        ----------
        dataset : RegionSequenceDataset in 'image' mode
        save_dir : output directory
        samples_path : path to samples.parquet (for recording hash in manifest)
        limit : optional limit for benchmark

        Returns
        -------
        Path to the embeddings directory.
        """
        import hashlib
        import json

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = save_dir / "manifest.json"
        if manifest_path.exists():
            logger.warning(
                "Existing manifest found at %s. Inactivating cache (complete=False, status='in_progress') "
                "before rebuild to ensure interrupted processes cannot leave a stale complete=True on disk.",
                manifest_path,
            )
        # Invalidate/initialize manifest immediately before creating or touching embeddings.npy
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"complete": False, "status": "in_progress"}, f, indent=2)

        total_n = len(dataset)
        N = min(total_n, limit) if limit is not None else total_n
        mmap_path = save_dir / "embeddings.npy"

        logger.info("Extracting embeddings for %d regions (total=%d, V=%d) → %s",
                     N, total_n, self.n_versions, mmap_path)

        # Real .npy file with header via np.lib.format.open_memmap
        emb_mmap = np.lib.format.open_memmap(
            str(mmap_path), mode="w+",
            dtype=np.float16, shape=(N, 1, 36, 2, 768)
        )

        index_data = []

        for i in tqdm(range(N), desc="Embedding"):
            data = dataset[i]
            region_id = data["region_id"]

            ppl_crops = data["ppl"]
            xpl_crops = data["xpl"]
            mask = data["mask"]

            v_out = self.extract_region(ppl_crops, xpl_crops, mask)
            emb_mmap[i] = v_out
            index_data.append({"region_id": region_id, "cache_row": i})

            if (i + 1) % 10 == 0:
                emb_mmap.flush()
            if (i + 1) % 50 == 0 or (i + 1) == N:
                logger.info("Embedded %d / %d regions (%.1f%%)", i + 1, N, (i + 1) * 100.0 / N)

        emb_mmap.flush()

        # Save index.parquet with region_id, cache_row
        index_df = pd.DataFrame(index_data)
        index_df.to_parquet(save_dir / "index.parquet", index=False)
        logger.info("Saved index.parquet (%d rows)", len(index_df))

        # Samples hash
        samples_hash = ""
        if samples_path and Path(samples_path).is_file():
            with open(samples_path, "rb") as f:
                samples_hash = hashlib.sha256(f.read()).hexdigest()

        # Manifest.json with complete: true ONLY at the very end
        manifest = {
            "N": int(N),
            "shape": list(emb_mmap.shape),
            "dtype": str(emb_mmap.dtype),
            "encoder": "dinov2_vits14",
            "crop_settings": self.cfg.get("dataset", {}),
            "samples_hash": samples_hash,
            "complete": mark_complete,
            "status": "complete" if mark_complete else "benchmark_incomplete",
        }
        manifest_path = save_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Embeddings saved: %s (shape %s) with complete manifest at %s",
                     mmap_path, emb_mmap.shape, manifest_path)
        return save_dir
