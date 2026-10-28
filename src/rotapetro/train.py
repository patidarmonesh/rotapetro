"""V1 Trainer — simplified training loop for mineral classification.

Changes from baseline:
- Only mineral_head loss (no group, property, or MIL losses)
- Reads hyperparams from v1.training config section
- label_smoothing=0 (disabled per V1 plan)
- Cosine LR with warmup
- Gradient clipping
- Mixed precision (AMP)
- Checkpoint best model by validation mean set-loss
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

logger = logging.getLogger(__name__)


def set_loss(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor | None:
    """Set-loss for partial credit learning over candidate family sets (§9b).

    Parameters
    ----------
    logits : [N, C] float tensor
    allowed : [N, C] bool tensor indicating allowable candidate classes

    Returns
    -------
    Negative log-likelihood of the allowed set, averaged across informative samples.
    Returns None if no rows have informative allowed sets.
    """
    keep = allowed.any(dim=1) & ~allowed.all(dim=1)
    if not keep.any():
        return None
    logp = torch.log_softmax(logits[keep].float(), dim=-1)
    lp_set = torch.logsumexp(logp.masked_fill(~allowed[keep], -1e4), dim=-1)
    return -lp_set.mean()


class Trainer:
    """V1 training loop with set-loss and fair static training (§9b)."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: dict,
        device: str = "cuda",
        output_dir: str | Path = "outputs/v1",
        arm: str | None = None,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg

        v1_train = cfg.get("v1", {}).get("training", {})
        self.arm = arm or v1_train.get("mode", v1_train.get("arm", getattr(model, "mode", "angle_aware_k12")))
        self.epochs = int(v1_train.get("epochs", 80))
        self.patience = int(v1_train.get("patience", 15))
        self.grad_clip = float(v1_train.get("grad_clip", 1.0))
        self.lr = float(v1_train.get("lr", 5e-4))
        self.wd = float(v1_train.get("weight_decay", 0.05))

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.wd)

        self.scaler = torch.amp.GradScaler(
            device=self.device.type, enabled=(self.device.type == "cuda"))

        self.history: list[dict] = []
        self.best_val_loss = float("inf")
        self.best_val_metrics: dict = {}
        self.patience_counter = 0

    def _cosine_lr(self, epoch: int) -> float:
        """Warmup (3 epochs) + cosine decay."""
        warmup = 3
        min_lr = self.lr * 0.01
        if epoch < warmup:
            return self.lr * ((epoch + 1) / warmup)
        progress = (epoch - warmup) / max(1, self.epochs - warmup)
        return min_lr + 0.5 * (self.lr - min_lr) * (1 + math.cos(math.pi * progress))

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        if hasattr(self.train_loader.dataset, "set_epoch"):
            self.train_loader.dataset.set_epoch(epoch)

        lr = self._cosine_lr(epoch)
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

        epoch_losses = []
        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)

            x_p = batch["emb_ppl"].to(self.device).float()   # [B, 12, 768]
            x_x = batch["emb_xpl"].to(self.device).float()   # [B, 12, 768]
            th = batch["theta_deg"].to(self.device).float()  # [B, 12]
            allowed = batch["allowed"].to(self.device)       # [B, C]

            with torch.amp.autocast(
                    device_type=self.device.type,
                    enabled=(self.device.type == "cuda")):
                if self.arm == "static_k1":
                    B, K = x_p.shape[:2]
                    logits = self.model(
                        x_p.reshape(B * K, 1, -1),
                        x_x.reshape(B * K, 1, -1),
                        th.reshape(B * K, 1)
                    )["mineral"]  # [B*K, C]
                    loss = set_loss(logits, allowed.repeat_interleave(K, 0))
                else:
                    logits = self.model(x_p, x_x, th)["mineral"]  # [B, C]
                    loss = set_loss(logits, allowed)

            if loss is None:
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            epoch_losses.append(loss.item())

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        return {"loss": avg_loss, "lr": lr}

    @torch.no_grad()
    def validate(self) -> dict[str, Any]:
        """Validate on validation set using mean set-loss (§9b)."""
        self.model.eval()
        val_loss_sum = 0.0
        val_n_samples = 0
        total_eval_k = None

        for batch in self.val_loader:
            x_p = batch["emb_ppl"].to(self.device).float()
            x_x = batch["emb_xpl"].to(self.device).float()
            th = batch["theta_deg"].to(self.device).float()
            allowed = batch["allowed"].to(self.device)
            bs = x_p.shape[0]

            K = x_p.shape[1]
            if total_eval_k is None:
                total_eval_k = K

            if self.arm == "static_k1" and K > 1:
                B, K_views = x_p.shape[:2]
                logits = self.model(
                    x_p.reshape(B * K_views, 1, -1),
                    x_x.reshape(B * K_views, 1, -1),
                    th.reshape(B * K_views, 1)
                )["mineral"]
                loss = set_loss(logits, allowed.repeat_interleave(K_views, 0))
            else:
                logits = self.model(x_p, x_x, th)["mineral"]
                loss = set_loss(logits, allowed)

            if loss is not None:
                val_loss_sum += loss.item() * bs
                val_n_samples += bs

        mean_val_loss = val_loss_sum / max(val_n_samples, 1) if val_n_samples > 0 else float("inf")
        logger.info("Validation [arm=%s, eval_K=%d]: mean set-loss = %.4f",
                    self.arm, total_eval_k or 0, mean_val_loss)
        return {
            "val_loss": mean_val_loss,
            "eval_k": total_eval_k,
        }

    def save_checkpoint(self, epoch: int, is_best: bool):
        """Save model checkpoint."""
        state = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "arm": self.arm,
            "v1_config": self.cfg.get("v1", {}),
        }
        torch.save(state, self.output_dir / "last.pt")
        if is_best:
            torch.save(state, self.output_dir / "best.pt")

    def train(self) -> dict[str, Any]:
        """Full training loop with validation set-loss early stopping (§9b)."""
        logger.info("Starting V1 training: arm=%s, %d epochs, patience=%d, lr=%.1e",
                     self.arm, self.epochs, self.patience, self.lr)

        for epoch in range(self.epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()
            val_loss = val_metrics["val_loss"]

            record = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_loss,
                "lr": train_metrics["lr"],
            }
            self.history.append(record)

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.best_val_metrics = val_metrics
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            self.save_checkpoint(epoch, is_best)

            logger.info(
                "Epoch %03d | train_loss=%.4f | val_loss=%.4f%s | lr=%.1e | pat=%d",
                epoch, train_metrics["loss"], val_loss,
                " *" if is_best else "",
                train_metrics["lr"],
                self.patience_counter,
            )

            if self.patience_counter >= self.patience:
                logger.info("Early stopping at epoch %d (best val_loss=%.4f)",
                            epoch, self.best_val_loss)
                break

        # Save logs
        pd.DataFrame(self.history).to_csv(
            self.output_dir / "train_log.csv", index=False)
        results = {
            "best_val_loss": self.best_val_loss,
            "best_val_metrics": self.best_val_metrics,
            "total_epochs": len(self.history),
            "arm": self.arm,
        }
        with open(self.output_dir / "train_summary.json", "w") as f:
            json.dump(results, f, indent=2)

        return results
