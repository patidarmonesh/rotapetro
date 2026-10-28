"""RotaPetroNet — Transformer classifier for petrographic mineral ID.

V1 changes:
- Three explicit modes: static_k1, unordered_k12, angle_aware_k12
- Angle embedding is configurable (enabled only for angle_aware_k12)
- Group head and property heads disabled for V1 (only mineral_head)
- Physics head remains optional (disabled for V1)
"""
import torch
import torch.nn as nn
import math


class AngleEmbedding(nn.Module):
    """Learnable angle encoding from sin/cos of 2θ and 4θ."""

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, d_model),
        )

    def forward(self, theta_deg: torch.Tensor) -> torch.Tensor:
        # theta_deg: [B, T]
        theta_rad = theta_deg * math.pi / 180.0
        s2 = torch.sin(2 * theta_rad)
        c2 = torch.cos(2 * theta_rad)
        s4 = torch.sin(4 * theta_rad)
        c4 = torch.cos(4 * theta_rad)
        x = torch.stack([s2, c2, s4, c4], dim=-1)  # [B, T, 4]
        return self.mlp(x)  # [B, T, d_model]


class RotaPetroNet(nn.Module):
    """V1 Transformer classifier with three training modes.

    Modes
    -----
    static_k1 : single angle (index from config), no angle embedding.
    unordered_k12 : 12 equally-spaced angles, no angle embedding.
        Angles are randomly permuted during training so the model
        cannot learn positional/angle information.
    angle_aware_k12 : 12 equally-spaced angles with angle embedding.
        The model receives the rotation angle as an input signal.

    Parameters
    ----------
    n_fine_classes : int
        Number of mineral classes for the mineral_head.
    cfg : dict
        Full project config.  Reads ``v1.training.mode`` to select mode.
    """

    VALID_MODES = {"static_k1", "unordered_k12", "angle_aware_k12"}

    def __init__(self, n_fine_classes: int, cfg: dict | None = None, arm: str | None = None):
        super().__init__()
        self.d_model = 256

        # Determine mode
        v1_cfg = (cfg or {}).get("v1", {}).get("training", {})
        self.mode = arm or v1_cfg.get("mode", v1_cfg.get("arm", "angle_aware_k12"))
        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown model mode '{self.mode}'. "
                f"Must be one of {self.VALID_MODES}")

        # Whether to use angle embedding
        self.use_angle_emb = (self.mode == "angle_aware_k12")

        # Input projections: separate linear for PPL and XPL embeddings
        self.w_p = nn.Linear(768, self.d_model)
        self.w_x = nn.Linear(768, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)

        # Angle embedding (only used when mode == angle_aware_k12)
        if self.use_angle_emb:
            self.angle_emb = AngleEmbedding(self.d_model)
        else:
            self.angle_emb = None

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2)

        # Pooling: concat CLS + mean-angle → d_model
        self.pool = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # ── Heads ───────────────────────────────────────────────
        # V1: only mineral_head is active.
        self.mineral_head = nn.Linear(self.d_model, n_fine_classes)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        import logging
        logging.getLogger(__name__).info(
            "RotaPetroNet initialized [arm=%s]: %d trainable parameters", self.mode, n_params
        )

        # V1: these are NOT created (saves memory, prevents accidental use)
        # group_head, prop_optic, prop_pleo, prop_biref, prop_ext, prop_twin
        # were in the baseline but are disabled per the plan.

    def forward(
        self,
        emb_ppl: torch.Tensor,
        emb_xpl: torch.Tensor,
        theta_deg: torch.Tensor,
        angle_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        emb_ppl : [B, T, 768]  PPL DINO embeddings
        emb_xpl : [B, T, 768]  XPL DINO embeddings
        theta_deg : [B, T]     rotation angles in degrees
        angle_mask : [B, T]    True = real angle, False = padding

        Returns
        -------
        dict with key ``'mineral'`` → logits [B, n_fine_classes].
        """
        B, T, _ = emb_ppl.shape

        # Combine PPL + XPL embeddings
        t_k = self.w_p(emb_ppl) + self.w_x(emb_xpl)
        t_k = self.norm(t_k)

        # Add angle embedding only for angle_aware_k12
        if self.use_angle_emb and self.angle_emb is not None:
            t_k = t_k + self.angle_emb(theta_deg)

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, t_k], dim=1)  # [B, T+1, d_model]

        # Transformer with optional padding mask
        if angle_mask is not None:
            padding_mask = torch.cat([
                torch.zeros(B, 1, dtype=torch.bool, device=x.device),
                ~angle_mask,  # True = masked out
            ], dim=1)
            out = self.transformer(x, src_key_padding_mask=padding_mask)
        else:
            out = self.transformer(x)

        # Pool: CLS + mean-over-angles
        cls_out = out[:, 0]
        if angle_mask is not None:
            angle_outs = out[:, 1:]
            mask_float = angle_mask.unsqueeze(-1).float()
            mean_angles = (
                (angle_outs * mask_float).sum(dim=1)
                / mask_float.sum(dim=1).clamp(min=1)
            )
        else:
            mean_angles = out[:, 1:].mean(dim=1)

        h = self.pool(torch.cat([cls_out, mean_angles], dim=-1))

        return {"mineral": self.mineral_head(h)}
