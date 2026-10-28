"""Unit tests for V1 model, set_loss, and training mechanics (§9).

Tests:
- Existing:
  - test_set_loss_singleton_equals_cross_entropy
  - test_set_loss_partial_set_and_all_class_skip
  - test_unordered_arm_permutation_invariance
  - test_model_arms_parameter_counts
- New checklist tests:
  - Test J2: Angle embedding is strictly isolated to angle_aware_k12 arm (modifying theta_deg
    does not alter outputs for static_k1 or unordered_k12).
  - Test J4: Padding mask: padded views with angle_mask=False produce identical output (diff <= 1e-6).
  - Test J6: Static arm shape safety: [B, 1, 768] input produces [B, 5].
  - Test K3: Partial set loss exact math: for allowed {0, 1}, set_loss equals -torch.log(p[0] + p[1]).
  - Test K4: Batch with all-empty or all-class rows returns None.
  - Test K6: Static training forward with reshape [B*12, 1, ...] equals average of 12 separate
    single-view forward losses.
  - Test K16: Save model checkpoint to disk, reload into fresh instance, verify identical logits.
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest
import torch
import torch.nn.functional as F

from src.rotapetro.model import RotaPetroNet
from src.rotapetro.train import set_loss


def test_set_loss_singleton_equals_cross_entropy():
    """Unit test: set_loss with singleton sets equals F.cross_entropy (§9b)."""
    torch.manual_seed(42)
    N, C = 20, 5
    logits = torch.randn(N, C)
    targets = torch.randint(0, C, (N,))
    allowed = F.one_hot(targets, C).bool()

    loss_set = set_loss(logits, allowed)
    loss_ce = F.cross_entropy(logits, targets)

    assert loss_set is not None
    assert torch.allclose(loss_set, loss_ce, atol=1e-6), (
        f"set_loss ({loss_set.item():.6f}) != cross_entropy ({loss_ce.item():.6f})"
    )


def test_set_loss_partial_set_and_all_class_skip():
    """Verify set_loss properly handles partial sets and skips uninformative rows."""
    torch.manual_seed(42)
    logits = torch.tensor([[2.0, 1.0, 0.0],
                           [0.0, 2.0, 1.0],
                           [1.0, 1.0, 1.0]])  # row 3 will be uninformative
    allowed = torch.tensor([[True, True, False],   # partial set {0, 1}
                            [False, True, False],  # singleton {1}
                            [True, True, True]])   # uninformative {0, 1, 2}
    loss = set_loss(logits, allowed)
    assert loss is not None
    assert loss.item() > 0.0

    # All uninformative or empty -> returns None
    empty_allowed = torch.zeros(3, 3, dtype=torch.bool)
    assert set_loss(logits, empty_allowed) is None

    all_allowed = torch.ones(3, 3, dtype=torch.bool)
    assert set_loss(logits, all_allowed) is None


def test_unordered_arm_permutation_invariance():
    """Padding/permutation test: shuffling the 12 views gives the same output for unordered arm (§9a)."""
    torch.manual_seed(42)
    model = RotaPetroNet(n_fine_classes=5, arm="unordered_k12")
    model.eval()

    B, K, D = 3, 12, 768
    x_p = torch.randn(B, K, D)
    x_x = torch.randn(B, K, D)
    th = torch.linspace(0, 165, K).unsqueeze(0).expand(B, -1)

    with torch.no_grad():
        out_orig = model(x_p, x_x, th)["mineral"]

    # Random permutation of the 12 views
    perm = torch.randperm(K)
    with torch.no_grad():
        out_shuf = model(x_p[:, perm], x_x[:, perm], th[:, perm])["mineral"]

    max_diff = (out_orig - out_shuf).abs().max().item()
    assert torch.allclose(out_orig, out_shuf, atol=1e-5), (
        f"Unordered arm is not permutation invariant! max_diff = {max_diff}"
    )


def test_model_arms_parameter_counts():
    """Verify parameter counts for static_k1, unordered_k12, and angle_aware_k12."""
    m_static = RotaPetroNet(n_fine_classes=5, arm="static_k1")
    m_unord = RotaPetroNet(n_fine_classes=5, arm="unordered_k12")
    m_angle = RotaPetroNet(n_fine_classes=5, arm="angle_aware_k12")

    p_static = sum(p.numel() for p in m_static.parameters() if p.requires_grad)
    p_unord = sum(p.numel() for p in m_unord.parameters() if p.requires_grad)
    p_angle = sum(p.numel() for p in m_angle.parameters() if p.requires_grad)

    # static and unordered have no angle_emb, so identical parameter counts
    assert p_static == p_unord
    # angle_aware has the angle_emb MLP, so strictly more parameters
    assert p_angle > p_unord


# ── Test J2: Angle embedding strictly isolated ─────────────────────────

def test_j2_angle_embedding_strictly_isolated():
    """Test J2: Modifying theta_deg does not alter outputs for static_k1 or unordered_k12, but alters angle_aware_k12."""
    torch.manual_seed(42)
    B, K, D = 2, 12, 768
    x_p = torch.randn(B, K, D)
    x_x = torch.randn(B, K, D)

    th1 = torch.linspace(0, 165, K).unsqueeze(0).expand(B, -1)
    th2 = torch.linspace(45, 210, K).unsqueeze(0).expand(B, -1)  # modified angles

    m_static = RotaPetroNet(n_fine_classes=5, arm="static_k1").eval()
    m_unord = RotaPetroNet(n_fine_classes=5, arm="unordered_k12").eval()
    m_angle = RotaPetroNet(n_fine_classes=5, arm="angle_aware_k12").eval()

    with torch.no_grad():
        out_static_1 = m_static(x_p, x_x, th1)["mineral"]
        out_static_2 = m_static(x_p, x_x, th2)["mineral"]
        assert torch.equal(out_static_1, out_static_2), "Modifying theta altered static_k1 output"

        out_unord_1 = m_unord(x_p, x_x, th1)["mineral"]
        out_unord_2 = m_unord(x_p, x_x, th2)["mineral"]
        assert torch.equal(out_unord_1, out_unord_2), "Modifying theta altered unordered_k12 output"

        out_angle_1 = m_angle(x_p, x_x, th1)["mineral"]
        out_angle_2 = m_angle(x_p, x_x, th2)["mineral"]
        assert not torch.equal(out_angle_1, out_angle_2), "Modifying theta did not alter angle_aware_k12 output"


# ── Test J4: Padding mask ──────────────────────────────────────────────

def test_j4_padding_mask_ignores_masked_views():
    """Test J4: Padded views with angle_mask=False produce identical output (diff <= 1e-6) when modified."""
    torch.manual_seed(42)
    model = RotaPetroNet(n_fine_classes=5, arm="angle_aware_k12").eval()

    B, K, D = 2, 12, 768
    x_p1 = torch.randn(B, K, D)
    x_x1 = torch.randn(B, K, D)
    th = torch.linspace(0, 165, K).unsqueeze(0).expand(B, -1)

    # First 6 views are real, last 6 are padded
    angle_mask = torch.zeros(B, K, dtype=torch.bool)
    angle_mask[:, :6] = True

    # Perturb the padded views heavily with arbitrary garbage
    x_p2 = x_p1.clone()
    x_x2 = x_x1.clone()
    x_p2[:, 6:] = torch.randn_like(x_p2[:, 6:]) * 50.0
    x_x2[:, 6:] = torch.randn_like(x_x2[:, 6:]) * 50.0

    with torch.no_grad():
        out1 = model(x_p1, x_x1, th, angle_mask=angle_mask)["mineral"]
        out2 = model(x_p2, x_x2, th, angle_mask=angle_mask)["mineral"]

    max_diff = (out1 - out2).abs().max().item()
    assert max_diff <= 1e-6, f"Padding mask failed to isolate masked views: max_diff = {max_diff}"


# ── Test J6: Static arm shape safety ───────────────────────────────────

def test_j6_static_arm_shape_safety():
    """Test J6: Static arm shape safety: [B, 1, 768] input produces [B, 5]."""
    torch.manual_seed(42)
    model = RotaPetroNet(n_fine_classes=5, arm="static_k1").eval()

    B = 7
    x_p = torch.randn(B, 1, 768)
    x_x = torch.randn(B, 1, 768)
    th = torch.zeros(B, 1)

    with torch.no_grad():
        out = model(x_p, x_x, th)["mineral"]

    assert out.shape == (B, 5), f"Expected shape ({B}, 5), got {out.shape}"


# ── Test K3: Partial set loss exact math ────────────────────────────────

def test_k3_partial_set_loss_exact_math():
    """Test K3: For allowed {0, 1}, set_loss equals -torch.log(p[0] + p[1])."""
    torch.manual_seed(42)
    logits = torch.tensor([[1.8, 2.2, -0.4, 0.1, 0.7]])
    allowed = torch.tensor([[True, True, False, False, False]])  # {0, 1}

    loss = set_loss(logits, allowed)
    probs = F.softmax(logits, dim=-1)
    expected = -torch.log(probs[0, 0] + probs[0, 1])

    assert loss is not None
    assert torch.allclose(loss, expected, atol=1e-6), (
        f"Computed set_loss {loss.item():.6f} != expected {expected.item():.6f}"
    )


# ── Test K4: Batch with all-empty or all-class rows returns None ────────

def test_k4_batch_with_all_empty_or_all_class_returns_none():
    """Test K4: Batch with all-empty or all-class rows returns None."""
    logits = torch.randn(4, 5)

    all_empty = torch.zeros(4, 5, dtype=torch.bool)
    assert set_loss(logits, all_empty) is None

    all_class = torch.ones(4, 5, dtype=torch.bool)
    assert set_loss(logits, all_class) is None

    mixed_uninformative = torch.cat([
        torch.zeros(2, 5, dtype=torch.bool),
        torch.ones(2, 5, dtype=torch.bool)
    ], dim=0)
    assert set_loss(logits, mixed_uninformative) is None


# ── Test K6: Static training forward with reshape equals average loss ──

def test_k6_static_training_forward_reshape_equals_average_loss():
    """Test K6: Static training forward with reshape [B*12, 1, ...] equals average of 12 separate single-view losses."""
    torch.manual_seed(42)
    model = RotaPetroNet(n_fine_classes=5, arm="static_k1").eval()

    B, K, D = 1, 12, 768
    x_p = torch.randn(B, K, D)
    x_x = torch.randn(B, K, D)
    th = torch.linspace(0, 165, K).unsqueeze(0).expand(B, -1)
    allowed = torch.tensor([[True, True, False, False, False]])  # [1, 5]

    # 1. Joint forward with reshape [B*K, 1, D]
    x_p_reshaped = x_p.reshape(B * K, 1, -1)
    x_x_reshaped = x_x.reshape(B * K, 1, -1)
    th_reshaped = th.reshape(B * K, 1)
    allowed_interleaved = allowed.repeat_interleave(K, 0)

    with torch.no_grad():
        logits_reshaped = model(x_p_reshaped, x_x_reshaped, th_reshaped)["mineral"]
    loss_joint = set_loss(logits_reshaped, allowed_interleaved)

    # 2. Average of 12 separate single-view forward passes
    losses_sep = []
    for k in range(K):
        with torch.no_grad():
            logits_k = model(x_p[:, k:k + 1], x_x[:, k:k + 1], th[:, k:k + 1])["mineral"]
        loss_k = set_loss(logits_k, allowed)
        assert loss_k is not None
        losses_sep.append(loss_k)

    loss_avg = torch.stack(losses_sep).mean()

    assert torch.allclose(loss_joint, loss_avg, atol=1e-5), (
        f"Joint reshaped loss {loss_joint.item():.6f} != average of 12 views {loss_avg.item():.6f}"
    )


# ── Test K16: Save and reload model checkpoint integrity ───────────────

def test_k16_save_checkpoint_reload_identical_logits(tmp_path: Path):
    """Test K16: Save model checkpoint to disk, reload into fresh instance, verify identical logits."""
    torch.manual_seed(42)
    model1 = RotaPetroNet(n_fine_classes=5, arm="angle_aware_k12").eval()

    ckpt_path = tmp_path / "model_ckpt.pt"
    torch.save(model1.state_dict(), ckpt_path)

    model2 = RotaPetroNet(n_fine_classes=5, arm="angle_aware_k12").eval()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model2.load_state_dict(state_dict)
    model2.eval()

    B, K, D = 3, 12, 768
    x_p = torch.randn(B, K, D)
    x_x = torch.randn(B, K, D)
    th = torch.randn(B, K)

    with torch.no_grad():
        out1 = model1(x_p, x_x, th)["mineral"]
        out2 = model2(x_p, x_x, th)["mineral"]

    assert torch.equal(out1, out2), "Reloaded model produced non-identical logits"


# ── Checklist Aliases & Explicit Tests (V1_CODE_CHECKLIST.md) ───────────

def test_k2_singleton_set_loss_equals_cross_entropy():
    """Test K2: set_loss with singleton sets equals F.cross_entropy within 1e-6 (§9b)."""
    test_set_loss_singleton_equals_cross_entropy()


def test_j3_unordered_permutation_invariance():
    """Test J3: Shuffling the 12 views gives the same logits within 1e-5."""
    test_unordered_arm_permutation_invariance()


def test_k7_training_views_identical_across_all_arms(tmp_path: Path):
    """Test K7: For the same seed and epoch, training-view angle indices are identical across all 3 arms."""
    from src.rotapetro.dataset import V1EmbeddingDataset
    import numpy as np
    import pandas as pd

    # Minimal dummy dataset frames
    index_df = pd.DataFrame([{"region_id": "r1", "code": "spec1", "cache_row": 0}])
    targets_df = pd.DataFrame([{"region_id": "r1", "code": "spec1", "state": "singleton", "allowed": ["quartz"]}])
    families = ["quartz", "feldspar", "mica", "pyroxene", "opaque_material"]

    emb_path = tmp_path / "emb_k7.npy"
    m = np.lib.format.open_memmap(str(emb_path), mode="w+", dtype=np.float16, shape=(1, 1, 36, 2, 768))
    m.flush()
    del m

    import json
    manifest = {
        "N": 1,
        "complete": True,
        "shape": [1, 1, 36, 2, 768],
        "dtype": "float16",
    }
    with open(tmp_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    thetas = {}
    datasets = []
    for arm in ["static_k1", "unordered_k12", "angle_aware_k12"]:
        ds = V1EmbeddingDataset(
            emb_path=emb_path,
            index_df=index_df,
            targets_df=targets_df,
            families=families,
            split_codes={"spec1"},
            phase="train",
            arm=arm,
            seed=42,
        )
        ds.set_epoch(3)
        thetas[arm] = ds[0]["theta_deg"]
        datasets.append(ds)

    assert torch.equal(thetas["static_k1"], thetas["unordered_k12"])
    assert torch.equal(thetas["unordered_k12"], thetas["angle_aware_k12"])

    # Release open memmap handles on Windows
    for ds in datasets:
        if hasattr(ds, "emb") and hasattr(ds.emb, "_mmap") and ds.emb._mmap is not None:
            ds.emb._mmap.close()
    del datasets
