"""
Unit and regression tests for Temporal GNN models and loss functions.
Validates:
- BinaryFocalLoss correctness and gradient flow
- TemporalGRU (14-day production default) architecture, attention, and inference
- MultiScaleTemporalGRU (60-day strategic optional) hierarchical gating and attention
- Checkpoint integrity and state dictionary compatibility
- SOC export schema validation (both 14d default and 60d optional)
"""

import json
import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
import torch.nn as nn

from models.losses import BinaryFocalLoss
from models.temporal import TemporalGRU, MultiScaleTemporalGRU


class TestBinaryFocalLoss(unittest.TestCase):
    def test_loss_computation(self):
        loss_fn = BinaryFocalLoss(gamma=1.5, alpha=0.75)
        logits = torch.tensor([2.0, -2.0, 0.0, 5.0], requires_grad=True)
        targets = torch.tensor([1.0, 0.0, 1.0, 0.0])

        loss = loss_fn(logits, targets)
        assert loss.ndim == 0
        assert loss.item() > 0.0

        # Verify gradient flow
        loss.backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_focal_downweighting(self):
        loss_fn = BinaryFocalLoss(gamma=2.0, alpha=0.5)
        # Easy positive (high logit) vs hard positive (low logit)
        easy_logit = torch.tensor([5.0])
        hard_logit = torch.tensor([-2.0])
        target = torch.tensor([1.0])

        loss_easy = loss_fn(easy_logit, target)
        loss_hard = loss_fn(hard_logit, target)
        assert loss_hard.item() > loss_easy.item()


class TestTemporalGRU(unittest.TestCase):
    def test_forward_and_attention(self):
        batch_size = 4
        window_size = 14
        input_dim = 66
        hidden_dim = 64

        model = TemporalGRU(input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2, dropout=0.2)
        model.eval()

        x = torch.randn(batch_size, window_size, input_dim)
        logits, attn = model(x)

        assert logits.shape == (batch_size,)
        assert attn.shape == (batch_size, window_size)

        # Attention weights should sum to 1.0 across time steps
        attn_sums = attn.sum(dim=1)
        torch.testing.assert_close(attn_sums, torch.ones_like(attn_sums), atol=1e-5, rtol=1e-5)

        # Predict risk probabilities in [0, 1]
        probs, attn_out = model.predict_risk(x)
        assert probs.shape == (batch_size,)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()


class TestMultiScaleTemporalGRU(unittest.TestCase):
    def test_multiscale_forward_and_gating(self):
        batch_size = 4
        window_size = 60
        input_dim = 66
        hidden_dim = 64

        model = MultiScaleTemporalGRU(
            input_dim=input_dim, hidden_dim=hidden_dim, num_layers=2, dropout=0.2
        )
        model.eval()

        x = torch.randn(batch_size, window_size, input_dim)
        logits, scale_weights, (w_3d, w_14d, w_60d) = model(x)

        assert logits.shape == (batch_size,)
        assert scale_weights.shape == (batch_size, 3)

        # Scale weights must sum to 1.0
        scale_sums = scale_weights.sum(dim=1)
        torch.testing.assert_close(scale_sums, torch.ones_like(scale_sums), atol=1e-5, rtol=1e-5)

        # Verify sub-attention shapes and sums
        assert w_3d.shape == (batch_size, 3)
        assert w_14d.shape == (batch_size, 14)
        assert w_60d.shape == (batch_size, 60)

        for w in [w_3d, w_14d, w_60d]:
            s = w.sum(dim=1)
            torch.testing.assert_close(s, torch.ones_like(s), atol=1e-5, rtol=1e-5)

        # Predict risk
        probs, sw_out, attns_out = model.predict_risk(x)
        assert probs.shape == (batch_size,)
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        assert "attn_3d" in attns_out and "attn_14d" in attns_out and "attn_60d" in attns_out


class TestCheckpointsAndArtifacts(unittest.TestCase):
    def test_14d_checkpoint_loads(self):
        ckpt_path = ROOT_DIR / "output" / "checkpoints" / "temporal_best.pt"
        assert ckpt_path.exists(), f"Missing 14-day production checkpoint at {ckpt_path}"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt.get("window_size", 14) == 14
        assert ckpt["hidden_dim"] == 64

        model = TemporalGRU(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy_in = torch.randn(2, 14, ckpt["input_dim"])
        probs, _ = model.predict_risk(dummy_in)
        assert probs.shape == (2,)

    def test_60d_multiscale_checkpoint_loads(self):
        ckpt_path = ROOT_DIR / "output" / "checkpoints" / "temporal_multiscale_best.pt"
        assert ckpt_path.exists(), f"Missing 60-day multi-scale checkpoint at {ckpt_path}"

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert ckpt.get("model_type") == "multiscale"
        assert ckpt.get("window_size") == 60

        model = MultiScaleTemporalGRU(
            input_dim=ckpt["input_dim"],
            hidden_dim=ckpt["hidden_dim"],
            num_layers=ckpt["num_layers"],
            dropout=ckpt["dropout"],
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        dummy_in = torch.randn(2, 60, ckpt["input_dim"])
        probs, sw, _ = model.predict_risk(dummy_in)
        assert probs.shape == (2,)
        assert sw.shape == (2, 3)

    def test_soc_data_schema(self):
        # 14-day default report
        rep_14d = ROOT_DIR / "output" / "reports" / "soc_dashboard_data.json"
        assert rep_14d.exists(), f"Missing {rep_14d}"
        with open(rep_14d, "r") as f:
            data_14d = json.load(f)

        assert "metadata" in data_14d
        assert data_14d["metadata"]["model_id"] == "14d"
        assert data_14d["metadata"]["is_default"] is True
        assert "kpi" in data_14d
        assert data_14d["kpi"]["temporal_pr_auc"] >= 0.67
        assert "incidents" in data_14d
        assert len(data_14d["incidents"]) > 0

        # 60-day optional report
        rep_60d = ROOT_DIR / "output" / "reports" / "soc_dashboard_data_60d.json"
        assert rep_60d.exists(), f"Missing {rep_60d}"
        with open(rep_60d, "r") as f:
            data_60d = json.load(f)

        assert "metadata" in data_60d
        assert data_60d["metadata"]["model_id"] == "60d"
        assert data_60d["metadata"]["is_default"] is False
        assert "kpi" in data_60d
        assert data_60d["kpi"]["temporal_pr_auc"] >= 0.57
        assert "incidents" in data_60d
        assert len(data_60d["incidents"]) > 0


if __name__ == "__main__":
    unittest.main()
