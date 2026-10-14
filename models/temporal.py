"""
Temporal Sequence Modeling for Insider Threat Detection.

Phase 3.2 in the master roadmap.
Processes rolling sequences of daily user embeddings (length W days) to detect:
  - Cumulative behavioral drift (Scenario 2, Scenario 4)
  - Multi-day attack escalation
  - Temporal attention attribution (which days triggered the alert)
"""
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Self-attention pooling mechanism over sequence dimension:
        e_t = w^T tanh(W_h h_t + b)
        a_t = softmax(e_t)
        context = sum_t a_t * h_t
    Returns (context, attn_weights) for explainability.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h: [batch_size, seq_len, hidden_dim]
            mask: [batch_size, seq_len] boolean mask (True for valid positions)
        Returns:
            context: [batch_size, hidden_dim]
            weights: [batch_size, seq_len]
        """
        energy = torch.tanh(self.proj(h))  # [B, T, H]
        scores = self.score(energy).squeeze(-1)  # [B, T]

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)  # [B, T]
        context = torch.bmm(weights.unsqueeze(1), h).squeeze(1)  # [B, H]
        return context, weights


class TemporalGRU(nn.Module):
    """
    Causal Temporal Gated Recurrent Unit with Attention Pooling.
    
    Architecture:
      1. Feature Projection (Linear + LayerNorm + ReLU + Dropout)
      2. Multi-layer causal GRU
      3. Attention Pooling over sequence steps
      4. Classification Head (BCE logit)
    
    Outputs:
      logits: [batch_size] threat score logit for the sequence
      attn_weights: [batch_size, seq_len] attention over days in window
    """

    def __init__(
        self,
        input_dim: int = 66,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Input projection & normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Causal GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention pooling
        self.attention = TemporalAttention(hidden_dim)

        # Prediction head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: optional [batch_size, seq_len]
        Returns:
            logits: [batch_size]
            attn_weights: [batch_size, seq_len]
        """
        projected = self.input_proj(x)  # [B, T, H]
        out, _ = self.gru(projected)     # [B, T, H]
        context, attn_weights = self.attention(out, mask=mask)  # [B, H], [B, T]
        logits = self.classifier(context).squeeze(-1)           # [B]
        return logits, attn_weights

    def predict_risk(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (probabilities in [0, 1], attention weights)."""
        logits, attn_weights = self.forward(x, mask=mask)
        probs = torch.sigmoid(logits)
        return probs, attn_weights


class CumulativeDriftDetector:
    """
    Statistical embedding drift detector for long-term behavioral shifting.
    Tracks a user's running baseline embedding mu_u and computes rolling
    L2 / Cosine deviation:
        drift(u, t) = 1 - cosine_similarity(h_{u, t}, mu_u)
    """

    def __init__(self, baseline_days: int = 30):
        self.baseline_days = baseline_days
        self.user_baselines = {}

    def fit_baseline(self, embeddings: torch.Tensor):
        """
        Computes baseline centroid for each user over the first `baseline_days`.
        Args:
            embeddings: [T, N_users, D]
        """
        T = min(self.baseline_days, embeddings.shape[0])
        # Average across initial baseline days
        self.user_baselines = embeddings[:T].mean(dim=0)  # [N_users, D]

    def compute_drift(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Computes cosine distance from baseline for all (T, N_users).
        Returns:
            drift: [T, N_users] in range [0, 2]
        """
        if self.user_baselines is None:
            raise ValueError("Baseline has not been fitted. Call fit_baseline first.")

        # Normalize baseline: [N_users, D]
        base_norm = F.normalize(self.user_baselines, p=2, dim=-1)
        # Normalize embeddings: [T, N_users, D]
        emb_norm = F.normalize(embeddings, p=2, dim=-1)

        # Cosine similarity: sum over feature dim
        # base_norm unsqueezed to [1, N_users, D]
        cos_sim = (emb_norm * base_norm.unsqueeze(0)).sum(dim=-1)  # [T, N_users]
        drift = 1.0 - cos_sim  # 0 = identical, 2 = opposite
        return drift


class MultiScaleTemporalGRU(nn.Module):
    """
    Multi-Scale Hierarchical Temporal GRU with Cross-Scale Attention Gating.

    Fuses three complementary temporal horizons:
      - Short (3-Day): Captures high-frequency exfiltration bursts & panic actions
      - Tactical (14-Day): Captures multi-stage staging & reconnaissance
      - Strategic (60-Day): Captures slow-burn behavioral drift & permission creep

    Outputs:
      logits: [batch_size]
      scale_weights: [batch_size, 3] adaptive attention weights (3d, 14d, 60d)
      attns: tuple of (w_3d, w_14d, w_60d)
    """

    def __init__(
        self,
        input_dim: int = 66,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Shared input projection & normalization
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # 3 Dedicated Temporal Scale Encoders
        self.gru_3d = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gru_14d = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.gru_60d = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Attention pooling per scale
        self.attn_3d = TemporalAttention(hidden_dim)
        self.attn_14d = TemporalAttention(hidden_dim)
        self.attn_60d = TemporalAttention(hidden_dim)

        # Hierarchical Cross-Scale Attention Gating
        self.scale_proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale_score = nn.Linear(hidden_dim, 1, bias=False)

        # Unified Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        x_60: torch.Tensor,
        mask_60: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            x_60: [batch_size, 60, input_dim] causal rolling window
            mask_60: [batch_size, 60] boolean mask (True for valid days, False for left-padded days)
        Returns:
            logits: [batch_size]
            scale_weights: [batch_size, 3] (weights for 3d, 14d, 60d)
            (w_3d, w_14d, w_60d): attention weights per scale
        """
        if mask_60 is None and mask is not None:
            mask_60 = mask
        proj = self.input_proj(x_60)  # [B, 60, H]

        # Multi-scale sequence slicing
        proj_3d = proj[:, -3:, :]    # [B, 3, H]
        proj_14d = proj[:, -14:, :]  # [B, 14, H]
        proj_60d = proj              # [B, 60, H]

        out_3d, _ = self.gru_3d(proj_3d)
        out_14d, _ = self.gru_14d(proj_14d)
        out_60d, _ = self.gru_60d(proj_60d)

        c_3d, w_3d = self.attn_3d(out_3d)
        c_14d, w_14d = self.attn_14d(out_14d)
        c_60d, w_60d = self.attn_60d(out_60d, mask=mask_60)

        # Stack scale context representations: [B, 3, H]
        contexts = torch.stack([c_3d, c_14d, c_60d], dim=1)

        # Compute adaptive cross-scale gating weights
        energy = torch.tanh(self.scale_proj(contexts))       # [B, 3, H]
        scale_logits = self.scale_score(energy).squeeze(-1)  # [B, 3]
        scale_weights = F.softmax(scale_logits, dim=-1)      # [B, 3]

        # Adaptively fuse representations across scales: [B, H]
        fused = torch.bmm(scale_weights.unsqueeze(1), contexts).squeeze(1)

        logits = self.classifier(fused).squeeze(-1)
        return logits, scale_weights, (w_3d, w_14d, w_60d)

    def predict_risk(
        self,
        x_60: torch.Tensor,
        mask_60: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Returns:
            probs: [batch_size] risk probabilities in [0, 1]
            scale_weights: [batch_size, 3] (proportions for 3d, 14d, 60d)
            attns: dict with keys 'attn_3d', 'attn_14d', 'attn_60d'
        """
        logits, scale_weights, (w_3d, w_14d, w_60d) = self.forward(x_60, mask=mask_60)
        probs = torch.sigmoid(logits)
        attns = {
            "attn_3d": w_3d,
            "attn_14d": w_14d,
            "attn_60d": w_60d,
        }
        return probs, scale_weights, attns

