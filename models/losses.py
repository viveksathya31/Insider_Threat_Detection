"""
Loss functions for extreme class imbalance in Insider Threat Detection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    Numerically stable Binary Focal Loss with Logits for extreme class imbalance.
    
    Down-weights easy routine negatives (benign user-days) to concentrate gradient
    updates on ambiguous boundary cases and rare malicious anomalies.
    
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma: float = 1.5, alpha: float = 0.75, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_weight = alpha_t * ((1.0 - p_t) ** self.gamma)
        loss = focal_weight * bce_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
