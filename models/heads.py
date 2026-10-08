"""
Task heads for Insider Threat Detection.

Phase 2 in the master roadmap.
Implements three distinct modeling paradigms on top of the shared HeteroGNNEncoder:
  1. AEHead             : Unsupervised behavioral reconstruction (reconstructs 139-dim interaction vector).
  2. OneClassHead (OC)  : Unsupervised Deep SVDD (encloses normal embeddings in a minimal hypersphere centered at c).
  3. ClassificationHead : Supervised classification (weighted BCE for 1:500 class imbalance).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AEHead(nn.Module):
    """
    Unsupervised Autoencoder Head.
    Decodes user embedding back into their 139-dim daily behavioral interaction vector.
    """

    def __init__(self, embedding_dim: int, target_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, target_dim),
        )

    def forward(self, user_embedding: torch.Tensor) -> torch.Tensor:
        return self.decoder(user_embedding)


class OneClassHead(nn.Module):
    """
    Unsupervised Deep SVDD (One-Class) Head.
    Maps user embeddings into a latent representation space where normal (benign)
    behavior is mapped as close as possible to a center vector c.
    
    The anomaly score is the squared Euclidean distance: ||phi(z) - c||^2.
    Anomalous users who behave abnormally will map far outside the hypersphere.
    """

    def __init__(self, embedding_dim: int, projection_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.projection_dim = projection_dim
        # Hypersphere center c is a fixed non-zero buffer (held constant during training to prevent collapse)
        self.register_buffer("c", torch.zeros(projection_dim))
        self.c_initialized = False

    def init_center(self, embeddings: torch.Tensor, eps: float = 0.1):
        """Initializes the hypersphere center c as the empirical mean of embeddings."""
        with torch.no_grad():
            if embeddings.shape[-1] != self.projection_dim:
                projected = self.projection(embeddings)
            else:
                projected = embeddings
            c = projected.mean(dim=0)
            # Avoid center being too close to zero (prevents trivial all-zero collapse)
            c[(c.abs() < eps) & (c >= 0)] = eps
            c[(c.abs() < eps) & (c < 0)] = -eps
            self.c.copy_(c)
            self.c_initialized = True

    def forward(self, user_embedding: torch.Tensor) -> tuple:
        """
        Returns:
            projected: Tensor[n_users, projection_dim]
            distances: Tensor[n_users] -- squared Euclidean distance to center c (the anomaly score)
        """
        projected = self.projection(user_embedding)
        diff = projected - self.c.unsqueeze(0)
        distances = (diff ** 2).sum(dim=1)  # [n_users]
        return projected, distances


class ClassificationHead(nn.Module):
    """
    Supervised Binary Classification Head.
    Maps user embedding to a single logit representing malicious probability.
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = 32, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, user_embedding: torch.Tensor) -> torch.Tensor:
        """Returns logits of shape [n_users]."""
        return self.classifier(user_embedding).squeeze(-1)
