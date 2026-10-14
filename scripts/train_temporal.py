"""
Training Script for Temporal GRU Insider Threat Detection (Phase 3.3).

Processes rolling sequences of daily multi-model user features (length W days)
to detect cumulative behavioral drift and multi-day insider attacks.
Uses weighted BCEWithLogitsLoss to handle extreme class imbalance.

Usage:
    venv/bin/python scripts/train_temporal.py --epochs 10 --window_size 14 --batch_size 512
"""
import argparse
import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from models import TemporalGRU, MultiScaleTemporalGRU, load_user_split_masks, BinaryFocalLoss

FEATURES_PATH = Path("output/temporal_features.pt")
CHECKPOINT_DIR = Path("output/checkpoints")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RollingWindowDataset(Dataset):
    """
    Zero-copy dataset providing rolling windows from precomputed tensors:
        features: [T, N_users, D]
        labels:   [T, N_users]
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 user_indices: torch.Tensor, window_size: int = 14):
        self.features = features  # [T, N_users, D]
        self.labels = labels      # [T, N_users]
        self.window_size = window_size
        self.T, self.N_users, self.D = features.shape

        samples = []
        user_list = user_indices.tolist()
        for t in range(window_size - 1, self.T):
            for u in user_list:
                samples.append((t, u))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        t, u = self.samples[idx]
        x = self.features[t - self.window_size + 1 : t + 1, u, :]  # [W, D]
        y = self.labels[t, u]                                      # scalar
        return x, y


class MultiScaleWindowDataset(Dataset):
    """
    Multi-Scale causal dataset providing 60-day historical windows with left-padding
    and boolean valid-day masking for early calendar days (days 13 to 59).
    """

    def __init__(self, features: torch.Tensor, labels: torch.Tensor,
                 user_indices: torch.Tensor, max_window: int = 60, min_window: int = 14):
        self.features = features
        self.labels = labels
        self.max_window = max_window
        self.min_window = min_window
        self.T, self.N_users, self.D = features.shape

        samples = []
        user_list = user_indices.tolist()
        for t in range(min_window - 1, self.T):
            for u in user_list:
                samples.append((t, u))
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        t, u = self.samples[idx]
        start_t = max(0, t - self.max_window + 1)
        actual_seq = self.features[start_t : t + 1, u, :]  # [L, D]
        actual_len = actual_seq.shape[0]

        if actual_len < self.max_window:
            pad_len = self.max_window - actual_len
            pad = torch.zeros(pad_len, self.D, dtype=actual_seq.dtype)
            x_60 = torch.cat([pad, actual_seq], dim=0)  # [60, D]
            mask_60 = torch.cat([
                torch.zeros(pad_len, dtype=torch.bool),
                torch.ones(actual_len, dtype=torch.bool),
            ], dim=0)
        else:
            x_60 = actual_seq
            mask_60 = torch.ones(self.max_window, dtype=torch.bool)

        y = self.labels[t, u]
        return x_60, mask_60, y


def compute_pos_weight(dataset: Dataset) -> float:
    pos_count = 0
    total_count = len(dataset)
    for t, u in dataset.samples:
        if dataset.labels[t, u] > 0.5:
            pos_count += 1
    neg_count = total_count - pos_count
    pos_weight = neg_count / max(pos_count, 1)
    print(f"[train_temporal] Dataset: {pos_count} positive / {neg_count} negative sequences (pos_weight: {pos_weight:.1f})", flush=True)
    return pos_weight


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
                    criterion: nn.Module, device: torch.device, grad_clip: float = 5.0) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        optimizer.zero_grad()
        if len(batch) == 3:
            x, mask, y = batch
            x, mask, y = x.to(device), mask.to(device), y.to(device).float()
            out = model(x, mask_60=mask) if hasattr(model, "gru_60d") else model(x)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            x, y = batch
            x, y = x.to(device), y.to(device).float()
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out

        loss = criterion(logits, y)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item() * len(y)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple:
    model.eval()
    all_probs, all_labels = [], []

    for batch in loader:
        if len(batch) == 3:
            x, mask, y = batch
            x, mask = x.to(device), mask.to(device)
            out = model(x, mask_60=mask) if hasattr(model, "gru_60d") else model(x)
            logits = out[0] if isinstance(out, tuple) else out
        else:
            x, y = batch
            x = x.to(device)
            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out

        probs = torch.sigmoid(logits).cpu().numpy()
        labels = y.numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    flat_probs = np.concatenate(all_probs)
    flat_labels = np.concatenate(all_labels)

    pr_auc = 0.0
    roc_auc = 0.0
    if flat_labels.sum() > 0:
        pr_auc = float(average_precision_score(flat_labels, flat_probs))
        roc_auc = float(roc_auc_score(flat_labels, flat_probs))

    return pr_auc, roc_auc


def main():
    parser = argparse.ArgumentParser(description="Train Temporal GRU Model for Insider Threat Detection")
    parser.add_argument("--model", type=str, default="14d", choices=["14d", "single", "60d", "multiscale"], help="Architecture: '14d' / 'single' (14-day production default) or '60d' / 'multiscale' (hierarchical 3d/14d/60d)")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--window_size", type=int, default=14, help="Rolling window size in days (for single model)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GRU hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "bce"], help="Loss function (focal or bce)")
    parser.add_argument("--gamma", type=float, default=1.5, help="Focal loss gamma parameter")
    parser.add_argument("--alpha", type=float, default=0.75, help="Focal loss alpha weighting")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[train_temporal] Using compute device: {device}", flush=True)

    # 1. Load Precomputed Temporal Features
    print(f"[train_temporal] Loading features from {FEATURES_PATH} ...", flush=True)
    cache = torch.load(FEATURES_PATH, weights_only=False)
    features = cache["features"]  # [501, 1000, 66]
    labels = cache["labels"]      # [501, 1000]
    input_dim = features.shape[-1]
    print(f"[train_temporal] Features loaded: {tuple(features.shape)} (dim={input_dim})", flush=True)

    # 2. User Splits
    masks = load_user_split_masks()
    train_indices = torch.where(masks["train"])[0]
    val_indices = torch.where(masks["val"])[0]

    is_multiscale = args.model in ["multiscale", "60d"]

    if is_multiscale:
        print("[train_temporal] Initializing MultiScaleWindowDataset (3d / 14d / 60d horizons with causal left-padding)...", flush=True)
        train_dataset = MultiScaleWindowDataset(features, labels, train_indices, max_window=60, min_window=14)
        val_dataset = MultiScaleWindowDataset(features, labels, val_indices, max_window=60, min_window=14)
    else:
        print(f"[train_temporal] Initializing RollingWindowDataset (Production default {args.window_size}-day window)...", flush=True)
        train_dataset = RollingWindowDataset(features, labels, train_indices, window_size=args.window_size)
        val_dataset = RollingWindowDataset(features, labels, val_indices, window_size=args.window_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    if args.loss == "focal":
        print(f"[train_temporal] Using BinaryFocalLoss (gamma={args.gamma}, alpha={args.alpha}) for hard boundary optimization", flush=True)
        criterion = BinaryFocalLoss(gamma=args.gamma, alpha=args.alpha)
    else:
        pos_weight = compute_pos_weight(train_dataset)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # 3. Model Architecture
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    if is_multiscale:
        model = MultiScaleTemporalGRU(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        best_checkpoint_path = CHECKPOINT_DIR / "temporal_multiscale_best.pt"
        desc = "Multi-Scale Hierarchical GRU (3d / 14d / 60d)"
    else:
        model = TemporalGRU(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
        ).to(device)
        best_checkpoint_path = CHECKPOINT_DIR / "temporal_best.pt"
        desc = f"Single-Scale Temporal GRU ({args.window_size}-Day Production Default)"

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    print("\n" + "=" * 70, flush=True)
    print(f" Starting {desc} Training ({args.epochs} Epochs)", flush=True)
    print("=" * 70, flush=True)

    best_val_pr_auc = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_pr_auc, val_roc_auc = evaluate(model, val_loader, device)

        scheduler.step(val_pr_auc)
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.5f} | "
              f"Val PR-AUC: {val_pr_auc:.4f} | "
              f"Val ROC-AUC: {val_roc_auc:.4f} | "
              f"LR: {current_lr:.1e}", flush=True)

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_type": args.model,
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "window_size": 60 if args.model == "multiscale" else args.window_size,
                "best_val_pr_auc": best_val_pr_auc,
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val PR-AUC: {val_pr_auc:.4f}) to {best_checkpoint_path}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[train_temporal] Early stopping triggered after {epoch} epochs.", flush=True)
                break

    print("=" * 70, flush=True)
    print(f" Temporal Training Complete! Best Validation PR-AUC: {best_val_pr_auc:.4f}", flush=True)
    print(f" Saved Checkpoint: {best_checkpoint_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
