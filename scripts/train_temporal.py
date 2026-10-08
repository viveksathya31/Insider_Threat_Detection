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

from models import TemporalGRU, load_user_split_masks

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

        # Build list of (day_t, user_idx) for valid days t >= window_size - 1
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


def compute_pos_weight(dataset: RollingWindowDataset) -> float:
    pos_count = 0
    total_count = len(dataset)
    # Fast vectorized calculation
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

    for x, y in loader:
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()
        logits, _ = model(x)
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

    for x, y in loader:
        x = x.to(device)
        logits, _ = model(x)
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
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--window_size", type=int, default=14, help="Rolling window size in days")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GRU hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GRU layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
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

    train_dataset = RollingWindowDataset(features, labels, train_indices, window_size=args.window_size)
    val_dataset = RollingWindowDataset(features, labels, val_indices, window_size=args.window_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    pos_weight = compute_pos_weight(train_dataset)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    # 3. Model Architecture
    model = TemporalGRU(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = CHECKPOINT_DIR / "temporal_best.pt"

    print("\n" + "=" * 70, flush=True)
    print(f" Starting Temporal GRU Training ({args.epochs} Epochs | Window: {args.window_size} days)", flush=True)
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
                "model_state_dict": model.state_dict(),
                "input_dim": input_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "window_size": args.window_size,
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
