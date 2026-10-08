"""
Training script for the Unsupervised Autoencoder (InsiderThreatAE).

Phase 1.2 in the master roadmap.
Trains on normal behavioral patterns across all 501 daily graphs, optimizing
reconstruction error strictly on train-split users (700 users). Monitors validation
loss on val-split users (150 users) with early stopping and saves the best model checkpoint.

Usage:
    venv/bin/python scripts/train_ae.py --epochs 15 --lr 1e-3 --hidden_dim 64
"""
import argparse
import json
import pickle
import random
import time
import sys
from pathlib import Path

# Add project root to sys.path so models package resolves cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score

from models import (
    InsiderThreatAE,
    prepare_graph,
    apply_edge_norm,
    infer_graph_metadata,
    load_user_split_masks,
)

GRAPHS_PATH = Path("output/daily_graphs_labeled.pkl")
STATS_PATH = Path("output/edge_norm_stats.pt")
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


def pre_transform_graphs(graphs: dict, norm_stats: dict) -> list:
    """Pre-computes ToUndirected() and edge z-score normalization once in memory
    so epochs run fast without re-normalizing every time."""
    transformed = []
    for day, g in graphs.items():
        data = prepare_graph(g)
        data = apply_edge_norm(data, norm_stats)
        transformed.append((day, data))
    return transformed


def train_one_epoch(model: nn.Module, day_graphs: list, train_mask: torch.Tensor,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    grad_clip: float = 5.0) -> float:
    model.train()
    total_loss = 0.0

    for _, data in day_graphs:
        data = data.to(device)
        optimizer.zero_grad()

        _, _, per_user_error = model(data)
        # Train loss is strictly computed over train-split users
        loss = per_user_error[train_mask].mean()

        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(day_graphs)


@torch.no_grad()
def evaluate_split(model: nn.Module, day_graphs: list, split_mask: torch.Tensor,
                   device: torch.device) -> tuple:
    model.eval()
    total_loss = 0.0
    all_scores = []
    all_labels = []

    for _, data in day_graphs:
        data = data.to(device)
        _, _, per_user_error = model(data)

        # Loss on this split's users
        loss = per_user_error[split_mask].mean()
        total_loss += loss.item()

        # Collect anomaly scores and ground-truth binary labels for ranking evaluation
        scores = per_user_error[split_mask].cpu().numpy()
        labels = data["user"].y[split_mask].cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels)

    avg_loss = total_loss / len(day_graphs)
    flat_scores = np.concatenate(all_scores)
    flat_labels = np.concatenate(all_labels)

    # Compute ranking metrics if positive examples exist in the split
    pr_auc = 0.0
    roc_auc = 0.0
    if flat_labels.sum() > 0:
        pr_auc = average_precision_score(flat_labels, flat_scores)
        roc_auc = roc_auc_score(flat_labels, flat_scores)

    return avg_loss, pr_auc, roc_auc


def main():
    parser = argparse.ArgumentParser(description="Train Unsupervised InsiderThreatAE")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GNN hidden dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[train_ae] Using compute device: {device}")

    # 1. Load graphs & norm stats
    print(f"[train_ae] Loading graphs from {GRAPHS_PATH} ...")
    with open(GRAPHS_PATH, "rb") as f:
        raw_graphs = pickle.load(f)
    print(f"[train_ae] Loaded {len(raw_graphs)} daily graphs.")

    if not STATS_PATH.exists():
        raise FileNotFoundError(f"Missing {STATS_PATH}. Run Phase 1.1 first to compute normalization stats.")
    norm_stats = torch.load(STATS_PATH, weights_only=False)
    print(f"[train_ae] Loaded normalization stats for {len(norm_stats)} edge types.")

    # 2. Split masks
    masks = load_user_split_masks()
    train_mask = masks["train"].to(device)
    val_mask = masks["val"].to(device)
    test_mask = masks["test"].to(device)
    print(f"[train_ae] User splits: {train_mask.sum().item()} train, "
          f"{val_mask.sum().item()} val, {test_mask.sum().item()} test users.")

    # 3. Pre-transform all graphs in memory
    print("[train_ae] Pre-transforming graphs (reverse edges + edge normalization) ...")
    t0 = time.time()
    day_graphs = pre_transform_graphs(raw_graphs, norm_stats)
    print(f"[train_ae] Ready in {time.time() - t0:.2f}s.")

    # 4. Infer metadata & instantiate model
    node_in_dims, edge_attr_dims = infer_graph_metadata(raw_graphs, n_check=5)
    model = InsiderThreatAE(
        node_in_dims=node_in_dims,
        edge_attr_dims=edge_attr_dims,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train_ae] InsiderThreatAE initialized ({n_params:,} parameters).")

    # 5. Optimizer & Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = CHECKPOINT_DIR / "ae_best.pt"

    # 6. Training loop
    print("\n" + "=" * 70)
    print(f" Starting Unsupervised AE Training ({args.epochs} Epochs)")
    print("=" * 70)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # Shuffle day presentation order each epoch to prevent day-order bias
        random.shuffle(day_graphs)

        train_loss = train_one_epoch(
            model=model,
            day_graphs=day_graphs,
            train_mask=train_mask,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_pr_auc, val_roc_auc = evaluate_split(
            model=model,
            day_graphs=day_graphs,
            split_mask=val_mask,
            device=device,
        )

        scheduler.step(val_loss)
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.5f} | "
              f"Val Loss: {val_loss:.5f} | "
              f"Val PR-AUC: {val_pr_auc:.4f} | "
              f"Val ROC-AUC: {val_roc_auc:.4f} | "
              f"LR: {current_lr:.1e}")

        # Early stopping and checkpointing on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "node_in_dims": node_in_dims,
                "edge_attr_dims": edge_attr_dims,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "best_val_loss": best_val_loss,
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.5f}) to {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[train_ae] Early stopping triggered after {epoch} epochs (patience={args.patience}).")
                break

    print("=" * 70)
    print(f" Training Complete! Best Validation Loss: {best_val_loss:.5f}")
    print(f" Saved Checkpoint: {best_checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
