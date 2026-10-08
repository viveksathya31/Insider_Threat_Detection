"""
Training script for the Supervised Classifier (InsiderThreatCLF).

Phase 2 in the master roadmap.
Trains the HeteroGNNEncoder + ClassificationHead using BCEWithLogitsLoss with
positive-class weighting (pos_weight ~ 500) to combat the severe 1:500 class imbalance.
Monitors validation PR-AUC and saves the best checkpoint.

Usage:
    venv/bin/python scripts/train_clf.py --epochs 10 --lr 1e-3
"""
import argparse
import pickle
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
from sklearn.metrics import average_precision_score, roc_auc_score

from models import (
    InsiderThreatCLF,
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
    transformed = []
    for day, g in graphs.items():
        data = prepare_graph(g)
        data = apply_edge_norm(data, norm_stats)
        transformed.append((day, data))
    return transformed


def compute_pos_weight(day_graphs: list, train_mask: torch.Tensor) -> float:
    total_pos = 0
    total_count = 0
    for _, data in day_graphs:
        y = data["user"].y[train_mask.to(data["user"].y.device)]
        total_pos += int(y.sum().item())
        total_count += y.numel()
    total_neg = total_count - total_pos
    pos_weight = total_neg / max(total_pos, 1)
    print(f"[train_clf] Train class balance: {total_pos} pos / {total_neg} neg user-days (pos_weight: {pos_weight:.1f})", flush=True)
    return pos_weight


def train_one_epoch(model: nn.Module, day_graphs: list, train_mask: torch.Tensor,
                    optimizer: torch.optim.Optimizer, criterion: nn.Module,
                    device: torch.device, grad_clip: float = 5.0) -> float:
    model.train()
    total_loss = 0.0

    for _, data in day_graphs:
        data = data.to(device)
        optimizer.zero_grad()

        logits = model(data)
        y = data["user"].y.to(device)
        loss = criterion(logits[train_mask], y[train_mask])

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
    all_probs, all_labels = [], []

    for _, data in day_graphs:
        data = data.to(device)
        logits = model(data)
        probs = torch.sigmoid(logits[split_mask]).cpu().numpy()
        labels = data["user"].y[split_mask].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels)

    flat_probs = np.concatenate(all_probs)
    flat_labels = np.concatenate(all_labels)

    pr_auc = 0.0
    roc_auc = 0.0
    if flat_labels.sum() > 0:
        pr_auc = average_precision_score(flat_labels, flat_probs)
        roc_auc = roc_auc_score(flat_labels, flat_probs)

    return pr_auc, roc_auc


def main():
    parser = argparse.ArgumentParser(description="Train Supervised Classifier (InsiderThreatCLF)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GNN hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Classifier dropout")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[train_clf] Using compute device: {device}")

    # Load data
    with open(GRAPHS_PATH, "rb") as f:
        raw_graphs = pickle.load(f)
    norm_stats = torch.load(STATS_PATH, weights_only=False)
    masks = load_user_split_masks()
    train_mask = masks["train"].to(device)
    val_mask = masks["val"].to(device)

    day_graphs = pre_transform_graphs(raw_graphs, norm_stats)
    node_in_dims, edge_attr_dims = infer_graph_metadata(raw_graphs, n_check=5)

    pos_weight = compute_pos_weight(day_graphs, train_mask)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    model = InsiderThreatCLF(
        node_in_dims=node_in_dims,
        edge_attr_dims=edge_attr_dims,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = CHECKPOINT_DIR / "clf_best.pt"

    print("\n" + "=" * 70, flush=True)
    print(f" Starting Supervised Classifier Training ({args.epochs} Epochs)", flush=True)
    print("=" * 70, flush=True)

    best_val_pr_auc = -1.0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        random.shuffle(day_graphs)

        train_loss = train_one_epoch(model, day_graphs, train_mask, optimizer, criterion, device)
        val_pr_auc, val_roc_auc = evaluate_split(model, day_graphs, val_mask, device)

        scheduler.step(val_pr_auc)
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.5f} | "
              f"Val PR-AUC: {val_pr_auc:.4f} | "
              f"Val ROC-AUC: {val_roc_auc:.4f} | "
              f"LR: {current_lr:.1e}", flush=True)

        # Checkpointing on validation PR-AUC
        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "node_in_dims": node_in_dims,
                "edge_attr_dims": edge_attr_dims,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "best_val_pr_auc": best_val_pr_auc,
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val PR-AUC: {val_pr_auc:.4f}) to {best_checkpoint_path}", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[train_clf] Early stopping triggered after {epoch} epochs.", flush=True)
                break

    print("=" * 70, flush=True)
    print(f" Training Complete! Best Validation PR-AUC: {best_val_pr_auc:.4f}", flush=True)
    print(f" Saved Checkpoint: {best_checkpoint_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
