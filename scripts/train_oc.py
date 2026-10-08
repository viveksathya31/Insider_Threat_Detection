"""
Training script for the One-Class / Deep SVDD Model (InsiderThreatOC).

Phase 2 in the master roadmap.
Trains the HeteroGNNEncoder + OneClassHead to enclose normal user behavior within
a compact hypersphere centered at c. Anomaly score is the squared distance ||phi(z) - c||^2.

Usage:
    venv/bin/python scripts/train_oc.py --epochs 10 --lr 1e-3
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
    InsiderThreatOC,
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


def init_hypersphere_center(model: InsiderThreatOC, day_graphs: list,
                            train_mask: torch.Tensor, device: torch.device,
                            n_warmup_days: int = 20):
    """Initializes the fixed hypersphere center c as the mean projection of train users."""
    model.eval()
    all_projections = []
    with torch.no_grad():
        for _, data in day_graphs[:n_warmup_days]:
            data = data.to(device)
            x_dict = model.encoder(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            user_emb = x_dict["user"]
            proj = model.oc_head.projection(user_emb[train_mask])
            all_projections.append(proj)
    combined = torch.cat(all_projections, dim=0)
    model.oc_head.init_center(combined)
    print(f"[train_oc] Hypersphere center c initialized from {combined.shape[0]} normal user projections.")


def train_one_epoch(model: nn.Module, day_graphs: list, train_mask: torch.Tensor,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    grad_clip: float = 5.0) -> float:
    model.train()
    total_loss = 0.0

    for _, data in day_graphs:
        data = data.to(device)
        optimizer.zero_grad()

        _, distances = model(data)
        # Deep SVDD loss: minimize distance to center for train users
        loss = distances[train_mask].mean()

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
    all_scores, all_labels = [], []

    for _, data in day_graphs:
        data = data.to(device)
        _, distances = model(data)

        loss = distances[split_mask].mean()
        total_loss += loss.item()

        scores = distances[split_mask].cpu().numpy()
        labels = data["user"].y[split_mask].cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels)

    avg_loss = total_loss / len(day_graphs)
    flat_scores = np.concatenate(all_scores)
    flat_labels = np.concatenate(all_labels)

    pr_auc = 0.0
    roc_auc = 0.0
    if flat_labels.sum() > 0:
        pr_auc = average_precision_score(flat_labels, flat_scores)
        roc_auc = roc_auc_score(flat_labels, flat_scores)

    return avg_loss, pr_auc, roc_auc


def main():
    parser = argparse.ArgumentParser(description="Train One-Class Deep SVDD (InsiderThreatOC)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden_dim", type=int, default=64, help="GNN hidden dimension")
    parser.add_argument("--projection_dim", type=int, default=32, help="SVDD hypersphere dimension")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"[train_oc] Using compute device: {device}")

    # Load data
    with open(GRAPHS_PATH, "rb") as f:
        raw_graphs = pickle.load(f)
    norm_stats = torch.load(STATS_PATH, weights_only=False)
    masks = load_user_split_masks()
    train_mask = masks["train"].to(device)
    val_mask = masks["val"].to(device)

    day_graphs = pre_transform_graphs(raw_graphs, norm_stats)
    node_in_dims, edge_attr_dims = infer_graph_metadata(raw_graphs, n_check=5)

    model = InsiderThreatOC(
        node_in_dims=node_in_dims,
        edge_attr_dims=edge_attr_dims,
        hidden_dim=args.hidden_dim,
        projection_dim=args.projection_dim,
    ).to(device)

    # Initialize SVDD center
    init_hypersphere_center(model, day_graphs, train_mask, device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = CHECKPOINT_DIR / "oc_best.pt"

    print("\n" + "=" * 70)
    print(f" Starting One-Class Deep SVDD Training ({args.epochs} Epochs)")
    print("=" * 70)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        random.shuffle(day_graphs)

        train_loss = train_one_epoch(model, day_graphs, train_mask, optimizer, device)
        val_loss, val_pr_auc, val_roc_auc = evaluate_split(model, day_graphs, val_mask, device)

        scheduler.step(val_loss)
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        print(f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] | "
              f"Train SVDD Loss: {train_loss:.5f} | "
              f"Val SVDD Loss: {val_loss:.5f} | "
              f"Val PR-AUC: {val_pr_auc:.4f} | "
              f"Val ROC-AUC: {val_roc_auc:.4f} | "
              f"LR: {current_lr:.1e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "node_in_dims": node_in_dims,
                "edge_attr_dims": edge_attr_dims,
                "hidden_dim": args.hidden_dim,
                "projection_dim": args.projection_dim,
                "best_val_loss": best_val_loss,
            }, best_checkpoint_path)
            print(f"  --> Saved new best checkpoint (Val Loss: {val_loss:.5f}) to {best_checkpoint_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[train_oc] Early stopping triggered after {epoch} epochs.")
                break

    print("=" * 70)
    print(f" Training Complete! Best Validation Loss: {best_val_loss:.5f}")
    print(f" Saved Checkpoint: {best_checkpoint_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
