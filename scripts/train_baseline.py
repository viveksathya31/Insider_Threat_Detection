"""
Baseline training loop for per-day (user, day) malicious classification.

Trains on every day in the train split (including pure-negative days, so
the model also learns what normal looks like), using weighted BCE loss to
counter the ~1:500 class imbalance, and validates each epoch with PR-AUC
(not accuracy -- meaningless at this imbalance).

Run from project root:
    python scripts/train_baseline.py
"""
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve
from torch_geometric.transforms import ToUndirected

sys.path.insert(0, "src")
from model import HeteroGNN, get_edge_dims

GRAPHS_PATH = Path("output/daily_graphs_labeled.pkl")
SPLITS_PATH = Path("data/processed/day_splits.json")
CHECKPOINT_PATH = Path("output/baseline_gnn_best.pt")

HIDDEN_DIM = 64
NUM_EPOCHS = 30
LEARNING_RATE = 1e-3
GRAD_CLIP_NORM = 5.0
SEED = 42
EARLY_STOP_PATIENCE = 10  # stop if val PR-AUC doesn't improve for this many epochs


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_edge_norm_stats(day_graphs):
    """Per-edge-type mean/std over the TRAIN split only, for z-score
    normalization. Computed once, then applied identically to train/val/test
    -- fitting on val/test would leak their distribution into training."""
    sums, sq_sums, counts = {}, {}, {}
    for _, data in day_graphs:
        for et in data.edge_types:
            if "edge_attr" not in data[et]:
                continue
            attr = data[et].edge_attr
            if et not in sums:
                sums[et] = attr.sum(dim=0)
                sq_sums[et] = (attr ** 2).sum(dim=0)
                counts[et] = attr.shape[0]
            else:
                sums[et] += attr.sum(dim=0)
                sq_sums[et] += (attr ** 2).sum(dim=0)
                counts[et] += attr.shape[0]

    stats = {}
    for et in sums:
        mean = sums[et] / counts[et]
        var = sq_sums[et] / counts[et] - mean ** 2
        std = var.clamp(min=1e-6).sqrt()
        stats[et] = (mean, std)
    return stats


def apply_edge_norm(data, stats):
    """Z-score normalize every relation's edge_attr in-place using the given
    per-edge-type (mean, std), skipping relations not seen in the stats
    (shouldn't happen if stats were computed on the same schema)."""
    for et in data.edge_types:
        if "edge_attr" not in data[et] or et not in stats:
            continue
        mean, std = stats[et]
        data[et].edge_attr = (data[et].edge_attr - mean.to(data[et].edge_attr.device)) \
            / std.to(data[et].edge_attr.device)
    return data


def load_split_graphs(graphs, dates, transform):
    """Returns list of (date, transformed HeteroData) for the given date strings."""
    out = []
    for d in dates:
        ts = pd.Timestamp(d)
        if ts not in graphs:
            print(f"[train] WARNING: {d} not found in graphs dict, skipping")
            continue
        out.append((ts, transform(graphs[ts])))
    return out


def compute_pos_weight(day_graphs):
    """neg/pos ratio over the given set of days, for BCEWithLogitsLoss(pos_weight=...)."""
    total_pos = 0
    total = 0
    for _, data in day_graphs:
        y = data["user"].y
        total_pos += y.sum().item()
        total += y.numel()
    total_neg = total - total_pos
    pos_weight = total_neg / max(total_pos, 1)
    print(f"[train] class balance: {int(total_pos)} positive / {int(total_neg)} negative "
          f"user-days (pos_weight={pos_weight:.1f})")
    return pos_weight


def run_epoch(model, day_graphs, optimizer, criterion, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for _, data in day_graphs:
            data = data.to(device)
            logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            y = data["user"].y

            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                optimizer.step()

            total_loss += loss.item()
            all_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    avg_loss = total_loss / len(day_graphs)
    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    pr_auc = average_precision_score(labels, probs)

    # Fixed 0.5 threshold is meaningless under a large pos_weight (model
    # learns to output high probabilities broadly). Instead, sweep the
    # actual PR curve and report precision/recall at the best-F1 threshold.
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.argmax(f1s)
    best_p, best_r, best_f1 = precisions[best_idx], recalls[best_idx], f1s[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 1.0

    return avg_loss, pr_auc, best_p, best_r, best_f1, best_threshold


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = get_device()
    print(f"[train] device: {device}")

    with open(GRAPHS_PATH, "rb") as f:
        graphs = pickle.load(f)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    transform = ToUndirected()
    print("[train] loading + transforming train/val graphs into memory...")
    train_graphs = load_split_graphs(graphs, splits["train"], transform)
    val_graphs = load_split_graphs(graphs, splits["val"], transform)
    print(f"[train] train days: {len(train_graphs)}, val days: {len(val_graphs)}")

    print("[train] computing edge feature normalization stats (train split only)...")
    edge_norm_stats = compute_edge_norm_stats(train_graphs)
    for et, (mean, std) in edge_norm_stats.items():
        print(f"  {et}: mean_range=[{mean.min():.2f}, {mean.max():.2f}], "
              f"std_range=[{std.min():.2f}, {std.max():.2f}]")

    train_graphs = [(d, apply_edge_norm(data, edge_norm_stats)) for d, data in train_graphs]
    val_graphs = [(d, apply_edge_norm(data, edge_norm_stats)) for d, data in val_graphs]

    EDGE_STATS_PATH = Path("output/edge_norm_stats.pt")
    EDGE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(edge_norm_stats, EDGE_STATS_PATH)
    print(f"[train] saved edge normalization stats -> {EDGE_STATS_PATH} "
          f"(reuse these for test-set eval -- never refit on test data)")

    pos_weight_value = compute_pos_weight(train_graphs)

    # Warmup forward pass -- materializes lazy input_proj params BEFORE the
    # optimizer is created, so those params actually get trained.
    sample_data = train_graphs[0][1].to(device)
    edge_dims = get_edge_dims(sample_data)
    model = HeteroGNN(sample_data.metadata(), edge_dims=edge_dims, hidden_dim=HIDDEN_DIM).to(device)
    with torch.no_grad():
        _ = model(sample_data.x_dict, sample_data.edge_index_dict, sample_data.edge_attr_dict)
    print(f"[train] model materialized, {sum(p.numel() for p in model.parameters())} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = torch.tensor(pos_weight_value, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_pr_auc = -1.0
    epochs_since_improvement = 0
    for epoch in range(1, NUM_EPOCHS + 1):
        random.shuffle(train_graphs)  # shuffle day order each epoch

        train_loss, train_pr_auc, train_p, train_r, train_f1, train_thr = run_epoch(
            model, train_graphs, optimizer, criterion, device, train=True
        )
        val_loss, val_pr_auc, val_p, val_r, val_f1, val_thr = run_epoch(
            model, val_graphs, optimizer, criterion, device, train=False
        )

        print(f"[epoch {epoch:02d}] "
              f"train loss={train_loss:.4f} PR-AUC={train_pr_auc:.4f} "
              f"bestF1 P={train_p:.3f} R={train_r:.3f} F1={train_f1:.3f} @thr={train_thr:.3f}  |  "
              f"val loss={val_loss:.4f} PR-AUC={val_pr_auc:.4f} "
              f"bestF1 P={val_p:.3f} R={val_r:.3f} F1={val_f1:.3f} @thr={val_thr:.3f}")

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            epochs_since_improvement = 0
            CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"[epoch {epoch:02d}] new best val PR-AUC ({val_pr_auc:.4f}), "
                  f"saved -> {CHECKPOINT_PATH}")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= EARLY_STOP_PATIENCE:
                print(f"\n[train] early stopping -- no val PR-AUC improvement "
                      f"for {EARLY_STOP_PATIENCE} epochs")
                break

    print(f"\n[train] done. best val PR-AUC: {best_val_pr_auc:.4f}")


if __name__ == "__main__":
    main()