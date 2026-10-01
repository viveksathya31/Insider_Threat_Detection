"""
Final, honest test-set evaluation of the baseline GNN.

Two things this script is careful about, both to avoid an inflated number:
  1. Edge normalization stats are loaded from what was fit on TRAIN only
     (saved by train_baseline.py) -- never refit on val or test.
  2. The classification threshold is chosen from VAL performance (best-F1
     point on the val PR curve), then applied AS-IS to test. We do not sweep
     thresholds on test itself -- doing so would leak test-set information
     into the "chosen" operating point and inflate the reported P/R/F1.
     PR-AUC (threshold-free) is reported on test regardless, as the primary
     summary number.

Run from project root (after train_baseline.py has produced a checkpoint):
    python scripts/evaluate_test.py
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, precision_recall_curve, confusion_matrix

sys.path.insert(0, "src")
from model import HeteroGNN, to_undirected, get_edge_dims

GRAPHS_PATH = Path("output/daily_graphs_labeled.pkl")
SPLITS_PATH = Path("data/processed/day_splits.json")
CHECKPOINT_PATH = Path("output/baseline_gnn_best.pt")
EDGE_STATS_PATH = Path("output/edge_norm_stats.pt")
HIDDEN_DIM = 64


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_split_graphs(graphs, dates, transform, edge_stats):
    out = []
    for d in dates:
        ts = pd.Timestamp(d)
        if ts not in graphs:
            print(f"[eval] WARNING: {d} not found in graphs dict, skipping")
            continue
        data = transform(graphs[ts])
        for et in data.edge_types:
            if "edge_attr" not in data[et] or et not in edge_stats:
                continue
            mean, std = edge_stats[et]
            data[et].edge_attr = (data[et].edge_attr - mean) / std
        out.append((ts, data))
    return out


def collect_predictions(model, day_graphs, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for _, data in day_graphs:
            data = data.to(device)
            logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = data["user"].y.cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels)
    return np.concatenate(all_probs), np.concatenate(all_labels)


def best_f1_threshold(labels, probs):
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    idx = np.argmax(f1s)
    thr = thresholds[idx] if idx < len(thresholds) else 1.0
    return thr, precisions[idx], recalls[idx], f1s[idx]


def report_at_threshold(labels, probs, threshold, name):
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall + 1e-12) if (precision + recall) > 0 else 0.0
    print(f"\n[{name}] at threshold={threshold:.4f}:")
    print(f"  precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}")
    print(f"  confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  (of {tp+fn} truly malicious user-days, caught {tp}; "
          f"of {tp+fp} flagged, {tp} were real)")
    return precision, recall, f1


def main():
    device = get_device()
    print(f"[eval] device: {device}")

    with open(GRAPHS_PATH, "rb") as f:
        graphs = pickle.load(f)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)

    edge_stats = torch.load(EDGE_STATS_PATH, weights_only=False)
    print(f"[eval] loaded edge norm stats (fit on train only) from {EDGE_STATS_PATH}")

    transform = to_undirected  # same shared instance used in training (model.py)

    val_graphs = load_split_graphs(graphs, splits["val"], transform, edge_stats)
    test_graphs = load_split_graphs(graphs, splits["test"], transform, edge_stats)
    print(f"[eval] val days: {len(val_graphs)}, test days: {len(test_graphs)}")

    sample_data = val_graphs[0][1]
    edge_dims = get_edge_dims(sample_data)
    model = HeteroGNN(sample_data.metadata(), edge_dims=edge_dims, hidden_dim=HIDDEN_DIM).to(device)

    # Materialize lazy layers with a dummy forward pass before loading weights
    with torch.no_grad():
        _ = model(sample_data.to(device).x_dict, sample_data.edge_index_dict, sample_data.edge_attr_dict)

    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"[eval] loaded checkpoint from {CHECKPOINT_PATH}")

    # Step 1: determine the operating threshold from VAL only
    val_probs, val_labels = collect_predictions(model, val_graphs, device)
    val_pr_auc = average_precision_score(val_labels, val_probs)
    threshold, _, _, _ = best_f1_threshold(val_labels, val_probs)
    print(f"\n[eval] val PR-AUC: {val_pr_auc:.4f}")
    report_at_threshold(val_labels, val_probs, threshold, "val (threshold source)")

    # Step 2: apply that SAME threshold to test -- no test-set threshold tuning
    test_probs, test_labels = collect_predictions(model, test_graphs, device)
    test_pr_auc = average_precision_score(test_labels, test_probs)
    print(f"\n{'='*60}")
    print(f"[eval] FINAL TEST-SET RESULT")
    print(f"{'='*60}")
    print(f"[eval] test PR-AUC: {test_pr_auc:.4f}  "
          f"(random baseline: {test_labels.mean():.4f}, "
          f"{test_pr_auc/max(test_labels.mean(), 1e-12):.1f}x better than chance)")
    report_at_threshold(test_labels, test_probs, threshold, "test (val-derived threshold)")


if __name__ == "__main__":
    main()