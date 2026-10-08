"""
Evaluation script for the Unsupervised Autoencoder (InsiderThreatAE).

Phase 1.3 in the master roadmap.
Evaluates the trained AE model strictly on the held-out TEST users (150 users)
across all 501 days (~75,000 user-days). Computes ranking metrics tailored for
extreme class imbalance (PR-AUC, ROC-AUC, Precision@K, score distributions).

Usage:
    venv/bin/python scripts/evaluate_ae.py
"""
import argparse
import json
import pickle
import time
import sys
from pathlib import Path

# Add project root to sys.path so models package resolves cleanly
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from models import (
    InsiderThreatAE,
    prepare_graph,
    apply_edge_norm,
    load_user_split_masks,
)

GRAPHS_PATH = Path("output/daily_graphs_labeled.pkl")
STATS_PATH = Path("output/edge_norm_stats.pt")
CHECKPOINT_PATH = Path("output/checkpoints/ae_best.pt")
REPORT_PATH = Path("output/reports/ae_baseline_report.json")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_precision_at_k(labels: np.ndarray, scores: np.ndarray, k_values: list) -> dict:
    """Computes Precision@K: fraction of top K highest-scoring user-days that are truly malicious."""
    top_indices = np.argsort(-scores)
    precisions = {}
    for k in k_values:
        if k > len(labels):
            continue
        top_k_labels = labels[top_indices[:k]]
        precisions[f"P@{k}"] = float(top_k_labels.sum() / k)
    return precisions


@torch.no_grad()
def run_evaluation(model: torch.nn.Module, graphs: dict, norm_stats: dict,
                   test_mask: torch.Tensor, device: torch.device) -> dict:
    model.eval()
    all_scores = []
    all_labels = []
    all_days = []
    all_users = []

    psych_df = pd.read_csv("data/raw/r4.2/psychometric.csv", usecols=["user_id"])
    ordered_users = sorted(set(psych_df["user_id"].astype(str)))
    test_user_indices = torch.where(test_mask)[0].cpu().numpy()
    test_user_ids = [ordered_users[i] for i in test_user_indices]

    for day, raw_g in graphs.items():
        data = prepare_graph(raw_g)
        data = apply_edge_norm(data, norm_stats)
        data = data.to(device)

        _, _, per_user_error = model(data)

        # Extract only test-user scores and ground-truth labels
        scores = per_user_error[test_mask].cpu().numpy()
        labels = data["user"].y[test_mask].cpu().numpy()

        all_scores.append(scores)
        all_labels.append(labels)
        all_days.extend([str(day.date())] * len(scores))
        all_users.extend(test_user_ids)

    scores_flat = np.concatenate(all_scores)
    labels_flat = np.concatenate(all_labels)

    # 1. Metrics calculation
    total_test_instances = len(labels_flat)
    malicious_instances = int(labels_flat.sum())
    benign_instances = total_test_instances - malicious_instances
    random_baseline = float(malicious_instances / max(total_test_instances, 1))

    pr_auc = float(average_precision_score(labels_flat, scores_flat))
    roc_auc = float(roc_auc_score(labels_flat, scores_flat))
    multiplier = float(pr_auc / max(random_baseline, 1e-12))

    # Precision@K
    k_list = [10, 20, 50, 100, 200, 500]
    p_at_k = compute_precision_at_k(labels_flat, scores_flat, k_list)

    # Score distributions
    benign_scores = scores_flat[labels_flat == 0]
    malicious_scores = scores_flat[labels_flat == 1]

    results = {
        "evaluation_split": "test_users",
        "num_test_users": len(test_user_indices),
        "total_test_user_days": total_test_instances,
        "malicious_user_days": malicious_instances,
        "benign_user_days": benign_instances,
        "random_baseline_prevalence": random_baseline,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "multiplier_over_chance": multiplier,
        "precision_at_k": p_at_k,
        "score_distribution": {
            "benign_mean": float(benign_scores.mean()),
            "benign_std": float(benign_scores.std()),
            "benign_median": float(np.median(benign_scores)),
            "malicious_mean": float(malicious_scores.mean()),
            "malicious_std": float(malicious_scores.std()),
            "malicious_median": float(np.median(malicious_scores)),
        },
    }
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Unsupervised InsiderThreatAE on Test Users")
    parser.add_argument("--checkpoint", type=str, default=str(CHECKPOINT_PATH), help="Model checkpoint path")
    args = parser.parse_args()

    device = get_device()
    print(f"[eval_ae] Using device: {device}")

    # 1. Load checkpoint
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}. Train the model first via train_ae.py.")

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"[eval_ae] Loaded checkpoint from {ckpt_path} (trained for {checkpoint.get('epoch', 'N/A')} epochs).")

    # 2. Re-instantiate model
    model = InsiderThreatAE(
        node_in_dims=checkpoint["node_in_dims"],
        edge_attr_dims=checkpoint["edge_attr_dims"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # 3. Load graphs & normalization stats
    print(f"[eval_ae] Loading graphs from {GRAPHS_PATH} ...")
    with open(GRAPHS_PATH, "rb") as f:
        graphs = pickle.load(f)
    norm_stats = torch.load(STATS_PATH, weights_only=False)
    masks = load_user_split_masks()
    test_mask = masks["test"].to(device)

    print(f"[eval_ae] Running evaluation over {len(graphs)} days for {test_mask.sum().item()} test users ...")
    t0 = time.time()
    results = run_evaluation(model, graphs, norm_stats, test_mask, device)
    elapsed = time.time() - t0

    # 4. Save and report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 65)
    print(" Phase 1.3: Unsupervised AE Baseline Evaluation Results")
    print("=" * 65)
    print(f" Evaluation Split      : Held-out Test Users ({results['num_test_users']} users)")
    print(f" Evaluated User-Days   : {results['total_test_user_days']:,} days ({results['malicious_user_days']} malicious)")
    print(f" Random Chance Baseline: {results['random_baseline_prevalence']:.5f}")
    print(f" Test PR-AUC           : {results['pr_auc']:.4f}  ({results['multiplier_over_chance']:.1f}x better than chance)")
    print(f" Test ROC-AUC          : {results['roc_auc']:.4f}")
    print("\n Precision @ K on Test Timeline:")
    for k_metric, val in results["precision_at_k"].items():
        print(f"   {k_metric:<6}: {val * 100:.2f}%")
    print("\n Anomaly Score Separation:")
    print(f"   Benign Mean Error   : {results['score_distribution']['benign_mean']:.5f} (median: {results['score_distribution']['benign_median']:.5f})")
    print(f"   Malicious Mean Error: {results['score_distribution']['malicious_mean']:.5f} (median: {results['score_distribution']['malicious_median']:.5f})")
    print(f"\n Report saved to: {REPORT_PATH}  (evaluated in {elapsed:.1f}s)")
    print("=" * 65)


if __name__ == "__main__":
    main()
