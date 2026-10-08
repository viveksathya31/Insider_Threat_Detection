"""
Multi-Head Model Comparison Evaluation Script.

Phase 2.3 in the master roadmap.
Evaluates AE (Autoencoder), OC (One-Class Deep SVDD), and CLF (Supervised Classifier)
side-by-side on the exact same 150 held-out test users across all 501 days (75,150 user-days).

Usage:
    venv/bin/python scripts/evaluate_comparison.py
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from models import (
    InsiderThreatAE,
    InsiderThreatOC,
    InsiderThreatCLF,
    prepare_graph,
    apply_edge_norm,
    load_user_split_masks,
)

GRAPHS_PATH = Path("output/daily_graphs_labeled.pkl")
STATS_PATH = Path("output/edge_norm_stats.pt")
CHECKPOINT_DIR = Path("output/checkpoints")
REPORT_PATH = Path("output/reports/model_comparison_report.json")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_precision_at_k(labels: np.ndarray, scores: np.ndarray, k_values: list) -> dict:
    top_indices = np.argsort(-scores)
    precisions = {}
    for k in k_values:
        if k > len(labels):
            continue
        top_k_labels = labels[top_indices[:k]]
        precisions[f"P@{k}"] = float(top_k_labels.sum() / k)
    return precisions


@torch.no_grad()
def score_model_on_test(model_type: str, model: torch.nn.Module, day_graphs: list,
                        test_mask: torch.Tensor, device: torch.device) -> tuple:
    model.eval()
    all_scores, all_labels = [], []

    for _, data in day_graphs:
        data = data.to(device)

        if model_type == "AE":
            _, _, error = model(data)
            scores = error[test_mask].cpu().numpy()
        elif model_type == "OC":
            _, distances = model(data)
            scores = distances[test_mask].cpu().numpy()
        elif model_type == "CLF":
            logits = model(data)
            scores = torch.sigmoid(logits[test_mask]).cpu().numpy()
        else:
            raise ValueError(f"Unknown model_type {model_type}")

        labels = data["user"].y[test_mask].cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels)

    scores_flat = np.concatenate(all_scores)
    labels_flat = np.concatenate(all_labels)
    return scores_flat, labels_flat


def evaluate_one_model(name: str, ckpt_path: Path, model_cls, day_graphs: list,
                       test_mask: torch.Tensor, device: torch.device) -> dict:
    if not ckpt_path.exists():
        print(f"[comparison] Checkpoint {ckpt_path} not found -- skipping {name}.")
        return None

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    kwargs = {
        "node_in_dims": ckpt["node_in_dims"],
        "edge_attr_dims": ckpt["edge_attr_dims"],
        "hidden_dim": ckpt.get("hidden_dim", 64),
    }
    if name == "OC":
        kwargs["projection_dim"] = ckpt.get("projection_dim", 32)
    elif name == "CLF":
        kwargs["dropout"] = ckpt.get("dropout", 0.2)

    model = model_cls(**kwargs).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    scores, labels = score_model_on_test(name, model, day_graphs, test_mask, device)

    total_instances = len(labels)
    malicious_count = int(labels.sum())
    baseline = float(malicious_count / max(total_instances, 1))

    pr_auc = float(average_precision_score(labels, scores))
    roc_auc = float(roc_auc_score(labels, scores))
    multiplier = float(pr_auc / max(baseline, 1e-12))
    p_at_k = compute_precision_at_k(labels, scores, [10, 20, 50, 100, 200])

    return {
        "model": name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "multiplier_over_chance": multiplier,
        "precision_at_k": p_at_k,
    }


def main():
    device = get_device()
    print(f"[comparison] Using compute device: {device}")

    # Load graphs & norm stats
    print(f"[comparison] Loading graphs from {GRAPHS_PATH} ...")
    with open(GRAPHS_PATH, "rb") as f:
        raw_graphs = pickle.load(f)
    norm_stats = torch.load(STATS_PATH, weights_only=False)
    masks = load_user_split_masks()
    test_mask = masks["test"].to(device)

    print("[comparison] Pre-transforming graphs ...")
    day_graphs = []
    for day, g in raw_graphs.items():
        data = prepare_graph(g)
        data = apply_edge_norm(data, norm_stats)
        day_graphs.append((day, data))

    models_to_eval = [
        ("AE", CHECKPOINT_DIR / "ae_best.pt", InsiderThreatAE),
        ("OC", CHECKPOINT_DIR / "oc_best.pt", InsiderThreatOC),
        ("CLF", CHECKPOINT_DIR / "clf_best.pt", InsiderThreatCLF),
    ]

    results = {}
    print("\n" + "=" * 78)
    print(" Evaluating Models on Held-out Test Users (75,150 user-days)")
    print("=" * 78)

    for name, path, cls in models_to_eval:
        res = evaluate_one_model(name, path, cls, day_graphs, test_mask, device)
        if res:
            results[name] = res

    # Comparison summary table
    print("\n" + "=" * 78)
    print(f"{'Model':<8} | {'Type':<14} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Vs. Chance':<10} | {'P@20':<8} | {'P@50':<8}")
    print("-" * 78)
    type_map = {"AE": "Unsupervised", "OC": "One-Class", "CLF": "Supervised"}
    for name, res in results.items():
        m_type = type_map.get(name, "Unknown")
        p20 = f"{res['precision_at_k'].get('P@20', 0) * 100:.1f}%"
        p50 = f"{res['precision_at_k'].get('P@50', 0) * 100:.1f}%"
        print(f"{name:<8} | {m_type:<14} | {res['pr_auc']:<8.4f} | {res['roc_auc']:<8.4f} | {res['multiplier_over_chance']:<10.1f}x | {p20:<8} | {p50:<8}")
    print("=" * 78)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved comparison report to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
