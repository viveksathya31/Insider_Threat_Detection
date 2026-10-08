"""
Temporal Model Evaluation & Benchmark Script (Phase 3.4).

Evaluates the Temporal GRU with Attention pooling on the 150 strictly held-out
test users across all valid sequence days (73,200 test instances).
Compares Temporal GRU vs. Static Snapshot CLF vs. Static AE.
Evaluates:
  - Test PR-AUC & ROC-AUC
  - Precision @ K (P@10, P@20, P@50, P@100, P@200)
  - Temporal Attention heatmaps for flagged users

Usage:
    venv/bin/python scripts/evaluate_temporal.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

from models import TemporalGRU, load_user_split_masks
from scripts.train_temporal import RollingWindowDataset

FEATURES_PATH = Path("output/temporal_features.pt")
CHECKPOINT_PATH = Path("output/checkpoints/temporal_best.pt")
REPORT_PATH = Path("output/reports/temporal_evaluation_report.json")


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
def main():
    device = get_device()
    print(f"[eval_temporal] Using compute device: {device}", flush=True)

    # 1. Load Precomputed Features
    print(f"[eval_temporal] Loading features from {FEATURES_PATH} ...", flush=True)
    cache = torch.load(FEATURES_PATH, weights_only=False)
    features = cache["features"]  # [501, 1000, 66]
    labels = cache["labels"]      # [501, 1000]

    # 2. Load Trained Temporal Model
    print(f"[eval_temporal] Loading checkpoint from {CHECKPOINT_PATH} ...", flush=True)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    window_size = ckpt.get("window_size", 14)

    model = TemporalGRU(
        input_dim=ckpt["input_dim"],
        hidden_dim=ckpt["hidden_dim"],
        num_layers=ckpt["num_layers"],
        dropout=ckpt["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Test Dataset
    masks = load_user_split_masks()
    test_indices = torch.where(masks["test"])[0]

    test_dataset = RollingWindowDataset(features, labels, test_indices, window_size=window_size)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    print(f"[eval_temporal] Evaluating on {len(test_dataset):,} test sequence instances (window={window_size} days) ...", flush=True)

    all_temporal_probs = []
    all_static_clf_probs = []
    all_static_ae_errors = []
    all_labels = []
    all_attn_weights = []

    start_time = time.time()
    for x, y in test_loader:
        # Static baselines from the current day t (last step in window):
        # x is [B, W, 66], where dim 64 is AE error, dim 65 is CLF prob
        static_ae = x[:, -1, 64].numpy()
        static_clf = x[:, -1, 65].numpy()

        x = x.to(device)
        probs, attn = model.predict_risk(x)

        all_temporal_probs.append(probs.cpu().numpy())
        all_static_clf_probs.append(static_clf)
        all_static_ae_errors.append(static_ae)
        all_labels.append(y.numpy())
        all_attn_weights.append(attn.cpu().numpy())

    temporal_scores = np.concatenate(all_temporal_probs)
    static_clf_scores = np.concatenate(all_static_clf_probs)
    static_ae_scores = np.concatenate(all_static_ae_errors)
    test_labels = np.concatenate(all_labels)

    eval_time = time.time() - start_time
    total_test = len(test_labels)
    malicious_count = int(test_labels.sum())
    chance_baseline = malicious_count / max(total_test, 1)

    print(f"[eval_temporal] Scored {total_test:,} instances in {eval_time:.1f}s. Positives: {malicious_count} ({chance_baseline * 100:.3f}%)", flush=True)

    # Metrics computation
    models_evaluated = {
        "Temporal_GRU": temporal_scores,
        "Static_CLF": static_clf_scores,
        "Static_AE": static_ae_scores,
    }

    results = {}
    k_vals = [10, 20, 50, 100, 200]

    for m_name, scores in models_evaluated.items():
        pr_auc = float(average_precision_score(test_labels, scores))
        roc_auc = float(roc_auc_score(test_labels, scores))
        p_at_k = compute_precision_at_k(test_labels, scores, k_vals)
        multiplier = float(pr_auc / max(chance_baseline, 1e-12))

        results[m_name] = {
            "pr_auc": pr_auc,
            "roc_auc": roc_auc,
            "multiplier_over_chance": multiplier,
            "precision_at_k": p_at_k,
        }

    # Print Comparison Table
    print("\n" + "=" * 80, flush=True)
    print(" Phase 3.4: Temporal vs. Static Benchmark on Test Sequences")
    print("=" * 80, flush=True)
    print(f"{'Model':<16} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Vs. Chance':<10} | {'P@10':<7} | {'P@20':<7} | {'P@50':<7}", flush=True)
    print("-" * 80, flush=True)

    for m_name, m_res in results.items():
        p10 = f"{m_res['precision_at_k'].get('P@10', 0) * 100:.1f}%"
        p20 = f"{m_res['precision_at_k'].get('P@20', 0) * 100:.1f}%"
        p50 = f"{m_res['precision_at_k'].get('P@50', 0) * 100:.1f}%"
        print(f"{m_name:<16} | {m_res['pr_auc']:<8.4f} | {m_res['roc_auc']:<8.4f} | {m_res['multiplier_over_chance']:<10.1f}x | {p10:<7} | {p20:<7} | {p50:<7}", flush=True)
    print("=" * 80, flush=True)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nReport saved to: {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
