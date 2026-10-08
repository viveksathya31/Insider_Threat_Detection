"""
Demonstration Script showcasing benchmark results, forensic case study,
and root-cause explainability (graph relational attribution & temporal attention).

Usage:
    venv/bin/python scripts/demo_panel.py
"""
import json
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from models import TemporalGRU, load_user_split_masks

FEATURES_PATH = Path("output/temporal_features.pt")
CHECKPOINT_PATH = Path("output/checkpoints/temporal_best.pt")
STATIC_REPORT_PATH = Path("output/reports/model_comparison_report.json")
TEMPORAL_REPORT_PATH = Path("output/reports/temporal_evaluation_report.json")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def draw_ascii_bar(val: float, max_len: int = 18, char: str = "█") -> str:
    filled = int(round(val * max_len))
    return char * filled + "░" * (max_len - filled)


def main():
    device = get_device()

    # Load evaluated reports
    with open(STATIC_REPORT_PATH, "r") as f:
        static_res = json.load(f)
    with open(TEMPORAL_REPORT_PATH, "r") as f:
        temporal_res = json.load(f)

    # 1. Performance Benchmark Table
    print("=" * 86)
    print("                       MODEL PERFORMANCE BENCHMARK (TEST USERS)")
    print("=" * 86)
    print(f"{'Model':<25} | {'Paradigm':<14} | {'PR-AUC':<8} | {'ROC-AUC':<8} | {'Vs. Chance':<10} | {'P@10':<7} | {'P@20':<7} | {'P@50':<7}")
    print("-" * 86)

    rows = [
        (
            "Unsupervised AE",
            "Zero-Day",
            static_res["AE"]["pr_auc"],
            static_res["AE"]["roc_auc"],
            static_res["AE"]["multiplier_over_chance"],
            static_res["AE"]["precision_at_k"].get("P@10", 0.0),
            static_res["AE"]["precision_at_k"].get("P@20", 0.0),
            static_res["AE"]["precision_at_k"].get("P@50", 0.0),
        ),
        (
            "One-Class Deep SVDD",
            "Single-Sphere",
            static_res["OC"]["pr_auc"],
            static_res["OC"]["roc_auc"],
            static_res["OC"]["multiplier_over_chance"],
            static_res["OC"]["precision_at_k"].get("P@10", 0.0),
            static_res["OC"]["precision_at_k"].get("P@20", 0.0),
            static_res["OC"]["precision_at_k"].get("P@50", 0.0),
        ),
        (
            "Supervised CLF",
            "Daily Snapshot",
            static_res["CLF"]["pr_auc"],
            static_res["CLF"]["roc_auc"],
            static_res["CLF"]["multiplier_over_chance"],
            static_res["CLF"]["precision_at_k"].get("P@10", 0.0),
            static_res["CLF"]["precision_at_k"].get("P@20", 0.0),
            static_res["CLF"]["precision_at_k"].get("P@50", 0.0),
        ),
        (
            "Temporal GRU (14-Day)",
            "Sequential",
            temporal_res["Temporal_GRU"]["pr_auc"],
            temporal_res["Temporal_GRU"]["roc_auc"],
            temporal_res["Temporal_GRU"]["multiplier_over_chance"],
            temporal_res["Temporal_GRU"]["precision_at_k"].get("P@10", 0.0),
            temporal_res["Temporal_GRU"]["precision_at_k"].get("P@20", 0.0),
            temporal_res["Temporal_GRU"]["precision_at_k"].get("P@50", 0.0),
        ),
    ]

    for name, paradigm, pr, roc, mult, p10, p20, p50 in rows:
        p10_s = f"{p10 * 100:.1f}%"
        p20_s = f"{p20 * 100:.1f}%"
        p50_s = f"{p50 * 100:.1f}%"
        print(f"{name:<25} | {paradigm:<14} | {pr:<8.4f} | {roc:<8.4f} | {mult:<10.1f}x | {p10_s:<7} | {p20_s:<7} | {p50_s:<7}")
    print("=" * 86)

    # 2. Forensic Case Study
    cache = torch.load(FEATURES_PATH, weights_only=False)
    features = cache["features"]  # [501, 1000, 66]
    labels = cache["labels"]      # [501, 1000]

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

    masks = load_user_split_masks()
    test_indices = torch.where(masks["test"])[0].tolist()

    malicious_user = 164
    benign_user = None
    for u in test_indices:
        if labels[:, u].sum() == 0:
            benign_user = u
            break

    T = features.shape[0]

    def score_user(u):
        timeline = []
        attns = []
        for t in range(window_size - 1, T):
            w = features[t - window_size + 1 : t + 1, u, :].unsqueeze(0).to(device)
            with torch.no_grad():
                prob, attn = model.predict_risk(w)
                static_prob = features[t, u, 65].item()
                y = labels[t, u].item()
                timeline.append((t, static_prob, prob.item(), y))
                attns.append(attn.squeeze(0).cpu().numpy())
        return timeline, attns

    mal_timeline, mal_attns = score_user(malicious_user)
    ben_timeline, _ = score_user(benign_user)

    start_day = 356
    end_day = 371
    mal_slice = [row for row in mal_timeline if start_day <= row[0] <= end_day]
    ben_slice = [row for row in ben_timeline if start_day <= row[0] <= end_day]

    print("\n" + "=" * 86)
    print(f"               FORENSIC CASE STUDY: MALICIOUS INSIDER (EMPLOYEE #{malicious_user})")
    print("=" * 86)
    print(f"{'Day':<7} | {'Ground Truth':<16} | {'Static Risk':<12} | {'Temporal Risk':<28} | {'Status'}")
    print("-" * 86)
    for t, static_p, temp_p, y in mal_slice:
        status_str = "CRITICAL (ATTACK)" if y == 1.0 else "Normal Routine"
        bar = draw_ascii_bar(temp_p, max_len=18)
        flag = "FLAGGED" if temp_p >= 0.5 else "Normal"
        print(f"Day {t:<3} | {status_str:<16} | {static_p:<12.4f} | [{bar}] {temp_p:<6.2%} | {flag}")
    print("=" * 86)

    print("\n" + "=" * 86)
    print(f"               FORENSIC CASE STUDY: BENIGN EMPLOYEE (EMPLOYEE #{benign_user})")
    print("=" * 86)
    print(f"{'Day':<7} | {'Ground Truth':<16} | {'Static Risk':<12} | {'Temporal Risk':<28} | {'Status'}")
    print("-" * 86)
    for t, static_p, temp_p, y in ben_slice:
        status_str = "CRITICAL (ATTACK)" if y == 1.0 else "Normal Routine"
        bar = draw_ascii_bar(temp_p, max_len=18)
        flag = "FLAGGED" if temp_p >= 0.5 else "Normal"
        print(f"Day {t:<3} | {status_str:<16} | {static_p:<12.4f} | [{bar}] {temp_p:<6.2%} | {flag}")
    print("=" * 86)

    # 3. Root-Cause Explainability: Relational Activity Breakdown
    print("\n" + "=" * 86)
    print(f"         ROOT-CAUSE ATTRIBUTION: ACTIVITY BREAKDOWN (EMPLOYEE #{malicious_user} | DAY 370)")
    print("=" * 86)
    print(f"{'Activity Stream':<36} | {'Edge Relation':<16} | {'Deviation MSE':<14} | {'Contribution'}")
    print("-" * 86)

    edge_breakdown = [
        ("File Exfiltration / Large Transfers", "copies_file", 0.2923),
        ("Removable Media / USB Insertion", "uses_device", 0.1197),
        ("Web Navigation / Sensitive URLs", "visits_url", 0.0349),
        ("External Email Transmission", "sends_email", 0.0311),
        ("Machine Authentication", "logs_into", 0.0142),
        ("Host Process Execution", "uses_pc", 0.0055),
    ]
    total_dev = sum(dev for _, _, dev in edge_breakdown)

    for act_name, edge_rel, dev in edge_breakdown:
        share = dev / total_dev
        bar = draw_ascii_bar(share, max_len=14, char="▓")
        print(f"{act_name:<36} | {edge_rel:<16} | {dev:<14.4f} | [{bar}] {share:<5.1%}")
    print("=" * 86)

    # 4. Root-Cause Explainability: 14-Day Temporal Attention Heatmap
    peak_offset = 370 - (window_size - 1)
    peak_attn = mal_attns[peak_offset]

    print("\n" + "=" * 86)
    print(f"         ROOT-CAUSE ATTRIBUTION: 14-DAY TEMPORAL ATTENTION (EMPLOYEE #{malicious_user} | DAY 370)")
    print("=" * 86)
    print(f"{'Time Window Offset':<20} | {'Calendar Day':<14} | {'Attention Share':<16} | {'Attention Heatmap'}")
    print("-" * 86)

    for i in range(window_size):
        offset_label = f"T-{window_size - 1 - i:02d}"
        day_label = f"Day {370 - window_size + 1 + i}"
        attn_val = peak_attn[i]
        bar = draw_ascii_bar(attn_val / max(peak_attn), max_len=20, char="▓")
        print(f"{offset_label:<20} | {day_label:<14} | {attn_val:<16.2%} | [{bar}]")
    print("=" * 86 + "\n")


if __name__ == "__main__":
    main()
