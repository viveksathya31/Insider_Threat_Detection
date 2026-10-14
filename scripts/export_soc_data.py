"""
Export Script for SOC Frontend Dashboard.

Executes the ExplainabilityParser across the dataset and compiles a complete,
interactive forensic dataset for the SOC web application.

Usage:
    venv/bin/python scripts/export_soc_data.py
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

from src.explainability_parser import ExplainabilityParser

WEB_DATA_DIR = ROOT_DIR / "web" / "data"
OUTPUT_REPORT_DIR = ROOT_DIR / "output" / "reports"


def export_model_dataset(model_choice: str = "14d"):
    start_time = time.time()
    is_ms = model_choice in ["60d", "multiscale"]
    ckpt_path = "output/checkpoints/temporal_multiscale_best.pt" if is_ms else "output/checkpoints/temporal_best.pt"
    model_name_desc = "Multi-Scale 3d/14d/60d GRU (Strategic Optional)" if is_ms else "14-Day Production GRU (Tactical Default)"

    print(f"\n[export] ======================================================================")
    print(f"[export] Starting SOC dashboard data compilation using: {model_name_desc}")
    print(f"[export] Checkpoint: {ckpt_path}")
    print(f"[export] ======================================================================")

    WEB_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

    parser = ExplainabilityParser(temporal_ckpt_path=ckpt_path)

    # Load evaluated benchmark reports
    with open(ROOT_DIR / "output/reports/model_comparison_report.json", "r") as f:
        static_report = json.load(f)
    with open(ROOT_DIR / "output/reports/temporal_evaluation_report.json", "r") as f:
        temporal_report = json.load(f)

    # 1. Score all test users across days
    print("[export] Scoring test instances and ranking alerts...")
    test_indices = torch.where(parser.masks["test"])[0].tolist()

    scored_instances = []
    user_timelines = {}

    for u in test_indices:
        timeline, attns = parser.compute_user_timeline(u)
        user_timelines[u] = timeline
        for item in timeline:
            scored_instances.append(
                {
                    "user_idx": u,
                    "day": item["day"],
                    "temporal_risk": item["temporal_risk"],
                    "static_risk": item["static_risk"],
                    "ae_error": item["ae_error"],
                    "is_malicious": item["is_malicious"],
                }
            )

    # Sort descending by temporal risk
    scored_instances.sort(key=lambda x: x["temporal_risk"], reverse=True)

    # 2. Extract Top Alert Incidents (e.g. top 50 highest risk incidents)
    print(f"[export] Compiling detailed dossiers for top 50 alerts...")
    top_alerts = scored_instances[:50]
    incidents = []

    for rank, inst in enumerate(top_alerts, start=1):
        u = inst["user_idx"]
        d = inst["day"]
        dossier = parser.build_incident_dossier(u, d, rank=rank)
        incidents.append(dossier)

    # 3. Identify Key Employees for Deep-Dive Explorer
    print("[export] Assembling employee catalog...")
    malicious_test_users = []
    for u in test_indices:
        if parser.labels[:, u].sum() > 0:
            malicious_test_users.append(u)

    # Pick representative benign users from diverse departments
    benign_candidates = []
    seen_depts = set()
    for u in test_indices:
        if parser.labels[:, u].sum() == 0:
            uid = parser.ordered_users[u]
            dept = parser.user_profiles.get(uid, {}).get("department", "Unknown")
            if dept not in seen_depts:
                seen_depts.add(dept)
                benign_candidates.append(u)
            if len(benign_candidates) >= 10:
                break

    selected_users = list(dict.fromkeys(malicious_test_users + benign_candidates))
    user_catalog = []

    for u in selected_users:
        uid = parser.ordered_users[u]
        profile = parser.user_profiles.get(uid, {})
        psych = parser.psychometrics.get(uid, {})
        scenario_info = None
        if uid in parser.scenarios:
            sc_raw = parser.scenarios[uid]
            meta = parser.get_scenario_metadata(sc_raw["scenario_id"])
            scenario_info = {
                "scenario_id": sc_raw["scenario_id"],
                "title": meta["title"],
                "pattern": meta["pattern"],
                "mitre_tactics": meta["mitre_tactics"],
                "start": sc_raw["start"],
                "end": sc_raw["end"],
            }

        user_timeline = user_timelines.get(u, [])
        max_temporal_risk = max((x["temporal_risk"] for x in user_timeline), default=0.0)
        max_static_risk = max((x["static_risk"] for x in user_timeline), default=0.0)
        total_malicious_days = int(parser.labels[:, u].sum().item())

        user_catalog.append(
            {
                "user_idx": u,
                "user_id": uid,
                "is_malicious": total_malicious_days > 0,
                "malicious_days_count": total_malicious_days,
                "max_temporal_risk": round(max_temporal_risk, 4),
                "max_static_risk": round(max_static_risk, 4),
                "profile": profile,
                "psychometrics": psych,
                "scenario": scenario_info,
                "timeline": user_timeline,
            }
        )

    # 4. Global KPIs & Benchmarks
    benchmarks = {
        "models": [
            {
                "name": "Unsupervised AE",
                "paradigm": "Zero-Day Behavioral",
                "pr_auc": static_report["AE"]["pr_auc"],
                "roc_auc": static_report["AE"]["roc_auc"],
                "multiplier": static_report["AE"]["multiplier_over_chance"],
                "p10": static_report["AE"]["precision_at_k"].get("P@10", 0.0),
                "p20": static_report["AE"]["precision_at_k"].get("P@20", 0.0),
                "p50": static_report["AE"]["precision_at_k"].get("P@50", 0.0),
                "p100": static_report["AE"]["precision_at_k"].get("P@100", 0.0),
            },
            {
                "name": "One-Class Deep SVDD",
                "paradigm": "Single-Sphere Latent",
                "pr_auc": static_report["OC"]["pr_auc"],
                "roc_auc": static_report["OC"]["roc_auc"],
                "multiplier": static_report["OC"]["multiplier_over_chance"],
                "p10": static_report["OC"]["precision_at_k"].get("P@10", 0.0),
                "p20": static_report["OC"]["precision_at_k"].get("P@20", 0.0),
                "p50": static_report["OC"]["precision_at_k"].get("P@50", 0.0),
                "p100": static_report["OC"]["precision_at_k"].get("P@100", 0.0),
            },
            {
                "name": "Supervised Snapshot CLF",
                "paradigm": "Static GNN Snapshot",
                "pr_auc": static_report["CLF"]["pr_auc"],
                "roc_auc": static_report["CLF"]["roc_auc"],
                "multiplier": static_report["CLF"]["multiplier_over_chance"],
                "p10": static_report["CLF"]["precision_at_k"].get("P@10", 0.0),
                "p20": static_report["CLF"]["precision_at_k"].get("P@20", 0.0),
                "p50": static_report["CLF"]["precision_at_k"].get("P@50", 0.0),
                "p100": static_report["CLF"]["precision_at_k"].get("P@100", 0.0),
            },
            {
                "name": "Sequential Temporal GRU (14-Day Default)",
                "paradigm": "Temporal Attention (14-Day)",
                "pr_auc": temporal_report["Temporal_GRU"]["pr_auc"],
                "roc_auc": temporal_report["Temporal_GRU"]["roc_auc"],
                "multiplier": temporal_report["Temporal_GRU"]["multiplier_over_chance"],
                "p10": temporal_report["Temporal_GRU"]["precision_at_k"].get("P@10", 0.0),
                "p20": temporal_report["Temporal_GRU"]["precision_at_k"].get("P@20", 0.0),
                "p50": temporal_report["Temporal_GRU"]["precision_at_k"].get("P@50", 0.0),
                "p100": temporal_report["Temporal_GRU"]["precision_at_k"].get("P@100", 0.0),
            },
        ]
    }

    if "MultiScale_GRU" in temporal_report:
        benchmarks["models"].append({
            "name": "Multi-Scale Temporal GRU (60-Day Optional)",
            "paradigm": "Hierarchical Gated (3d/14d/60d)",
            "pr_auc": temporal_report["MultiScale_GRU"]["pr_auc"],
            "roc_auc": temporal_report["MultiScale_GRU"]["roc_auc"],
            "multiplier": temporal_report["MultiScale_GRU"]["multiplier_over_chance"],
            "p10": temporal_report["MultiScale_GRU"]["precision_at_k"].get("P@10", 0.0),
            "p20": temporal_report["MultiScale_GRU"]["precision_at_k"].get("P@20", 0.0),
            "p50": temporal_report["MultiScale_GRU"]["precision_at_k"].get("P@50", 0.0),
            "p100": temporal_report["MultiScale_GRU"]["precision_at_k"].get("P@100", 0.0),
        })

    if is_ms:
        benchmarks["kpi"] = {
            "total_monitored_users": 1000,
            "evaluation_days": 501,
            "test_users": len(test_indices),
            "test_instances_evaluated": len(scored_instances),
            "model_in_use": "60d_multiscale",
            "temporal_pr_auc": round(temporal_report["MultiScale_GRU"]["pr_auc"], 4),
            "temporal_roc_auc": round(temporal_report["MultiScale_GRU"]["roc_auc"], 4),
            "temporal_multiplier": round(temporal_report["MultiScale_GRU"]["multiplier_over_chance"], 1),
            "precision_at_10": round(temporal_report["MultiScale_GRU"]["precision_at_k"]["P@10"] * 100, 1),
            "precision_at_20": round(temporal_report["MultiScale_GRU"]["precision_at_k"]["P@20"] * 100, 1),
            "precision_at_50": round(temporal_report["MultiScale_GRU"]["precision_at_k"]["P@50"] * 100, 1),
            "false_positive_reduction": "5.6x vs Static CLF",
        }
    else:
        benchmarks["kpi"] = {
            "total_monitored_users": 1000,
            "evaluation_days": 501,
            "test_users": len(test_indices),
            "test_instances_evaluated": len(scored_instances),
            "model_in_use": "14d_default",
            "temporal_pr_auc": round(temporal_report["Temporal_GRU"]["pr_auc"], 4),
            "temporal_roc_auc": round(temporal_report["Temporal_GRU"]["roc_auc"], 4),
            "temporal_multiplier": round(temporal_report["Temporal_GRU"]["multiplier_over_chance"], 1),
            "precision_at_10": round(temporal_report["Temporal_GRU"]["precision_at_k"]["P@10"] * 100, 1),
            "precision_at_20": round(temporal_report["Temporal_GRU"]["precision_at_k"]["P@20"] * 100, 1),
            "precision_at_50": round(temporal_report["Temporal_GRU"]["precision_at_k"]["P@50"] * 100, 1),
            "false_positive_reduction": "6.5x vs Static CLF",
        }

    soc_payload = {
        "metadata": {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "dataset": "CMU CERT Insider Threat Testbed (r4.2)",
            "pipeline": "Temporal Heterogeneous GNN + Attention GRU",
            "model_id": "60d" if is_ms else "14d",
            "model_name": "Multi-Scale Hierarchical GRU (3d/14d/60d)" if is_ms else "Sequential Temporal GRU (14-Day)",
            "is_default": not is_ms,
        },
        "kpi": benchmarks["kpi"],
        "benchmarks": benchmarks["models"],
        "incidents": incidents,
        "users": user_catalog,
    }

    # Determine output destinations
    targets = []
    if is_ms:
        targets.append(OUTPUT_REPORT_DIR / "soc_dashboard_data_60d.json")
        if WEB_DATA_DIR.exists():
            targets.append(WEB_DATA_DIR / "soc_data_60d.json")
    else:
        # Default files + explicit 14d files
        targets.append(OUTPUT_REPORT_DIR / "soc_dashboard_data.json")
        targets.append(OUTPUT_REPORT_DIR / "soc_dashboard_data_14d.json")
        if WEB_DATA_DIR.exists():
            targets.append(WEB_DATA_DIR / "soc_data.json")
            targets.append(WEB_DATA_DIR / "soc_data_14d.json")

    for tgt in targets:
        print(f"[export] Writing payload to {tgt}...")
        with open(tgt, "w") as f:
            json.dump(soc_payload, f, indent=2)

    elapsed = time.time() - start_time
    print(f"[export] SUCCESS: Generated {len(incidents)} alert dossiers and {len(user_catalog)} employee profiles in {elapsed:.2f}s.")
    return soc_payload


def main():
    arg_parser = argparse.ArgumentParser(description="Export SOC Frontend Dashboard Dataset")
    arg_parser.add_argument(
        "--model",
        type=str,
        default="14d",
        choices=["14d", "single", "60d", "multiscale", "all"],
        help="Model to export: '14d' (production default), '60d' (strategic optional), or 'all' (compiles both)",
    )
    args = arg_parser.parse_args()

    if args.model == "all":
        print("[export] Running full export for both 14-day default and 60-day optional models...")
        export_model_dataset("14d")
        export_model_dataset("60d")
        print("\n[export] All models successfully exported and synchronized.")
    else:
        export_model_dataset(args.model)


if __name__ == "__main__":
    main()
