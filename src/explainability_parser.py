"""
Explainability Parser and Forensic Data Extraction Module.

Extracts multi-faceted explainability signals across:
  1. 14-day temporal self-attention distributions (when did the risk accumulate).
  2. Relational activity attribution via Autoencoder MSE slices (what actions drove the anomaly).
  3. Real employee organizational profiles and Big-5 psychometrics from LDAP.
  4. Ground-truth CERT attack scenario alignment and chronological incident tracking.
  5. Cross-paradigm benchmark metrics for comparative SOC display.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from torch_geometric.data import HeteroData
from models import (
    InsiderThreatAE,
    TemporalGRU,
    apply_edge_norm,
    load_user_split_masks,
    prepare_graph,
)
from models.nf_gnn import FORWARD_EDGE_TYPES

# Slices in the 139-dim reconstruction target for each forward relation
RELATION_SLICES = {
    "copies_file": (0, 22, "File Exfiltration & Large Transfers"),
    "logs_into": (22, 44, "Machine Authentication & Logon Activity"),
    "uses_device": (44, 66, "Removable Media & USB Insertion"),
    "visits_url": (66, 77, "Web Navigation & External URLs"),
    "sends_email": (77, 108, "Outbound Email Transmission"),
    "uses_pc": (108, 139, "Host Process & Workstation Usage"),
}


class ExplainabilityParser:
    """
    Parses and packages deep learning explainability artifacts for SOC frontend consumption.
    """

    def __init__(
        self,
        features_path: str = "output/temporal_features.pt",
        temporal_ckpt_path: str = "output/checkpoints/temporal_best.pt",
        ae_ckpt_path: str = "output/checkpoints/ae_best.pt",
        edge_stats_path: str = "output/edge_norm_stats.pt",
        graphs_path: str = "output/daily_graphs_labeled.pkl",
        device: Optional[torch.device] = None,
    ):
        self.device = device or (
            torch.device("mps")
            if torch.backends.mps.is_available()
            else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        )

        print(f"[parser] Initializing ExplainabilityParser on device: {self.device}")

        # 1. Load pre-extracted temporal features and labels
        cache = torch.load(features_path, weights_only=False)
        self.features = cache["features"]  # [501, 1000, 66]
        self.labels = cache["labels"]  # [501, 1000]

        # 2. Load Temporal GRU
        temporal_ckpt = torch.load(temporal_ckpt_path, map_location=self.device, weights_only=False)
        self.window_size = temporal_ckpt.get("window_size", 14)
        self.temporal_model = TemporalGRU(
            input_dim=temporal_ckpt["input_dim"],
            hidden_dim=temporal_ckpt["hidden_dim"],
            num_layers=temporal_ckpt["num_layers"],
            dropout=temporal_ckpt["dropout"],
        ).to(self.device)
        self.temporal_model.load_state_dict(temporal_ckpt["model_state_dict"])
        self.temporal_model.eval()

        # 3. Load AE Model
        ae_ckpt = torch.load(ae_ckpt_path, map_location=self.device, weights_only=False)
        self.ae_model = InsiderThreatAE(
            node_in_dims=ae_ckpt["node_in_dims"],
            edge_attr_dims=ae_ckpt["edge_attr_dims"],
            hidden_dim=ae_ckpt.get("hidden_dim", 64),
        ).to(self.device)
        self.ae_model.load_state_dict(ae_ckpt["model_state_dict"])
        self.ae_model.eval()

        self.norm_stats = torch.load(edge_stats_path, map_location=self.device, weights_only=False)
        self.graphs_path = Path(graphs_path)
        self.raw_graphs = None  # Lazy load

        # 4. User and Split mappings
        self.masks = load_user_split_masks()
        psych_df = pd.read_csv("data/raw/r4.2/psychometric.csv")
        self.ordered_users = sorted(set(psych_df["user_id"].astype(str)))
        self.user_to_idx = {uid: i for i, uid in enumerate(self.ordered_users)}

        # Load psychometrics lookup
        self.psychometrics = {}
        for _, row in psych_df.iterrows():
            uid = str(row["user_id"])
            self.psychometrics[uid] = {
                "name": str(row.get("employee_name", uid)),
                "O": int(row.get("O", 25)),
                "C": int(row.get("C", 25)),
                "E": int(row.get("E", 25)),
                "A": int(row.get("A", 25)),
                "N": int(row.get("N", 25)),
            }

        # Load LDAP profiles
        self.user_profiles = self._load_ldap_profiles()

        # Load Scenarios
        self.scenarios = self._load_scenarios()

    def _load_graphs(self):
        if self.raw_graphs is None:
            import pickle

            print(f"[parser] Loading daily graphs from {self.graphs_path}...")
            with open(self.graphs_path, "rb") as f:
                self.raw_graphs = pickle.load(f)
            self.sorted_days = sorted(self.raw_graphs.keys())

    def _load_ldap_profiles(self) -> Dict[str, Dict[str, str]]:
        profiles = {}
        ldap_dir = Path("data/raw/r4.2/LDAP")
        if not ldap_dir.exists():
            return profiles

        ldap_files = sorted(list(ldap_dir.glob("*.csv")))
        for f in ldap_files:
            df = pd.read_csv(f)
            for _, row in df.iterrows():
                uid = str(row["user_id"])
                profiles[uid] = {
                    "employee_name": str(row.get("employee_name", uid)),
                    "email": str(row.get("email", f"{uid}@dtaa.com")),
                    "role": str(row.get("role", "Employee")),
                    "functional_unit": str(row.get("functional_unit", "Corporate")),
                    "department": str(row.get("department", "General")),
                    "team": str(row.get("team", "Operations")),
                    "supervisor": str(row.get("supervisor", "Executive Management")),
                }
        return profiles

    def _load_scenarios(self) -> Dict[str, Dict[str, Any]]:
        scenarios = {}
        answers_path = Path("data/raw/answers/insiders.csv")
        if answers_path.exists():
            df = pd.read_csv(answers_path)
            r42 = df[df["dataset"] == 4.2]
            for _, row in r42.iterrows():
                uid = str(row["user"])
                scenario_id = int(row["scenario"])
                scenarios[uid] = {
                    "scenario_id": scenario_id,
                    "start": str(row["start"]),
                    "end": str(row["end"]),
                    "details_file": str(row.get("details", "")),
                }
        return scenarios

    def get_scenario_metadata(self, scenario_id: int) -> Dict[str, str]:
        metadata = {
            1: {
                "title": "Unauthorized Exfiltration via Removable Media & Web Leaks",
                "pattern": "After-hours logins, unauthorized USB drive insertion, bulk data exfiltration to external upload targets.",
                "mitre_tactics": ["Initial Access (TA0001)", "Exfiltration Over Physical Medium (T1052)", "Exfiltration to Cloud Storage (T1567)"],
                "threat_level": "CRITICAL",
            },
            2: {
                "title": "IP Theft & Solicit Competitor Employment Prior to Resignation",
                "pattern": "Browsing job boards and competitor domains, escalating thumb drive transfers to copy trade secrets before resignation.",
                "mitre_tactics": ["Collection (TA0009)", "Data Staged (T1074)", "Exfiltration via Removable Media (T1052)"],
                "threat_level": "CRITICAL",
            },
            3: {
                "title": "Privileged Insider Sabotage & Supervisory Account Impersonation",
                "pattern": "System administrator accesses keylogging tools, transfers payloads to supervisor PC, impersonates management for mass panic.",
                "mitre_tactics": ["Credential Access (TA0006)", "Privilege Escalation (TA0004)", "Account Manipulation (T1098)"],
                "threat_level": "CRITICAL",
            },
        }
        return metadata.get(
            scenario_id,
            {
                "title": "Anomalous Behavioral Deviation",
                "pattern": "Elevated statistical discrepancy across structural graph relations.",
                "mitre_tactics": ["Anomalous Activity"],
                "threat_level": "HIGH",
            },
        )

    def compute_user_timeline(self, user_idx: int) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """
        Computes the complete daily risk timeline (Days 13..500) and attention matrices for a given user.
        """
        T = self.features.shape[0]
        W = self.window_size

        windows = torch.stack(
            [self.features[t - W + 1 : t + 1, user_idx, :] for t in range(W - 1, T)]
        ).to(self.device)

        with torch.no_grad():
            probs, attns = self.temporal_model.predict_risk(windows)

        probs_np = probs.cpu().numpy()
        attns_np = attns.cpu().numpy()

        timeline = []
        for i in range(len(probs_np)):
            t = i + W - 1
            temporal_p = float(probs_np[i])
            static_p = float(self.features[t, user_idx, 65].item())
            ae_err = float(self.features[t, user_idx, 64].item())
            label = int(self.labels[t, user_idx].item())

            timeline.append(
                {
                    "day": t,
                    "temporal_risk": round(temporal_p, 4),
                    "static_risk": round(static_p, 4),
                    "ae_error": round(ae_err, 4),
                    "is_malicious": label,
                }
            )

        return timeline, attns_np

    def compute_relational_attribution(
        self, user_idx: int, day_idx: int
    ) -> List[Dict[str, Any]]:
        """
        Decomposes the daily behavioral anomaly into exact per-relation MSE contributions
        and raw activity frequencies.
        """
        self._load_graphs()
        g = self.raw_graphs[self.sorted_days[day_idx]]
        data = prepare_graph(g)
        data = apply_edge_norm(data, self.norm_stats)
        data = data.to(self.device)

        with torch.no_grad():
            recon, target, per_user_err = self.ae_model(data)

        sq_err = ((recon[user_idx] - target[user_idx]) ** 2).cpu().numpy()

        breakdown = []
        total_mse = 0.0

        for rel, (start, end, stream_name) in RELATION_SLICES.items():
            rel_mse = float(sq_err[start:end].mean())
            total_mse += rel_mse

            # Count actual edges for user
            edge_count = 0
            for et in g.edge_types:
                if et[0] == "user" and et[1] == rel:
                    mask = g[et].edge_index[0] == user_idx
                    edge_count = int(mask.sum().item())
                    break

            breakdown.append(
                {
                    "relation": rel,
                    "stream_name": stream_name,
                    "mse": rel_mse,
                    "edge_count": edge_count,
                }
            )

        # Normalize share
        safe_total = total_mse if total_mse > 1e-6 else 1e-6
        for item in breakdown:
            item["share_pct"] = round((item["mse"] / safe_total) * 100.0, 1)
            item["mse"] = round(item["mse"], 4)

        # Sort descending by contribution share
        breakdown.sort(key=lambda x: x["share_pct"], reverse=True)
        return breakdown

    def build_incident_dossier(
        self, user_idx: int, day_idx: int, rank: int = 1
    ) -> Dict[str, Any]:
        """
        Builds a comprehensive, SOC-ready incident report for a specific alert.
        """
        self._load_graphs()
        uid = self.ordered_users[user_idx]
        profile = self.user_profiles.get(
            uid,
            {
                "employee_name": uid,
                "email": f"{uid}@dtaa.com",
                "role": "Unknown",
                "functional_unit": "Corporate",
                "department": "Unknown",
                "team": "Operations",
                "supervisor": "Executive",
            },
        )
        psych = self.psychometrics.get(
            uid, {"name": uid, "O": 25, "C": 25, "E": 25, "A": 25, "N": 25}
        )

        # 14-day window features
        W = self.window_size
        w = self.features[day_idx - W + 1 : day_idx + 1, user_idx, :].unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob, attn = self.temporal_model.predict_risk(w)

        temporal_prob = float(prob.item())
        static_prob = float(self.features[day_idx, user_idx, 65].item())
        ae_error = float(self.features[day_idx, user_idx, 64].item())
        is_mal = int(self.labels[day_idx, user_idx].item())
        attn_weights = attn.squeeze(0).cpu().numpy().tolist()

        # Attention Heatmap Breakdown
        attention_heatmap = []
        for i, weight in enumerate(attn_weights):
            offset_day = day_idx - W + 1 + i
            date_str = self.sorted_days[offset_day].strftime("%Y-%m-%d")

            # Determine stage semantics
            if weight >= 0.22:
                stage = "Peak Attack Exfiltration"
                stage_color = "crimson"
            elif weight >= 0.14:
                stage = "Escalation & Staging"
                stage_color = "amber"
            elif weight >= 0.05:
                stage = "Reconnaissance / Probe"
                stage_color = "cyan"
            else:
                stage = "Baseline Routine"
                stage_color = "slate"

            attention_heatmap.append(
                {
                    "offset": f"T-{W - 1 - i:02d}",
                    "day": offset_day,
                    "date": date_str,
                    "attention_share": round(weight * 100.0, 2),
                    "stage": stage,
                    "stage_color": stage_color,
                    "static_risk": round(float(self.features[offset_day, user_idx, 65].item()), 4),
                    "is_malicious": int(self.labels[offset_day, user_idx].item()),
                }
            )

        # Relational activity breakdown
        relational_attribution = self.compute_relational_attribution(user_idx, day_idx)

        # Scenario details
        scenario_info = None
        if uid in self.scenarios:
            sc_raw = self.scenarios[uid]
            meta = self.get_scenario_metadata(sc_raw["scenario_id"])
            scenario_info = {
                "scenario_id": sc_raw["scenario_id"],
                "title": meta["title"],
                "pattern": meta["pattern"],
                "mitre_tactics": meta["mitre_tactics"],
                "start": sc_raw["start"],
                "end": sc_raw["end"],
            }

        # Severity classification
        if temporal_prob >= 0.95:
            severity = "CRITICAL"
        elif temporal_prob >= 0.80:
            severity = "HIGH"
        elif temporal_prob >= 0.50:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        # Split info
        split = "test" if self.masks["test"][user_idx] else ("val" if self.masks["val"][user_idx] else "train")

        incident_id = f"INC-{day_idx:03d}-{uid}"

        return {
            "incident_id": incident_id,
            "rank": rank,
            "user_idx": user_idx,
            "user_id": uid,
            "day": day_idx,
            "calendar_date": self.sorted_days[day_idx].strftime("%Y-%m-%d"),
            "severity": severity,
            "temporal_risk": round(temporal_prob, 4),
            "static_risk": round(static_prob, 4),
            "ae_error": round(ae_error, 4),
            "is_malicious": is_mal,
            "split": split,
            "profile": profile,
            "psychometrics": psych,
            "scenario": scenario_info,
            "attention_heatmap": attention_heatmap,
            "relational_attribution": relational_attribution,
        }
