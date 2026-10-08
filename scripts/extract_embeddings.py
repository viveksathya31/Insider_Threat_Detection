"""
Pre-extract Daily User Embeddings and Multi-Model Features across all 501 days.

Phase 3.1 in the master roadmap.
Extracts:
  - 64-dim structural user embedding from trained HeteroGNNEncoder (CLF)
  - 1-dim unsupervised reconstruction error from trained InsiderThreatAE
  - 1-dim threat probability from trained InsiderThreatCLF
  - 1-dim ground truth label per user-day

Output shape:
  features: [501, 1000, 66] (day x user x feature)
  labels:   [501, 1000]

Usage:
    venv/bin/python scripts/extract_embeddings.py
"""
import pickle
import sys
import time
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch
from models import (
    InsiderThreatAE,
    InsiderThreatCLF,
    prepare_graph,
    apply_edge_norm,
)

GRAPHS_PATH = Path("output/daily_graphs_labeled.pkl")
STATS_PATH = Path("output/edge_norm_stats.pt")
CHECKPOINT_DIR = Path("output/checkpoints")
OUTPUT_PATH = Path("output/temporal_features.pt")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def main():
    device = get_device()
    print(f"[extract] Using compute device: {device}", flush=True)

    # 1. Load Data
    print(f"[extract] Loading graphs from {GRAPHS_PATH} ...", flush=True)
    with open(GRAPHS_PATH, "rb") as f:
        raw_graphs = pickle.load(f)
    print(f"[extract] Loaded {len(raw_graphs)} daily graphs.", flush=True)

    norm_stats = torch.load(STATS_PATH, weights_only=False)

    # 2. Load Trained Models
    ae_ckpt = torch.load(CHECKPOINT_DIR / "ae_best.pt", map_location=device, weights_only=False)
    ae_model = InsiderThreatAE(
        node_in_dims=ae_ckpt["node_in_dims"],
        edge_attr_dims=ae_ckpt["edge_attr_dims"],
        hidden_dim=ae_ckpt.get("hidden_dim", 64),
    ).to(device)
    ae_model.load_state_dict(ae_ckpt["model_state_dict"])
    ae_model.eval()

    clf_ckpt = torch.load(CHECKPOINT_DIR / "clf_best.pt", map_location=device, weights_only=False)
    clf_model = InsiderThreatCLF(
        node_in_dims=clf_ckpt["node_in_dims"],
        edge_attr_dims=clf_ckpt["edge_attr_dims"],
        hidden_dim=clf_ckpt.get("hidden_dim", 64),
        dropout=clf_ckpt.get("dropout", 0.2),
    ).to(device)
    clf_model.load_state_dict(clf_ckpt["model_state_dict"])
    clf_model.eval()

    print("[extract] Pre-transforming and extracting features across all 501 days ...", flush=True)
    start_time = time.time()

    all_features = []
    all_labels = []
    days_list = []

    sorted_days = sorted(raw_graphs.keys())

    for idx, day in enumerate(sorted_days):
        g = raw_graphs[day]
        data = prepare_graph(g)
        data = apply_edge_norm(data, norm_stats)
        data = data.to(device)

        # AE Error
        _, _, ae_error = ae_model(data)  # [1000]

        # CLF Encoder Embedding + Logit
        x_dict = clf_model.encoder(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        user_emb = x_dict["user"]  # [1000, 64]
        clf_logits = clf_model.clf_head(user_emb)  # [1000]
        clf_prob = torch.sigmoid(clf_logits)  # [1000]

        # Combine: 64 emb + 1 ae_error + 1 clf_prob = 66
        combined = torch.cat([
            user_emb,
            ae_error.unsqueeze(-1),
            clf_prob.unsqueeze(-1),
        ], dim=-1).cpu()  # [1000, 66]

        y = data["user"].y.cpu()  # [1000]

        all_features.append(combined)
        all_labels.append(y)
        days_list.append(day)

        if (idx + 1) % 100 == 0 or idx == len(sorted_days) - 1:
            elapsed = time.time() - start_time
            print(f"  Processed {idx + 1:3d}/{len(sorted_days)} days ({elapsed:.1f}s)", flush=True)

    # Stack into tensors
    features_tensor = torch.stack(all_features, dim=0)  # [501, 1000, 66]
    labels_tensor = torch.stack(all_labels, dim=0)      # [501, 1000]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": features_tensor,
        "labels": labels_tensor,
        "days": days_list,
        "feature_dim": features_tensor.shape[-1],
        "feature_names": [f"gnn_emb_{i}" for i in range(64)] + ["ae_error", "clf_prob"],
    }, OUTPUT_PATH)

    total_time = time.time() - start_time
    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 70, flush=True)
    print(f" Feature Extraction Complete in {total_time:.1f}s!", flush=True)
    print(f" Saved: {OUTPUT_PATH} ({file_size_mb:.1f} MB)", flush=True)
    print(f" Tensor Shape: features {tuple(features_tensor.shape)}, labels {tuple(labels_tensor.shape)}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
