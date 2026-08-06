"""
Attaches ground-truth (user, day) -> is_malicious labels onto the daily HeteroData
graphs built by build_graphs.py, using the labels from src/labels.py.

graph_builder.py does NOT store user_id ordering on the graph objects -- it builds
id_maps["user"] internally as sorted(set(agg_df["user_id"]) | set(psych_df.index))
and discards it after use. We reconstruct that EXACT same ordering here from the
saved aggregated_edges.csv + psychometric.csv, since it's a deterministic sorted
set union (no rerun of build_graphs.py needed).

Adds graphs[day]['user'].y : FloatTensor[num_users], aligned to that reconstructed
user ordering (graph_builder.py keeps node indices stable across all days).

Usage:
    python3 scripts/attach_labels.py
"""
import pickle
from pathlib import Path
import pandas as pd
import torch

GRAPHS_PATH = Path("output/daily_graphs.pkl")
AGG_EDGES_PATH = Path("output/aggregated_edges.csv")
PSYCH_PATH = Path("data/raw/r4.2/psychometric.csv")
LABELS_PATH = Path("data/processed/user_day_labels.csv")
OUT_PATH = Path("output/daily_graphs_labeled.pkl")


def reconstruct_user_ordering() -> list:
    """Must exactly match src/graph_builder.py's _build_id_maps() user ordering."""
    agg_user_ids = pd.read_csv(AGG_EDGES_PATH, usecols=["user_id"])["user_id"].astype(str)
    psych_user_ids = pd.read_csv(PSYCH_PATH, usecols=["user_id"])["user_id"].astype(str)
    user_ids = sorted(set(agg_user_ids) | set(psych_user_ids))
    return user_ids


def main():
    print("=== Reconstructing user node ordering ===")
    user_ids = reconstruct_user_ordering()
    print(f"Reconstructed {len(user_ids)} user ids (should match graphs['user'].x row count)")

    print("\n=== Loading graphs and labels ===")
    with open(GRAPHS_PATH, "rb") as f:
        graphs = pickle.load(f)
    print(f"Loaded {len(graphs)} daily graphs")

    sample_day = next(iter(graphs))
    n_user_nodes = graphs[sample_day]["user"].x.shape[0]
    if n_user_nodes != len(user_ids):
        raise RuntimeError(
            f"Mismatch: reconstructed {len(user_ids)} user ids but graphs have "
            f"{n_user_nodes} user nodes. The reconstruction logic must not match "
            f"_build_id_maps() exactly -- STOP and re-check src/graph_builder.py "
            f"before trusting any labels attached from this script."
        )
    print(f"Verified: {n_user_nodes} user nodes matches reconstructed ordering")

    labels_df = pd.read_csv(LABELS_PATH, parse_dates=["day"])
    labels_df["user_id"] = labels_df["user_id"].astype(str)
    print(f"Loaded {len(labels_df)} malicious (user, day) label rows, "
          f"{labels_df['user_id'].nunique()} unique users")

    unmatched_users = set(labels_df["user_id"]) - set(user_ids)
    if unmatched_users:
        print(f"WARNING: {len(unmatched_users)} labeled users not found in graph "
              f"user ordering (e.g. {list(unmatched_users)[:5]}) -- their labels "
              f"will be silently dropped. Investigate before trusting results.")

    malicious_set = set(zip(labels_df["day"], labels_df["user_id"]))

    print("\n=== Attaching labels per day ===")
    total_malicious = 0
    days_with_positives = 0

    for day, g in graphs.items():
        y = torch.zeros(len(user_ids), dtype=torch.float32)
        n_pos = 0
        for idx, uid in enumerate(user_ids):
            if (day, uid) in malicious_set:
                y[idx] = 1.0
                n_pos += 1
        g["user"].y = y
        total_malicious += n_pos
        if n_pos > 0:
            days_with_positives += 1

    print(f"Total malicious (user, day) pairs labeled: {total_malicious}")
    print(f"Days with at least 1 malicious user: {days_with_positives} / {len(graphs)}")

    if total_malicious != len(labels_df):
        print(f"NOTE: {len(labels_df) - total_malicious} label rows did not match "
              f"any graph day (likely dates outside the {len(graphs)}-day built range, "
              f"or unmatched users listed above) -- expected if labels.py's date range "
              f"differs slightly from build_graphs.py's.")

    with open(OUT_PATH, "wb") as f:
        pickle.dump(graphs, f)
    print(f"\nSaved labeled graphs -> {OUT_PATH}")


if __name__ == "__main__":
    main()