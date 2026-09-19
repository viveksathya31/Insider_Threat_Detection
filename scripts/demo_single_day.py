"""
Live demo: pick a day from the TEST split, run the trained model on it, and
show its predictions side-by-side with ground truth. Designed to run fast
(single day, not the full split) for use in a live walkthrough.

Usage:
    python scripts/demo_single_day.py                  # auto-picks an
                                                         # illustrative test day
    python scripts/demo_single_day.py --day 2011-03-15  # a specific day
    python scripts/demo_single_day.py --list            # list test days with
                                                         # malicious activity,
                                                         # to pick one ahead of time
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, "src")
from model import HeteroGNN, to_undirected, get_edge_dims

GRAPHS_PATH = Path("output/daily_graphs_labeled.pkl")
SPLITS_PATH = Path("data/processed/day_splits.json")
CHECKPOINT_PATH = Path("output/baseline_gnn_best.pt")
EDGE_STATS_PATH = Path("output/edge_norm_stats.pt")
HIDDEN_DIM = 64
THRESHOLD = 0.9938  # val-derived threshold from evaluate_test.py -- keep in sync


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(sample_data, device):
    edge_dims = get_edge_dims(sample_data)
    model = HeteroGNN(sample_data.metadata(), edge_dims=edge_dims, hidden_dim=HIDDEN_DIM).to(device)
    with torch.no_grad():
        _ = model(sample_data.x_dict, sample_data.edge_index_dict, sample_data.edge_attr_dict)
    state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--day", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    with open(GRAPHS_PATH, "rb") as f:
        graphs = pickle.load(f)
    with open(SPLITS_PATH) as f:
        splits = json.load(f)
    edge_stats = torch.load(EDGE_STATS_PATH, weights_only=False)
    labels_df = pd.read_csv("data/processed/user_day_labels.csv", parse_dates=["day"])

    test_days = [pd.Timestamp(d) for d in splits["test"]]

    if args.list:
        print("Test days with malicious activity (pick one with --day):")
        for d in test_days:
            n = (labels_df["day"] == d).sum()
            if n > 0:
                print(f"  {d.date()}: {n} malicious user(s)")
        return

    if args.day:
        day = pd.Timestamp(args.day)
        assert day in test_days, f"{args.day} is not in the test split"
    else:
        # auto-pick: a test day with malicious activity, for an illustrative demo
        counts = [(d, (labels_df["day"] == d).sum()) for d in test_days]
        counts = [c for c in counts if c[1] > 0]
        day = sorted(counts, key=lambda c: -c[1])[0][0]  # most malicious users that day

    print(f"\n{'='*60}")
    print(f"  DEMO: {day.date()}  (from TEST split -- model never trained on this day)")
    print(f"{'='*60}\n")

    device = get_device()
    data = to_undirected(graphs[day])
    for et in data.edge_types:
        if "edge_attr" not in data[et] or et not in edge_stats:
            continue
        mean, std = edge_stats[et]
        data[et].edge_attr = (data[et].edge_attr - mean) / std
    data = data.to(device)

    model = load_model(data, device)

    with torch.no_grad():
        logits = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)
        probs = torch.sigmoid(logits).cpu().numpy()

    true_labels = data["user"].y.cpu().numpy()

    # Need real user_ids for display -- reconstruct the same index ordering
    # graph_builder.py / attach_labels.py use.
    agg = pd.read_csv("output/aggregated_edges.csv", usecols=["user_id"]).drop_duplicates()
    psych_users = pd.read_csv("data/raw/r4.2/psychometric.csv", usecols=["user_id"])
    all_user_ids = sorted(set(agg["user_id"]) | set(psych_users["user_id"]))

    results = pd.DataFrame({
        "user_id": all_user_ids,
        "predicted_prob": probs,
        "actually_malicious": true_labels.astype(bool),
    }).sort_values("predicted_prob", ascending=False)

    print(f"Total users: {len(results)}  |  Actually malicious today: {true_labels.sum():.0f}")
    print(f"Operating threshold: {THRESHOLD}\n")

    print("Top 10 highest-risk users (model's ranking):")
    print(results.head(10).to_string(index=False))

    flagged = results[results["predicted_prob"] >= THRESHOLD]
    caught = flagged[flagged["actually_malicious"]]
    missed = results[results["actually_malicious"] & (results["predicted_prob"] < THRESHOLD)]

    print(f"\nAt threshold {THRESHOLD}:")
    print(f"  Flagged {len(flagged)} users, {len(caught)} were truly malicious")
    if len(caught) > 0:
        print(f"  Caught: {caught['user_id'].tolist()}")
    if len(missed) > 0:
        print(f"  Missed: {missed['user_id'].tolist()}")
    else:
        print(f"  Missed: none -- caught every truly malicious user today")


if __name__ == "__main__":
    main()