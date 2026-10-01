"""
Splits users into train/val/test sets, stratified by whether they were EVER
malicious (at least one malicious day in data/processed/user_day_labels.csv).
Ensures each split gets a proportional share of the 72 malicious users, since a
naive random split risks starving val/test of positive examples entirely.

Split is by USER, not by day -- a user's full history stays in exactly one split,
to avoid leaking their behavioral baseline across train/test.

Usage:
    python3 scripts/split_users.py
"""
import json
import random
from pathlib import Path
import pandas as pd

AGG_EDGES_PATH = Path("output/aggregated_edges.csv")
PSYCH_PATH = Path("data/raw/r4.2/psychometric.csv")
LABELS_PATH = Path("data/processed/user_day_labels.csv")
OUT_PATH = Path("data/processed/user_splits.json")

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC is the remainder (0.15)
SEED = 42


def reconstruct_user_ordering() -> list:
    """Must exactly match src/graph_builder.py's _build_id_maps() user ordering
    (same logic reused in scripts/attach_labels.py)."""
    agg_user_ids = pd.read_csv(AGG_EDGES_PATH, usecols=["user_id"])["user_id"].astype(str)
    psych_user_ids = pd.read_csv(PSYCH_PATH, usecols=["user_id"])["user_id"].astype(str)
    return sorted(set(agg_user_ids) | set(psych_user_ids))


def stratified_user_split(all_users: list, malicious_users: set, seed: int):
    rng = random.Random(seed)

    malicious = [u for u in all_users if u in malicious_users]
    benign = [u for u in all_users if u not in malicious_users]
    rng.shuffle(malicious)
    rng.shuffle(benign)

    def split_list(lst):
        n = len(lst)
        n_train = round(n * TRAIN_FRAC)
        n_val = round(n * VAL_FRAC)
        return lst[:n_train], lst[n_train:n_train + n_val], lst[n_train + n_val:]

    mal_train, mal_val, mal_test = split_list(malicious)
    ben_train, ben_val, ben_test = split_list(benign)

    return {
        "train": sorted(mal_train + ben_train),
        "val": sorted(mal_val + ben_val),
        "test": sorted(mal_test + ben_test),
    }


def main():
    print("=== Reconstructing user ordering ===")
    all_users = reconstruct_user_ordering()
    print(f"Total users: {len(all_users)}")

    print("\n=== Loading labels ===")
    labels_df = pd.read_csv(LABELS_PATH)
    labels_df["user_id"] = labels_df["user_id"].astype(str)
    malicious_users = set(labels_df["user_id"].unique())
    print(f"Malicious users: {len(malicious_users)}")

    print("\n=== Splitting (stratified by malicious flag, seed=%d) ===" % SEED)
    splits = stratified_user_split(all_users, malicious_users, SEED)

    for name, users in splits.items():
        n_mal = len(set(users) & malicious_users)
        print(f"{name}: {len(users)} users ({n_mal} malicious, "
              f"{n_mal / len(users) * 100:.1f}%)")

    # Sanity check: no user appears in more than one split, every user accounted for
    all_split_users = splits["train"] + splits["val"] + splits["test"]
    assert len(all_split_users) == len(set(all_split_users)) == len(all_users), \
        "Split integrity check failed -- user(s) missing or duplicated across splits"
    print("\nIntegrity check passed: every user in exactly one split, no duplicates.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()