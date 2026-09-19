"""
Builds a strictly chronological train/val/test split over the full 501-day
graph timeline, choosing split boundaries by cumulative positive-label count
(not raw calendar percentage) so each split gets a fair share of malicious
(user, day) pairs while preserving temporal order (train days < val days <
test days -- no leakage).

Run from project root:
    python scripts/build_split.py

Writes: data/processed/day_splits.json
  {"train": ["2010-01-02", ...], "val": [...], "test": [...]}
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")

GRAPH_START = "2010-01-02"
GRAPH_END = "2011-05-17"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.85  # cumulative -- val ends here, test is the remainder

OUT_PATH = Path("data/processed/day_splits.json")


def main():
    labels_path = Path("data/processed/user_day_labels.csv")
    labels = pd.read_csv(labels_path, parse_dates=["day"])

    all_days = pd.date_range(GRAPH_START, GRAPH_END, freq="D")
    print(f"[split] full graph timeline: {len(all_days)} days "
          f"({GRAPH_START} -> {GRAPH_END})")
    if len(all_days) != 501:
        print(f"[split] WARNING: expected 501 days, got {len(all_days)} -- "
              f"check GRAPH_START/GRAPH_END match your actual daily_graphs.pkl range")

    # Positive count per day across the FULL timeline (0 for days with no label)
    pos_per_day = labels.groupby("day").size().reindex(all_days, fill_value=0)
    total_pos = pos_per_day.sum()
    cum_pos = pos_per_day.cumsum()

    train_end_idx = (cum_pos >= total_pos * TRAIN_FRAC).idxmax()
    val_end_idx = (cum_pos >= total_pos * VAL_FRAC).idxmax()

    train_days = all_days[all_days <= train_end_idx]
    val_days = all_days[(all_days > train_end_idx) & (all_days <= val_end_idx)]
    test_days = all_days[all_days > val_end_idx]

    print(f"\n[split] boundaries chosen by cumulative positive-label count "
          f"({int(TRAIN_FRAC*100)}% / {int((VAL_FRAC-TRAIN_FRAC)*100)}% / "
          f"{int((1-VAL_FRAC)*100)}%):")
    print(f"  train: {train_days[0].date()} -> {train_days[-1].date()}  "
          f"({len(train_days)} days)")
    print(f"  val:   {val_days[0].date()} -> {val_days[-1].date()}  "
          f"({len(val_days)} days)")
    print(f"  test:  {test_days[0].date()} -> {test_days[-1].date()}  "
          f"({len(test_days)} days)")

    print("\n[split] positive-label coverage per split:")
    for name, days in [("train", train_days), ("val", val_days), ("test", test_days)]:
        subset = labels[labels["day"].isin(days)]
        print(f"  {name}: {len(subset)} malicious pairs "
              f"({len(subset)/total_pos*100:.1f}% of total), "
              f"{subset['user_id'].nunique()} unique users, "
              f"{subset['day'].nunique()} malicious days")

    # Sanity: no day appears in more than one split, and all days covered
    assert len(train_days) + len(val_days) + len(test_days) == len(all_days)
    assert set(train_days).isdisjoint(set(val_days))
    assert set(val_days).isdisjoint(set(test_days))
    print("\n[split] sanity check passed: splits are disjoint and cover all days")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "train": [d.strftime("%Y-%m-%d") for d in train_days],
            "val": [d.strftime("%Y-%m-%d") for d in val_days],
            "test": [d.strftime("%Y-%m-%d") for d in test_days],
        }, f, indent=2)
    print(f"[split] saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()