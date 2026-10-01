"""
Inspect the temporal distribution of malicious (user, day) labels before
designing a train/val/test split. Run from project root:

    python scripts/check_label_distribution.py

Answers:
  - How many of the 501 days have >=1 malicious user?
  - Are malicious days clustered in specific date ranges (per scenario)?
  - What does a 70/15/15 chronological split look like in terms of
    positive-label coverage in each split?
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")


def main():
    labels_path = Path("data/processed/user_day_labels.csv")
    if not labels_path.exists():
        print(f"[check] {labels_path} not found -- run `python src/labels.py` first")
        return

    labels = pd.read_csv(labels_path, parse_dates=["day"])
    labels = labels.sort_values("day")

    print(f"[check] total malicious (user, day) pairs: {len(labels)}")
    print(f"[check] unique malicious users: {labels['user_id'].nunique()}")
    print(f"[check] date range: {labels['day'].min()} -> {labels['day'].max()}")

    # Malicious days per month -- reveals clustering
    by_month = labels.groupby(labels["day"].dt.to_period("M")).size()
    print("\n[check] malicious (user, day) pairs per month:")
    print(by_month.to_string())

    # Days with at least 1 malicious user
    malicious_days = sorted(labels["day"].unique())
    print(f"\n[check] distinct days with >=1 malicious user: {len(malicious_days)}")

    # Simulate a 70/15/15 chronological split over the full graph date range.
    # NOTE: this assumes daily_graphs.pkl's date range -- adjust start/end if
    # your actual graph range differs from what's printed here.
    all_days = pd.date_range(labels["day"].min(), labels["day"].max(), freq="D")
    n = len(all_days)
    train_end = all_days[int(n * 0.70)]
    val_end = all_days[int(n * 0.85)]

    print(f"\n[check] simulated chronological split (70/15/15 by calendar day):")
    print(f"  train: {all_days[0].date()} -> {train_end.date()}")
    print(f"  val:   {train_end.date()} -> {val_end.date()}")
    print(f"  test:  {val_end.date()} -> {all_days[-1].date()}")

    for name, start, end in [
        ("train", all_days[0], train_end),
        ("val", train_end, val_end),
        ("test", val_end, all_days[-1]),
    ]:
        mask = (labels["day"] > start) & (labels["day"] <= end) if name != "train" else (labels["day"] <= end)
        subset = labels[mask]
        print(f"  {name}: {len(subset)} malicious pairs, "
              f"{subset['user_id'].nunique()} unique users")

    # Per-scenario check -- does each user's malicious window fall entirely
    # within one split, or does it straddle a boundary? (Straddling is fine
    # for a day-level task, but good to know.)
    print("\n[check] per-user malicious date range (first 15 users):")
    per_user = labels.groupby("user_id")["day"].agg(["min", "max", "count"])
    print(per_user.head(15).to_string())


if __name__ == "__main__":
    main()