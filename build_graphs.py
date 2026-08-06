"""
End-to-end pipeline: raw CERT CSVs -> sequence of daily heterogeneous graphs.

Usage:
    python3 build_graphs.py                              # full run, all data
    python3 build_graphs.py --start 2010-01-02 --end 2010-01-09   # 1-week test slice

Small sources (logon, device, file, psychometric) are loaded directly into memory.
Large sources (http, email) are streamed chunk-by-chunk to data/interim/*_events.csv
via src/streaming.py, then aggregated off-disk via DuckDB (src/aggregator.py).
"""
import argparse
import pickle
from pathlib import Path
import pandas as pd

from src.schema import parse_all
from src.parsers import (  # noqa: F401
    load_psychometric, parse_logon, parse_device, parse_file,
)
from src.aggregator import aggregate_edges, aggregate_events_file_duckdb
from src.graph_builder import build_daily_graphs
from src.streaming import parse_source_to_events_file

RAW_DIR = Path("data/raw/r4.2")
INTERIM_DIR = Path("data/interim")
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

SMALL_SOURCES = ["logon", "device", "file"]
LARGE_SOURCES = ["http", "email"]


def _date_filter(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if start is None and end is None:
        return df
    ts = pd.to_datetime(df["date"], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= ts >= pd.Timestamp(start)
    if end is not None:
        mask &= ts <= pd.Timestamp(end)
    return df.loc[mask]


def main(start=None, end=None):
    slice_note = f" (slice {start} to {end})" if (start or end) else " (FULL RUN)"
    print(f"=== Build config{slice_note} ===")

    print("\n=== 1. Loading small sources + psychometric ===")
    psych_raw = pd.read_csv(RAW_DIR / "psychometric.csv")
    psych_df = load_psychometric(psych_raw)
    print(f"psychometric: {psych_raw.shape}, users with OCEAN features: {len(psych_df)}")

    small_frames = {}
    for name in SMALL_SOURCES:
        df = pd.read_csv(RAW_DIR / f"{name}.csv")
        df = _date_filter(df, start, end)
        small_frames[name] = df
        print(f"{name}: {df.shape} (after date filter)")

    print("\n=== 2. Parsing small sources into activity records ===")
    small_records = parse_all(small_frames)
    print(f"Small-source activity records: {len(small_records)}")

    print("\n=== 3. Aggregating small-source records (in-memory) ===")
    small_agg_df = aggregate_edges(small_records)
    print(f"Small-source aggregated edges: {small_agg_df.shape}")

    print("\n=== 4. Streaming + aggregating large sources (http, email) ===")
    large_agg_frames = []
    for name in LARGE_SOURCES:
        csv_path = RAW_DIR / f"{name}.csv"
        events_path = INTERIM_DIR / f"{name}_events.csv"
        print(f"\n-- {name} --")
        if start or end:
            _stream_with_date_filter(name, csv_path, events_path, start, end)
        else:
            n = parse_source_to_events_file(name, csv_path, events_path)
            print(f"{name}: streamed {n} total events -> {events_path}")

        agg = aggregate_events_file_duckdb(events_path)
        print(f"{name}: aggregated to {agg.shape}")
        large_agg_frames.append(agg)

    print("\n=== 5. Combining all aggregated edges ===")
    agg_df = pd.concat([small_agg_df] + large_agg_frames, ignore_index=True, sort=False)
    print(f"Total aggregated edges: {agg_df.shape}")
    print(f"Edge types: {agg_df['edge_type'].unique().tolist()}")
    print(f"Days covered: {agg_df['day'].nunique()}")

    print("\n=== 6. Building daily heterogeneous graphs ===")
    graphs = build_daily_graphs(agg_df, psych_df)
    print(f"Built {len(graphs)} daily graphs")

    if graphs:
        first_day = next(iter(graphs))
        print(f"\nExample graph ({first_day.date()}):")
        print(graphs[first_day])

    print("\n=== 7. Saving ===")
    suffix = "_slice" if (start or end) else ""
    with open(OUT_DIR / f"daily_graphs{suffix}.pkl", "wb") as f:
        pickle.dump(graphs, f)
    agg_df.to_csv(OUT_DIR / f"aggregated_edges{suffix}.csv", index=False)
    print(f"Saved {len(graphs)} graphs to {OUT_DIR / f'daily_graphs{suffix}.pkl'}")
    print(f"Saved aggregated edge table to {OUT_DIR / f'aggregated_edges{suffix}.csv'}")


def _stream_with_date_filter(name, csv_path, events_path, start, end):
    from src.streaming import VECTORIZED_PARSERS
    parser = VECTORIZED_PARSERS[name]
    events_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    first_chunk = True

    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=500_000)):
        chunk = _date_filter(chunk, start, end)
        if chunk.empty:
            continue
        events = parser(chunk)
        events.to_csv(events_path, mode="w" if first_chunk else "a",
                       header=first_chunk, index=False)
        first_chunk = False
        total += len(events)
        print(f"[streaming/slice] {name}: chunk {i} -> {len(events)} events "
              f"in range, running total {total}")

    if first_chunk:
        pd.DataFrame(columns=["user_id", "timestamp", "day", "target_type",
                               "target_id", "edge_type"]).to_csv(events_path, index=False)
        print(f"[streaming/slice] {name}: no events in range, wrote empty events file")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=None, help="e.g. 2010-01-02")
    ap.add_argument("--end", type=str, default=None, help="e.g. 2010-01-09")
    args = ap.parse_args()
    main(start=args.start, end=args.end)