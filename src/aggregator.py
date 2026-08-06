"""
Collapses raw per-event ActivityRecords into aggregated edges, one row per
(day, user, edge_type, target_type, target_id), with mean/std/skew/kurtosis/median
of every numeric feature -- the same 5-moment recipe NF-GNN uses for flow edges.

This is the layer that stays IDENTICAL no matter how many log sources you add later:
every parser just needs to emit ActivityRecords with a 'features' dict, and whatever
keys are in there get aggregated automatically.
"""
from typing import List
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import duckdb
from src.schema import ActivityRecord

MOMENTS = ["mean", "std", "skew", "kurtosis", "median"]


def _agg_moments(values: np.ndarray) -> dict:
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "skew": 0.0, "kurtosis": 0.0, "median": values[0]}
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "skew": float(skew(values)) if np.std(values) > 0 else 0.0,
        "kurtosis": float(kurtosis(values)) if np.std(values) > 0 else 0.0,
        "median": float(np.median(values)),
    }


def records_to_dataframe(records: List[ActivityRecord]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {
            "user_id": r.user_id,
            "timestamp": r.timestamp,
            "day": r.timestamp.normalize(),
            "target_type": r.target_type,
            "target_id": r.target_id,
            "edge_type": r.edge_type,
        }
        row.update(r.features)
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_edges(records: List[ActivityRecord]) -> pd.DataFrame:
    df = records_to_dataframe(records)
    if df.empty:
        return df

    group_keys = ["day", "user_id", "edge_type", "target_type", "target_id"]
    feature_cols = [c for c in df.columns if c not in group_keys + ["timestamp"]]

    out_rows = []
    for keys, group in df.groupby(group_keys):
        row = dict(zip(group_keys, keys))
        row["count"] = len(group)
        for col in feature_cols:
            moments = _agg_moments(group[col].to_numpy(dtype=float))
            for m_name, m_val in moments.items():
                row[f"{col}_{m_name}"] = m_val
        out_rows.append(row)

    return pd.DataFrame(out_rows)


def aggregate_events_file_duckdb(events_csv_path) -> pd.DataFrame:
    con = duckdb.connect()
    cols = con.execute(f"SELECT * FROM read_csv_auto('{events_csv_path}') LIMIT 0").df().columns
    group_keys = ["day", "user_id", "edge_type", "target_type", "target_id"]
    feature_cols = [c for c in cols if c not in group_keys + ["timestamp"]]

    agg_list_cols = ", ".join(f"list({c}) AS {c}_vals" for c in feature_cols)
    query = f"""
        SELECT day, user_id, edge_type, target_type, target_id, count(*) AS count,
               {agg_list_cols}
        FROM read_csv_auto('{events_csv_path}')
        GROUP BY day, user_id, edge_type, target_type, target_id
    """
    grouped = con.execute(query).df()
    con.close()

    out_rows = []
    for _, row in grouped.iterrows():
        rec = {
            "day": row["day"], "user_id": row["user_id"], "edge_type": row["edge_type"],
            "target_type": row["target_type"], "target_id": row["target_id"],
            "count": row["count"],
        }
        for col in feature_cols:
            vals = np.array(row[f"{col}_vals"], dtype=float)
            moments = _agg_moments(vals)
            for m_name, m_val in moments.items():
                rec[f"{col}_{m_name}"] = m_val
        out_rows.append(rec)
    return pd.DataFrame(out_rows)