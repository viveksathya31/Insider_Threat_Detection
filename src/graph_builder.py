"""
Turns aggregated edge statistics + psychometric features into a sequence of
PyTorch Geometric HeteroData graphs, one per day (day 1, day 2, ... day N).

That sequence is exactly what feeds NF-GNN-AE / -OC / -CLF per day, and later
a GRU/attention layer over the sequence for the temporal extension.

Node types   : 'user', 'domain', 'pc'   (extend this set when new sources are added,
               e.g. 'file', 'usb_device')
Edge types   : ('user','sends_email','domain'), ('user','uses_pc','pc')

Extensibility: to add a new edge type once a new source is registered (e.g. logon.csv
producing ('user','logs_into','pc')), nothing here needs to change -- build_daily_graphs()
derives node/edge types directly from whatever is present in the aggregated dataframe.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData


def _build_id_maps(agg_df: pd.DataFrame, psych_df: pd.DataFrame):
    """Global (stable across all days) integer id maps per node type, so a user/domain/pc
    keeps the same index in every daily graph -- required for the temporal GRU later."""
    user_ids = sorted(set(agg_df["user_id"]) | set(psych_df.index))
    id_maps = {"user": {u: i for i, u in enumerate(user_ids)}}

    for target_type in agg_df["target_type"].unique():
        ids = sorted(agg_df.loc[agg_df["target_type"] == target_type, "target_id"].unique())
        id_maps[target_type] = {v: i for i, v in enumerate(ids)}

    return id_maps


def _edge_feature_cols(agg_df: pd.DataFrame, edge_type: str) -> List[str]:
    """Only columns that are actually populated (non-null) for this edge_type.
    agg_df is a single dataframe covering ALL edge types, so columns unique to one
    edge type (e.g. 'size_mean' from email) show up as all-NaN for other edge types
    (e.g. uses_pc) unless we filter them out here."""
    sub = agg_df[agg_df["edge_type"] == edge_type]
    exclude = {"day", "user_id", "edge_type", "target_type", "target_id"}
    cols = [c for c in sub.columns if c not in exclude]
    return [c for c in cols if sub[c].notna().all()]


def build_user_node_features(psych_df: pd.DataFrame, id_maps: Dict) -> torch.Tensor:
    """Static OCEAN features per user, aligned to the global user id map.
    Users present in activity logs but missing from psychometric.csv get zero vectors
    (flag this in EDA -- shouldn't happen with a complete CERT release, but the pipeline
    won't crash if it does)."""
    n_users = len(id_maps["user"])
    n_feat = psych_df.shape[1]
    x = np.zeros((n_users, n_feat), dtype=np.float32)
    for user_id, idx in id_maps["user"].items():
        if user_id in psych_df.index:
            x[idx] = psych_df.loc[user_id].to_numpy(dtype=np.float32)
    return torch.tensor(x)


def build_daily_graphs(agg_df: pd.DataFrame, psych_df: pd.DataFrame) -> Dict[pd.Timestamp, HeteroData]:
    """
    Returns {day_timestamp: HeteroData}, sorted chronologically.
    Each HeteroData has:
      - data['user'].x            : static OCEAN features (same every day)
      - data[<other node type>].x : placeholder ones (extend with real features later,
                                     e.g. domain reputation score, pc criticality tier)
      - data[edge_type].edge_index, data[edge_type].edge_attr : aggregated stats for that day
    """
    if agg_df.empty:
        raise ValueError("aggregate_edges() returned no rows -- check parsers/raw data")

    id_maps = _build_id_maps(agg_df, psych_df)
    user_x = build_user_node_features(psych_df, id_maps)

    graphs: Dict[pd.Timestamp, HeteroData] = {}

    for day, day_df in agg_df.groupby("day"):
        data = HeteroData()
        data["user"].x = user_x  # same node set/features every day; only edges vary

        for node_type, mapping in id_maps.items():
            if node_type == "user":
                continue
            n = len(mapping)
            # Placeholder 1-dim feature for now (node exists / is active).
            # Swap in real attributes (e.g. domain risk score) as that data becomes available.
            data[node_type].x = torch.ones((n, 1), dtype=torch.float32)

        for edge_type in day_df["edge_type"].unique():
            edge_rows = day_df[day_df["edge_type"] == edge_type]
            target_type = edge_rows["target_type"].iloc[0]
            feat_cols = _edge_feature_cols(agg_df, edge_type)

            src = edge_rows["user_id"].map(id_maps["user"]).to_numpy()
            dst = edge_rows["target_id"].map(id_maps[target_type]).to_numpy()
            edge_index = torch.tensor(np.vstack([src, dst]), dtype=torch.long)
            edge_attr = torch.tensor(edge_rows[feat_cols].to_numpy(dtype=np.float32))

            data["user", edge_type, target_type].edge_index = edge_index
            data["user", edge_type, target_type].edge_attr = edge_attr

        graphs[day] = data

    return dict(sorted(graphs.items()))