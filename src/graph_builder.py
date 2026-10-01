"""
Turns aggregated edge statistics + psychometric + LDAP features into a sequence of
PyTorch Geometric HeteroData graphs, one per day (day 1, day 2, ... day N).

User node features are now PER-MONTH (not fully static): 5 static OCEAN traits +
5 LDAP-derived features (4 normalized categorical + is_present), since LDAP data
changes month to month (role changes, promotions, departures) while OCEAN does not.
Each day pulls its user feature tensor from whichever calendar month it falls in.

Node types   : 'user', 'domain', 'pc'
Edge types   : derived from whatever is present in the aggregated dataframe.
"""
from typing import Dict, List
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

LDAP_FEATURE_COLS = [
    "role_norm", "functional_unit_norm", "department_norm", "team_norm", "is_present",
]


def _build_id_maps(agg_df: pd.DataFrame, psych_df: pd.DataFrame):
    user_ids = sorted(set(agg_df["user_id"]) | set(psych_df.index))
    id_maps = {"user": {u: i for i, u in enumerate(user_ids)}}

    for target_type in agg_df["target_type"].unique():
        ids = sorted(agg_df.loc[agg_df["target_type"] == target_type, "target_id"].unique())
        id_maps[target_type] = {v: i for i, v in enumerate(ids)}

    return id_maps


def _edge_feature_cols(agg_df: pd.DataFrame, edge_type: str) -> List[str]:
    sub = agg_df[agg_df["edge_type"] == edge_type]
    exclude = {"day", "user_id", "edge_type", "target_type", "target_id"}
    cols = [c for c in sub.columns if c not in exclude]
    return [c for c in cols if sub[c].notna().all()]

def build_user_node_features_by_month(psych_df: pd.DataFrame, ldap_df: pd.DataFrame,
                                       id_maps: Dict) -> Dict[str, torch.Tensor]:
    """
    Returns {month_str ('2010-01'): Tensor[n_users, 10]} -- OCEAN (static, 5 cols)
    concatenated with that month's LDAP features (5 cols).

    Vectorized via pandas reindex() per month instead of per-(month, user) .loc
    lookups -- the earlier per-row approach made 18,000 individual MultiIndex
    lookups and took over 2 hours; this does one reindex per month (18 total).
    """
    n_users = len(id_maps["user"])
    n_ocean = psych_df.shape[1]

    # Order users by their graph node index (0..n_users-1), not alphabetically,
    # so row i here lines up with node index i in every graph.
    user_order = sorted(id_maps["user"], key=lambda u: id_maps["user"][u])

    ocean_x = np.zeros((n_users, n_ocean), dtype=np.float32)
    for user_id, idx in id_maps["user"].items():
        if user_id in psych_df.index:
            ocean_x[idx] = psych_df.loc[user_id].to_numpy(dtype=np.float32)

    result = {}
    months = sorted(ldap_df["month"].unique())
    for month in months:
        month_df = ldap_df[ldap_df["month"] == month].set_index("user_id")
        # reindex aligns to user_order in one vectorized pass; missing users
        # (not in this month's snapshot) get NaN, filled with -1.0 sentinel
        aligned = month_df.reindex(user_order)[LDAP_FEATURE_COLS].fillna(-1.0)
        ldap_x = aligned.to_numpy(dtype=np.float32)

        combined = np.concatenate([ocean_x, ldap_x], axis=1)
        result[month] = torch.tensor(combined)

    return result


def _day_to_month(day: pd.Timestamp) -> str:
    return day.strftime("%Y-%m")


def build_daily_graphs(agg_df: pd.DataFrame, psych_df: pd.DataFrame,
                        ldap_df: pd.DataFrame) -> Dict[pd.Timestamp, HeteroData]:
    """
    Returns {day_timestamp: HeteroData}, sorted chronologically.
      - data['user'].x : OCEAN (static) + that day's month's LDAP features (10 dims
                          total, up from 5) -- VARIES month to month, unlike before.
      - data[<other node type>].x : placeholder ones.
      - data[edge_type].edge_index, data[edge_type].edge_attr : aggregated stats for that day.
    """
    if agg_df.empty:
        raise ValueError("aggregate_edges() returned no rows -- check parsers/raw data")

    id_maps = _build_id_maps(agg_df, psych_df)
    user_x_by_month = build_user_node_features_by_month(psych_df, ldap_df, id_maps)
    available_months = set(user_x_by_month.keys())

    graphs: Dict[pd.Timestamp, HeteroData] = {}

    for day, day_df in agg_df.groupby("day"):
        month = _day_to_month(day)
        if month not in available_months:
            raise ValueError(
                f"No LDAP snapshot covers month {month} (day {day.date()}) -- "
                f"available months: {sorted(available_months)}. Check LDAP coverage "
                f"against the full graph date range before rerunning."
            )

        data = HeteroData()
        data["user"].x = user_x_by_month[month]

        for node_type, mapping in id_maps.items():
            if node_type == "user":
                continue
            n = len(mapping)
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