"""
Core abstractions shared by every log-source parser.

Design goal: adding a NEW CERT log file (logon.csv, device.csv, http.csv, file.csv, ...)
in the future should require writing exactly one new parser function and registering it
below -- nothing else in the pipeline (aggregation, graph construction, model) changes.
"""
from dataclasses import dataclass, field
from typing import Dict, Callable, List
import pandas as pd


@dataclass
class ActivityRecord:
    """
    One normalized 'edge event' extracted from any raw log file.

    user_id     : source node id (always a CERT user_id)
    timestamp   : pandas.Timestamp
    target_type : node type of the OTHER endpoint of this activity, e.g. 'domain', 'pc', 'recipient'
    target_id   : id of that node, e.g. 'dtaa.com', 'PC-4275'
    edge_type   : relation name, e.g. 'sends_email', 'logs_into'
    features    : dict of numeric features for THIS single event (aggregated later into
                  mean/std/skew/kurtosis/median per (user, target, day), matching NF-GNN)
    """
    user_id: str
    timestamp: pd.Timestamp
    target_type: str
    target_id: str
    edge_type: str
    features: Dict[str, float] = field(default_factory=dict)


# --- Registry -----------------------------------------------------------
# Each entry: source_name -> parser function.
# A parser function takes a raw dataframe (as loaded from CSV) and yields ActivityRecords.
PARSER_REGISTRY: Dict[str, Callable[[pd.DataFrame], List[ActivityRecord]]] = {}


def register_parser(source_name: str):
    """Decorator: @register_parser('email') on a function registers it as a log source."""
    def _wrap(fn):
        PARSER_REGISTRY[source_name] = fn
        return fn
    return _wrap


def parse_all(raw_frames: Dict[str, pd.DataFrame]) -> List[ActivityRecord]:
    """
    raw_frames: {'email': email_df, 'logon': logon_df, ...}
    Only sources present in raw_frames AND registered in PARSER_REGISTRY are parsed.
    Missing future sources are silently skipped -- add the CSV + parser later, no
    other code changes needed.
    """
    records: List[ActivityRecord] = []
    for source_name, df in raw_frames.items():
        parser = PARSER_REGISTRY.get(source_name)
        if parser is None:
            print(f"[schema] WARNING: no parser registered for source '{source_name}', skipping.")
            continue
        recs = parser(df)
        print(f"[schema] {source_name}: parsed {len(recs)} activity records")
        records.extend(recs)
    return records