"""
Concrete parsers for the log sources we currently have.

psychometric.csv -> USER NODE FEATURES (not activity edges -- handled separately,
                     see load_psychometric() below).
email.csv        -> ActivityRecords with edge_type='sends_email', target_type='domain'.

To add a future source (e.g. logon.csv with columns id,date,user,pc,activity):
    @register_parser('logon')
    def parse_logon(df: pd.DataFrame) -> List[ActivityRecord]:
        records = []
        for row in df.itertuples():
            records.append(ActivityRecord(
                user_id=row.user, timestamp=pd.to_datetime(row.date),
                target_type='pc', target_id=row.pc, edge_type='logs_into',
                features={'is_logon': 1.0 if row.activity == 'Logon' else 0.0}
            ))
        return records
That's it -- parse_all() and everything downstream picks it up automatically.
"""
from typing import List
import numpy as np
import pandas as pd
from src.schema import ActivityRecord, register_parser

INTERNAL_DOMAIN = "dtaa.com"


def _split_addresses(cell) -> List[str]:
    if pd.isna(cell) or cell == "":
        return []
    return [a.strip() for a in str(cell).split(";") if a.strip()]


def _domain_of(addr: str) -> str:
    return addr.split("@")[-1].lower() if "@" in addr else "unknown"


def load_psychometric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by user_id with OCEAN traits as static node features.
    Called directly by the graph builder (not through the activity-record registry,
    since these are per-user static attributes, not timestamped events).
    """
    out = df[["user_id", "O", "C", "E", "A", "N"]].copy()
    out = out.set_index("user_id")
    # Normalize 0-50 scale to 0-1 for stable GNN training
    for col in ["O", "C", "E", "A", "N"]:
        out[col] = out[col].astype(float) / 50.0
    return out


@register_parser("email")
def parse_email(df: pd.DataFrame) -> List[ActivityRecord]:
    """
    One ActivityRecord per (sender, recipient) pair per email, so a single email with
    3 recipients produces 3 edge events. Edge target is the RECIPIENT'S DOMAIN (internal
    'dtaa.com' vs external domains), which keeps the graph compact -- this can be swapped
    to per-recipient-address nodes later if finer resolution is needed.

    Per-event numeric features (aggregated later into mean/std/skew/kurtosis/median):
      - size          : email size in bytes
      - attachments   : attachment count
      - n_recipients  : total recipients on this email (to + cc)
      - is_external   : 1.0 if recipient domain != dtaa.com
      - hour_of_day   : 0-23, captures off-hours activity
      - has_cc        : 1.0 if email has any cc
    """
    records: List[ActivityRecord] = []
    ts_col = pd.to_datetime(df["date"], errors="coerce")

    for idx, row in df.iterrows():
        ts = ts_col.loc[idx]
        if pd.isna(ts):
            continue

        to_addrs = _split_addresses(row.get("to", ""))
        cc_addrs = _split_addresses(row.get("cc", ""))
        all_recipients = to_addrs + cc_addrs
        if not all_recipients:
            continue

        n_recipients = len(all_recipients)
        has_cc = 1.0 if cc_addrs else 0.0
        size = float(row.get("size", 0) or 0)
        attachments = float(row.get("attachments", 0) or 0)
        hour = ts.hour

        for addr in all_recipients:
            domain = _domain_of(addr)
            is_external = 0.0 if domain == INTERNAL_DOMAIN else 1.0
            records.append(ActivityRecord(
                user_id=str(row["user"]),
                timestamp=ts,
                target_type="domain",
                target_id=domain,
                edge_type="sends_email",
                features={
                    "size": size,
                    "attachments": attachments,
                    "n_recipients": float(n_recipients),
                    "is_external": is_external,
                    "hour_of_day": float(hour),
                    "has_cc": has_cc,
                },
            ))

        # Secondary edge: user -> pc (which machine they sent from). Useful once we add
        # logon/device data, since it links activity across log sources on the same node.
        pc = row.get("pc", None)
        if pc and not pd.isna(pc):
            records.append(ActivityRecord(
                user_id=str(row["user"]),
                timestamp=ts,
                target_type="pc",
                target_id=str(pc),
                edge_type="uses_pc",
                features={"hour_of_day": float(hour)},
            ))

    return records


@register_parser("logon")
def parse_logon(df: pd.DataFrame) -> List[ActivityRecord]:
    """Vectorized. Real schema: id, date, user, pc, activity (Logon/Logoff)."""
    ts = pd.to_datetime(df["date"], errors="coerce")
    valid = ts.notna()
    df = df.loc[valid]
    ts = ts.loc[valid]
    hour = ts.dt.hour

    is_logon = (df["activity"].astype(str).str.strip().str.lower() == "logon").astype(float)
    is_after_hours = ((hour < 6) | (hour >= 19)).astype(float)

    return [
        ActivityRecord(
            user_id=str(u), timestamp=t, target_type="pc", target_id=str(p),
            edge_type="logs_into",
            features={"is_logon": il, "hour_of_day": float(h), "is_after_hours": iah},
        )
        for u, t, p, il, h, iah in zip(
            df["user"], ts, df["pc"], is_logon, hour, is_after_hours
        )
    ]



@register_parser("device")
def parse_device(df: pd.DataFrame) -> List[ActivityRecord]:
    """Vectorized. Real schema: id, date, user, pc, activity (connect/disconnect)."""
    ts = pd.to_datetime(df["date"], errors="coerce")
    valid = ts.notna()
    df = df.loc[valid]
    ts = ts.loc[valid]
    hour = ts.dt.hour

    is_connect = (df["activity"].astype(str).str.strip().str.lower() == "connect").astype(float)
    is_after_hours = ((hour < 6) | (hour >= 19)).astype(float)

    return [
        ActivityRecord(
            user_id=str(u), timestamp=t, target_type="pc", target_id=str(p),
            edge_type="uses_device",
            features={"is_connect": ic, "hour_of_day": float(h), "is_after_hours": iah},
        )
        for u, t, p, ic, h, iah in zip(
            df["user"], ts, df["pc"], is_connect, hour, is_after_hours
        )
    ]


@register_parser("http")
def parse_http(df: pd.DataFrame) -> List[ActivityRecord]:
    """Real schema: id, date, user, pc, url, content.
    Domain is extracted from the URL as the edge target (user -> domain), same pattern
    as email recipients, so both end up in the same 'domain' node type. 'content' (topic
    keywords) is left unused for now -- natural extension point for topic-drift features."""
    records: List[ActivityRecord] = []
    ts_col = pd.to_datetime(df["date"], errors="coerce")
    for idx, row in df.iterrows():
        ts = ts_col.loc[idx]
        if pd.isna(ts):
            continue
        url = str(row.get("url", ""))
        parts = url.split("/")
        domain = parts[2].lower() if url.startswith("http") and len(parts) > 2 else "unknown"
        content = str(row.get("content", "") or "")
        hour = ts.hour
        records.append(ActivityRecord(
            user_id=str(row["user"]),
            timestamp=ts,
            target_type="domain",
            target_id=domain,
            edge_type="visits_url",
            features={
                "hour_of_day": float(hour),
                "content_len": float(len(content.split())),  # crude proxy until real topic features
            },
        ))
    return records


@register_parser("file")
def parse_file(df: pd.DataFrame) -> List[ActivityRecord]:
    """Vectorized. Real schema: id, date, user, pc, filename, content."""
    ts = pd.to_datetime(df["date"], errors="coerce")
    valid = ts.notna()
    df = df.loc[valid]
    ts = ts.loc[valid]
    hour = ts.dt.hour

    filename = df["filename"].fillna("").astype(str)
    ext = filename.str.rsplit(".", n=1).str[-1].str.lower()
    ext = ext.where(filename.str.contains(r"\."), "none")
    office_exts = {"doc", "docx", "xls", "xlsx", "ppt", "pptx"}
    is_office = ext.isin(office_exts).astype(float)
    content_len = df["content"].fillna("").astype(str).str.split().str.len().astype(float)

    return [
        ActivityRecord(
            user_id=str(u), timestamp=t, target_type="pc", target_id=str(p),
            edge_type="copies_file",
            features={"hour_of_day": float(h), "is_office_doc": io, "content_len": cl},
        )
        for u, t, p, h, io, cl in zip(
            df["user"], ts, df["pc"], hour, is_office, content_len
        )
    ]