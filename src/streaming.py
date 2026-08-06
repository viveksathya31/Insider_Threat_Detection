"""
Chunked reading + VECTORIZED parsing for the two large log sources (http.csv: 28.4M
rows/14.5GB, email.csv: 2.6M rows/1.36GB). Row-by-row iterrows() parsing (as used for
the small sources in src/parsers.py) does not scale to these sizes -- this module
re-implements http/email parsing using vectorized pandas ops instead, and streams
output straight to disk so the full parsed event set is never held in memory.

Output event CSVs land in data/interim/<source>_events.csv with columns:
    user_id, timestamp, day, target_type, target_id, edge_type, <feature columns>
which is exactly what aggregate_events_file_duckdb() (src/aggregator.py) expects.
"""
from pathlib import Path
import pandas as pd

INTERNAL_DOMAIN = "dtaa.com"


def _extract_url_domain(url_series: pd.Series) -> pd.Series:
    is_http = url_series.str.startswith("http", na=False)
    parts = url_series.str.split("/", n=3, expand=True)
    domain = parts[2].str.lower()
    return domain.where(is_http, "unknown").fillna("unknown")


def _vectorized_parse_http(chunk: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(chunk["date"], errors="coerce")
    valid = ts.notna()
    chunk = chunk.loc[valid].copy()
    ts = ts.loc[valid]

    out = pd.DataFrame({
        "user_id": chunk["user"].astype(str),
        "timestamp": ts,
        "day": ts.dt.normalize(),
        "target_type": "domain",
        "target_id": _extract_url_domain(chunk["url"]),
        "edge_type": "visits_url",
        "hour_of_day": ts.dt.hour.astype(float),
        "content_len": chunk["content"].fillna("").str.split().str.len().astype(float),
    })
    return out


def _vectorized_parse_email(chunk: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(chunk["date"], errors="coerce")
    valid = ts.notna()
    chunk = chunk.loc[valid].copy()
    ts = ts.loc[valid]

    to_lists = chunk["to"].fillna("").apply(lambda s: [a.strip() for a in s.split(";") if a.strip()])
    cc_lists = chunk["cc"].fillna("").apply(lambda s: [a.strip() for a in s.split(";") if a.strip()])
    all_recipients = to_lists + cc_lists

    base = pd.DataFrame({
        "user_id": chunk["user"].astype(str),
        "timestamp": ts,
        "day": ts.dt.normalize(),
        "hour_of_day": ts.dt.hour.astype(float),
        "size": pd.to_numeric(chunk["size"], errors="coerce").fillna(0.0),
        "attachments": pd.to_numeric(chunk["attachments"], errors="coerce").fillna(0.0),
        "n_recipients": all_recipients.apply(len).astype(float),
        "has_cc": cc_lists.apply(lambda L: 1.0 if L else 0.0),
        "recipients": all_recipients,
    })
    base = base[base["recipients"].apply(len) > 0]

    exploded = base.explode("recipients").rename(columns={"recipients": "addr"})
    domain = exploded["addr"].str.split("@").str[-1].str.lower().fillna("unknown")
    exploded["target_type"] = "domain"
    exploded["target_id"] = domain
    exploded["edge_type"] = "sends_email"
    exploded["is_external"] = (domain != INTERNAL_DOMAIN).astype(float)
    email_edges = exploded.drop(columns=["addr"])

    pc = chunk["pc"]
    pc_edges = pd.DataFrame({
        "user_id": chunk["user"].astype(str),
        "timestamp": ts,
        "day": ts.dt.normalize(),
        "target_type": "pc",
        "target_id": pc.astype(str),
        "edge_type": "uses_pc",
        "hour_of_day": ts.dt.hour.astype(float),
    })
    pc_edges = pc_edges[pc.notna()]

    return pd.concat([email_edges, pc_edges], ignore_index=True, sort=False)


VECTORIZED_PARSERS = {
    "http": _vectorized_parse_http,
    "email": _vectorized_parse_email,
}


def parse_source_to_events_file(source_name: str, csv_path: Path, out_path: Path,
                                 chunksize: int = 500_000) -> int:
    """
    Reads csv_path in chunks, vectorized-parses each chunk, appends to out_path.
    Returns total event rows written. Caller must pass source_name in
    VECTORIZED_PARSERS (currently 'http', 'email').
    """
    parser = VECTORIZED_PARSERS.get(source_name)
    if parser is None:
        raise ValueError(f"No vectorized parser registered for '{source_name}'")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    first_chunk = True
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=chunksize)):
        events = parser(chunk)
        events.to_csv(out_path, mode="w" if first_chunk else "a",
                       header=first_chunk, index=False)
        first_chunk = False
        total += len(events)
        print(f"[streaming] {source_name}: chunk {i} ({len(chunk)} rows) "
              f"-> {len(events)} events, running total {total}")
    return total