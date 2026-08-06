"""
Parses data/raw/answers/ (insiders.csv + per-incident detail files) into a
(user, day) -> is_malicious label table.

insiders.csv columns: dataset, scenario, details, user, start, end
  - only rows with dataset == '4.2' apply to our local r4.2 raw data (insiders.csv
    spans multiple CERT releases we don't have locally).
  - 'details' is the filename of that incident's per-event detail file, living
    under data/raw/answers/r4.2-<scenario>/<details>.

Per readme.txt: each detail file is NOT proper CSV -- rows are variable-length,
interleaved by event type (email/http/device/file/logon), with the first field
indicating that row's type.

Row layout by type (fields after type):
  email : id, date, user, pc, to, cc, bcc, from, size, attachments, content
  http  : id, date, user, pc, url, content
  device: id, date, user, pc, activity
  file  : id, date, user, pc, filename, content
  logon : id, date, user, pc, activity

We only need (type, id, date, user) from each row:
  - (type, id) is the precise key for matching against a raw r4.2/*.csv event
    later (ids are unique only WITHIN a source file -- see project handoff).
  - (user, date) gives the (user, day) label directly, no id-matching needed.
"""
from pathlib import Path
import csv
import pandas as pd

ANSWERS_DIR = Path("data/raw/answers")
INSIDERS_CSV = ANSWERS_DIR / "insiders.csv"
RELEASE = "4.2"


def _incident_file_path(scenario: str, details: str) -> Path:
    return ANSWERS_DIR / f"r4.2-{scenario}" / details


def load_insiders_index() -> pd.DataFrame:
    df = pd.read_csv(INSIDERS_CSV, dtype=str)
    return df[df["dataset"] == RELEASE].copy()


def parse_incident_file(path: Path):
    """Yields dicts: {source_type, event_id, timestamp, user} for one incident file."""
    rows = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for fields in csv.reader(f):
            if len(fields) < 4:
                continue  # malformed/short row, skip defensively
            ts = pd.to_datetime(fields[2].strip(), errors="coerce")
            if pd.isna(ts):
                continue
            rows.append({
                "source_type": fields[0].strip(),
                "event_id": fields[1].strip(),
                "timestamp": ts,
                "user": fields[3].strip(),
            })
    return rows


def build_labels():
    """
    Returns:
      user_day_labels: DataFrame[user_id, day, is_malicious=1] -- rows present here
        ARE malicious; union against your full (user, day) grid, everything else = 0.
      malicious_event_ids: dict[source_type] -> set(event_id), for precise row-level
        matching later (edge attribution / explainability ground truth).
    """
    index = load_insiders_index()
    print(f"[labels] {len(index)} incident entries in dataset {RELEASE}, "
          f"scenarios {sorted(index['scenario'].unique().tolist())}")

    all_rows = []
    for _, r in index.iterrows():
        path = _incident_file_path(r["scenario"], r["details"])
        if not path.exists():
            print(f"[labels] WARNING: missing incident file {path}")
            continue
        all_rows.extend(parse_incident_file(path))

    if not all_rows:
        raise RuntimeError("No incident rows parsed -- check answers/ layout")

    ev_df = pd.DataFrame(all_rows)
    malicious_event_ids = {
        st: set(group["event_id"]) for st, group in ev_df.groupby("source_type")
    }

    ev_df["day"] = ev_df["timestamp"].dt.normalize()
    user_day_labels = (
        ev_df[["user", "day"]]
        .drop_duplicates()
        .rename(columns={"user": "user_id"})
        .assign(is_malicious=1)
        .reset_index(drop=True)
    )

    print(f"[labels] {len(user_day_labels)} malicious (user, day) pairs, "
          f"{user_day_labels['user_id'].nunique()} unique users")
    for st, ids in malicious_event_ids.items():
        print(f"[labels]   {st}: {len(ids)} malicious event ids")

    return user_day_labels, malicious_event_ids


if __name__ == "__main__":
    user_day_labels, malicious_event_ids = build_labels()
    out_path = Path("data/processed/user_day_labels.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    user_day_labels.to_csv(out_path, index=False)
    print(f"[labels] saved -> {out_path}")