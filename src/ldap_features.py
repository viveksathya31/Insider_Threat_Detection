"""
Parses the monthly LDAP snapshots (data/raw/r4.2/LDAP/*.csv) into per-(user, month)
numeric feature rows, for use as additional user node features alongside the static
OCEAN psychometric scores.

Unlike psychometric data, these fields change over time (role changes, promotions,
departures) -- so features here are keyed by (user_id, month), not just user_id.

Categorical columns (role, functional_unit, department, team) are label-encoded with
a STABLE mapping built from ALL months combined, so the same category always maps to
the same integer regardless of which month it's drawn from -- required for a model
to meaningfully compare the same category's index across time.

Also derives `is_present`: 1.0 if the user appears in that month's snapshot, 0.0 if
not. A user who stops appearing in later snapshots has likely left the organization --
several CERT scenarios explicitly describe this ("leaves the organization shortly
thereafter"), so this is a real signal, not just a data-completeness flag.

Usage:
    python3 -m src.ldap_features
"""
from pathlib import Path
import pandas as pd

LDAP_DIR = Path("data/raw/r4.2/LDAP")
OUT_PATH = Path("data/processed/ldap_features.csv")

CATEGORICAL_COLS = ["role", "functional_unit", "department", "team"]


def load_all_snapshots() -> dict:
    """Returns {month_str ('2010-01'): raw DataFrame indexed by user_id}."""
    snapshots = {}
    for path in sorted(LDAP_DIR.glob("*.csv")):
        month = path.stem  # e.g. '2010-01'
        df = pd.read_csv(path, dtype=str)
        df = df.set_index("user_id")
        snapshots[month] = df
    return snapshots


def build_stable_category_maps(snapshots: dict) -> dict:
    """One label-encoding map per categorical column, built from the UNION of values
    across all months, so a category's integer index is consistent over time."""
    maps = {}
    for col in CATEGORICAL_COLS:
        values = set()
        for df in snapshots.values():
            values.update(df[col].dropna().unique())
        maps[col] = {v: i for i, v in enumerate(sorted(values))}
    return maps


def build_ldap_features() -> pd.DataFrame:
    snapshots = load_all_snapshots()
    print(f"[ldap_features] Loaded {len(snapshots)} monthly snapshots: "
          f"{min(snapshots)} to {max(snapshots)}")

    cat_maps = build_stable_category_maps(snapshots)
    for col, mapping in cat_maps.items():
        print(f"[ldap_features] {col}: {len(mapping)} distinct categories")

    # Full set of users ever seen, across all months -- needed so a user who's
    # absent in a given month still gets a row (is_present=0, categorical cols NaN).
    all_users = sorted(set().union(*(df.index for df in snapshots.values())))

    rows = []
    for month, df in snapshots.items():
        present_users = set(df.index)
        for user_id in all_users:
            is_present = user_id in present_users
            row = {"user_id": user_id, "month": month, "is_present": 1.0 if is_present else 0.0}
            if is_present:
                user_row = df.loc[user_id]
                for col in CATEGORICAL_COLS:
                    val = user_row[col]
                    row[f"{col}_idx"] = float(cat_maps[col].get(val, -1))
                    n_cats = max(len(cat_maps[col]), 1)
                    row[f"{col}_norm"] = row[f"{col}_idx"] / n_cats  # 0-1 scale, same style as OCEAN
            else:
                for col in CATEGORICAL_COLS:
                    row[f"{col}_idx"] = -1.0
                    row[f"{col}_norm"] = -1.0  # sentinel: user not in org this month
            rows.append(row)

    out = pd.DataFrame(rows)
    print(f"[ldap_features] Built {len(out)} (user, month) feature rows "
          f"for {len(all_users)} unique users across {len(snapshots)} months")
    return out


if __name__ == "__main__":
    features_df = build_ldap_features()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUT_PATH, index=False)
    print(f"[ldap_features] Saved -> {OUT_PATH}")