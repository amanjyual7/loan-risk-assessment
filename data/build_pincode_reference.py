#!/usr/bin/env python3
"""Export the pincode -> tier Google Sheet to a committed SQLite artefact.

Run ONCE at build time, commit the output, and never call the sheet from the
app. A Streamlit Community Cloud app that reads a live Google Sheet on every
rerun is one quota error away from being broken in front of a reviewer.

    python data/build_pincode_reference.py
    python data/build_pincode_reference.py --sample   # rebuild the committed sample

Reads the first six columns matched by EXACT header name and ignores every
column beyond the sixth.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.pincode import REQUIRED_COLUMNS, TierScale  # noqa: E402

SHEET_ID = "1EI1nGuOtpq95O1M86VA0lIvVF0Hdytu5gzjC7YTvfGk"
TAB = "Data"
CSV_URL = (
    f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&sheet={TAB}"
)
OUT = REPO_ROOT / "data" / "pincode_reference.sqlite"
TIER_ORDER = ["Metro", "Tier 1", "Tier 2", "Tier 3"]


def fetch() -> pd.DataFrame:
    """The sheet must be link-viewable for this to work."""
    print(f"reading {CSV_URL}")
    return pd.read_csv(CSV_URL, dtype=str)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: str(c).strip().lower() for c in df.columns})

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(
            f"sheet is missing required headers: {missing}\n"
            f"found: {list(df.columns)}"
        )
    # Exact header match on the first six; everything else is ignored.
    df = df[REQUIRED_COLUMNS].copy()

    before = len(df)
    df["pincode"] = pd.to_numeric(df["pincode"], errors="coerce")
    df = df.dropna(subset=["pincode"])
    df["pincode"] = df["pincode"].astype("int64")
    print(f"dropped {before - len(df)} rows with unparseable pincodes")

    scale = TierScale(tuple(TIER_ORDER))
    df["tier"] = df["tier"].map(scale.normalise)

    for col in ("state", "city", "region"):
        df[col] = df[col].fillna("").str.strip()

    before = len(df)
    df = df.drop_duplicates(subset=["pincode"], keep="first")
    print(f"dropped {before - len(df)} duplicate pincodes")

    assert df["pincode"].is_unique, "pincode index is not unique after dedupe"
    return df.sort_values("pincode").reset_index(drop=True)


def write(df: pd.DataFrame, out: Path = OUT) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    conn = sqlite3.connect(out)
    try:
        conn.execute(
            """
            CREATE TABLE pincode (
                pincode    INTEGER PRIMARY KEY,
                state      TEXT,
                city       TEXT,
                metro_flag TEXT,
                tier       TEXT NOT NULL,
                region     TEXT
            )
            """
        )
        conn.executemany(
            "INSERT INTO pincode (pincode, state, city, metro_flag, tier, region) "
            "VALUES (?,?,?,?,?,?)",
            df[["pincode", "state", "city", "metro_flag", "tier", "region"]]
            .itertuples(index=False, name=None),
        )
        conn.execute("CREATE UNIQUE INDEX idx_pincode ON pincode(pincode)")
        conn.execute(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.executemany(
            "INSERT INTO meta VALUES (?,?)",
            [("source_sheet", SHEET_ID), ("rows", str(len(df))), ("tab", TAB)],
        )
        conn.commit()
    finally:
        conn.close()
    print(f"wrote {len(df):,} pincodes to {out}")


SAMPLE_ROWS = [
    # pincode, state, city, metro_flag, tier, region
    (110001, "Delhi", "New Delhi", "Y", "Metro", "North"),
    (122001, "Haryana", "Gurugram", "Y", "Metro", "North"),
    (201301, "Uttar Pradesh", "Noida", "Y", "Metro", "North"),
    (226001, "Uttar Pradesh", "Lucknow", "N", "Tier 1", "North"),
    (250001, "Uttar Pradesh", "Meerut", "N", "Tier 2", "North"),
    (302001, "Rajasthan", "Jaipur", "N", "Tier 1", "West"),
    (380001, "Gujarat", "Ahmedabad", "Y", "Metro", "West"),
    (400001, "Maharashtra", "Mumbai", "Y", "Metro", "West"),
    (411001, "Maharashtra", "Pune", "N", "Tier 1", "West"),
    (440001, "Maharashtra", "Nagpur", "N", "Tier 2", "West"),
    (452001, "Madhya Pradesh", "Indore", "N", "Tier 1", "West"),
    (500001, "Telangana", "Hyderabad", "Y", "Metro", "South"),
    (504273, "Telangana", "Adilabad", "N", "Tier 3", "South"),
    (560001, "Karnataka", "Bengaluru", "Y", "Metro", "South"),
    (600001, "Tamil Nadu", "Chennai", "Y", "Metro", "South"),
    (641001, "Tamil Nadu", "Coimbatore", "N", "Tier 2", "South"),
    (682001, "Kerala", "Kochi", "N", "Tier 2", "South"),
    (700001, "West Bengal", "Kolkata", "Y", "Metro", "East"),
    (751001, "Odisha", "Bhubaneswar", "N", "Tier 2", "East"),
    (800001, "Bihar", "Patna", "N", "Tier 2", "East"),
    (855101, "Bihar", "Katihar", "N", "Tier 3", "East"),
    (781001, "Assam", "Guwahati", "N", "Tier 2", "Northeast"),
    (799001, "Tripura", "Agartala", "N", "Tier 3", "Northeast"),
    (190001, "Jammu and Kashmir", "Srinagar", "N", "Tier 2", "North"),
]


def build_sample() -> pd.DataFrame:
    """A 24-row stand-in so the app and tests run before the real export.

    Clearly labelled in meta so nobody mistakes it for the full table.
    """
    return pd.DataFrame(SAMPLE_ROWS, columns=[
        "pincode", "state", "city", "metro_flag", "tier", "region"
    ])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true",
                    help="write the small committed sample instead of fetching")
    args = ap.parse_args()

    if args.sample:
        frame = build_sample()
        write(frame)
        conn = sqlite3.connect(OUT)
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('sample','true')")
        conn.commit()
        conn.close()
        print("NOTE: this is the SAMPLE artefact. Run without --sample for the "
              "full ~20k-pincode table.")
    else:
        write(transform(fetch()))
