"""01 — Load, define the target, build temporal cohorts.

Written as a script with `# %%` cell markers, so it opens as a notebook in
VS Code or Jupyter (via jupytext) and also runs headless.

    python notebooks/01_load_and_target.py --raw ~/data/lending_club/accepted.csv

Raw training data is NEVER committed. See .gitignore.
"""

# %%
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from notebooks._io import load_frame, save_frame  # noqa: E402
from notebooks.leakage import audit  # noqa: E402

WORK = REPO_ROOT / "notebooks" / "_work"       # gitignored scratch space
WORK.mkdir(exist_ok=True)

# %%
# ---------------------------------------------------------------------------
# Target definition.
#
# Binary: charged-off / defaulted == 1, fully repaid == 0.
#
# Everything still in progress is DROPPED, not treated as good. A loan that
# is 8 months into a 60-month term has no known outcome, and coding it as
# repaid is the second most common mistake in this dataset after leakage —
# it dilutes the positive class with loans that will default later.
# ---------------------------------------------------------------------------
DEFAULT_STATUSES = {"Charged Off", "Default",
                    "Does not meet the credit policy. Status:Charged Off"}
REPAID_STATUSES = {"Fully Paid",
                   "Does not meet the credit policy. Status:Fully Paid"}
IN_PROGRESS_STATUSES = {"Current", "In Grace Period", "Late (16-30 days)",
                        "Late (31-120 days)", "Issued"}


def load_raw(path: Path, nrows: int | None = None) -> pd.DataFrame:
    print(f"reading {path}")
    frame = pd.read_csv(path, low_memory=False, nrows=nrows)
    print(f"  {len(frame):,} rows × {frame.shape[1]} columns")
    return frame


def define_target(frame: pd.DataFrame) -> pd.DataFrame:
    counts = frame["loan_status"].value_counts(dropna=False)
    print("\nloan_status distribution:")
    print(counts.to_string())

    unknown = set(counts.index.dropna()) - (
        DEFAULT_STATUSES | REPAID_STATUSES | IN_PROGRESS_STATUSES
    )
    if unknown:
        raise SystemExit(
            f"unmapped loan_status values: {sorted(unknown)}. Map them "
            "explicitly rather than letting them fall into a default branch."
        )

    resolved = frame[frame["loan_status"].isin(DEFAULT_STATUSES | REPAID_STATUSES)].copy()
    dropped = len(frame) - len(resolved)
    print(f"\ndropped {dropped:,} loans with no known outcome "
          f"({dropped / len(frame):.1%} of the file)")

    resolved["target"] = resolved["loan_status"].isin(DEFAULT_STATUSES).astype(int)
    rate = resolved["target"].mean()
    print(f"observed default rate on resolved loans: {rate:.3%}  "
          f"({resolved['target'].sum():,} of {len(resolved):,})")
    print("  ^ put this number in the README model card")
    return resolved


# %%
# ---------------------------------------------------------------------------
# Temporal cohorts.
#
# issue_d is why this dataset was chosen over Home Credit: it gives an
# absolute origination date, so a genuine train-on-earlier / test-on-later
# split is possible. Home Credit's application table carries only DAYS_*
# fields relative to each application, which makes temporal validation
# impossible.
# ---------------------------------------------------------------------------
def add_cohorts(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["issue_date"] = pd.to_datetime(frame["issue_d"], format="%b-%Y",
                                         errors="coerce")
    missing = frame["issue_date"].isna().sum()
    if missing:
        print(f"dropping {missing:,} rows with an unparseable issue_d")
        frame = frame.dropna(subset=["issue_date"])

    frame["cohort_year"] = frame["issue_date"].dt.year
    frame["cohort_quarter"] = frame["issue_date"].dt.to_period("Q").astype(str)

    print("\ndefault rate by origination year:")
    summary = frame.groupby("cohort_year")["target"].agg(["size", "mean"])
    summary.columns = ["loans", "default_rate"]
    print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
    print(
        "\nNote the shape here: later cohorts look better simply because the "
        "worse-performing loans in them have not resolved yet and were "
        "dropped above. That survivorship effect is the reason the test "
        "cohort must not be the most recent incomplete one."
    )
    return frame


# %%
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="path to Lending Club accepted_*.csv")
    ap.add_argument("--nrows", type=int, default=None,
                    help="sample for a fast first pass")
    args = ap.parse_args()

    raw = load_raw(Path(args.raw).expanduser(), args.nrows)

    report = audit(raw.columns)
    print(f"\nleakage audit: {len(report['still_present'])} excluded columns "
          f"present in the raw file (expected — they are dropped in 03), "
          f"{len(report['not_in_data'])} on the list but absent from this "
          "dataset version")
    if report["not_in_data"]:
        print("  absent:", ", ".join(report["not_in_data"][:10]))

    resolved = add_cohorts(define_target(raw))

    out = save_frame(resolved, WORK / "resolved")
    print(f"\nwrote {out} ({len(resolved):,} rows)")


if __name__ == "__main__":
    main()
