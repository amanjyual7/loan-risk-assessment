"""02 — EDA.

Deliberately narrow: only the columns that survive the leakage list are
explored, because exploring a leaking column is how a leaking column talks
its way into the model.

    python notebooks/02_eda.py
"""

# %%
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from notebooks._io import load_frame, save_frame  # noqa: E402
from notebooks.leakage import excluded_columns  # noqa: E402

WORK = REPO_ROOT / "notebooks" / "_work"
FIGS = WORK / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

CANDIDATES = [
    "fico_range_low", "fico_range_high", "annual_inc", "dti", "emp_length",
    "home_ownership", "revol_util", "open_acc", "delinq_2yrs",
    "inq_last_6mths", "pub_rec", "loan_amnt", "term", "purpose",
]


# %%
def univariate(frame: pd.DataFrame) -> None:
    numeric = frame[CANDIDATES].select_dtypes("number")
    print("\nnumeric predictors:")
    print(numeric.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99])
          .T.to_string(float_format=lambda v: f"{v:,.2f}"))

    print("\nmissingness (kept columns only):")
    miss = frame[CANDIDATES].isna().mean().sort_values(ascending=False)
    print(miss[miss > 0].to_string(float_format=lambda v: f"{v:.2%}")
          or "  none")


# %%
def default_rate_by_bucket(frame: pd.DataFrame, column: str, bins: int = 10):
    """Default rate across deciles of a predictor.

    This is the plot to look at before fitting anything: a predictor with a
    flat or non-monotone relationship here will not behave monotonically in
    the model either, and that is worth knowing before the monotonicity test
    in tests/ catches it.
    """
    series = frame[column]
    if pd.api.types.is_numeric_dtype(series):
        buckets = pd.qcut(series, bins, duplicates="drop")
    else:
        buckets = series.fillna("missing")

    table = frame.groupby(buckets, observed=True)["target"].agg(["size", "mean"])
    table.columns = ["loans", "default_rate"]
    print(f"\n{column}:")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(range(len(table)), table["default_rate"], color="#a61b1b")
    ax.set_xticks(range(len(table)))
    ax.set_xticklabels([str(i) for i in table.index], rotation=45,
                       ha="right", fontsize=7)
    ax.set_ylabel("default rate")
    ax.set_title(f"Default rate by {column}")
    fig.tight_layout()
    path = FIGS / f"default_rate_{column}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return table


# %%
def cohort_volume(frame: pd.DataFrame) -> None:
    by_quarter = frame.groupby("cohort_quarter")["target"].agg(["size", "mean"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    top.bar(range(len(by_quarter)), by_quarter["size"], color="#4f6d7a")
    top.set_ylabel("loans originated")
    bottom.plot(range(len(by_quarter)), by_quarter["mean"], color="#a61b1b")
    bottom.set_ylabel("default rate")
    bottom.set_xticks(range(0, len(by_quarter), 4))
    bottom.set_xticklabels(by_quarter.index[::4], rotation=45, ha="right",
                           fontsize=7)
    fig.suptitle("Origination volume and default rate by cohort")
    fig.tight_layout()
    fig.savefig(FIGS / "cohorts.png", dpi=110)
    plt.close(fig)
    print(f"\nwrote {FIGS / 'cohorts.png'}")


# %%
def main() -> None:
    frame = load_frame(WORK / "resolved")
    print(f"{len(frame):,} resolved loans")

    leaking = set(frame.columns) & excluded_columns()
    print(f"{len(leaking)} excluded columns still present in the working file "
          "— they are dropped in 03, and nothing in this notebook touches them")

    univariate(frame)
    for column in ("fico_range_low", "dti", "revol_util", "annual_inc",
                   "inq_last_6mths", "term", "purpose", "home_ownership"):
        if column in frame:
            default_rate_by_bucket(frame, column)
    cohort_volume(frame)
    print(f"\nfigures in {FIGS}")


if __name__ == "__main__":
    main()
