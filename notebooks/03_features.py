"""03 — Feature construction.

The output of this notebook must match `engine.features.FEATURE_NAMES`
exactly, in order. That is asserted here rather than discovered at runtime,
because a silent schema drift between training and serving is the failure
that produces a model which loads fine and scores nonsense.

    python notebooks/03_features.py
"""

# %%
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.features import (  # noqa: E402
    FEATURE_NAMES,
    HIGH_RISK_PURPOSES,
    LOW_RISK_PURPOSES,
)
from notebooks._io import load_frame, save_frame  # noqa: E402
from notebooks.leakage import audit  # noqa: E402

WORK = REPO_ROOT / "notebooks" / "_work"


# %%
def parse_emp_length(series: pd.Series) -> pd.Series:
    """'10+ years' -> 10, '< 1 year' -> 0, NaN -> NaN.

    Left as NaN rather than filled: emp_length is missing for roughly 5% of
    Lending Club rows, and those rows are not randomly distributed. The
    imputation decision is made explicitly in 05, not hidden here.
    """
    cleaned = (
        series.astype(str)
        .str.replace("+", "", regex=False)
        .str.replace("< 1", "0", regex=False)
        .str.extract(r"(\d+)")[0]
    )
    return pd.to_numeric(cleaned, errors="coerce")


def parse_term(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )


# %%
def build(frame: pd.DataFrame) -> pd.DataFrame:
    report = audit(frame.columns)
    print(f"dropping {len(report['still_present'])} excluded columns")
    frame = frame.drop(columns=report["still_present"], errors="ignore")

    still = set(frame.columns) & set(report["still_present"])
    assert not still, f"excluded columns survived the drop: {sorted(still)}"

    out = pd.DataFrame(index=frame.index)

    # FICO is published as a 5-point range; the midpoint is the conventional
    # single-value treatment and both endpoints together are collinear.
    out["fico_avg"] = (frame["fico_range_low"] + frame["fico_range_high"]) / 2.0

    annual_inc = pd.to_numeric(frame["annual_inc"], errors="coerce")
    out["log_annual_inc"] = np.log(annual_inc.clip(lower=1))

    out["dti"] = pd.to_numeric(frame["dti"], errors="coerce")
    out["emp_length_years"] = parse_emp_length(frame["emp_length"])

    ownership = frame["home_ownership"].astype(str).str.upper()
    # OWN and MORTGAGE are both "not renting" for our purposes; the app's
    # residence_type has three levels and maps onto this pair of indicators.
    out["home_rent"] = (ownership == "RENT").astype(float)
    out["home_other"] = (~ownership.isin(["RENT", "OWN", "MORTGAGE"])).astype(float)

    out["revol_util"] = pd.to_numeric(
        frame["revol_util"].astype(str).str.rstrip("%"), errors="coerce"
    )
    out["open_acc"] = pd.to_numeric(frame["open_acc"], errors="coerce")
    out["delinq_2yrs"] = pd.to_numeric(frame["delinq_2yrs"], errors="coerce")
    out["inq_last_6mths"] = pd.to_numeric(frame["inq_last_6mths"], errors="coerce")
    out["pub_rec"] = pd.to_numeric(frame["pub_rec"], errors="coerce")

    loan_amnt = pd.to_numeric(frame["loan_amnt"], errors="coerce")
    out["log_loan_amnt"] = np.log(loan_amnt.clip(lower=1))

    term = parse_term(frame["term"])
    out["term_60"] = (term > 36).astype(float)

    purpose = frame["purpose"].astype(str).str.replace("_", " ").str.lower()
    out["purpose_high_risk"] = purpose.isin(HIGH_RISK_PURPOSES).astype(float)
    out["purpose_low_risk"] = purpose.isin(LOW_RISK_PURPOSES).astype(float)

    out["loan_to_income"] = loan_amnt / annual_inc.clip(lower=1)

    # Carried through for splitting and evaluation, not as features.
    out["target"] = frame["target"].values
    out["issue_date"] = frame["issue_date"].values
    out["cohort_year"] = frame["cohort_year"].values

    missing = [c for c in FEATURE_NAMES if c not in out.columns]
    extra = [c for c in out.columns
             if c not in set(FEATURE_NAMES) | {"target", "issue_date", "cohort_year"}]
    assert not missing, f"feature schema mismatch, missing: {missing}"
    assert not extra, f"feature schema mismatch, unexpected: {extra}"
    assert list(out[list(FEATURE_NAMES)].columns) == list(FEATURE_NAMES)

    print(f"\nbuilt {len(FEATURE_NAMES)} features for {len(out):,} loans")
    print(out[list(FEATURE_NAMES)].isna().mean()
          .sort_values(ascending=False)
          .to_string(float_format=lambda v: f"{v:.3%}"))
    return out


# %%
def standardisation_stats(features: pd.DataFrame) -> dict:
    """Median and IQR/2 per feature, to be stamped into the artefact.

    Median and IQR rather than mean and standard deviation: annual income
    and loan-to-income both have long right tails, and a mean-centred scale
    lets a handful of outliers move every applicant's standardised value.
    Support bounds are p1/p99, used by the app to detect extrapolation.
    """
    stats = {}
    for name in FEATURE_NAMES:
        column = features[name].dropna()
        q1, median, q3 = column.quantile([0.25, 0.5, 0.75])
        scale = (q3 - q1) / 2.0 or 1.0
        stats[name] = {
            "center": float(median),
            "scale": float(scale),
            "support": [float(column.quantile(0.01)), float(column.quantile(0.99))],
        }
    return stats


# %%
def main() -> None:
    frame = load_frame(WORK / "resolved")
    features = build(frame)
    features_path = save_frame(features, WORK / "features")

    stats = standardisation_stats(features)
    (WORK / "standardisation.json").write_text(json.dumps(stats, indent=2))

    print(f"\nwrote {features_path}")
    print(f"wrote {WORK / 'standardisation.json'}")
    print(
        "\nCompare the support bounds here against the FeatureSpec values in "
        "engine/features.py and update them — the app uses them to decide "
        "when it is extrapolating."
    )


if __name__ == "__main__":
    main()
