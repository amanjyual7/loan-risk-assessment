#!/usr/bin/env python3
"""00 — Generate a synthetic Lending-Club-shaped dataset.

    python notebooks/00_synthetic_data.py --rows 240000

=============================================================================
READ THIS BEFORE USING THE OUTPUT FOR ANYTHING
=============================================================================
A model trained on this file learns the relationships written below, because
this script is where those relationships come from. Any "finding" you get
out of it is an assumption read back to you. It is CIRCULAR.

What it is legitimately for:
  * proving the pipeline runs end to end before you have real data
  * exercising the leakage audit, the temporal split, the calibration step
    and the band derivation on data whose true structure you know
  * checking the monotonicity tests catch a sign error

What it is NOT for:
  * the README model card
  * any claim about what predicts default
  * a deployed portfolio app

Replace it with the real Lending Club file before publishing:
    kaggle datasets download -d wordsforthewise/lending-club \\
        -f accepted_2007_to_2018Q4.csv.gz -p data/raw
The downstream notebooks need no changes — this file has the same column
names and dtypes.
=============================================================================

Three things here are deliberate and worth knowing about.

1. The signal is WEAK on purpose. It is easy to generate synthetic credit
   data that yields AUC 0.95, and that number is a tell: real consumer
   credit models land around 0.65-0.72. The noise term is sized to put this
   in that range. A reviewer who sees AUC 0.94 on a credit model assumes
   leakage, and they are usually right.

2. There is a macro shock in the middle cohorts. Without it, a temporal
   split and a random split give the same answer, and the whole argument for
   temporal validation goes untested.

3. It emits leaking columns (grade, int_rate, recoveries, total_pymnt,
   last_pymnt_d) computed FROM the outcome. They exist so that
   notebooks/leakage.py has something real to drop, and so that you can see
   what happens if it does not: train with them included and AUC jumps to
   ~0.99, which is exactly the failure the exclusion list prevents.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "data" / "raw" / "synthetic_loans.csv.gz"

# ---------------------------------------------------------------------------
# The true data-generating process. These are the coefficients the model will
# rediscover, which is the whole reason the exercise is circular.
# Units: log-odds per standard deviation of the feature.
# ---------------------------------------------------------------------------
TRUE_COEFFICIENTS = {
    "fico": -0.58,
    "log_income": -0.16,
    "dti": 0.26,
    "emp_length": -0.09,
    "rent": 0.08,
    "revol_util": 0.22,
    "open_acc": 0.04,
    "delinq": 0.30,
    "inq": 0.15,
    "pub_rec": 0.17,
    "log_amount": 0.13,
    "term_60": 0.38,
    "purpose_high": 0.23,
    "loan_to_income": 0.14,
}
BASE_DEFAULT_RATE = 0.16
NOISE_SD = 0.95          # sized to land AUC in the 0.66-0.72 range
MACRO_SHOCK = {2013: 0.45, 2014: 0.35}   # extra log-odds by origination year

PURPOSES = {
    "debt_consolidation": 0.52, "credit_card": 0.20, "home_improvement": 0.07,
    "other": 0.07, "major_purchase": 0.04, "medical": 0.03, "car": 0.03,
    "small_business": 0.02, "moving": 0.01, "vacation": 0.01,
}
HIGH_RISK = {"small_business", "moving", "debt_consolidation", "other",
             "renewable_energy"}


def generate(rows: int, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # -- origination dates, 2012-2018, volume growing over time ------------
    # Window ends well before "today" so that most cohorts have had time to
    # resolve. Generate 2010-2016 against a 2018-12 observation date and the
    # test split still has real volume; generate up to 2018 and everything
    # recent is Current, leaving nothing to test on.
    months = pd.date_range("2010-01-01", "2016-12-01", freq="MS")
    weights = np.linspace(0.4, 1.6, len(months))
    weights /= weights.sum()
    issue = rng.choice(months, size=rows, p=weights)
    issue = pd.to_datetime(issue)
    year = issue.year

    # -- a latent credit-quality factor, so features are CORRELATED --------
    # Independent features are the second tell of synthetic credit data:
    # in reality income, bureau score and utilisation all move together.
    quality = rng.normal(0, 1, rows)

    fico = np.clip(690 + 42 * quality + rng.normal(0, 22, rows), 612, 848)
    fico = (fico // 5) * 5  # bureau scores arrive in 5-point bands
    log_income = 11.05 + 0.34 * quality + rng.normal(0, 0.42, rows)
    income = np.exp(log_income).round(-2)

    dti = np.clip(18 - 4.2 * quality + rng.normal(0, 7.5, rows), 0, 45)
    revol_util = np.clip(52 - 14 * quality + rng.normal(0, 21, rows), 0, 99.9)
    open_acc = np.clip(
        rng.poisson(np.clip(11 - 1.1 * quality, 2, None)), 1, 45
    )
    emp_length_years = np.clip(
        rng.poisson(np.clip(5 + 1.2 * quality, 0.4, None)), 0, 10
    )
    delinq = rng.poisson(np.clip(0.28 - 0.16 * quality, 0.01, None))
    inq6 = rng.poisson(np.clip(0.85 - 0.30 * quality, 0.02, None))
    pub_rec = rng.poisson(np.clip(0.12 - 0.06 * quality, 0.005, None))

    ownership = rng.choice(
        ["MORTGAGE", "RENT", "OWN"], size=rows, p=[0.49, 0.40, 0.11]
    )
    is_rent = (ownership == "RENT").astype(float)

    term = rng.choice([36, 60], size=rows, p=[0.72, 0.28])
    amount = np.clip(
        np.exp(9.3 + 0.28 * quality + rng.normal(0, 0.55, rows)), 1000, 40000
    )
    amount = (amount / 25).round() * 25
    purpose = rng.choice(list(PURPOSES), size=rows, p=list(PURPOSES.values()))
    lti = amount / income

    # -- the outcome -------------------------------------------------------
    def z(values):
        values = np.asarray(values, dtype=float)
        return (values - values.mean()) / (values.std() or 1.0)

    c = TRUE_COEFFICIENTS
    linear = (
        c["fico"] * z(fico)
        + c["log_income"] * z(log_income)
        + c["dti"] * z(dti)
        + c["emp_length"] * z(emp_length_years)
        + c["rent"] * is_rent
        + c["revol_util"] * z(revol_util)
        + c["open_acc"] * z(open_acc)
        + c["delinq"] * z(np.minimum(delinq, 5))
        + c["inq"] * z(np.minimum(inq6, 8))
        + c["pub_rec"] * z(np.minimum(pub_rec, 3))
        + c["log_amount"] * z(np.log(amount))
        + c["term_60"] * (term == 60)
        + c["purpose_high"] * np.isin(purpose, list(HIGH_RISK))
        + c["loan_to_income"] * z(lti)
    )

    # one genuine interaction: high utilisation hurts a thin file more
    linear += 0.18 * ((revol_util > 80) & (open_acc < 6))

    for shock_year, extra in MACRO_SHOCK.items():
        linear = linear + extra * (year == shock_year)

    linear += rng.normal(0, NOISE_SD, rows)

    # Solve the intercept numerically rather than setting it to
    # logit(base_rate). Because the mean of a sigmoid is not the sigmoid of
    # the mean, a logit intercept overshoots badly once the linear predictor
    # has any real spread — here it lands near 29% against a 16% target.
    lo, hi = -8.0, 4.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(linear + mid)))).mean() < BASE_DEFAULT_RATE:
            lo = mid
        else:
            hi = mid
    linear = linear + (lo + hi) / 2

    pd_true = 1 / (1 + np.exp(-linear))
    defaulted = rng.random(rows) < pd_true

    # -- status: recent cohorts are still running --------------------------
    # 01_load_and_target.py must have in-progress loans to drop, or its
    # survivorship handling is never exercised.
    months_elapsed = (pd.Timestamp("2018-12-01") - issue).days / 30.44
    resolved = months_elapsed >= term * 0.92
    status = np.where(
        resolved,
        np.where(defaulted, "Charged Off", "Fully Paid"),
        np.where(rng.random(rows) < 0.04, "Late (31-120 days)", "Current"),
    )

    frame = pd.DataFrame({
        "id": np.arange(1, rows + 1),
        "loan_status": status,
        "issue_d": issue.strftime("%b-%Y"),
        "fico_range_low": fico.astype(int),
        "fico_range_high": (fico + 4).astype(int),
        "annual_inc": income,
        "dti": dti.round(2),
        "emp_length": [
            "< 1 year" if v == 0 else "10+ years" if v >= 10 else f"{v} years"
            for v in emp_length_years
        ],
        "home_ownership": ownership,
        "revol_util": [f"{v:.1f}%" for v in revol_util],
        "open_acc": open_acc,
        "delinq_2yrs": delinq,
        "inq_last_6mths": inq6,
        "pub_rec": pub_rec,
        "loan_amnt": amount,
        "term": [f" {t} months" for t in term],
        "purpose": purpose,
    })

    # -- leaking columns, computed FROM the outcome ------------------------
    # Present so the exclusion list has something to remove. Include them in
    # training and AUC goes to ~0.99; that is the point.
    grade_index = np.clip(
        ((pd_true - pd_true.min()) / (pd_true.max() - pd_true.min()) * 6.99),
        0, 6.99,
    ).astype(int)
    frame["grade"] = [list("ABCDEFG")[i] for i in grade_index]
    frame["sub_grade"] = [
        f"{g}{rng.integers(1, 6)}" for g in frame["grade"]
    ]
    frame["int_rate"] = [f"{6.0 + 3.2 * i + rng.normal(0, 0.4):.2f}%"
                         for i in grade_index]
    frame["recoveries"] = np.where(
        resolved & defaulted, (amount * rng.uniform(0.02, 0.22, rows)).round(2), 0.0
    )
    frame["total_pymnt"] = np.where(
        resolved & defaulted,
        (amount * rng.uniform(0.15, 0.75, rows)).round(2),
        (amount * rng.uniform(1.05, 1.42, rows)).round(2),
    )
    frame["total_rec_prncp"] = np.where(
        resolved & defaulted, (amount * rng.uniform(0.1, 0.6, rows)).round(2),
        amount.round(2),
    )
    frame["last_pymnt_d"] = np.where(
        resolved & defaulted,
        (issue + pd.to_timedelta(rng.integers(60, 400, rows), unit="D")).strftime("%b-%Y"),
        (issue + pd.to_timedelta((term * 30).astype(int), unit="D")).strftime("%b-%Y"),
    )
    frame["addr_state"] = rng.choice(
        ["CA", "NY", "TX", "FL", "IL", "NJ", "PA", "OH", "GA", "NC"], size=rows
    )
    frame["zip_code"] = [f"{rng.integers(100, 999)}xx" for _ in range(rows)]
    frame["emp_title"] = rng.choice(
        ["Manager", "Teacher", "Driver", "Nurse", "Engineer", "Owner", None],
        size=rows,
    )

    # -- realistic missingness ---------------------------------------------
    # Not random: emp_length is missing more often for lower-quality
    # applicants, which is why 03 leaves it as NaN and 05 imputes explicitly
    # rather than hiding the decision.
    missing_emp = rng.random(rows) < np.clip(0.04 - 0.02 * quality, 0.005, 0.15)
    frame.loc[missing_emp, "emp_length"] = None
    frame.loc[rng.random(rows) < 0.004, "revol_util"] = None
    frame.loc[rng.random(rows) < 0.002, "dti"] = None

    return frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=240_000)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    frame = generate(args.rows, args.seed)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT, index=False, compression="gzip")

    resolved = frame[frame["loan_status"].isin(["Charged Off", "Fully Paid"])]
    print(f"wrote {OUT}  ({len(frame):,} rows, {frame.shape[1]} columns)")
    print(f"\nloan_status:\n{frame['loan_status'].value_counts().to_string()}")
    print(f"\nresolved loans: {len(resolved):,}")
    print(f"observed default rate on resolved: "
          f"{(resolved['loan_status'] == 'Charged Off').mean():.3%}")
    print("\ndefault rate by origination year (note the 2013-14 shock):")
    years = pd.to_datetime(resolved["issue_d"], format="%b-%Y").dt.year
    print((resolved["loan_status"] == "Charged Off").groupby(years)
          .agg(["size", "mean"]).to_string(float_format=lambda v: f"{v:.3f}"))
    print(
        "\nREMINDER: this data is synthetic and the model that learns from it "
        "is circular. Do not put its metrics in the README."
    )


if __name__ == "__main__":
    main()
