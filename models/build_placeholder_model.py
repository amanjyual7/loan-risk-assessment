#!/usr/bin/env python3
"""Build the PLACEHOLDER scorecard artefact.

This exists so the repo is runnable — app, demo profiles and tests — before
the Lending Club model is trained. It is NOT a model:

  * coefficients are hand-set from domain reasoning, not fitted
  * the calibrator is an identity-ish shrink, not fitted on held-out data
  * metrics are empty, and the UI shows a warning banner

It implements the same artefact contract as the trained model, so
notebooks/05_train.py drops in without touching engine/ or app/.

    python models/build_placeholder_model.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.features import FEATURE_NAMES, SPEC_BY_NAME  # noqa: E402

OUT = REPO_ROOT / "models" / "pd_model.joblib"

# Log-odds per standardised unit. Signs are the point: every one is set from
# credit reasoning, and the monotonicity test in tests/ checks that raising
# income or lowering DTI never raises PD.
COEFFICIENTS = {
    "fico_avg":          -0.62,   # higher bureau score, lower risk
    "log_annual_inc":    -0.18,
    "dti":               +0.30,
    "emp_length_years":  -0.11,
    "home_rent":         +0.09,
    "home_other":        +0.14,
    "revol_util":        +0.21,
    "open_acc":          +0.05,
    "delinq_2yrs":       +0.27,
    "inq_last_6mths":    +0.14,
    "pub_rec":           +0.18,
    "log_loan_amnt":     +0.16,
    "term_60":           +0.42,
    "purpose_high_risk": +0.25,
    "purpose_low_risk":  -0.12,
    "loan_to_income":    +0.12,
}

BASE_RATE = 0.148  # placeholder; the trained artefact carries the observed rate


def build() -> dict:
    coef = np.array([[COEFFICIENTS[n] for n in FEATURE_NAMES]], dtype=float)
    intercept = np.array([np.log(BASE_RATE / (1 - BASE_RATE))])

    est = LogisticRegression()
    # Fit on two trivial rows purely to populate sklearn's internal
    # attributes, then overwrite the parameters with the hand-set ones.
    est.fit(np.zeros((2, len(FEATURE_NAMES))) + [[0.0], [1.0]], [0, 1])
    est.coef_ = coef
    est.intercept_ = intercept
    est.classes_ = np.array([0, 1])

    # Monotone increasing, so it cannot break the monotonicity guarantee.
    grid = np.linspace(0.001, 0.999, 400)
    cal = IsotonicRegression(out_of_bounds="clip", increasing=True)
    cal.fit(grid, 0.88 * grid + 0.015)

    return {
        "model_version": "placeholder-0.1.0",
        "trained_at": date.today().isoformat(),
        "dataset": "NONE — hand-specified placeholder scorecard",
        "kind": "logistic",
        "feature_names": list(FEATURE_NAMES),
        "standardisation": {
            n: [SPEC_BY_NAME[n].center, SPEC_BY_NAME[n].scale] for n in FEATURE_NAMES
        },
        "support": {n: list(SPEC_BY_NAME[n].support) for n in FEATURE_NAMES},
        "estimator": est,
        "calibrator": cal,
        "metrics": {"note": "no metrics: this artefact was not trained"},
        "is_placeholder": True,
    }


if __name__ == "__main__":
    artefact = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artefact, OUT)
    print(f"wrote placeholder artefact -> {OUT}")
    print("Replace it by running notebooks/05_train.py against Lending Club data.")
