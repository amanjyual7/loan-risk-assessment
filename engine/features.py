"""Map the questionnaire onto the model's training schema.

The questionnaire collects more than the model can consume. This module is
the boundary: anything that reaches a FeatureVector is a MODEL FEATURE.
Everything else is a POLICY SIGNAL and goes to engine.policy instead.

Deliberately absent, and it is a design decision rather than an oversight:

  pincode / city / state / region / tier
      Expense estimation and display only. Never a model feature and never
      one-hot encoded into training. Geographic risk pricing is the
      structural form of redlining, and pincode in India proxies caste and
      religion closely enough that the model would learn them.

  dependent children / dependent adults
      Expense estimation and the policy layer only. Lending Club carries no
      household-size field, so there is nothing to map them onto. (Had the
      Home Credit dataset been chosen, CNT_CHILDREN and CNT_FAM_MEMBERS
      would have made these legitimate model features — see the README for
      why Lending Club was chosen anyway.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from engine.credit import CreditBehaviour
from engine.types import Applicant


class FeatureValidationError(ValueError):
    """Raised when a feature is missing or out of range.

    Deliberately raises rather than silently imputing. A median-imputed
    bureau score is indistinguishable, downstream, from a real one.
    """


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    plain: str            # plain-language label for the result screen
    lo: float             # hard validity bound: outside this, raise
    hi: float
    center: float         # standardisation centre (training-set median)
    scale: float          # standardisation scale (training-set IQR/2)
    is_indicator: bool = False
    # Training support, roughly p1/p99 of the training distribution. Values
    # outside it are clamped and the clamp is surfaced as a caveat — the
    # model has never seen an applicant out there, and silently
    # extrapolating a GBM past the edge of its support produces confident
    # nonsense. Indian education-loan amounts sit well above Lending Club's
    # loan range even after the income bridge, so this fires in practice.
    sup_lo: float | None = None
    sup_hi: float | None = None

    @property
    def support(self) -> tuple[float, float]:
        return (
            self.lo if self.sup_lo is None else self.sup_lo,
            self.hi if self.sup_hi is None else self.sup_hi,
        )


# Training-set centres, scales and support bounds are stamped into the model
# artefact after training and override whatever is written here; these are
# the placeholder values, and keeping them here documents the schema.
FEATURE_SCHEMA: tuple[FeatureSpec, ...] = (
    FeatureSpec("fico_avg", "Bureau score", 300, 850, 692, 30, False, 660, 830),
    FeatureSpec("log_annual_inc", "Annual income", 8.0, 15.0, 11.07, 0.45, False, 10.09, 12.61),
    FeatureSpec("dti", "Debt-to-income at application", 0, 200, 17.8, 8.5, False, 0, 40),
    FeatureSpec("emp_length_years", "Years in current job", 0, 40, 5.0, 3.5, False, 0, 10),
    FeatureSpec("home_rent", "Lives in rented accommodation", 0, 1, 0, 1, True),
    FeatureSpec("home_other", "Residence type other than owned or rented", 0, 1, 0, 1, True),
    FeatureSpec("revol_util", "Revolving credit utilisation", 0, 200, 52.0, 24.0, False, 0, 99),
    FeatureSpec("open_acc", "Number of open accounts", 0, 90, 11, 5, False, 2, 30),
    FeatureSpec("delinq_2yrs", "Delinquencies in last 2 years", 0, 40, 0, 1, False, 0, 4),
    FeatureSpec("inq_last_6mths", "Credit enquiries in last 6 months", 0, 40, 0, 1, False, 0, 5),
    FeatureSpec("pub_rec", "Public records", 0, 40, 0, 1, False, 0, 3),
    FeatureSpec("log_loan_amnt", "Loan amount requested", 5.0, 15.0, 9.55, 0.62, False, 6.9, 10.6),
    FeatureSpec("term_60", "Tenure longer than 36 months", 0, 1, 0, 1, True),
    FeatureSpec("purpose_high_risk", "Loan purpose in a higher-risk category", 0, 1, 0, 1, True),
    FeatureSpec("purpose_low_risk", "Loan purpose in a lower-risk category", 0, 1, 0, 1, True),
    FeatureSpec("loan_to_income", "Loan amount relative to annual income", 0, 20, 0.22, 0.14, False, 0.01, 0.75),
)

FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in FEATURE_SCHEMA)
SPEC_BY_NAME: dict[str, FeatureSpec] = {f.name: f for f in FEATURE_SCHEMA}

# Lending Club purpose buckets, collapsed. Grouping rather than one-hotting
# all 14 keeps the baseline logistic model readable.
HIGH_RISK_PURPOSES = {
    "small business", "business", "moving", "debt consolidation",
    "renewable energy", "other",
}
LOW_RISK_PURPOSES = {
    "credit card refinance", "car", "auto", "home improvement", "major purchase",
}


def bridge_bureau_score(score: Optional[int], cfg: dict) -> Optional[float]:
    """CIBIL 300-900 -> FICO 300-850.

    A linear rescale. Crude, and the two distributions are not the same
    shape, but it is monotone, which is what the model needs. The result
    screen carries this caveat.
    """
    if score is None:
        return None
    s_lo, s_hi = cfg["bureau_bridge"]["source_range"]
    t_lo, t_hi = cfg["bureau_bridge"]["target_range"]
    if not (s_lo <= score <= s_hi):
        raise FeatureValidationError(
            f"bureau score {score} outside expected range {s_lo}-{s_hi}"
        )
    frac = (score - s_lo) / (s_hi - s_lo)
    return t_lo + frac * (t_hi - t_lo)


def bridge_amount(inr: float, cfg: dict) -> float:
    """INR -> training-currency units, anchored on income, not FX.

    See config/config.yaml currency_bridge for why FX is the wrong
    conversion here.
    """
    divisor = float(cfg["currency_bridge"]["inr_per_training_unit"])
    return inr / divisor


def build_features(
    applicant: Applicant,
    credit: CreditBehaviour,
    dti_pct: float,
    cfg: dict,
) -> dict[str, float]:
    """Produce the model feature vector. Raises on anything unusable."""
    a = applicant

    fico = bridge_bureau_score(a.bureau.score, cfg)
    if fico is None:
        raise FeatureValidationError(
            "no bureau score: this applicant must take the thin-file path, "
            "not the model"
        )

    annual_inc = bridge_amount(a.income.gross_monthly * 12.0, cfg)
    if annual_inc <= 0:
        raise FeatureValidationError("gross monthly income must be positive")
    loan_amnt = bridge_amount(a.loan.amount, cfg)

    purpose = a.loan.purpose.strip().lower()
    residence = a.identity.residence_type

    features = {
        "fico_avg": fico,
        "log_annual_inc": math.log(annual_inc),
        "dti": dti_pct,
        "emp_length_years": min(a.employment.years_current_job, 40.0),
        "home_rent": 1.0 if residence == "rented" else 0.0,
        "home_other": 1.0 if residence == "parental" else 0.0,
        "revol_util": (
            credit.revolving_utilisation_pct
            if credit.revolving_utilisation_pct is not None
            else 0.0
        ),
        "open_acc": float(max(credit.live_accounts, a.bureau.active_accounts)),
        "delinq_2yrs": float(credit.delinq_accounts_24m),
        "inq_last_6mths": float(a.bureau.enquiries_6m),
        # Not collected by the questionnaire — suits filed and write-offs are
        # collected instead and routed to the policy layer, because they are
        # not the same thing as a US public record.
        "pub_rec": 0.0,
        "log_loan_amnt": math.log(max(loan_amnt, 1.0)),
        "term_60": 1.0 if a.loan.tenure_months > 36 else 0.0,
        "purpose_high_risk": 1.0 if purpose in HIGH_RISK_PURPOSES else 0.0,
        "purpose_low_risk": 1.0 if purpose in LOW_RISK_PURPOSES else 0.0,
        "loan_to_income": loan_amnt / annual_inc,
    }
    validate_features(features)
    return features


def clamp_to_training_support(
    features: dict[str, float]
) -> tuple[dict[str, float], list[str]]:
    """Clamp features to the training support, reporting what was clamped.

    Returns (clamped_features, notes). The notes go on the result screen as
    caveats: a clamped feature means the model is being asked about an
    applicant outside the range it learned from, and the reader should know.
    """
    out: dict[str, float] = {}
    notes: list[str] = []
    for name, value in features.items():
        spec = SPEC_BY_NAME[name]
        lo, hi = spec.support
        if value < lo:
            out[name] = lo
            notes.append(
                f"{spec.plain} sits below the training range "
                f"({value:.2f} < {lo:.2f}); clamped to the edge."
            )
        elif value > hi:
            out[name] = hi
            notes.append(
                f"{spec.plain} sits above the training range "
                f"({value:.2f} > {hi:.2f}); clamped to the edge."
            )
        else:
            out[name] = value
    return out, notes


def validate_features(features: dict[str, float]) -> None:
    missing = set(FEATURE_NAMES) - set(features)
    if missing:
        raise FeatureValidationError(f"missing features: {sorted(missing)}")
    extra = set(features) - set(FEATURE_NAMES)
    if extra:
        raise FeatureValidationError(f"unexpected features: {sorted(extra)}")
    for name, value in features.items():
        spec = SPEC_BY_NAME[name]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise FeatureValidationError(f"{name} is missing")
        if not (spec.lo <= value <= spec.hi):
            raise FeatureValidationError(
                f"{name}={value!r} outside schema range [{spec.lo}, {spec.hi}]"
            )
