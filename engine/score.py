"""The centrepiece: a pure scoring function.

    assess(applicant, model=..., reference=..., cfg=...) -> Assessment

No file reads, no database writes, no Streamlit imports, no clock reads
unless you pass `as_of`. The model and the pincode reference are loaded by
the caller and injected, which is what makes the whole thing testable.
"""

from __future__ import annotations

import dataclasses
from datetime import date, datetime, timezone
from typing import Optional

from engine.affordability import (
    Affordability,
    affordability_metrics,
    compute_affordability,
)
from engine.bands import Band, band_for_pd, band_penalty, load_bands, worsen
from engine.credit import compute_credit_behaviour, credit_metrics
from engine.expenses import estimate_expenses, expense_metrics
from engine.features import (
    FeatureValidationError,
    build_features,
    clamp_to_training_support,
)
from engine.model import PDModel, plain_language
from engine.pincode import PincodeReference
from engine.policy import evaluate
from engine.types import (
    Applicant,
    Assessment,
    Contribution,
    Location,
    Metric,
    PolicyFlag,
    Suggestion,
)

# Suggestion search space. Each entry mutates the applicant and is only
# reported if it actually improves the band.
_CANDIDATES = [
    ("Loan amount", 0.90, "reduce the requested amount by 10%"),
    ("Loan amount", 0.80, "reduce the requested amount by 20%"),
    ("Loan amount", 0.70, "reduce the requested amount by 30%"),
]


def assess(
    applicant: Applicant,
    *,
    model: PDModel,
    reference: PincodeReference,
    cfg: dict,
    as_of: Optional[date] = None,
    with_suggestions: bool = True,
) -> Assessment:
    as_of = as_of or date.today()

    location = reference.lookup(
        applicant.identity.pincode,
        default_tier=cfg["pincode"]["default_tier_on_miss"],
        miss_label=cfg["pincode"]["miss_label"],
        manual_state=applicant.manual_state,
    )

    expense = estimate_expenses(
        applicant.household, location.tier, applicant.identity.residence_type, cfg
    )
    co_income = (
        applicant.co_applicant.net_monthly_income if applicant.co_applicant else 0.0
    )
    afford = compute_affordability(
        applicant.income, applicant.loan, applicant.trade_lines, expense, cfg, co_income
    )
    credit = compute_credit_behaviour(applicant.trade_lines, applicant.bureau, as_of)

    metrics: list[Metric] = [
        *affordability_metrics(afford),
        *expense_metrics(expense),
        *credit_metrics(credit),
    ]

    policy_flags, knockouts = evaluate(applicant, afford, credit, location, cfg)

    caveats: list[str] = []
    if location.assumed:
        caveats.append(
            f"Pincode {location.pincode} is not in the reference data. Tier "
            f"'{location.tier}' is {cfg['pincode']['miss_label']}, so the expense "
            "floor and everything derived from it are approximate."
        )
    if model.is_placeholder:
        caveats.append(
            "The loaded model artefact is the PLACEHOLDER scorecard, not a "
            "trained model. Probabilities are illustrative only."
        )
    if applicant.bureau.score is not None:
        caveats.append(
            "The bureau score was linearly rescaled from the Indian 300–900 "
            "range onto the training data's 300–850 range. The two "
            "distributions are not the same shape."
        )

    # ---------------- thin file: do not pretend to have a PD --------------
    if applicant.thin_file and cfg["thin_file"]["enabled"]:
        return _thin_file_assessment(
            applicant, afford, credit, location, metrics, policy_flags,
            knockouts, caveats, model, cfg,
        )

    # ---------------- model path ------------------------------------------
    dti_pct = (afford.existing_emis / applicant.income.gross_monthly * 100.0) \
        if applicant.income.gross_monthly > 0 else 0.0
    features = build_features(applicant, credit, dti_pct, cfg)
    features, clamp_notes = clamp_to_training_support(features)
    caveats.extend(clamp_notes)
    pred = model.predict(features)

    model_band = band_for_pd(pred.pd, cfg)
    penalty = band_penalty(policy_flags, cfg)
    if knockouts:
        penalty = max(penalty, int(cfg["policy"]["band_penalty"]["knockout"]))
    final_band = worsen(model_band, penalty, cfg)

    contributions = _contributions(pred.contributions, features)

    suggestions: tuple[Suggestion, ...] = ()
    if with_suggestions:
        suggestions = _suggestions(
            applicant, final_band, model=model, reference=reference, cfg=cfg, as_of=as_of
        )

    if pred.contribution_method.startswith("unavailable"):
        caveats.append(
            "Per-applicant attribution is unavailable for this model kind; "
            "install shap to enable it."
        )

    return Assessment(
        band=final_band.code,
        band_label=final_band.label,
        band_observed_default_rate=final_band.observed_default_rate,
        pd=pred.pd,
        pd_is_meaningful=True,
        model_band=model_band.code,
        band_raised_by_policy=final_band.code != model_band.code,
        thin_file=False,
        thin_file_view=None,
        location=location,
        metrics=tuple(metrics),
        contributions=contributions,
        policy_flags=tuple(policy_flags),
        knockouts=tuple(knockouts),
        suggestions=suggestions,
        model_features=features,
        caveats=tuple(caveats),
        config_version=cfg["config_version"],
        model_version=model.version,
        assessed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# --------------------------------------------------------------------------
def _contributions(
    raw: dict[str, float], features: dict[str, float]
) -> tuple[Contribution, ...]:
    items = [
        Contribution(
            feature=name,
            plain_language=plain_language(name, features[name]),
            value=features[name],
            log_odds_delta=delta,
        )
        for name, delta in raw.items()
    ]
    # Largest absolute effect first, so the result screen leads with what
    # actually moved the estimate.
    items.sort(key=lambda c: abs(c.log_odds_delta), reverse=True)
    return tuple(items)


def _thin_file_assessment(
    applicant, afford, credit, location, metrics, policy_flags, knockouts,
    caveats, model, cfg,
) -> Assessment:
    """No bureau history: fall back to income and cash-flow signals and say
    so, rather than scoring zero or imputing a median score.

    A thin file is never assigned the best band. Absence of adverse history
    is not evidence of good history.
    """
    tf = cfg["thin_file"]
    bands = load_bands(cfg)
    checks = {
        "FOIR within the thin-file ceiling": (
            afford.foir is not None and afford.foir <= float(tf["max_foir"])
        ),
        "Surplus covers the proposed EMI with margin": (
            afford.surplus >= float(tf["min_surplus_multiple_of_emi"]) * afford.proposed_emi
        ),
        "Regular income credits observed": (
            applicant.cash_flow.salary_credits_6m is not None
            and applicant.cash_flow.salary_credits_6m >= int(tf["min_salary_credits_6m"])
        ),
        "Co-applicant with income": (
            applicant.co_applicant is not None
            and applicant.co_applicant.net_monthly_income > 0
        ),
    }
    passed = sum(checks.values())

    # Best available outcome for a thin file is the middle band. Cap, then
    # worsen from there by the same policy penalty as everyone else.
    floor_index = next(i for i, b in enumerate(bands) if b.code == bands[len(bands) // 2].code)
    start = bands[floor_index] if passed >= 3 else bands[min(floor_index + 1, len(bands) - 1)]
    penalty = band_penalty(policy_flags, cfg)
    if knockouts:
        penalty = max(penalty, int(cfg["policy"]["band_penalty"]["knockout"]))
    final = worsen(start, penalty, cfg)

    caveats = list(caveats) + [
        "No bureau history, so the model was not used. This band comes from "
        "income, affordability and cash-flow signals only, and is capped at "
        f"band {start.code} — an absent credit file is not a clean one.",
    ]

    return Assessment(
        band=final.code,
        band_label=final.label,
        band_observed_default_rate=final.observed_default_rate,
        pd=None,
        pd_is_meaningful=False,
        model_band=start.code,
        band_raised_by_policy=final.code != start.code,
        thin_file=True,
        thin_file_view={
            "checks": checks,
            "checks_passed": passed,
            "checks_total": len(checks),
            "net_monthly_income": afford.net_income,
            "household_income": afford.household_income,
            "proposed_emi": afford.proposed_emi,
            "surplus": afford.surplus,
            "average_monthly_balance": applicant.cash_flow.average_monthly_balance,
            "salary_credits_6m": applicant.cash_flow.salary_credits_6m,
            "bounced_debits_6m": applicant.cash_flow.bounced_debits_6m,
        },
        location=location,
        metrics=tuple(metrics),
        contributions=(),
        policy_flags=tuple(policy_flags),
        knockouts=tuple(knockouts),
        suggestions=(),
        model_features={},
        caveats=tuple(caveats),
        config_version=cfg["config_version"],
        model_version=model.version,
        assessed_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def _suggestions(
    applicant: Applicant,
    current: Band,
    *,
    model: PDModel,
    reference: PincodeReference,
    cfg: dict,
    as_of: date,
) -> tuple[Suggestion, ...]:
    """'What would change this' — the two or three input changes that move
    the applicant to a better band.

    Every candidate is re-scored through the same pure function, so a
    suggestion can never claim an improvement the engine would not actually
    produce. Knockouts are not counterfactualled: a written-off account is
    not something to tune away.
    """
    bands = [b.code for b in load_bands(cfg)]
    if current.code == bands[0]:
        return ()

    def better(code: str) -> bool:
        return bands.index(code) < bands.index(current.code)

    out: list[Suggestion] = []

    def try_variant(label: str, change: str, mutated: Applicant) -> None:
        if len(out) >= 3:
            return
        try:
            res = assess(
                mutated, model=model, reference=reference, cfg=cfg,
                as_of=as_of, with_suggestions=False,
            )
        except (FeatureValidationError, ValueError, KeyError):
            return
        if better(res.band) and res.pd is not None:
            out.append(Suggestion(label, change, res.band, res.pd))

    # 1. Smaller loan
    for label, factor, text in _CANDIDATES:
        if len(out) >= 3:
            break
        loan = dataclasses.replace(applicant.loan, amount=applicant.loan.amount * factor)
        try_variant(
            label,
            f"{text} (to ₹{loan.amount:,.0f})",
            dataclasses.replace(applicant, loan=loan),
        )

    # 2. Lower revolving utilisation to 30% on live revolving lines
    revolving = [t for t in applicant.trade_lines if t.is_live and t.is_revolving]
    if revolving and len(out) < 3:
        new_lines = tuple(
            dataclasses.replace(t, current_outstanding=min(t.current_outstanding,
                                                           0.30 * t.sanctioned_amount))
            if (t.is_live and t.is_revolving) else t
            for t in applicant.trade_lines
        )
        try_variant(
            "Revolving utilisation",
            "bring card and overdraft balances down to 30% of the limit",
            dataclasses.replace(applicant, trade_lines=new_lines),
        )

    # 3. Clear the smallest existing obligation
    live = [t for t in applicant.trade_lines if t.is_live and t.emi > 0]
    if live and len(out) < 3:
        smallest = min(live, key=lambda t: t.emi)
        new_lines = tuple(t for t in applicant.trade_lines if t is not smallest)
        try_variant(
            "Existing obligations",
            f"close the {smallest.account_type} account carrying an EMI of "
            f"₹{smallest.emi:,.0f}",
            dataclasses.replace(applicant, trade_lines=new_lines),
        )

    # 4. Longer tenure (lower EMI, more interest — stated as a trade-off)
    if len(out) < 3:
        longer = min(applicant.loan.tenure_months + 24,
                     int(cfg["pricing"]["max_tenure_months"]))
        if longer > applicant.loan.tenure_months:
            loan = dataclasses.replace(applicant.loan, tenure_months=longer)
            try_variant(
                "Tenure",
                f"extend the tenure to {longer} months, which lowers the EMI "
                "but raises total interest paid",
                dataclasses.replace(applicant, loan=loan),
            )

    return tuple(out)
