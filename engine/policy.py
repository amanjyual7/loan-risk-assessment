"""Policy layer: deterministic rules over the signals the public training
data does not carry.

These are ASSERTED, not learned. Each flag names the value that fired it so
a reviewer can disagree with the threshold rather than with the output.

Knockouts are kept separate from graded flags throughout. An active
written-off account is a different kind of fact from a slightly high FOIR,
and averaging the two into one score is the mistake this split exists to
avoid.
"""

from __future__ import annotations

from typing import Optional, Sequence

from engine.affordability import Affordability
from engine.credit import CreditBehaviour
from engine.types import Applicant, Location, PolicyFlag


def evaluate(
    applicant: Applicant,
    afford: Affordability,
    credit: CreditBehaviour,
    location: Location,
    cfg: dict,
) -> tuple[list[PolicyFlag], list[PolicyFlag]]:
    """Returns (graded_flags, knockouts)."""
    p = cfg["policy"]
    flags: list[PolicyFlag] = []
    knockouts: list[PolicyFlag] = []

    def flag(code, message, severity, triggered_by):
        flags.append(PolicyFlag(code, message, severity, str(triggered_by)))

    def knockout(code, message, triggered_by):
        knockouts.append(
            PolicyFlag(code, message, "knockout", str(triggered_by), is_knockout=True)
        )

    # -- hard knockouts -----------------------------------------------------
    ko = p["knockouts"]
    bad_statuses = set(ko["account_status_in"])
    for t in applicant.trade_lines:
        if t.status in bad_statuses and t.is_live:
            knockout(
                "active_impaired_account",
                f"Live {t.account_type} account carries status '{t.status}'",
                f"{t.account_type}: {t.status}, outstanding "
                f"{t.current_outstanding:,.0f}",
            )
        if t.is_live and t.current_dpd >= int(ko["current_dpd_at_or_above"]):
            knockout(
                "severe_current_dpd",
                f"Live {t.account_type} account is {t.current_dpd} days past due",
                f"current DPD = {t.current_dpd}",
            )

    if ko.get("suit_filed_any") and applicant.adverse.suits_filed > 0:
        knockout("suit_filed", "Suit filed against the applicant in the last 12 months",
                 f"{applicant.adverse.suits_filed} suit(s)")
    if ko.get("active_written_off_account") and applicant.adverse.write_offs > 0:
        knockout("write_off_12m", "Write-off recorded in the last 12 months",
                 f"{applicant.adverse.write_offs} write-off(s)")
    if ko.get("settlement_last_12m") and applicant.adverse.settlements > 0:
        knockout("settlement_12m", "Account settled in the last 12 months",
                 f"{applicant.adverse.settlements} settlement(s)")

    # -- affordability ------------------------------------------------------
    aff = p["affordability"]
    if afford.foir is not None:
        if afford.foir >= float(aff["foir_severe"]):
            flag("foir_severe", "FOIR is far above the policy ceiling", "high",
                 f"FOIR = {afford.foir:.0%} vs ceiling {float(aff['foir_max']):.0%}")
        elif afford.foir > float(aff["foir_max"]):
            flag("foir_high", "FOIR above the policy ceiling", "medium",
                 f"FOIR = {afford.foir:.0%} vs ceiling {float(aff['foir_max']):.0%}")

    if (
        afford.expense_adjusted_foir is not None
        and afford.expense_adjusted_foir > float(aff["expense_adjusted_foir_max"])
    ):
        flag("eaf_high",
             "Obligations plus the household expense floor exceed the ceiling",
             "medium",
             f"expense-adjusted FOIR = {afford.expense_adjusted_foir:.0%}")

    # The reason the dependent fields exist. Plain FOIR can pass while the
    # household is short every month.
    if afford.surplus < float(aff["min_surplus_inr"]):
        flag("negative_surplus",
             "Negative surplus: household income does not cover the expense "
             "floor plus all EMIs",
             "high",
             f"surplus = {afford.surplus:,.0f}/month with "
             f"{applicant.household.dependent_children} child(ren) and "
             f"{applicant.household.dependent_adults} dependent adult(s) at "
             f"{location.tier}")
    elif afford.surplus < float(aff["min_residual_income_inr"]):
        flag("thin_surplus", "Surplus is positive but thin", "low",
             f"surplus = {afford.surplus:,.0f}/month")

    if afford.residual_income < float(aff["min_residual_income_inr"]):
        flag("low_residual", "Residual income after all EMIs is below the floor",
             "medium", f"residual = {afford.residual_income:,.0f}/month")

    # -- credit behaviour ---------------------------------------------------
    cb = p["credit_behaviour"]
    if credit.current_dpd_max >= int(cb["current_dpd_severe"]):
        flag("current_dpd_severe", "Severe current delinquency", "high",
             f"current DPD = {credit.current_dpd_max}")
    elif credit.current_dpd_max >= int(cb["current_dpd_flag"]):
        flag("current_dpd", "Currently delinquent", "medium",
             f"current DPD = {credit.current_dpd_max}")
    elif credit.worst_dpd_24m >= int(cb["worst_dpd_24m_flag"]):
        # Recovered, but recent. Graded lower than a live delinquency.
        flag("recovered_delinquency",
             "Delinquent within the last 24 months, now current", "low",
             f"worst DPD 24m = {credit.worst_dpd_24m}")

    if (
        credit.revolving_utilisation_pct is not None
        and credit.revolving_utilisation_pct > float(cb["revolving_utilisation_flag_pct"])
    ):
        flag("high_utilisation", "Revolving utilisation is very high", "medium",
             f"utilisation = {credit.revolving_utilisation_pct:.0f}%")

    if (
        credit.credit_vintage_months is not None
        and credit.credit_vintage_months < int(cb["min_credit_vintage_months"])
    ):
        flag("short_vintage", "Very short credit history", "low",
             f"vintage = {credit.credit_vintage_months} months")

    if applicant.bureau.enquiries_3m >= int(cb["enquiries_3m_flag"]):
        flag("enquiry_burst", "Burst of credit enquiries in the last 3 months",
             "medium", f"{applicant.bureau.enquiries_3m} enquiries in 3 months")
    elif applicant.bureau.enquiries_12m >= int(cb["enquiries_12m_flag"]):
        flag("enquiry_velocity", "Elevated enquiry activity over 12 months", "low",
             f"{applicant.bureau.enquiries_12m} enquiries in 12 months")

    # -- adverse history ----------------------------------------------------
    ad = p["adverse_12m"]
    nach = applicant.adverse.nach_bounces
    if nach >= int(ad["nach_bounce_severe_count"]):
        flag("nach_bounce_severe", "Repeated NACH mandate failures", "high",
             f"{nach} bounces in 12 months")
    elif nach >= int(ad["nach_bounce_flag_count"]):
        flag("nach_bounce", "NACH mandate failures", "medium",
             f"{nach} bounces in 12 months")
    if applicant.adverse.cheque_bounces >= int(ad["cheque_bounce_flag_count"]):
        flag("cheque_bounce", "Cheque returns in the last 12 months", "medium",
             f"{applicant.adverse.cheque_bounces} returns")

    # -- cash flow (optional section) --------------------------------------
    cf = p["cash_flow"]
    c = applicant.cash_flow
    if c.salary_credits_6m is not None and c.salary_credits_6m < int(cf["min_salary_credits_6m"]):
        flag("irregular_credits", "Fewer regular income credits than expected",
             "medium", f"{c.salary_credits_6m} credits in 6 months")
    if c.bounced_debits_6m is not None and c.bounced_debits_6m >= int(cf["bounced_debits_6m_flag"]):
        flag("bounced_debits", "Debits bouncing against the account", "medium",
             f"{c.bounced_debits_6m} in 6 months")

    if applicant.income.income_varies and applicant.employment.employment_type in (
        "self-employed", "business",
    ):
        flag("variable_income", "Income varies month to month", "low",
             f"{applicant.employment.employment_type}, self-declared variable")

    # -- co-applicant -------------------------------------------------------
    if applicant.co_applicant is None:
        if applicant.loan.purpose.strip().lower() in ("education", "educational"):
            flag("no_co_applicant",
                 "No co-applicant on an education loan, where the co-applicant "
                 "normally carries the repayment capacity", "medium",
                 "co-applicant not provided")
    else:
        ca = applicant.co_applicant
        if ca.net_monthly_income <= 0:
            flag("co_applicant_no_income",
                 "Co-applicant provided but contributes no income", "low",
                 f"{ca.relationship}, income 0")

    return flags, knockouts


def flags_summary(flags: Sequence[PolicyFlag]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in flags:
        out[f.severity] = out.get(f.severity, 0) + 1
    return out
