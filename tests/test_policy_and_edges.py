"""Expense model, policy layer and edge cases."""

from __future__ import annotations

import dataclasses
from datetime import date

from tests.context import band_rank, cfg, profiles, score

from engine.affordability import proposed_emi
from engine.expenses import estimate_expenses
from engine.types import (
    AdverseHistory,
    BureauSummary,
    CashFlow,
    Employment,
    Household,
    Identity,
    Income,
    LoanRequest,
    TradeLine,
)


def _minimal(**overrides):
    """A mid-income applicant with nothing remarkable about it.

    Used where a test needs headroom to move one variable at a time. The
    demo profiles are deliberately extreme, which makes them poor probes:
    the prime borrower earns enough that no plausible number of dependents
    drives the surplus negative.
    """
    payload = {
        "identity": Identity("Edge Case", 30, 560001, "rented", 1.0),
        "household": Household(0, 0),
        "employment": Employment("salaried", "X", "Y", 2.0, 5.0),
        "income": Income(50000, 42000),
        "loan": LoanRequest(300000, "education", 60),
        "bureau": BureauSummary(score=700, active_accounts=1, enquiries_3m=0,
                                enquiries_6m=0, enquiries_12m=0),
        "trade_lines": (),
        "adverse": AdverseHistory(),
        "cash_flow": CashFlow(),
        "co_applicant": None,
    }
    payload.update(overrides)
    from engine.types import Applicant

    return Applicant(**payload)


# ---------------------------------------------------------------- expenses
def test_surplus_goes_negative_as_dependents_rise_and_the_flag_fires():
    base = _minimal(
        trade_lines=(
            TradeLine("personal", "bank", 200000, 90000, 3500,
                      date(2023, 1, 1), None, "individual", 0, 0, "standard"),
        )
    )
    seen_negative = False
    previous = None

    for children in range(0, 9):
        variant = dataclasses.replace(
            base, household=Household(dependent_children=children,
                                      dependent_adults=0)
        )
        r = score(variant, with_suggestions=False)
        surplus = next(m for m in r.metrics if m.key == "surplus").value

        if previous is not None:
            assert surplus < previous, (
                f"surplus did not fall when children went to {children}"
            )
        previous = surplus

        if surplus < 0:
            seen_negative = True
            codes = {f.code for f in r.policy_flags}
            assert "negative_surplus" in codes, (
                f"surplus is {surplus:,.0f} but no negative_surplus flag fired"
            )

    assert seen_negative, "surplus never went negative across 0-8 children"


def test_expense_floor_respects_tier_ordering():
    c = cfg()
    hh = Household(dependent_children=1, dependent_adults=1)
    totals = {
        tier: estimate_expenses(hh, tier, "rented", c).total
        for tier in c["pincode"]["tier_order"]
    }
    ordered = list(totals.values())
    assert ordered == sorted(ordered, reverse=True), (
        f"a Metro household should not cost less than Tier 3: {totals}"
    )


def test_children_and_dependent_adults_are_costed_separately():
    c = cfg()
    one_child = estimate_expenses(Household(1, 0), "Metro", "rented", c).total
    one_adult = estimate_expenses(Household(0, 1), "Metro", "rented", c).total
    assert one_child != one_adult, (
        "a child and an elderly parent must not carry the same cost"
    )


def test_owned_residence_lowers_the_expense_floor():
    c = cfg()
    hh = Household(1, 0)
    assert (
        estimate_expenses(hh, "Metro", "owned", c).total
        < estimate_expenses(hh, "Metro", "rented", c).total
    )


# ------------------------------------------------------------------ policy
def test_policy_can_only_raise_severity_never_lower_it():
    for key, applicant in profiles().items():
        r = score(applicant, with_suggestions=False)
        assert band_rank(r.band) >= band_rank(r.model_band), (
            f"{key}: policy improved the band from {r.model_band} to {r.band}"
        )


def test_knockouts_force_the_worst_band_whatever_the_model_says():
    base = profiles()["clean_prime"]  # band A on the model
    impaired = TradeLine(
        account_type="personal", lender_type="NBFC", sanctioned_amount=200000,
        current_outstanding=180000, emi=0, opened_date=date(2021, 1, 1),
        closed_date=None, ownership="individual", current_dpd=120,
        worst_dpd_24m=180, status="written-off",
    )
    variant = dataclasses.replace(
        base, trade_lines=base.trade_lines + (impaired,)
    )
    r = score(variant, with_suggestions=False)
    assert r.knockouts
    assert r.band == "E"
    assert r.band_raised_by_policy


def test_suggestions_never_promise_an_improvement_the_engine_would_not_make():
    for key, applicant in profiles().items():
        r = score(applicant)
        for s in r.suggestions:
            assert band_rank(s.resulting_band) < band_rank(r.band), (
                f"{key}: suggestion claims band {s.resulting_band} which is "
                f"not better than {r.band}"
            )


# --------------------------------------------------------------- edge cases
def test_zero_income_does_not_crash_and_does_not_score_well():
    applicant = _minimal(income=Income(0, 0))
    r = score(applicant, with_suggestions=False)
    # FOIR is undefined rather than infinite
    foir = next(m for m in r.metrics if m.key == "foir").value
    assert foir is None
    assert r.policy_flags, "zero income must raise something"
    assert band_rank(r.band) >= band_rank("C")


def test_no_trade_lines_with_a_bureau_score_is_not_treated_as_thin_file_silently():
    r = score(_minimal(), with_suggestions=False)
    # bureau.no_credit_history is False but the grid is empty, so the engine
    # takes the thin-file path and says so rather than scoring an empty file
    assert r.thin_file
    assert any("not used" in c for c in r.caveats)


def test_single_trade_line_at_90_dpd():
    line = TradeLine(
        "credit card", "bank", 100000, 95000, 0, date(2020, 1, 1), None,
        "individual", 95, 95, "standard",
    )
    r = score(_minimal(trade_lines=(line,)), with_suggestions=False)
    assert r.knockouts
    assert r.band == "E"


def test_foir_over_100_percent():
    line = TradeLine(
        "personal", "NBFC", 900000, 700000, 60000, date(2022, 1, 1), None,
        "individual", 0, 0, "standard",
    )
    r = score(_minimal(trade_lines=(line,)), with_suggestions=False)
    foir = next(m for m in r.metrics if m.key == "foir").value
    assert foir > 1.0
    codes = {f.code for f in r.policy_flags}
    assert "foir_severe" in codes
    # Relative, not absolute: an applicant spending more than their whole
    # income on EMIs must be downgraded from wherever the model put them.
    assert band_rank(r.band) > band_rank(r.model_band)


def test_applicant_and_co_applicant_both_thin_file():
    from engine.types import CoApplicant

    applicant = _minimal(
        bureau=BureauSummary(no_credit_history=True),
        co_applicant=CoApplicant("Parent", "mother", "salaried", 0.0, 55),
        cash_flow=CashFlow(average_monthly_balance=8000, salary_credits_6m=2,
                           bounced_debits_6m=0),
    )
    r = score(applicant, with_suggestions=False)
    assert r.thin_file
    assert r.pd is None
    view = r.thin_file_view
    assert view["checks_passed"] < view["checks_total"]
    codes = {f.code for f in r.policy_flags}
    assert "co_applicant_no_income" in codes
    assert band_rank(r.band) >= band_rank("C")


def test_emi_formula_matches_a_known_value():
    # 10,00,000 at 12.5% over 120 months
    emi = proposed_emi(1_000_000, 12.5, 120)
    assert abs(emi - 14_640) < 40, emi
    # zero-rate edge case degrades to simple division
    assert abs(proposed_emi(120_000, 0.0, 12) - 10_000) < 1e-6


def test_assessment_is_stamped_with_config_and_model_version():
    r = score(profiles()["clean_prime"], with_suggestions=False)
    assert r.config_version == cfg()["config_version"]
    assert r.model_version
