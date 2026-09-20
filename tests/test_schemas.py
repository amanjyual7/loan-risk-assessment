"""Input validation tests.

Imports Pydantic at module level on purpose: in an environment without it,
both pytest and tests/run.py skip this module rather than failing, and the
engine tests still run. The engine never imports Pydantic.
"""

from __future__ import annotations

from pydantic import ValidationError  # noqa: F401  (import guard)

from engine.schemas import ApplicantIn

BASE = {
    "identity": {
        "full_name": "Test Person", "age": 35, "pincode": "560001",
        "residence_type": "rented", "years_at_address": 3,
    },
    "household": {"dependent_children": 1, "dependent_adults": 0},
    "employment": {
        "employment_type": "salaried", "employer_name": "X", "industry": "Y",
        "years_current_job": 4, "total_experience_years": 10,
    },
    "income": {"gross_monthly": 90000, "net_monthly": 74000},
    "loan": {"amount": 600000, "purpose": "education", "tenure_months": 60},
    "bureau": {
        "score": 740, "active_accounts": 2, "closed_accounts": 1,
        "enquiries_3m": 1, "enquiries_6m": 2, "enquiries_12m": 3,
    },
    "trade_lines": [],
}


def _with(section: str, **changes) -> dict:
    payload = {k: (dict(v) if isinstance(v, dict) else v) for k, v in BASE.items()}
    payload[section] = {**payload[section], **changes}
    return payload


def _expect_invalid(payload: dict, because: str) -> None:
    try:
        ApplicantIn(**payload)
    except Exception:
        return
    raise AssertionError(f"accepted an invalid payload: {because}")


def test_valid_payload_round_trips_to_an_engine_record():
    record = ApplicantIn(**BASE).to_record()
    assert record.identity.pincode == 560001
    assert isinstance(record.identity.pincode, int)
    assert record.trade_lines == ()
    assert record.household.total_dependents == 1


def test_net_above_gross_is_rejected():
    _expect_invalid(_with("income", gross_monthly=50000, net_monthly=60000),
                    "net take-home above gross")


def test_malformed_pincode_is_rejected_at_the_schema_boundary():
    for bad in ("999999", "012345", "12345", "abc123"):
        _expect_invalid(_with("identity", pincode=bad), f"pincode {bad}")


def test_job_tenure_above_total_experience_is_rejected():
    _expect_invalid(
        _with("employment", years_current_job=12, total_experience_years=5),
        "more years in the current job than total experience",
    )


def test_enquiry_counts_must_be_non_decreasing():
    _expect_invalid(
        _with("bureau", enquiries_3m=5, enquiries_6m=2, enquiries_12m=9),
        "more enquiries in 3 months than in 6",
    )


def test_no_credit_history_conflicts_are_rejected():
    _expect_invalid(_with("bureau", no_credit_history=True, score=700),
                    "thin file with a bureau score")

    payload = {**BASE, "bureau": {"no_credit_history": True}}
    payload["trade_lines"] = [{
        "account_type": "credit card", "sanctioned_amount": 100000,
        "current_outstanding": 20000, "emi": 0,
    }]
    _expect_invalid(payload, "thin file with trade lines")


def test_closed_account_cannot_carry_an_emi():
    payload = {**BASE}
    payload["trade_lines"] = [{
        "account_type": "personal", "sanctioned_amount": 200000,
        "current_outstanding": 0, "emi": 5000,
        "opened_date": "2020-01-01", "closed_date": "2023-01-01",
    }]
    _expect_invalid(payload, "closed account with a live EMI")


def test_worst_dpd_below_current_dpd_is_rejected():
    payload = {**BASE}
    payload["trade_lines"] = [{
        "account_type": "personal", "sanctioned_amount": 200000,
        "current_outstanding": 100000, "emi": 4000,
        "current_dpd": 90, "worst_dpd_24m": 30,
    }]
    _expect_invalid(payload, "worst 24-month DPD below current DPD")


def test_errors_are_field_level_so_the_ui_can_place_them():
    try:
        ApplicantIn(**_with("income", gross_monthly=50000, net_monthly=60000))
    except Exception as exc:
        errors = exc.errors()
        assert errors and "loc" in errors[0]
        return
    raise AssertionError("expected a validation error")
