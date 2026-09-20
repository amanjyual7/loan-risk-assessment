"""Session state -> validated applicant payload.

The UI collects flat dictionaries per screen; the engine wants a nested,
validated record. This module is the only place that knows both shapes, so
neither the views nor the engine has to.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

import pandas as pd


def _date(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str):
        return value or None
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.date().isoformat()
    return None


def trade_lines_from_frame(grid: Optional[pd.DataFrame]) -> list[dict[str, Any]]:
    if grid is None or grid.empty:
        return []
    out: list[dict[str, Any]] = []
    for _, row in grid.iterrows():
        if not row.get("account_type"):
            continue  # a half-typed row the user has not finished
        out.append(
            {
                "account_type": row["account_type"],
                "lender_type": row.get("lender_type") or "",
                "sanctioned_amount": float(row.get("sanctioned_amount") or 0),
                "current_outstanding": float(row.get("current_outstanding") or 0),
                "emi": float(row.get("emi") or 0),
                "opened_date": _date(row.get("opened_date")),
                "closed_date": _date(row.get("closed_date")),
                "ownership": row.get("ownership") or "individual",
                "current_dpd": int(row.get("current_dpd") or 0),
                "worst_dpd_24m": int(row.get("worst_dpd_24m") or 0),
                "status": row.get("status") or "standard",
            }
        )
    return out


def build_payload(
    applicant_form: dict[str, Any],
    credit_form: dict[str, Any],
    grid: Optional[pd.DataFrame],
) -> dict[str, Any]:
    a, c = applicant_form, credit_form
    thin = bool(c.get("no_credit_history", False))

    payload: dict[str, Any] = {
        "identity": {
            "full_name": a.get("full_name", ""),
            "age": int(a.get("age", 0) or 0),
            "pincode": str(a.get("pincode", "")),
            "residence_type": a.get("residence_type", "rented"),
            "years_at_address": float(a.get("years_at_address", 0) or 0),
        },
        "household": {
            "dependent_children": int(a.get("dependent_children", 0) or 0),
            "dependent_adults": int(a.get("dependent_adults", 0) or 0),
        },
        "employment": {
            "employment_type": a.get("employment_type", "salaried"),
            "employer_name": a.get("employer_name", ""),
            "industry": a.get("industry", ""),
            "years_current_job": float(a.get("years_current_job", 0) or 0),
            "total_experience_years": float(a.get("total_experience_years", 0) or 0),
        },
        "income": {
            "gross_monthly": float(a.get("gross_monthly", 0) or 0),
            "net_monthly": float(a.get("net_monthly", 0) or 0),
            "income_source": a.get("income_source", ""),
            "income_varies": bool(a.get("income_varies", False)),
            "other_household_income": float(a.get("other_household_income", 0) or 0),
        },
        "loan": {
            "amount": float(a.get("amount", 0) or 0),
            "purpose": a.get("purpose", "education"),
            "tenure_months": int(a.get("tenure_months", 12) or 12),
            "collateral_offered": a.get("collateral_offered", "none"),
        },
        "bureau": (
            {"no_credit_history": True}
            if thin
            else {
                "no_credit_history": False,
                "score": int(c.get("score", 720) or 720),
                "score_date": _date(c.get("score_date")),
                "active_accounts": int(c.get("active_accounts", 0) or 0),
                "closed_accounts": int(c.get("closed_accounts", 0) or 0),
                "enquiries_3m": int(c.get("enquiries_3m", 0) or 0),
                "enquiries_6m": int(c.get("enquiries_6m", 0) or 0),
                "enquiries_12m": int(c.get("enquiries_12m", 0) or 0),
            }
        ),
        "trade_lines": [] if thin else trade_lines_from_frame(grid),
        "adverse": (
            {}
            if thin
            else {
                "write_offs": int(c.get("write_offs", 0) or 0),
                "settlements": int(c.get("settlements", 0) or 0),
                "suits_filed": int(c.get("suits_filed", 0) or 0),
                "nach_bounces": int(c.get("nach_bounces", 0) or 0),
                "cheque_bounces": int(c.get("cheque_bounces", 0) or 0),
            }
        ),
        "cash_flow": {
            "average_monthly_balance": (
                float(c["average_monthly_balance"])
                if c.get("average_monthly_balance") else None
            ),
            "salary_credits_6m": (
                int(c["salary_credits_6m"]) if c.get("salary_credits_6m") else None
            ),
            "bounced_debits_6m": (
                int(c["bounced_debits_6m"])
                if c.get("bounced_debits_6m") is not None else None
            ),
        },
        "manual_state": a.get("manual_state") or None,
    }

    if a.get("has_co_applicant") and a.get("co_name"):
        payload["co_applicant"] = {
            "name": a["co_name"],
            "relationship": a.get("co_relationship", ""),
            "employment_type": a.get("co_employment_type", "salaried"),
            "net_monthly_income": float(a.get("co_net_monthly_income", 0) or 0),
            "age": int(a.get("co_age", 40) or 40),
        }
    return payload


def payload_to_forms(payload: dict[str, Any]) -> tuple[dict, dict, pd.DataFrame]:
    """Inverse of build_payload, for loading a demo profile or resuming a
    saved assessment into the two screens."""
    ident = payload.get("identity", {})
    hh = payload.get("household", {})
    emp = payload.get("employment", {})
    inc = payload.get("income", {})
    loan = payload.get("loan", {})
    bureau = payload.get("bureau", {})
    adverse = payload.get("adverse", {})
    cash = payload.get("cash_flow", {})
    co = payload.get("co_applicant")

    applicant_form = {
        **{k: ident.get(k) for k in
           ("full_name", "age", "residence_type", "years_at_address")},
        "pincode": str(ident.get("pincode", "")),
        "dependent_children": hh.get("dependent_children", 0),
        "dependent_adults": hh.get("dependent_adults", 0),
        **{k: emp.get(k) for k in
           ("employment_type", "employer_name", "industry", "years_current_job",
            "total_experience_years")},
        **{k: inc.get(k) for k in
           ("gross_monthly", "net_monthly", "income_source", "income_varies",
            "other_household_income")},
        **{k: loan.get(k) for k in
           ("amount", "purpose", "tenure_months", "collateral_offered")},
        "manual_state": payload.get("manual_state"),
        "has_co_applicant": co is not None,
        "co_name": (co or {}).get("name", ""),
        "co_relationship": (co or {}).get("relationship", ""),
        "co_employment_type": (co or {}).get("employment_type", "salaried"),
        "co_net_monthly_income": (co or {}).get("net_monthly_income", 0.0),
        "co_age": (co or {}).get("age", 45),
    }

    credit_form = {
        "no_credit_history": bool(bureau.get("no_credit_history", False)),
        "score": bureau.get("score", 720),
        "score_date": (
            date.fromisoformat(bureau["score_date"])
            if bureau.get("score_date") else date.today()
        ),
        "active_accounts": bureau.get("active_accounts", 0),
        "closed_accounts": bureau.get("closed_accounts", 0),
        "enquiries_3m": bureau.get("enquiries_3m", 0),
        "enquiries_6m": bureau.get("enquiries_6m", 0),
        "enquiries_12m": bureau.get("enquiries_12m", 0),
        "write_offs": adverse.get("write_offs", 0),
        "settlements": adverse.get("settlements", 0),
        "suits_filed": adverse.get("suits_filed", 0),
        "nach_bounces": adverse.get("nach_bounces", 0),
        "cheque_bounces": adverse.get("cheque_bounces", 0),
        "average_monthly_balance": cash.get("average_monthly_balance") or 0.0,
        "salary_credits_6m": cash.get("salary_credits_6m") or 0,
        "bounced_debits_6m": cash.get("bounced_debits_6m") or 0,
    }

    lines = payload.get("trade_lines", [])
    grid = pd.DataFrame(lines) if lines else pd.DataFrame(
        columns=[
            "account_type", "lender_type", "sanctioned_amount",
            "current_outstanding", "emi", "opened_date", "closed_date",
            "ownership", "current_dpd", "worst_dpd_24m", "status",
        ]
    )
    for col in ("opened_date", "closed_date"):
        if col in grid:
            grid[col] = pd.to_datetime(grid[col], errors="coerce")
    return applicant_form, credit_form, grid
