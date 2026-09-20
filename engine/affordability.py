"""Affordability arithmetic. Pure functions, no I/O."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from engine.expenses import ExpenseEstimate
from engine.types import Income, LoanRequest, Metric, TradeLine


def proposed_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Standard reducing-balance EMI."""
    if tenure_months <= 0:
        raise ValueError("tenure must be positive")
    if principal <= 0:
        return 0.0
    r = annual_rate_pct / 12.0 / 100.0
    if r == 0:
        return principal / tenure_months
    factor = (1 + r) ** tenure_months
    return principal * r * factor / (factor - 1)


def existing_obligations(trade_lines: Iterable[TradeLine]) -> float:
    """Sum of EMIs on live trade lines only. A closed account has no EMI
    however the grid was filled in."""
    return float(sum(t.emi for t in trade_lines if t.is_live))


@dataclass(frozen=True)
class Affordability:
    proposed_emi: float
    existing_emis: float
    total_emis: float
    net_income: float
    household_income: float
    foir: Optional[float]
    residual_income: float
    expense: float
    surplus: float
    expense_adjusted_foir: Optional[float]
    assumed_rate_pct: float


def compute_affordability(
    income: Income,
    loan: LoanRequest,
    trade_lines: Iterable[TradeLine],
    expense: ExpenseEstimate,
    cfg: dict,
    co_applicant_income: float = 0.0,
) -> Affordability:
    rate = float(cfg["pricing"]["assumed_annual_rate_pct"])
    emi = proposed_emi(loan.amount, rate, loan.tenure_months)
    existing = existing_obligations(trade_lines)
    total = existing + emi

    net = float(income.net_monthly)
    household = net + float(income.other_household_income) + float(co_applicant_income)

    # FOIR is deliberately on the applicant's own net income. Co-applicant
    # income enters the household surplus, not the applicant's own ratio,
    # so the two can be read separately.
    foir = (total / net) if net > 0 else None
    residual = net - total
    surplus = household - expense.total - total
    eaf = ((total + expense.total) / net) if net > 0 else None

    return Affordability(
        proposed_emi=emi,
        existing_emis=existing,
        total_emis=total,
        net_income=net,
        household_income=household,
        foir=foir,
        residual_income=residual,
        expense=expense.total,
        surplus=surplus,
        expense_adjusted_foir=eaf,
        assumed_rate_pct=rate,
    )


def affordability_metrics(a: Affordability) -> list[Metric]:
    return [
        Metric(
            "proposed_emi",
            "Proposed EMI",
            a.proposed_emi,
            "INR/month",
            f"P·r·(1+r)^n / ((1+r)^n − 1), r = {a.assumed_rate_pct}%/12, n = tenure",
            "Affordability",
        ),
        Metric(
            "existing_emis",
            "Existing obligations",
            a.existing_emis,
            "INR/month",
            "Σ EMI of live trade lines",
            "Affordability",
        ),
        Metric(
            "foir",
            "FOIR / DTI",
            a.foir,
            "ratio",
            "(existing EMIs + proposed EMI) ÷ net monthly income",
            "Affordability",
        ),
        Metric(
            "residual_income",
            "Residual income",
            a.residual_income,
            "INR/month",
            "net monthly income − all EMIs",
            "Affordability",
        ),
        Metric(
            "expense_adjusted_foir",
            "Expense-adjusted FOIR",
            a.expense_adjusted_foir,
            "ratio",
            "(all EMIs + estimated household expense) ÷ net monthly income",
            "Expense floor",
        ),
        Metric(
            "surplus",
            "Surplus",
            a.surplus,
            "INR/month",
            "net income + other household income − expense − all EMIs",
            "Expense floor",
        ),
    ]
