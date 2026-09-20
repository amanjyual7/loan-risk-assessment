"""Credit behaviour metrics derived from the trade-line grid."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional, Sequence

from engine.types import BureauSummary, Metric, TradeLine


@dataclass(frozen=True)
class CreditBehaviour:
    current_dpd_max: int
    worst_dpd_24m: int
    delinquency_recency: Optional[str]
    revolving_utilisation_pct: Optional[float]
    credit_vintage_months: Optional[int]
    secured_share_pct: Optional[float]
    unsecured_share_pct: Optional[float]
    enquiry_velocity_3m: float
    live_accounts: int
    delinq_accounts_24m: int
    total_outstanding: float


def _months_between(earlier: date, later: date) -> int:
    return (later.year - earlier.year) * 12 + (later.month - earlier.month)


def compute_credit_behaviour(
    trade_lines: Sequence[TradeLine],
    bureau: BureauSummary,
    as_of: Optional[date] = None,
) -> CreditBehaviour:
    as_of = as_of or date.today()
    live = [t for t in trade_lines if t.is_live]

    current_max = max((t.current_dpd for t in live), default=0)
    worst_24 = max((t.worst_dpd_24m for t in trade_lines), default=0)

    if current_max >= 30:
        recency = "currently delinquent"
    elif worst_24 >= 30:
        recency = "delinquent within last 24 months, now current"
    elif trade_lines:
        recency = "no delinquency in last 24 months"
    else:
        recency = None

    revolving = [t for t in live if t.is_revolving]
    sanctioned = sum(t.sanctioned_amount for t in revolving)
    outstanding = sum(t.current_outstanding for t in revolving)
    util = (outstanding / sanctioned * 100.0) if sanctioned > 0 else None

    opened = [t.opened_date for t in trade_lines if t.opened_date]
    vintage = _months_between(min(opened), as_of) if opened else None

    total_bal = sum(t.current_outstanding for t in live)
    secured_bal = sum(t.current_outstanding for t in live if t.is_secured)
    secured_share = (secured_bal / total_bal * 100.0) if total_bal > 0 else None

    return CreditBehaviour(
        current_dpd_max=current_max,
        worst_dpd_24m=worst_24,
        delinquency_recency=recency,
        revolving_utilisation_pct=util,
        credit_vintage_months=vintage,
        secured_share_pct=secured_share,
        unsecured_share_pct=(100.0 - secured_share) if secured_share is not None else None,
        # Enquiries per month over the last quarter. A burst of recent
        # enquiries on a thin file is the classic credit-hungry signature.
        enquiry_velocity_3m=bureau.enquiries_3m / 3.0,
        live_accounts=len(live),
        delinq_accounts_24m=sum(1 for t in trade_lines if t.worst_dpd_24m >= 30),
        total_outstanding=total_bal,
    )


def credit_metrics(c: CreditBehaviour) -> list[Metric]:
    return [
        Metric("current_dpd_max", "Current DPD (worst live account)", c.current_dpd_max,
               "days", "max(current DPD) over live trade lines", "Credit behaviour"),
        Metric("worst_dpd_24m", "Worst DPD, last 24 months", c.worst_dpd_24m,
               "days", "max(worst DPD 24m) over all trade lines", "Credit behaviour"),
        Metric("revolving_utilisation", "Revolving utilisation",
               c.revolving_utilisation_pct, "%",
               "Σ outstanding ÷ Σ sanctioned, cards and overdrafts only",
               "Credit behaviour"),
        Metric("credit_vintage", "Credit vintage", c.credit_vintage_months,
               "months", "months since earliest account opened", "Credit behaviour"),
        Metric("secured_share", "Secured share of balances", c.secured_share_pct,
               "%", "secured outstanding ÷ total outstanding (auto, home, gold)",
               "Credit behaviour"),
        Metric("enquiry_velocity", "Enquiry velocity", c.enquiry_velocity_3m,
               "per month", "enquiries in last 3 months ÷ 3", "Credit behaviour"),
    ]
