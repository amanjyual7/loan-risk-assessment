"""Canonical record types.

Deliberately stdlib-only. The engine consumes these, not Pydantic models and
not Streamlit widgets, so scoring can be imported and tested without either
installed. `engine.schemas` validates raw UI input and emits these.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Any, Literal, Optional

ResidenceType = Literal["owned", "rented", "parental"]
EmploymentType = Literal["salaried", "self-employed", "business", "student"]
Ownership = Literal["individual", "joint", "guarantor"]
AccountStatus = Literal[
    "standard", "substandard", "doubtful", "loss", "written-off", "settled"
]
AccountType = Literal[
    "credit card", "personal", "auto", "home", "education", "gold", "overdraft"
]


@dataclass(frozen=True)
class Location:
    """Resolved from pincode only. Display + expense estimation only."""

    pincode: int
    state: Optional[str]
    city: Optional[str]
    region: Optional[str]
    tier: str
    assumed: bool = False
    assumption_note: Optional[str] = None


@dataclass(frozen=True)
class Identity:
    full_name: str
    age: int
    pincode: int
    residence_type: ResidenceType
    years_at_address: float


@dataclass(frozen=True)
class Household:
    dependent_children: int = 0
    dependent_adults: int = 0

    @property
    def total_dependents(self) -> int:
        return self.dependent_children + self.dependent_adults


@dataclass(frozen=True)
class Employment:
    employment_type: EmploymentType
    employer_name: str = ""
    industry: str = ""
    years_current_job: float = 0.0
    total_experience_years: float = 0.0


@dataclass(frozen=True)
class Income:
    gross_monthly: float
    net_monthly: float
    income_source: str = ""
    income_varies: bool = False
    other_household_income: float = 0.0


@dataclass(frozen=True)
class LoanRequest:
    amount: float
    purpose: str
    tenure_months: int
    collateral_offered: str = "none"


@dataclass(frozen=True)
class CoApplicant:
    name: str
    relationship: str
    employment_type: EmploymentType
    net_monthly_income: float
    age: int


@dataclass(frozen=True)
class TradeLine:
    account_type: AccountType
    lender_type: str
    sanctioned_amount: float
    current_outstanding: float
    emi: float
    opened_date: Optional[date]
    closed_date: Optional[date]
    ownership: Ownership
    current_dpd: int
    worst_dpd_24m: int
    status: AccountStatus

    @property
    def is_live(self) -> bool:
        return self.closed_date is None

    @property
    def is_revolving(self) -> bool:
        return self.account_type in ("credit card", "overdraft")

    @property
    def is_secured(self) -> bool:
        return self.account_type in ("auto", "home", "gold")


@dataclass(frozen=True)
class AdverseHistory:
    """Counts over the last 12 months."""

    write_offs: int = 0
    settlements: int = 0
    suits_filed: int = 0
    nach_bounces: int = 0
    cheque_bounces: int = 0


@dataclass(frozen=True)
class CashFlow:
    average_monthly_balance: Optional[float] = None
    salary_credits_6m: Optional[int] = None
    bounced_debits_6m: Optional[int] = None

    @property
    def provided(self) -> bool:
        return any(
            v is not None
            for v in (
                self.average_monthly_balance,
                self.salary_credits_6m,
                self.bounced_debits_6m,
            )
        )


@dataclass(frozen=True)
class BureauSummary:
    no_credit_history: bool = False
    score: Optional[int] = None
    score_date: Optional[date] = None
    active_accounts: int = 0
    closed_accounts: int = 0
    enquiries_3m: int = 0
    enquiries_6m: int = 0
    enquiries_12m: int = 0


@dataclass(frozen=True)
class Applicant:
    """Everything the two screens collect, validated and normalised."""

    identity: Identity
    household: Household
    employment: Employment
    income: Income
    loan: LoanRequest
    bureau: BureauSummary
    trade_lines: tuple[TradeLine, ...] = ()
    adverse: AdverseHistory = field(default_factory=AdverseHistory)
    cash_flow: CashFlow = field(default_factory=CashFlow)
    co_applicant: Optional[CoApplicant] = None
    # Set when the pincode was not in the reference data and the user picked
    # a state by hand.
    manual_state: Optional[str] = None

    @property
    def thin_file(self) -> bool:
        return self.bureau.no_credit_history or not self.trade_lines


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Metric:
    """A derived number shown with its formula, per the spec."""

    key: str
    label: str
    value: Optional[float]
    unit: str
    formula: str
    group: str


@dataclass(frozen=True)
class Contribution:
    """One per-applicant model contribution, in plain language."""

    feature: str
    plain_language: str
    value: Any
    log_odds_delta: float

    @property
    def direction(self) -> str:
        return "up-risk" if self.log_odds_delta > 0 else "down-risk"


@dataclass(frozen=True)
class PolicyFlag:
    code: str
    message: str
    severity: str  # knockout | high | medium | low
    triggered_by: str  # the actual value that fired it
    is_knockout: bool = False


@dataclass(frozen=True)
class Suggestion:
    """A 'what would change this' item."""

    field_label: str
    change: str
    resulting_band: str
    resulting_pd: float


@dataclass(frozen=True)
class Assessment:
    band: str
    band_label: str
    band_observed_default_rate: Optional[float]
    pd: Optional[float]
    pd_is_meaningful: bool
    model_band: str
    band_raised_by_policy: bool
    thin_file: bool
    thin_file_view: Optional[dict[str, Any]]
    location: Location
    metrics: tuple[Metric, ...]
    contributions: tuple[Contribution, ...]
    policy_flags: tuple[PolicyFlag, ...]
    knockouts: tuple[PolicyFlag, ...]
    suggestions: tuple[Suggestion, ...]
    model_features: dict[str, float]
    caveats: tuple[str, ...]
    config_version: str
    model_version: str
    assessed_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
