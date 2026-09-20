"""Pydantic schemas for raw questionnaire input.

Validation happens here, before scoring, so the UI can render field-level
errors instead of a stack trace. The engine itself never imports Pydantic —
`to_record()` hands it plain dataclasses.
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from engine import types as T
from engine.pincode import is_well_formed


class IdentityIn(BaseModel):
    full_name: str = Field(min_length=1, max_length=120)
    age: int = Field(ge=18, le=80)
    pincode: str
    residence_type: Literal["owned", "rented", "parental"]
    years_at_address: float = Field(ge=0, le=80)

    @field_validator("pincode")
    @classmethod
    def _pin(cls, v: str) -> str:
        if not is_well_formed(v):
            raise ValueError("pincode must be 6 digits and cannot start with 0")
        return str(v).strip()

    @model_validator(mode="after")
    def _address_vs_age(self):
        if self.years_at_address > self.age - 17:
            raise ValueError("years at address cannot exceed the applicant's adult life")
        return self


class HouseholdIn(BaseModel):
    dependent_children: int = Field(default=0, ge=0, le=12)
    dependent_adults: int = Field(default=0, ge=0, le=12)


class EmploymentIn(BaseModel):
    employment_type: Literal["salaried", "self-employed", "business", "student"]
    employer_name: str = ""
    industry: str = ""
    years_current_job: float = Field(default=0, ge=0, le=60)
    total_experience_years: float = Field(default=0, ge=0, le=60)

    @model_validator(mode="after")
    def _tenure(self):
        if self.years_current_job > self.total_experience_years:
            raise ValueError(
                "years in current job cannot exceed total work experience"
            )
        return self


class IncomeIn(BaseModel):
    gross_monthly: float = Field(ge=0, le=100_000_000)
    net_monthly: float = Field(ge=0, le=100_000_000)
    income_source: str = ""
    income_varies: bool = False
    other_household_income: float = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _net_vs_gross(self):
        # Asked separately rather than inferred, so the relationship is
        # checked rather than assumed.
        if self.net_monthly > self.gross_monthly:
            raise ValueError("net take-home cannot exceed gross monthly income")
        return self


class LoanRequestIn(BaseModel):
    amount: float = Field(gt=0, le=100_000_000)
    purpose: str = Field(min_length=1)
    tenure_months: int = Field(ge=6, le=180)
    collateral_offered: str = "none"


class CoApplicantIn(BaseModel):
    name: str = Field(min_length=1)
    relationship: str = Field(min_length=1)
    employment_type: Literal["salaried", "self-employed", "business", "student"]
    net_monthly_income: float = Field(ge=0)
    age: int = Field(ge=18, le=90)


class TradeLineIn(BaseModel):
    account_type: Literal["credit card", "personal", "auto", "home", "education",
                          "gold", "overdraft"]
    lender_type: str = ""
    sanctioned_amount: float = Field(ge=0)
    current_outstanding: float = Field(ge=0)
    emi: float = Field(ge=0)
    opened_date: Optional[date] = None
    closed_date: Optional[date] = None
    ownership: Literal["individual", "joint", "guarantor"] = "individual"
    current_dpd: int = Field(default=0, ge=0, le=1500)
    worst_dpd_24m: int = Field(default=0, ge=0, le=1500)
    status: Literal["standard", "substandard", "doubtful", "loss", "written-off",
                    "settled"] = "standard"

    @model_validator(mode="after")
    def _consistency(self):
        if self.worst_dpd_24m < self.current_dpd:
            raise ValueError("worst DPD in 24 months cannot be below current DPD")
        if self.closed_date and self.opened_date and self.closed_date < self.opened_date:
            raise ValueError("closed date precedes opened date")
        if self.closed_date and self.emi > 0:
            raise ValueError("a closed account cannot carry an EMI")
        return self


class AdverseIn(BaseModel):
    write_offs: int = Field(default=0, ge=0)
    settlements: int = Field(default=0, ge=0)
    suits_filed: int = Field(default=0, ge=0)
    nach_bounces: int = Field(default=0, ge=0)
    cheque_bounces: int = Field(default=0, ge=0)


class CashFlowIn(BaseModel):
    average_monthly_balance: Optional[float] = Field(default=None, ge=0)
    salary_credits_6m: Optional[int] = Field(default=None, ge=0, le=60)
    bounced_debits_6m: Optional[int] = Field(default=None, ge=0, le=200)


class BureauIn(BaseModel):
    no_credit_history: bool = False
    score: Optional[int] = Field(default=None, ge=300, le=900)
    score_date: Optional[date] = None
    active_accounts: int = Field(default=0, ge=0)
    closed_accounts: int = Field(default=0, ge=0)
    enquiries_3m: int = Field(default=0, ge=0)
    enquiries_6m: int = Field(default=0, ge=0)
    enquiries_12m: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _monotone_enquiries(self):
        if not (self.enquiries_3m <= self.enquiries_6m <= self.enquiries_12m):
            raise ValueError(
                "enquiry counts must be non-decreasing across 3, 6 and 12 months"
            )
        if self.no_credit_history and self.score is not None:
            raise ValueError("'no credit history' is set but a score was supplied")
        return self


class ApplicantIn(BaseModel):
    identity: IdentityIn
    household: HouseholdIn = HouseholdIn()
    employment: EmploymentIn
    income: IncomeIn
    loan: LoanRequestIn
    bureau: BureauIn = BureauIn()
    trade_lines: list[TradeLineIn] = []
    adverse: AdverseIn = AdverseIn()
    cash_flow: CashFlowIn = CashFlowIn()
    co_applicant: Optional[CoApplicantIn] = None
    manual_state: Optional[str] = None

    @model_validator(mode="after")
    def _no_history_means_no_lines(self):
        if self.bureau.no_credit_history and self.trade_lines:
            raise ValueError(
                "'no credit history' is set but trade lines were entered"
            )
        return self

    def to_record(self) -> T.Applicant:
        """Hand the validated input to the engine as plain dataclasses."""
        return T.Applicant(
            identity=T.Identity(
                full_name=self.identity.full_name,
                age=self.identity.age,
                pincode=int(self.identity.pincode),
                residence_type=self.identity.residence_type,
                years_at_address=self.identity.years_at_address,
            ),
            household=T.Household(
                dependent_children=self.household.dependent_children,
                dependent_adults=self.household.dependent_adults,
            ),
            employment=T.Employment(**self.employment.model_dump()),
            income=T.Income(**self.income.model_dump()),
            loan=T.LoanRequest(**self.loan.model_dump()),
            bureau=T.BureauSummary(**self.bureau.model_dump()),
            trade_lines=tuple(T.TradeLine(**t.model_dump()) for t in self.trade_lines),
            adverse=T.AdverseHistory(**self.adverse.model_dump()),
            cash_flow=T.CashFlow(**self.cash_flow.model_dump()),
            co_applicant=(
                T.CoApplicant(**self.co_applicant.model_dump())
                if self.co_applicant else None
            ),
            manual_state=self.manual_state,
        )
