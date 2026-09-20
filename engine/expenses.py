"""Household expense floor.

Why this exists: plain FOIR scores a single earner with no dependents
identically to a household of five at the same income. That is wrong in a
way that matters for education lending, where the co-applicant parent is
often supporting siblings on the same salary.

The floor is built from four numbers, all in config/config.yaml with a dated
provenance comment:
    base cost for a single adult at the applicant's city tier
  + per dependent child
  + per dependent adult
  - a discount when the residence is owned (no rent component)

Children and dependent adults are kept separate on purpose: a child and an
elderly parent do not carry the same cost, and averaging them into a single
"dependents" count throws away the distinction the questionnaire collects.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.types import Household, Metric


@dataclass(frozen=True)
class ExpenseEstimate:
    base: float
    children_component: float
    dependent_adults_component: float
    owned_discount: float
    total: float
    tier: str
    formula: str


def estimate_expenses(
    household: Household,
    tier: str,
    residence_type: str,
    cfg: dict,
) -> ExpenseEstimate:
    ex = cfg["expenses"]
    by_tier = ex["base_single_adult_by_tier"]
    if tier not in by_tier:
        raise KeyError(f"no expense base configured for tier {tier!r}")

    base = float(by_tier[tier])
    children = float(ex["per_dependent_child"]) * household.dependent_children
    adults = float(ex["per_dependent_adult"]) * household.dependent_adults

    gross = base + children + adults
    discount = 0.0
    if residence_type == "owned":
        discount = gross * float(ex["owned_residence_discount_pct"]) / 100.0

    total = gross - discount

    formula = (
        f"base({tier})={base:,.0f} "
        f"+ {household.dependent_children}×child({ex['per_dependent_child']:,}) "
        f"+ {household.dependent_adults}×dep_adult({ex['per_dependent_adult']:,})"
    )
    if discount:
        formula += f" − owned_discount({ex['owned_residence_discount_pct']}%)"

    return ExpenseEstimate(
        base=base,
        children_component=children,
        dependent_adults_component=adults,
        owned_discount=discount,
        total=total,
        tier=tier,
        formula=formula,
    )


def expense_metrics(est: ExpenseEstimate) -> list[Metric]:
    return [
        Metric(
            key="household_expense",
            label="Estimated household expense",
            value=est.total,
            unit="INR/month",
            formula=est.formula,
            group="Expense floor",
        )
    ]
