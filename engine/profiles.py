"""Seeded demo profiles.

Synthetic. No real applicant data appears anywhere in this repo. The point
is that a reviewer can see the app work in fifteen seconds without typing.

`record_from_dict` is stdlib-only on purpose: the test suite and the demo
loader must not depend on Pydantic or Streamlit being installed.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

from engine import types as T


def _d(value: Optional[str]) -> Optional[date]:
    return date.fromisoformat(value) if value else None


def record_from_dict(raw: dict[str, Any]) -> T.Applicant:
    return T.Applicant(
        identity=T.Identity(
            full_name=raw["identity"]["full_name"],
            age=int(raw["identity"]["age"]),
            pincode=int(raw["identity"]["pincode"]),
            residence_type=raw["identity"]["residence_type"],
            years_at_address=float(raw["identity"]["years_at_address"]),
        ),
        household=T.Household(**raw.get("household", {})),
        employment=T.Employment(**raw["employment"]),
        income=T.Income(**raw["income"]),
        loan=T.LoanRequest(**raw["loan"]),
        bureau=T.BureauSummary(
            **{
                **raw.get("bureau", {}),
                "score_date": _d(raw.get("bureau", {}).get("score_date")),
            }
        ),
        trade_lines=tuple(
            T.TradeLine(
                **{
                    **t,
                    "opened_date": _d(t.get("opened_date")),
                    "closed_date": _d(t.get("closed_date")),
                }
            )
            for t in raw.get("trade_lines", [])
        ),
        adverse=T.AdverseHistory(**raw.get("adverse", {})),
        cash_flow=T.CashFlow(**raw.get("cash_flow", {})),
        co_applicant=(
            T.CoApplicant(**raw["co_applicant"]) if raw.get("co_applicant") else None
        ),
        manual_state=raw.get("manual_state"),
    )


def load_profiles(path: str | Path) -> dict[str, dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return {p["key"]: p for p in payload["profiles"]}


def load_profile_records(path: str | Path) -> dict[str, T.Applicant]:
    return {k: record_from_dict(v["applicant"]) for k, v in load_profiles(path).items()}
