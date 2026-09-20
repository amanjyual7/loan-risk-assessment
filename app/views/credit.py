"""Screen 2 — credit data.

Same form-boundary trade-off as screen 1, for the same reason: the footer
has to show live derived values as the user types, and a widget inside
st.form cannot trigger the rerun that would recompute them. So the trade-line
editor and the footer sit outside the form and the scalar bureau fields sit
inside it.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import streamlit as st

from engine.affordability import proposed_emi

ACCOUNT_TYPES = ["credit card", "personal", "auto", "home", "education", "gold",
                 "overdraft"]
STATUSES = ["standard", "substandard", "doubtful", "loss", "written-off", "settled"]
OWNERSHIP = ["individual", "joint", "guarantor"]

EMPTY_GRID = pd.DataFrame(
    {
        "account_type": pd.Series(dtype="object"),
        "lender_type": pd.Series(dtype="object"),
        "sanctioned_amount": pd.Series(dtype="float"),
        "current_outstanding": pd.Series(dtype="float"),
        "emi": pd.Series(dtype="float"),
        "opened_date": pd.Series(dtype="object"),
        "closed_date": pd.Series(dtype="object"),
        "ownership": pd.Series(dtype="object"),
        "current_dpd": pd.Series(dtype="int"),
        "worst_dpd_24m": pd.Series(dtype="int"),
        "status": pd.Series(dtype="object"),
    }
)

COLUMN_CONFIG = {
    "account_type": st.column_config.SelectboxColumn(
        "Account type", options=ACCOUNT_TYPES, required=True),
    "lender_type": st.column_config.TextColumn("Lender type"),
    "sanctioned_amount": st.column_config.NumberColumn(
        "Sanctioned ₹", min_value=0, step=10000, format="%d"),
    "current_outstanding": st.column_config.NumberColumn(
        "Outstanding ₹", min_value=0, step=10000, format="%d"),
    "emi": st.column_config.NumberColumn("EMI ₹", min_value=0, step=500, format="%d"),
    "opened_date": st.column_config.DateColumn("Opened"),
    "closed_date": st.column_config.DateColumn("Closed"),
    "ownership": st.column_config.SelectboxColumn(
        "Ownership", options=OWNERSHIP, default="individual"),
    "current_dpd": st.column_config.NumberColumn(
        "Current DPD", min_value=0, max_value=1500, step=1),
    "worst_dpd_24m": st.column_config.NumberColumn(
        "Worst DPD 24m", min_value=0, max_value=1500, step=1),
    "status": st.column_config.SelectboxColumn(
        "Status", options=STATUSES, default="standard"),
}


def render(cfg: dict) -> None:
    st.subheader("2 · Credit data")
    data = st.session_state.setdefault("credit_form", {})
    applicant = st.session_state.get("applicant_form", {})

    # ---- no-credit-history toggle, outside the form so it can hide the
    #      rest of the section immediately ---------------------------------
    data["no_credit_history"] = st.toggle(
        "No credit history (thin file)",
        value=bool(data.get("no_credit_history", False)),
        help="Skips the rest of this section. The result screen switches to "
             "an income and cash-flow view rather than scoring zero.",
    )
    thin = data["no_credit_history"]

    if not thin:
        with st.form("screen2_bureau"):
            st.markdown("**Bureau summary**")
            b1, b2, b3, b4 = st.columns(4)
            data["score"] = b1.number_input(
                "Bureau score (300–900)", 300, 900, int(data.get("score", 720)))
            data["score_date"] = b2.date_input(
                "Score date", value=data.get("score_date", date.today()))
            data["active_accounts"] = b3.number_input(
                "Active accounts", 0, 100, int(data.get("active_accounts", 2)))
            data["closed_accounts"] = b4.number_input(
                "Closed accounts", 0, 100, int(data.get("closed_accounts", 1)))
            e1, e2, e3 = st.columns(3)
            data["enquiries_3m"] = e1.number_input(
                "Enquiries, 3m", 0, 99, int(data.get("enquiries_3m", 0)))
            data["enquiries_6m"] = e2.number_input(
                "Enquiries, 6m", 0, 99, int(data.get("enquiries_6m", 1)))
            data["enquiries_12m"] = e3.number_input(
                "Enquiries, 12m", 0, 99, int(data.get("enquiries_12m", 2)))

            st.markdown("**Adverse history, last 12 months**")
            a1, a2, a3, a4, a5 = st.columns(5)
            data["write_offs"] = a1.number_input(
                "Write-offs", 0, 50, int(data.get("write_offs", 0)))
            data["settlements"] = a2.number_input(
                "Settlements", 0, 50, int(data.get("settlements", 0)))
            data["suits_filed"] = a3.number_input(
                "Suits filed", 0, 50, int(data.get("suits_filed", 0)))
            data["nach_bounces"] = a4.number_input(
                "NACH bounces", 0, 200, int(data.get("nach_bounces", 0)))
            data["cheque_bounces"] = a5.number_input(
                "Cheque bounces", 0, 200, int(data.get("cheque_bounces", 0)))
            st.form_submit_button("Apply bureau fields")

        st.markdown("**Trade lines**")
        st.caption("Add or remove rows. Closed accounts must carry a zero EMI.")
        grid = st.data_editor(
            st.session_state.get("trade_lines_df", EMPTY_GRID.copy()),
            column_config=COLUMN_CONFIG,
            num_rows="dynamic",
            use_container_width=True,
            key="trade_lines_editor",
        )
        st.session_state["trade_lines_df"] = grid
    else:
        grid = EMPTY_GRID.copy()
        st.session_state["trade_lines_df"] = grid
        st.info(
            "Thin file: no bureau summary, trade lines or adverse history "
            "collected. Cash-flow fields below become the main evidence."
        )

    with st.form("screen2_cashflow"):
        st.markdown("**Cash flow** (optional)")
        c1, c2, c3 = st.columns(3)
        data["average_monthly_balance"] = c1.number_input(
            "Average monthly balance (₹)", 0.0, 1e9,
            float(data.get("average_monthly_balance", 0.0)), step=1000.0)
        data["salary_credits_6m"] = c2.number_input(
            "Salary credits, last 6 months", 0, 60,
            int(data.get("salary_credits_6m", 0)))
        data["bounced_debits_6m"] = c3.number_input(
            "Bounced debits, last 6 months", 0, 200,
            int(data.get("bounced_debits_6m", 0)))
        st.form_submit_button("Apply cash-flow fields")

    st.session_state["credit_form"] = data

    # ---- live footer ------------------------------------------------------
    st.divider()
    obligations = _sum_emis(grid)
    emi = proposed_emi(
        float(applicant.get("amount", 0) or 0),
        float(cfg["pricing"]["assumed_annual_rate_pct"]),
        int(applicant.get("tenure_months", 12) or 12),
    )
    net = float(applicant.get("net_monthly", 0) or 0)
    foir = (obligations + emi) / net if net > 0 else None

    f1, f2, f3 = st.columns(3)
    f1.metric("Existing obligations", f"₹{obligations:,.0f}")
    f2.metric("Proposed EMI", f"₹{emi:,.0f}",
              help=f"at {cfg['pricing']['assumed_annual_rate_pct']}% assumed rate")
    f3.metric("FOIR", f"{foir:.1%}" if foir is not None else "—")

    nav1, nav2 = st.columns([1, 1])
    if nav1.button("← Back to applicant details"):
        st.session_state["step"] = "applicant"
        st.rerun()
    if nav2.button("Assess risk →", type="primary"):
        st.session_state["step"] = "result"
        st.rerun()


def _sum_emis(grid: pd.DataFrame) -> float:
    if grid is None or grid.empty or "emi" not in grid:
        return 0.0
    live = grid[grid.get("closed_date").isna()] if "closed_date" in grid else grid
    return float(pd.to_numeric(live["emi"], errors="coerce").fillna(0).sum())
