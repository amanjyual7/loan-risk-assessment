"""Result screen."""

from __future__ import annotations

import streamlit as st

from app import components as ui
from app.mapping import build_payload
from engine.export import to_json, to_pdf
from engine.model import PDModel
from engine.pincode import PincodeReference
from engine.score import assess
from engine.types import Assessment


def render(model: PDModel, reference: PincodeReference, cfg: dict, store) -> None:
    payload = build_payload(
        st.session_state.get("applicant_form", {}),
        st.session_state.get("credit_form", {}),
        st.session_state.get("trade_lines_df"),
    )

    try:
        from engine.schemas import ApplicantIn

        validated = ApplicantIn(**payload)
    except Exception as exc:  # pydantic ValidationError, shown field by field
        st.error("The questionnaire has validation errors. Nothing was scored.")
        _render_validation_errors(exc)
        if st.button("← Back to credit data"):
            st.session_state["step"] = "credit"
            st.rerun()
        return

    record = validated.to_record()
    result = assess(record, model=model, reference=reference, cfg=cfg)

    _render(result, record, cfg)

    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button(
        "Export JSON", to_json(result),
        file_name=f"assessment_{result.band}.json", mime="application/json")
    c2.download_button(
        "Export PDF", to_pdf(result, record.identity.full_name),
        file_name=f"assessment_{result.band}.pdf", mime="application/pdf")
    if c3.button("Save assessment"):
        store.save(payload, result)
        st.toast("Saved. Reload it from the sidebar.")
    if c4.button("← Edit inputs"):
        st.session_state["step"] = "credit"
        st.rerun()


def _render_validation_errors(exc: Exception) -> None:
    errors = getattr(exc, "errors", None)
    if not callable(errors):
        st.write(str(exc))
        return
    for err in errors():
        where = " → ".join(str(p) for p in err.get("loc", ()))
        st.markdown(f"- **{where or 'form'}**: {err.get('msg', '')}")


def _render(result: Assessment, record, cfg: dict) -> None:
    # 1 — band, prominent, with the historical default rate beside it,
    #     and the full scale underneath with this band picked out
    ui.band_header(result.band, result.band_label,
                   result.band_observed_default_rate,
                   result.band_raised_by_policy, result.model_band)
    st.write("")
    ui.band_strip(cfg, current=result.band)

    # 2 — calibrated PD with an honest caveat
    st.write("")
    if result.pd_is_meaningful and result.pd is not None:
        p1, p2 = st.columns([1, 3])
        p1.metric("Calibrated PD", f"{result.pd:.1%}")
        p2.caption(
            "This is a calibrated probability, not a score: if the model says "
            "7%, roughly 7 of 100 similar applicants in the held-out data "
            "defaulted. The interval around a single applicant's estimate is "
            "wide, and it is wider still here because the model was trained "
            "on a different lending population — see the caveats below."
        )
    else:
        st.info(
            "No calibrated PD is shown for this applicant. See the thin-file "
            "section below for what was used instead."
        )

    # 6 — hard knockouts, shown apart from scored factors
    if result.knockouts:
        st.write("")
        st.error(
            "**Hard knockouts.** These are facts, not ratios, and they are "
            "kept apart from the scored factors on purpose.", icon="⛔")
        ui.flag_list(result.knockouts)

    tabs = st.tabs([
        "Summary of details", "What moved the estimate", "Policy flags",
        "What would change this", "Metrics & formulas", "Caveats",
    ])

    # 3 — readable recap
    with tabs[0]:
        _summary(result, record)

    # 4 — per-applicant contributions
    with tabs[1]:
        if result.thin_file:
            st.info("The model was not used for this applicant.")
        else:
            st.caption(
                f"Attribution method: {ui.factor_tag('model')} contributions in "
                "log-odds, largest first."
            )
            ui.contribution_bars(result.contributions)

    # 5 — policy flags, separately, with severity and trigger value
    with tabs[2]:
        st.caption(
            f"{ui.factor_tag('policy')} These are asserted rules over signals "
            "the public training data does not carry. They can raise a band, "
            "never lower one."
        )
        ui.flag_list(result.policy_flags)

    # 7 — what would change this
    with tabs[3]:
        if not result.suggestions:
            st.info(
                "No single input change moved this applicant to a better band. "
                "That is itself informative: the band is not resting on one "
                "marginal number."
                if result.band != "A" else
                "Already in the best band."
            )
        else:
            st.caption(
                "Each suggestion was re-scored through the same engine, so "
                "none of them claims an improvement the model would not "
                "actually produce."
            )
            for s in result.suggestions:
                st.markdown(
                    f"**{s.field_label}** — {s.change}  \n"
                    f"→ band **{s.resulting_band}**, PD {s.resulting_pd:.1%}"
                )

    with tabs[4]:
        for group in ("Affordability", "Expense floor", "Credit behaviour"):
            st.markdown(f"**{group}**")
            ui.metric_table(result.metrics, group)
            st.write("")

    with tabs[5]:
        for c in result.caveats:
            st.markdown(f"- {c}")
        st.caption(
            f"config {result.config_version} · model {result.model_version} · "
            f"assessed {result.assessed_at}"
        )

    # 8 — thin-file path
    if result.thin_file and result.thin_file_view:
        st.divider()
        st.subheader("Thin-file assessment")
        view = result.thin_file_view
        st.caption(
            "No bureau history, so the model was not used. These are the "
            "income and cash-flow signals the band came from instead."
        )
        cols = st.columns(4)
        cols[0].metric("Net income", ui.money(view["net_monthly_income"]))
        cols[1].metric("Household income", ui.money(view["household_income"]))
        cols[2].metric("Proposed EMI", ui.money(view["proposed_emi"]))
        cols[3].metric("Surplus", ui.money(view["surplus"]))
        st.markdown(
            f"**{view['checks_passed']} of {view['checks_total']} "
            "thin-file checks passed**"
        )
        for name, ok in view["checks"].items():
            st.markdown(f"{'✅' if ok else '❌'} {name}")


def _summary(result: Assessment, record) -> None:
    by_key = {m.key: m for m in result.metrics}

    def val(key: str) -> str:
        m = by_key.get(key)
        return ui.fmt_metric(m) if m else "—"

    loc = result.location
    loc_text = (
        f"{loc.city or '—'}, {loc.state or '—'} · {loc.region or '—'} · "
        f"tier {loc.tier}"
    )
    if loc.assumed:
        loc_text += f"  ⚠️ {loc.assumption_note}"

    co = record.co_applicant
    rows = [
        ("Applicant", f"{record.identity.full_name}, age {record.identity.age}"),
        ("Employment", f"{record.employment.employment_type} · "
                       f"{record.employment.employer_name or '—'} · "
                       f"{record.employment.years_current_job:g} yrs in role"),
        ("Income", f"gross {ui.money(record.income.gross_monthly)} · "
                   f"net {ui.money(record.income.net_monthly)}"
                   + (" · varies month to month" if record.income.income_varies else "")),
        ("Location and tier", loc_text),
        ("Household", f"{record.household.dependent_children} child(ren), "
                      f"{record.household.dependent_adults} dependent adult(s)"),
        ("Loan requested", f"{ui.money(record.loan.amount)} over "
                           f"{record.loan.tenure_months} months for "
                           f"{record.loan.purpose}"),
        ("Proposed EMI", val("proposed_emi")),
        ("Existing obligations", val("existing_emis")),
        ("FOIR", val("foir")),
        ("Estimated household expense", val("household_expense")),
        ("Surplus", val("surplus")),
        ("Residual income", val("residual_income")),
        ("Bureau score", str(record.bureau.score) if record.bureau.score
                         else "no credit history"),
        ("Credit vintage", val("credit_vintage")),
        ("Revolving utilisation", val("revolving_utilisation")),
        ("Worst DPD, 24 months", val("worst_dpd_24m")),
        ("Co-applicant", f"{co.name} ({co.relationship}), net "
                         f"{ui.money(co.net_monthly_income)}" if co else "none"),
    ]
    st.markdown(
        "\n".join(
            ["| | |", "| --- | --- |"]
            + [f"| {label} | {value} |" for label, value in rows]
        )
    )
