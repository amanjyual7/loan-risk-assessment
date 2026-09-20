"""Shared render helpers. No business logic lives here."""

from __future__ import annotations

from typing import Iterable, Sequence

import streamlit as st

from engine.types import Contribution, Metric, PolicyFlag

BAND_COLOUR = {
    "A": "#1b7f3b",
    "B": "#4f9c28",
    "C": "#c08b00",
    "D": "#d35400",
    "E": "#a61b1b",
}

SEVERITY_ICON = {
    "knockout": "⛔",
    "high": "🔴",
    "medium": "🟠",
    "low": "🟡",
}


def money(value: float | None) -> str:
    if value is None:
        return "—"
    return f"₹{value:,.0f}"


def fmt_metric(m: Metric) -> str:
    if m.value is None:
        return "—"
    if m.unit == "ratio":
        return f"{m.value:.1%}"
    if m.unit == "%":
        return f"{m.value:.0f}%"
    if m.unit == "INR/month":
        return money(m.value)
    if m.unit in ("days", "months"):
        return f"{m.value:.0f} {m.unit}"
    return f"{m.value:,.2f}"


def metric_table(metrics: Sequence[Metric], group: str | None = None) -> None:
    """Every metric with its formula visible, per the spec."""
    rows = [m for m in metrics if group is None or m.group == group]
    if not rows:
        return
    st.markdown(
        "\n".join(
            ["| Metric | Value | Formula |", "| --- | --- | --- |"]
            + [
                f"| {m.label} | **{fmt_metric(m)}** | `{m.formula}` |"
                for m in rows
            ]
        )
    )


def band_header(band: str, label: str, observed: float | None,
                raised_by_policy: bool, model_band: str) -> None:
    colour = BAND_COLOUR.get(band, "#555555")
    observed_txt = (
        f"observed default rate in this band: <b>{observed:.1%}</b>"
        if observed is not None
        else "observed default rate: not yet available (model untrained)"
    )
    st.markdown(
        f"""
        <div style="border-left:6px solid {colour};padding:12px 16px;
                    background:rgba(0,0,0,0.03);border-radius:4px">
          <div style="font-size:0.8rem;letter-spacing:0.08em;color:#666">
            RISK BAND</div>
          <div style="font-size:2.4rem;font-weight:700;color:{colour};
                      line-height:1.1">{band} · {label}</div>
          <div style="font-size:0.85rem;color:#444;margin-top:4px">
            {observed_txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if raised_by_policy:
        st.caption(
            f"The model placed this applicant in band {model_band}. The policy "
            f"layer raised it to {band}. Policy can only ever raise severity."
        )


def flag_list(flags: Iterable[PolicyFlag]) -> None:
    flags = list(flags)
    if not flags:
        st.success("No policy flags fired.")
        return
    order = {"knockout": 0, "high": 1, "medium": 2, "low": 3}
    for f in sorted(flags, key=lambda x: order.get(x.severity, 9)):
        icon = SEVERITY_ICON.get(f.severity, "•")
        st.markdown(
            f"{icon} **{f.message}**  \n"
            f"<span style='color:#666;font-size:0.85rem'>severity "
            f"{f.severity} · triggered by: {f.triggered_by}</span>",
            unsafe_allow_html=True,
        )


def contribution_bars(contributions: Sequence[Contribution]) -> None:
    """Up-risk and down-risk split, largest first, plain language."""
    if not contributions:
        st.info("Per-applicant attribution is not available for this assessment.")
        return
    up = [c for c in contributions if c.log_odds_delta > 0]
    down = [c for c in contributions if c.log_odds_delta < 0]
    peak = max((abs(c.log_odds_delta) for c in contributions), default=1.0) or 1.0

    def render(items: Sequence[Contribution], colour: str) -> None:
        for c in items:
            width = abs(c.log_odds_delta) / peak * 100
            st.markdown(
                f"<div style='font-size:0.85rem;margin-bottom:2px'>"
                f"{c.plain_language} "
                f"<span style='color:#888'>({c.log_odds_delta:+.2f} log-odds)</span>"
                f"</div>"
                f"<div style='height:6px;width:{width:.0f}%;background:{colour};"
                f"border-radius:3px;margin-bottom:8px'></div>",
                unsafe_allow_html=True,
            )

    left, right = st.columns(2)
    with left:
        st.markdown("**Pushed risk up**")
        render(up, "#a61b1b")
    with right:
        st.markdown("**Pulled risk down**")
        render(down, "#1b7f3b")


def factor_tag(kind: str) -> str:
    """Mark each factor in the UI as model or policy, per the spec."""
    if kind == "model":
        return "`MODEL`"
    return "`POLICY`"
