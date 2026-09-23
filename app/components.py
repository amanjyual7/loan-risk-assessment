"""Shared render helpers. No business logic lives here."""

from __future__ import annotations

from typing import Iterable, Sequence

import streamlit as st

from engine.types import Contribution, Metric, PolicyFlag

# Tuned for a dark background: the light-theme greens and reds were too
# dark to read against #0e1621.
BAND_COLOUR = {
    "A": "#3fb98f",
    "B": "#d4b920",
    "C": "#e08a3c",
    "D": "#dd5f5f",
    "E": "#c8384f",
}

SEVERITY_ICON = {
    "knockout": "⛔",
    "high": "🔴",
    "medium": "🟠",
    "low": "🟡",
}


STYLESHEET = """
<style>
  /* Tighter, more deliberate type than Streamlit's defaults. */
  h1 { font-weight: 700; letter-spacing: -0.02em; }
  h2, h3 { font-weight: 600; letter-spacing: -0.01em; }

  /* Section labels: small, spaced, muted — used for "RISK BAND" etc. */
  .lra-eyebrow {
    font-size: 0.72rem; letter-spacing: 0.10em; text-transform: uppercase;
    color: #8b9bb0; font-weight: 600;
  }

  /* The band reference strip. Cells share a border so they read as one
     table rather than five floating cards. */
  .lra-bandstrip { display: flex; border: 1px solid #263243; border-radius: 6px;
                   overflow: hidden; margin: 4px 0 18px 0; }
  .lra-bandcell  { flex: 1; padding: 12px 14px 14px 14px;
                   border-right: 1px solid #263243; }
  .lra-bandcell:last-child { border-right: none; }
  .lra-bandcell.is-current { background: rgba(79, 209, 197, 0.07); }
  .lra-bandbar   { height: 9px; border-radius: 3px; margin-bottom: 10px; }
  .lra-bandname  { font-weight: 700; font-size: 0.95rem; }
  .lra-bandrate  { font-family: ui-monospace, "SF Mono", Menlo, monospace;
                   font-size: 1.05rem; color: #e6edf3; margin-top: 2px; }
  .lra-bandnote  { font-size: 0.72rem; color: #7d8ea4; line-height: 1.3; }

  /* Make the step tabs look like tabs rather than a sentence with arrows. */
  .lra-steps { display: flex; gap: 2px; border-bottom: 1px solid #263243;
               margin-bottom: 18px; }
  .lra-step  { padding: 9px 18px; font-size: 0.9rem; color: #7d8ea4;
               border-bottom: 2px solid transparent; }
  .lra-step.is-active { color: #e6edf3; font-weight: 600;
                        border-bottom-color: #4fd1c5;
                        background: rgba(255,255,255,0.03); }

  /* Metric tables: monospace numbers so columns line up. */
  .stMarkdown table { font-size: 0.85rem; }
  .stMarkdown table td:nth-child(2) {
    font-family: ui-monospace, "SF Mono", Menlo, monospace;
  }
  .stMarkdown table code { font-size: 0.74rem; color: #8b9bb0;
                           background: transparent; }
</style>
"""


def inject_styles() -> None:
    """Called once per rerun from the entry point."""
    st.markdown(STYLESHEET, unsafe_allow_html=True)


def band_strip(cfg: dict, current: str | None = None) -> None:
    """The full band scale with the default rate observed in each.

    Shown above the questionnaire so the reader knows what the bands mean
    before they see one assigned, and again on the result screen with the
    assigned band highlighted. A band label alone ("High") asserts a
    judgement; a band label next to 18.5% observed defaults is a
    measurement.
    """
    cells = []
    for d in cfg["bands"]["definitions"]:
        code = d["band"]
        observed = d.get("observed_default_rate")
        rate = f"{observed:.1%}" if observed is not None else "—"
        note = "observed default" if observed is not None else "not yet measured"
        active = " is-current" if code == current else ""
        cells.append(
            f"<div class='lra-bandcell{active}'>"
            f"<div class='lra-bandbar' style='background:{BAND_COLOUR[code]}'></div>"
            f"<div class='lra-bandname'>Band {code} · {d['label']}</div>"
            f"<div class='lra-bandrate'>{rate}</div>"
            f"<div class='lra-bandnote'>{note}</div>"
            f"</div>"
        )
    st.markdown(
        f"<div class='lra-bandstrip'>{''.join(cells)}</div>",
        unsafe_allow_html=True,
    )


def step_tabs(steps: dict[str, str], current: str) -> None:
    """Render the three-screen progress as tabs."""
    parts = []
    for i, (key, label) in enumerate(steps.items(), start=1):
        # Built without nesting quotes inside the f-string: that syntax is
        # 3.12+ only and the deploy target's Python version is not ours to
        # choose.
        active = " is-active" if key == current else ""
        parts.append(f"<div class='lra-step{active}'>{i}. {label}</div>")
    items = "".join(parts)
    st.markdown(f"<div class='lra-steps'>{items}</div>", unsafe_allow_html=True)


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
        <div style="border:1px solid #263243;border-left:5px solid {colour};
                    padding:16px 20px;background:#161f2c;border-radius:6px">
          <div class="lra-eyebrow">Risk band</div>
          <div style="font-size:2.5rem;font-weight:700;color:{colour};
                      line-height:1.15;letter-spacing:-0.02em">
            {band} · {label}</div>
          <div style="font-size:0.85rem;color:#8b9bb0;margin-top:2px">
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
            f"<span style='color:#8b9bb0;font-size:0.85rem'>severity "
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
                f"<span style='color:#7d8ea4'>({c.log_odds_delta:+.2f} log-odds)</span>"
                f"</div>"
                f"<div style='height:6px;width:{width:.0f}%;background:{colour};"
                f"border-radius:3px;margin-bottom:8px'></div>",
                unsafe_allow_html=True,
            )

    left, right = st.columns(2)
    with left:
        st.markdown("**Pushed risk up**")
        render(up, "#c8384f")
    with right:
        st.markdown("**Pulled risk down**")
        render(down, "#3fb98f")


def factor_tag(kind: str) -> str:
    """Mark each factor in the UI as model or policy, per the spec."""
    if kind == "model":
        return "`MODEL`"
    return "`POLICY`"
