"""Export an assessment as JSON or PDF.

The JSON export is the full structured result including the config and model
version, so an exported assessment is reproducible. The PDF is a readable
one-pager for sharing.
"""

from __future__ import annotations

import io
import json
from dataclasses import asdict
from typing import Any

from engine.types import Assessment


def to_json(assessment: Assessment, *, indent: int = 2) -> str:
    def default(obj: Any):
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return str(obj)

    return json.dumps(asdict(assessment), default=default, indent=indent,
                      ensure_ascii=False)


def to_pdf(assessment: Assessment, applicant_name: str = "") -> bytes:
    """One-page PDF. Uses reportlab's platypus so text wraps."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4, title="Loan Risk Assessment",
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
    )
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=16, spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=11, spaceBefore=10,
                        spaceAfter=4, textColor=colors.HexColor("#333333"))
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=8.5, leading=11,
                          alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=7.5,
                           textColor=colors.HexColor("#666666"))

    flow: list[Any] = [
        Paragraph("Loan Risk Assessment", h1),
        Paragraph(
            "Assessment tool, not a credit decision. Synthetic data. "
            f"config {assessment.config_version} · model {assessment.model_version} "
            f"· {assessment.assessed_at}",
            small,
        ),
        HRFlowable(width="100%", color=colors.HexColor("#cccccc"), spaceAfter=8),
    ]

    pd_text = (
        f"{assessment.pd:.1%}" if assessment.pd is not None
        else "not applicable (thin file)"
    )
    header_rows = [
        ["Applicant", applicant_name or "—"],
        ["Risk band", f"{assessment.band} — {assessment.band_label}"],
        ["Calibrated PD", pd_text],
        ["Band from model", assessment.model_band],
        ["Raised by policy", "yes" if assessment.band_raised_by_policy else "no"],
        ["Location", (
            f"{assessment.location.city or '—'}, {assessment.location.state or '—'} "
            f"({assessment.location.tier}"
            f"{', ' + assessment.location.assumption_note if assessment.location.assumed else ''})"
        )],
    ]
    table = Table(header_rows, colWidths=[35 * mm, 135 * mm])
    table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#555555")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    flow.append(table)

    def section(title: str, lines: list[str]) -> None:
        if not lines:
            return
        flow.append(Paragraph(title, h2))
        for line in lines:
            flow.append(Paragraph(line, body))

    section("Derived metrics", [
        f"<b>{m.label}</b>: "
        + (f"{m.value:,.2f} {m.unit}" if m.value is not None else "—")
        + f"  <font color='#888888'>[{m.formula}]</font>"
        for m in assessment.metrics
    ])

    if assessment.knockouts:
        section("Hard knockouts", [
            f"<b>{k.message}</b> — {k.triggered_by}" for k in assessment.knockouts
        ])

    section("Policy flags", [
        f"[{f.severity}] {f.message} — {f.triggered_by}"
        for f in assessment.policy_flags
    ])

    up = [c for c in assessment.contributions if c.log_odds_delta > 0][:6]
    down = [c for c in assessment.contributions if c.log_odds_delta < 0][:6]
    section("What moved the estimate — up-risk", [
        f"{c.plain_language} ({c.log_odds_delta:+.2f})" for c in up
    ])
    section("What moved the estimate — down-risk", [
        f"{c.plain_language} ({c.log_odds_delta:+.2f})" for c in down
    ])

    section("What would change this", [
        f"{s.field_label}: {s.change} → band {s.resulting_band} "
        f"(PD {s.resulting_pd:.1%})"
        for s in assessment.suggestions
    ])

    section("Caveats", list(assessment.caveats))

    flow.append(Spacer(1, 8))
    flow.append(HRFlowable(width="100%", color=colors.HexColor("#cccccc")))
    flow.append(Paragraph(
        "Illustrative only. Not a production credit decision engine and no "
        "claim of regulatory compliance.", small))

    doc.build(flow)
    return buf.getvalue()
