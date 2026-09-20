"""Risk bands.

Cutoffs live in config, derived from PD deciles on the held-out test set —
not round numbers, because round numbers imply a policy choice that was
never made.

The policy layer may only ever move an applicant to a WORSE band. There is
no path by which an asserted rule improves a learned estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Band:
    code: str
    label: str
    pd_upper: float
    observed_default_rate: Optional[float]


def load_bands(cfg: dict) -> list[Band]:
    bands = [
        Band(d["band"], d["label"], float(d["pd_upper"]), d.get("observed_default_rate"))
        for d in cfg["bands"]["definitions"]
    ]
    uppers = [b.pd_upper for b in bands]
    if uppers != sorted(uppers):
        raise ValueError("band cutoffs in config are not ascending")
    return bands


def band_for_pd(pd_value: float, cfg: dict) -> Band:
    for band in load_bands(cfg):
        if pd_value < band.pd_upper:
            return band
    return load_bands(cfg)[-1]


def worsen(band: Band, steps: int, cfg: dict) -> Band:
    """Move `steps` bands toward the severe end. Never improves."""
    if steps <= 0:
        return band
    bands = load_bands(cfg)
    idx = next(i for i, b in enumerate(bands) if b.code == band.code)
    return bands[min(idx + steps, len(bands) - 1)]


_SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2, "knockout": 3}


def band_penalty(flags, cfg: dict) -> int:
    """Downgrade implied by the graded policy flags.

    Sub-additive and capped. The worst single flag sets the base; several
    flags of substance stacking up adds one more step; the total is capped
    so the band scale keeps its resolution. Knockouts are handled separately
    by the caller and always force the worst band.
    """
    table = cfg["policy"]["band_penalty"]
    if not flags:
        return 0

    base = max(int(table.get(f.severity, 0)) for f in flags)

    threshold = _SEVERITY_RANK[str(table.get("stacking_threshold_severity", "medium"))]
    substantive = sum(
        1 for f in flags if _SEVERITY_RANK.get(f.severity, 0) >= threshold
    )
    if substantive >= int(table.get("stacking_threshold_count", 3)):
        base += int(table.get("stacking_extra_step", 1))

    return min(base, int(table.get("max_graded_downgrade", 2)))
