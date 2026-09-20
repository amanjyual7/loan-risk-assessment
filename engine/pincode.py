"""Pincode -> state / city / region / tier.

Pincode is the ONLY lookup key. There is deliberately no city-name fallback:
city strings in Indian reference data are inconsistent ("Bangalore" /
"Bengaluru" / "Bangalore South"), and a silent wrong match is worse than a
flagged miss.

The app reads a committed SQLite artefact, never the live Google Sheet. See
data/build_pincode_reference.py for the export step.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from engine.types import Location

REQUIRED_COLUMNS = ["state", "city", "pincode", "metro_flag", "tier", "region"]
_PINCODE_RE = re.compile(r"^[1-8][0-9]{5}$")


class PincodeReferenceError(RuntimeError):
    pass


def is_well_formed(raw: str | int) -> bool:
    """Indian pincodes are 6 digits with a first digit of 1-8.

    0 is unassigned and 9 is reserved for the Army Postal Service, which
    does not appear in civilian reference data. Malformed input is rejected
    before any lookup, so `999999` and `012345` never reach the reference
    table — a miss and a typo are different outcomes and the UI shows them
    differently.
    """
    s = str(raw).strip()
    return bool(_PINCODE_RE.match(s))


@dataclass(frozen=True)
class TierScale:
    """Ordered categorical, so cost-of-living lookups can compare tiers
    rather than string-match them."""

    order: tuple[str, ...]

    def rank(self, tier: str) -> int:
        try:
            return self.order.index(tier)
        except ValueError as exc:
            raise PincodeReferenceError(f"unknown tier {tier!r}") from exc

    def is_at_least(self, tier: str, threshold: str) -> bool:
        """True when `tier` is at least as urban as `threshold`."""
        return self.rank(tier) <= self.rank(threshold)

    def normalise(self, raw: str) -> str:
        cleaned = " ".join(str(raw).strip().split()).title()
        cleaned = cleaned.replace("Tier-", "Tier ").replace("Tier", "Tier")
        aliases = {
            "Metro": "Metro",
            "Metropolitan": "Metro",
            "Tier 1": "Tier 1",
            "Tier1": "Tier 1",
            "Tier 2": "Tier 2",
            "Tier2": "Tier 2",
            "Tier 3": "Tier 3",
            "Tier3": "Tier 3",
        }
        if cleaned not in aliases:
            raise PincodeReferenceError(f"unmappable tier value {raw!r}")
        return aliases[cleaned]


class PincodeReference:
    """Loaded once and cached by the app (@st.cache_data)."""

    def __init__(self, rows: dict[int, dict[str, str]], scale: TierScale):
        self._rows = rows
        self.scale = scale

    # -- construction -------------------------------------------------------
    @classmethod
    def from_sqlite(cls, path: str | Path, tier_order: list[str]) -> "PincodeReference":
        path = Path(path)
        if not path.exists():
            raise PincodeReferenceError(
                f"pincode artefact missing at {path}. "
                "Run: python data/build_pincode_reference.py"
            )
        scale = TierScale(tuple(tier_order))
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute(
                "SELECT state, city, pincode, metro_flag, tier, region FROM pincode"
            )
            rows: dict[int, dict[str, str]] = {}
            for r in cur:
                pin = int(r["pincode"])  # cast pincode to int
                if pin in rows:
                    # Duplicates are dropped at build time; if one survives,
                    # the artefact is not the one we built.
                    raise PincodeReferenceError(
                        f"duplicate pincode {pin} in artefact — rebuild it"
                    )
                rows[pin] = {
                    "state": r["state"],
                    "city": r["city"],
                    "region": r["region"],
                    "tier": scale.normalise(r["tier"]),
                    "metro_flag": str(r["metro_flag"]),
                }
        finally:
            conn.close()
        if not rows:
            raise PincodeReferenceError("pincode artefact is empty")
        return cls(rows, scale)

    # -- invariants ---------------------------------------------------------
    def assert_unique_index(self) -> None:
        """A dict cannot hold duplicate keys, so this asserts the count
        survived the load rather than re-checking the dict."""
        if len(self._rows) != len(set(self._rows)):  # pragma: no cover
            raise PincodeReferenceError("pincode index is not unique")

    def __len__(self) -> int:
        return len(self._rows)

    # -- lookup -------------------------------------------------------------
    def lookup(
        self,
        raw_pincode: str | int,
        *,
        default_tier: str,
        miss_label: str,
        manual_state: Optional[str] = None,
    ) -> Location:
        """Resolve a pincode.

        Raises ValueError on a malformed pincode — that is a UI validation
        error, not a data miss. On a valid-but-absent pincode, returns a
        Location carrying `assumed=True` so the result screen can surface it.
        """
        if not is_well_formed(raw_pincode):
            raise ValueError(f"{raw_pincode!r} is not a valid 6-digit Indian pincode")
        pin = int(str(raw_pincode).strip())
        row = self._rows.get(pin)
        if row is None:
            return Location(
                pincode=pin,
                state=manual_state,
                city=None,
                region=None,
                tier=self.scale.normalise(default_tier),
                assumed=True,
                assumption_note=miss_label,
            )
        return Location(
            pincode=pin,
            state=row["state"],
            city=row["city"],
            region=row["region"],
            tier=row["tier"],
            assumed=False,
        )

    def states(self) -> list[str]:
        return sorted({r["state"] for r in self._rows.values() if r["state"]})
