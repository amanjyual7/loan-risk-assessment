"""Shared test environment.

Deliberately plain functions rather than pytest fixtures, so the suite can
be run either with pytest or with `python tests/run.py` in an environment
where pytest is not installed.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from engine.config import load_config, resolve  # noqa: E402
from engine.model import PDModel  # noqa: E402
from engine.pincode import PincodeReference  # noqa: E402
from engine.profiles import load_profile_records  # noqa: E402
from engine.score import assess  # noqa: E402


@lru_cache(maxsize=1)
def cfg() -> dict:
    return load_config()


@lru_cache(maxsize=1)
def model() -> PDModel:
    return PDModel.from_config(cfg())


@lru_cache(maxsize=1)
def reference() -> PincodeReference:
    return PincodeReference.from_sqlite(
        resolve(cfg(), "pincode_reference"), cfg()["pincode"]["tier_order"]
    )


@lru_cache(maxsize=1)
def profiles() -> dict:
    return load_profile_records(resolve(cfg(), "demo_profiles"))


def score(applicant, **kwargs):
    return assess(
        applicant, model=model(), reference=reference(), cfg=cfg(), **kwargs
    )


def band_rank(code: str) -> int:
    return [b["band"] for b in cfg()["bands"]["definitions"]].index(code)
