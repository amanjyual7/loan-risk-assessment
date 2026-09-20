"""Config loading. One file, one loader, no defaults scattered in code."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"


@lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    with p.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    _validate(cfg)
    return cfg


def resolve(cfg: dict, key: str) -> Path:
    """Resolve a path from cfg['paths'] against the repo root."""
    return REPO_ROOT / cfg["paths"][key]


def _validate(cfg: dict) -> None:
    required = ["config_version", "pricing", "expenses", "bands", "policy",
                "thin_file", "currency_bridge", "bureau_bridge", "paths", "pincode"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"config missing sections: {missing}")

    tiers = set(cfg["pincode"]["tier_order"])
    expense_tiers = set(cfg["expenses"]["base_single_adult_by_tier"])
    if tiers != expense_tiers:
        raise ValueError(
            "every tier needs an expense base: "
            f"tier_order={sorted(tiers)} expenses={sorted(expense_tiers)}"
        )
