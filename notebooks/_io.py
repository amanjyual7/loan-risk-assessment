"""Intermediate-file IO.

Parquet when pyarrow is available, pickle otherwise. The training pipeline
should not fall over because one optional dependency is missing, and these
are throwaway intermediates in a gitignored folder either way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import pyarrow  # noqa: F401
    _PARQUET = True
except ImportError:
    _PARQUET = False


def save_frame(frame: pd.DataFrame, stem: Path) -> Path:
    if _PARQUET:
        path = stem.with_suffix(".parquet")
        frame.to_parquet(path, index=False)
    else:
        path = stem.with_suffix(".pkl")
        frame.to_pickle(path)
    return path


def load_frame(stem: Path) -> pd.DataFrame:
    for suffix in (".parquet", ".pkl"):
        path = stem.with_suffix(suffix)
        if path.exists():
            return (pd.read_parquet(path) if suffix == ".parquet"
                    else pd.read_pickle(path))
    raise FileNotFoundError(
        f"no intermediate file at {stem}.parquet or {stem}.pkl — "
        "run the earlier notebooks first"
    )
