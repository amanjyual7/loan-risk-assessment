"""06 — Band cutoffs from held-out PD deciles.

    python notebooks/06_bands.py [--apply]

Cutoffs come from the data, not from round numbers. A 5% boundary implies
someone decided 5% was meaningful; a decile boundary of 0.0413 admits that
the bands are a partition of the observed distribution and nothing more.

Prints the config block to paste into config/config.yaml, or writes it
directly with --apply.
"""

# %%
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from notebooks._io import load_frame, save_frame  # noqa: E402
from engine.features import FEATURE_NAMES  # noqa: E402

WORK = REPO_ROOT / "notebooks" / "_work"
CONFIG = REPO_ROOT / "config" / "config.yaml"
ARTEFACT = REPO_ROOT / "models" / "pd_model.joblib"

# Five bands over ten deciles: the boundaries fall at the 2nd, 4th, 6th and
# 8th deciles, so each band holds two deciles of the test population.
BAND_QUANTILES = [0.2, 0.4, 0.6, 0.8]
BAND_LABELS = [("A", "Low"), ("B", "Moderate"), ("C", "Elevated"),
               ("D", "High"), ("E", "Severe")]


# %%
def pds_from_reference_grid(cfg: dict) -> np.ndarray:
    """PD distribution of the CURRENT artefact over the reference grid.

    Used when no trained model exists yet. Band cutoffs and the model that
    produces the PDs must come from the same distribution — cutoffs derived
    from one distribution and applied to another are not a partition of
    anything, and the visible symptom is most applicants collapsing into
    the worst band.

    There are no observed outcomes on a synthetic grid, so
    observed_default_rate stays null. That is the honest answer, and it is
    why this mode is a stopgap rather than a substitute for 05_train.py.
    """
    from notebooks.population_grid import grid_rows

    from engine.features import clamp_to_training_support
    from engine.model import PDModel

    model = PDModel.from_config(cfg)
    out = []
    for row in grid_rows():
        features, _ = clamp_to_training_support(row)
        out.append(model.predict(features).pd)
    return np.asarray(out)


# %%
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="write the new cutoffs into config/config.yaml")
    ap.add_argument("--calib-end", type=int, default=2015)
    ap.add_argument("--from-reference-grid", action="store_true",
                    help="derive cutoffs from the synthetic reference grid "
                         "instead of the held-out test cohort. Use only "
                         "while the placeholder artefact is in place.")
    args = ap.parse_args()

    if args.from_reference_grid:
        from engine.config import load_config

        cfg = load_config()
        pd_hat = pds_from_reference_grid(cfg)
        observed_available = False
        test = None
        source = (
            "PD quantiles on the synthetic reference grid — NO observed "
            "outcomes, placeholder only"
        )
        print(f"reference grid: {len(pd_hat):,} synthetic applicants, "
              f"PD {pd_hat.min():.2%} to {pd_hat.max():.2%}")
    else:
        artefact = joblib.load(ARTEFACT)

        # The gate is whether a scorable test cohort exists, not whether the
        # artefact carries the placeholder flag. A synthetic-trained model
        # has a real test cohort and real deciles; it just has no claim to
        # being about the world.
        try:
            frame = load_frame(WORK / "features")
        except FileNotFoundError:
            raise SystemExit(
                "no feature file to score, so cutoffs cannot come from a "
                "test cohort. Run notebooks 01-05, or pass "
                "--from-reference-grid for cutoffs that are at least "
                "internally consistent with the current artefact."
            )
        if artefact.get("is_placeholder"):
            print(
                "WARNING: the artefact is flagged as placeholder or "
                "synthetic. These cutoffs partition a distribution that was "
                "invented, and the observed default rates below are outcomes "
                "of an invented process. Do not put them in the README."
            )
        stats = json.loads((WORK / "standardisation.json").read_text())
        test = frame[frame["cohort_year"] > args.calib_end]
        if test.empty:
            raise SystemExit("test split is empty — check --calib-end")

        x_test = np.column_stack([
            (test[name].astype(float) - stats[name]["center"]) / stats[name]["scale"]
            for name in FEATURE_NAMES
        ])
        raw = artefact["estimator"].predict_proba(x_test)[:, 1]
        pd_hat = (
            artefact["calibrator"].predict(raw) if artefact.get("calibrator") else raw
        )
        observed_available = True
        source = (
            f"PD deciles on the held-out test cohort, "
            f"model {artefact['model_version']}"
        )

        shift = yaml.safe_load(CONFIG.read_text()).get("calibration", {}).get(
            "population_shift_log_odds", 0.0)
        if shift:
            print(
                f"NOTE: config carries a population shift of {shift:+.3f} "
                "log-odds for the India transfer. Cutoffs here are derived on "
                "the US test population WITHOUT it, which is correct: the "
                "bands partition the model's own distribution, and the shift "
                "then moves Indian applicants through those same bands."
            )

    cutoffs = np.quantile(pd_hat, BAND_QUANTILES)
    edges = [0.0, *cutoffs, 1.01]

    rows = []
    for i, (code, label) in enumerate(BAND_LABELS):
        mask = (pd_hat >= edges[i]) & (pd_hat < edges[i + 1])
        observed = None
        if observed_available and mask.any():
            observed = float(test["target"].to_numpy()[mask].mean())
        rows.append({
            "band": code,
            "label": label,
            "pd_upper": float(round(edges[i + 1], 4)),
            "loans": int(mask.sum()),
            "mean_predicted_pd": float(pd_hat[mask].mean()) if mask.any() else None,
            "observed_default_rate": observed,
        })

    table = pd.DataFrame(rows)
    print(f"\nbands derived from: {source}")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if not observed_available:
        print(
            "\nobserved_default_rate is null throughout because a synthetic "
            "grid has no outcomes. These cutoffs make the bands internally "
            "consistent with the current artefact; they do not make them "
            "meaningful. Re-derive from the test cohort after training."
        )
    else:
        print(
            "\nCheck two things. mean_predicted_pd and observed_default_rate "
            "should track each other closely — if they diverge inside a band, "
            "calibration has failed locally even if it looks fine in "
            "aggregate. And observed_default_rate must increase "
            "monotonically down the bands; if it does not, the bands are not measuring risk."
    )

    block = {
        "bands": {
            "source": source,
            "definitions": [
                {
                    "band": r["band"],
                    "label": r["label"],
                    "pd_upper": r["pd_upper"],
                    "observed_default_rate": (
                        round(r["observed_default_rate"], 4)
                        if r["observed_default_rate"] is not None else None
                    ),
                }
                for r in rows
            ],
        }
    }

    print("\n--- paste into config/config.yaml ---")
    print(yaml.safe_dump(block, sort_keys=False, default_flow_style=False))

    if args.apply:
        cfg = yaml.safe_load(CONFIG.read_text())
        cfg["bands"] = block["bands"]
        parts = cfg["config_version"].split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        cfg["config_version"] = ".".join(parts)
        CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"wrote {CONFIG} at config_version {cfg['config_version']}")
        print(
            "WARNING: --apply rewrites the YAML through the parser, which "
            "drops the provenance comments. Reinstate them from git before "
            "committing — the comments are the reason anyone can audit the "
            "numbers."
        )


if __name__ == "__main__":
    main()
