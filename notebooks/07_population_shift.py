"""07 — Derive the population shift.

    python notebooks/07_population_shift.py

The problem, restated. The model is fitted on US personal loans and served
to Indian education-loan applicants. Three features behave structurally
differently in the target population:

  term_60          fires for essentially every applicant, because Indian
                   education loans are not written under 36 months. A
                   feature with no variance contributes a constant.
  log_loan_amnt    clamps at the top of the training support for most
                   applicants.
  loan_to_income   same, for the same reason.

Left alone they add a near-constant penalty to everyone and the whole
portfolio lands in the worst band. The minimum correct response is to
re-fit the intercept against the target population, which is what this
script estimates.

What this is NOT: a fix for the coefficients. Those are still US
coefficients and this script cannot validate them. The only thing that
would is observed default outcomes on Indian education loans, which is
stated as a limitation in the README rather than papered over.
"""

# %%
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from engine.config import load_config  # noqa: E402
from engine.features import SPEC_BY_NAME, clamp_to_training_support  # noqa: E402
from engine.model import PDModel  # noqa: E402

CONFIG = REPO_ROOT / "config" / "config.yaml"


# %%
# A grid standing in for the applicant population the app will actually see.
# Replace these with the real marginal distribution as soon as you have one —
# a pull of accepted education-loan applications is enough. Until then this
# is an assumption, and it is labelled as one in config.
from notebooks.population_grid import grid_rows  # noqa: E402


# %%
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-rate", type=float, default=None,
                    help="assumed portfolio default rate; defaults to "
                         "calibration.reference_base_rate in config")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    target = args.target_rate or float(cfg["calibration"]["reference_base_rate"])

    # Load with NO shift, so we measure the raw transfer gap.
    unshifted = PDModel.load(REPO_ROOT / cfg["paths"]["model_artefact"],
                             population_shift=0.0)

    log_odds = []
    clamped_counts = {name: 0 for name in SPEC_BY_NAME}
    total = 0
    for row in grid_rows():
        features, notes = clamp_to_training_support(row)
        for note in notes:
            for name, spec in SPEC_BY_NAME.items():
                if spec.plain in note:
                    clamped_counts[name] += 1
        prediction = unshifted.predict(features)
        p = min(max(prediction.raw_pd, 1e-9), 1 - 1e-9)
        log_odds.append(math.log(p / (1 - p)))
        total += 1

    log_odds = np.asarray(log_odds)
    median_log_odds = float(np.median(log_odds))
    current_median_pd = 1 / (1 + math.exp(-median_log_odds))
    target_log_odds = math.log(target / (1 - target))
    shift = target_log_odds - median_log_odds

    print(f"reference grid: {total:,} synthetic applicants\n")
    print("features clamped to the training support:")
    for name, count in sorted(clamped_counts.items(), key=lambda kv: -kv[1]):
        if count:
            print(f"  {name:20s} {count:>7,} / {total:,} ({count / total:.0%})")
    print(
        "\n^ any feature clamped for most of the grid is carrying no "
        "information in this population. Consider dropping it from the "
        "feature set at training time instead of relying on the shift to "
        "absorb it."
    )

    print(f"\nmedian PD on the grid, unshifted: {current_median_pd:.2%}")
    print(f"assumed portfolio base rate:      {target:.2%}")
    print(f"required shift:                   {shift:+.3f} log-odds")
    print(f"\nconfig block:\n  population_shift_log_odds: {shift:.2f}")

    shifted_pds = 1 / (1 + np.exp(-(log_odds + shift)))
    print(f"\nafter the shift, the grid spans "
          f"{shifted_pds.min():.2%} to {shifted_pds.max():.2%} "
          f"(median {np.median(shifted_pds):.2%})")
    if shifted_pds.max() > 0.9:
        print(
            "WARNING: the top of the grid is above 90%. A constant shift "
            "cannot fix a spread that wide — that is a coefficient problem, "
            "and it means the US coefficients are too steep for this "
            "population. Note it in the README limitations."
        )

    if args.apply:
        cfg_raw = yaml.safe_load(CONFIG.read_text())
        cfg_raw["calibration"]["population_shift_log_odds"] = round(shift, 2)
        cfg_raw["calibration"]["source"] = (
            f"derived from a {total}-point reference grid against an assumed "
            f"base rate of {target:.2%}"
        )
        CONFIG.write_text(yaml.safe_dump(cfg_raw, sort_keys=False))
        print(f"\nwrote {CONFIG} — reinstate the comments from git before "
              "committing")


if __name__ == "__main__":
    main()
