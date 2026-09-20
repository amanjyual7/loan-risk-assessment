"""05 — Train, calibrate, evaluate, write the artefact.

    python notebooks/05_train.py

Produces models/pd_model.joblib with the same contract as the placeholder,
so nothing in engine/ or app/ changes when this replaces it.

Three things here are non-negotiable and each has a comment saying why:
temporal splitting, class weights rather than oversampling, and calibration
on a set the model never saw.
"""

# %%
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from notebooks._io import load_frame, save_frame  # noqa: E402
from engine.features import FEATURE_NAMES  # noqa: E402

WORK = REPO_ROOT / "notebooks" / "_work"
FIGS = WORK / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
ARTEFACT = REPO_ROOT / "models" / "pd_model.joblib"


# %%
# ---------------------------------------------------------------------------
# Temporal split.
#
# A random split leaks the macro environment: loans originated in the same
# month share an unemployment rate, a house-price trend and a Lending Club
# underwriting policy, so a randomly held-out loan has near-identical
# siblings in the training set. Reported AUC comes out several points higher
# than anything achievable on genuinely future applicants.
#
# Calibration needs its own slice, again earlier than the test set, or the
# calibrator is fitted on the same data the metrics are reported on.
# ---------------------------------------------------------------------------
def temporal_split(frame: pd.DataFrame, train_end: int, calib_end: int):
    train = frame[frame["cohort_year"] <= train_end]
    calib = frame[(frame["cohort_year"] > train_end)
                  & (frame["cohort_year"] <= calib_end)]
    test = frame[frame["cohort_year"] > calib_end]

    for name, part in (("train", train), ("calibration", calib), ("test", test)):
        if part.empty:
            raise SystemExit(f"{name} split is empty — adjust the cohort years")
        print(f"{name:12s} {len(part):>9,} loans  "
              f"{part['cohort_year'].min()}-{part['cohort_year'].max()}  "
              f"default rate {part['target'].mean():.3%}")
    return train, calib, test


# %%
def ks_statistic(y_true, scores) -> float:
    order = np.argsort(scores)
    y = np.asarray(y_true)[order]
    positives = np.cumsum(y) / max(y.sum(), 1)
    negatives = np.cumsum(1 - y) / max((1 - y).sum(), 1)
    return float(np.max(np.abs(positives - negatives)))


def decile_lift(y_true, scores) -> pd.DataFrame:
    frame = pd.DataFrame({"y": np.asarray(y_true), "pd": np.asarray(scores)})
    frame["decile"] = pd.qcut(frame["pd"], 10, labels=False, duplicates="drop")
    table = frame.groupby("decile").agg(
        loans=("y", "size"),
        defaults=("y", "sum"),
        default_rate=("y", "mean"),
        mean_pd=("pd", "mean"),
    )
    base = frame["y"].mean()
    table["lift"] = table["default_rate"] / base
    return table


def report(name: str, y_true, scores) -> dict:
    auc = roc_auc_score(y_true, scores)
    ks = ks_statistic(y_true, scores)
    metrics = {"auc": float(auc), "gini": float(2 * auc - 1), "ks": ks,
               "base_rate": float(np.mean(y_true)), "n": int(len(y_true))}
    print(f"\n{name}: AUC {auc:.4f} · Gini {2 * auc - 1:.4f} · KS {ks:.4f} "
          f"· base rate {np.mean(y_true):.3%} · n={len(y_true):,}")
    print(decile_lift(y_true, scores).to_string(
        float_format=lambda v: f"{v:.4f}"))
    return metrics


# %%
def build_models(train_y: np.ndarray) -> dict:
    """Class imbalance is handled with class weights, never by oversampling.

    Oversampling the minority class before splitting duplicates rows across
    train and validation, which inflates every metric. Doing it after the
    split still distorts the predicted probabilities, and this app displays
    a probability — so weights, which leave the ranking intact and let the
    calibrator fix the level.
    """
    return {
        "logistic": Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("model", LogisticRegression(
                penalty="l2", C=0.5, max_iter=2000, solver="lbfgs",
                class_weight="balanced",
            )),
        ]),
        "gbm": Pipeline([
            # HistGradientBoosting handles NaN natively, but the app's
            # feature validation refuses to serve a missing value anyway, so
            # the imputer keeps train and serve consistent.
            ("impute", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
                min_samples_leaf=100, l2_regularization=1.0,
                class_weight="balanced", random_state=7,
            )),
        ]),
    }


# %%
def plot_calibration(y_true, raw, calibrated, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    for scores, label, colour in ((raw, "uncalibrated", "#4f6d7a"),
                                  (calibrated, "calibrated", "#a61b1b")):
        true_frac, pred_frac = calibration_curve(y_true, scores, n_bins=15,
                                                 strategy="quantile")
        ax.plot(pred_frac, true_frac, "o-", label=label, color=colour)
    ax.plot([0, 1], [0, 1], "--", color="#999999", label="perfect")
    upper = max(max(calibrated), max(raw)) * 1.05
    ax.set_xlim(0, upper)
    ax.set_ylim(0, upper)
    ax.set_xlabel("predicted PD")
    ax.set_ylabel("observed default rate")
    ax.set_title(f"Calibration — {name}")
    ax.legend()
    fig.tight_layout()
    path = FIGS / f"calibration_{name}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# %%
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-end", type=int, default=2014,
                    help="last origination year in the training set")
    ap.add_argument("--calib-end", type=int, default=2015,
                    help="last origination year in the calibration set")
    ap.add_argument("--choose", choices=["logistic", "gbm"], default="gbm",
                    help="which model to ship; both are always evaluated")
    ap.add_argument("--dataset-label", default="lendingclub",
                    help="goes into the model version string and the model "
                         "card. Pass 'synthetic' when training on "
                         "notebooks/00_synthetic_data.py output, so a "
                         "circular model can never be mistaken for a real "
                         "one on the result screen.")
    args = ap.parse_args()

    frame = load_frame(WORK / "features")
    stats = json.loads((WORK / "standardisation.json").read_text())

    print("temporal split:")
    train, calib, test = temporal_split(frame, args.train_end, args.calib_end)

    def standardise(part: pd.DataFrame) -> np.ndarray:
        return np.column_stack([
            (part[name].astype(float) - stats[name]["center"]) / stats[name]["scale"]
            for name in FEATURE_NAMES
        ])

    x_train, y_train = standardise(train), train["target"].to_numpy()
    x_calib, y_calib = standardise(calib), calib["target"].to_numpy()
    x_test, y_test = standardise(test), test["target"].to_numpy()

    results = {}
    fitted = {}
    for name, pipeline in build_models(y_train).items():
        print(f"\n{'=' * 62}\nfitting {name}")
        pipeline.fit(x_train, y_train)
        fitted[name] = pipeline
        raw_test = pipeline.predict_proba(x_test)[:, 1]
        results[name] = {
            "test_uncalibrated": report(f"{name} — test, uncalibrated",
                                        y_test, raw_test),
        }

    # -- the trade-off, measured rather than assumed --------------------
    print(f"\n{'=' * 62}\nbaseline vs boosted on the held-out test cohort")
    for name in results:
        m = results[name]["test_uncalibrated"]
        print(f"  {name:10s} AUC {m['auc']:.4f}  Gini {m['gini']:.4f}  "
              f"KS {m['ks']:.4f}")
    print(
        "\nIf the gap is under ~0.01 AUC, ship the logistic model: an "
        "interpretable scorecard whose coefficients can be read off and "
        "argued with is worth more here than a marginal ranking gain, and "
        "the app's attribution is then exact rather than approximated."
    )

    chosen_name = args.choose
    chosen = fitted[chosen_name]

    # -- calibration on the held-out middle slice ------------------------
    raw_calib = chosen.predict_proba(x_calib)[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip", increasing=True)
    calibrator.fit(raw_calib, y_calib)

    raw_test = chosen.predict_proba(x_test)[:, 1]
    calibrated_test = calibrator.predict(raw_test)

    final = report(f"{chosen_name} — test, CALIBRATED", y_test, calibrated_test)
    figure = plot_calibration(y_test, raw_test, calibrated_test, chosen_name)
    print(f"\ncalibration plot: {figure}")

    mean_predicted = float(np.mean(calibrated_test))
    observed = float(np.mean(y_test))
    print(f"mean predicted PD {mean_predicted:.3%} vs observed "
          f"{observed:.3%} — the gap is the calibration error in aggregate")

    artefact = {
        "model_version": (
            f"{chosen_name}-{args.dataset_label}-{date.today():%Y.%m}.1"
        ),
        "trained_at": date.today().isoformat(),
        "dataset": (
            ("SYNTHETIC — generated by notebooks/00_synthetic_data.py, "
             "relationships are invented and the model is circular. "
             if args.dataset_label == "synthetic" else
             "Lending Club accepted loans, ")
            + f"origination cohorts "
            f"{train['cohort_year'].min()}-{test['cohort_year'].max()}; "
            f"train {len(train):,} / calib {len(calib):,} / test {len(test):,}; "
            f"observed default rate {frame['target'].mean():.3%}"
        ),
        "kind": "logistic" if chosen_name == "logistic" else "gbm",
        "feature_names": list(FEATURE_NAMES),
        "standardisation": {
            n: [stats[n]["center"], stats[n]["scale"]] for n in FEATURE_NAMES
        },
        "support": {n: stats[n]["support"] for n in FEATURE_NAMES},
        "estimator": chosen,
        "calibrator": calibrator,
        "metrics": {
            "chosen": chosen_name,
            "test_calibrated": final,
            "comparison": {k: v["test_uncalibrated"] for k, v in results.items()},
            "mean_predicted_pd": mean_predicted,
            "observed_default_rate": observed,
            "split": {
                "train_end": args.train_end,
                "calib_end": args.calib_end,
                "method": "temporal by origination year",
            },
        },
        # Synthetic training is not a trained model in any meaningful sense,
        # so it keeps the placeholder banner on in the UI.
        "is_placeholder": args.dataset_label == "synthetic",
    }
    joblib.dump(artefact, ARTEFACT)
    print(f"\nwrote {ARTEFACT}")
    print("\nNext: python notebooks/06_bands.py, then "
          "python notebooks/07_population_shift.py")


if __name__ == "__main__":
    main()
