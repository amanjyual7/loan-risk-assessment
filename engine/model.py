"""Model wrapper.

The artefact is trained offline in notebooks/ and committed. Nothing here
trains, fits, or touches the network. Loading is the only I/O, and the app
wraps it in @st.cache_resource.

Artefact contract (dict, joblib-serialised):
    model_version      str, e.g. "lr-lendingclub-2026.09.1"
    trained_at         ISO date
    dataset            free text, goes in the model card
    kind               "logistic" | "gbm"
    feature_names      list[str], must equal engine.features.FEATURE_NAMES
    standardisation    {name: [center, scale]}
    estimator          fitted sklearn estimator
    calibrator         fitted isotonic/Platt calibrator, or None
    metrics            {auc, ks, gini, base_rate, ...}
    is_placeholder     bool — drives a warning banner in the UI
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np

from engine.features import FEATURE_NAMES, SPEC_BY_NAME, validate_features


class ModelArtefactError(RuntimeError):
    pass


@dataclass(frozen=True)
class Prediction:
    pd: float
    raw_pd: float
    log_odds: float
    base_log_odds: float
    contributions: dict[str, float]  # log-odds delta per feature
    contribution_method: str


class PDModel:
    def __init__(
        self,
        artefact: dict[str, Any],
        population_shift: float = 0.0,
        pd_floor: float = 0.0,
        pd_ceiling: float = 1.0,
    ):
        for key in ("model_version", "kind", "feature_names", "standardisation",
                    "estimator"):
            if key not in artefact:
                raise ModelArtefactError(f"artefact missing {key!r}")
        if tuple(artefact["feature_names"]) != FEATURE_NAMES:
            raise ModelArtefactError(
                "artefact feature schema does not match engine.features.\n"
                f"  artefact: {tuple(artefact['feature_names'])}\n"
                f"  engine:   {FEATURE_NAMES}"
            )
        self._a = artefact
        self.population_shift = float(population_shift)
        self.pd_floor = float(pd_floor)
        self.pd_ceiling = float(pd_ceiling)
        self.version: str = artefact["model_version"]
        self.kind: str = artefact["kind"]
        self.is_placeholder: bool = bool(artefact.get("is_placeholder", False))
        self.metrics: dict[str, Any] = artefact.get("metrics", {})
        self.dataset: str = artefact.get("dataset", "unspecified")
        self.trained_at: str = artefact.get("trained_at", "unknown")

    # -- loading ------------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path, population_shift: float = 0.0,
             pd_floor: float = 0.0, pd_ceiling: float = 1.0) -> "PDModel":
        path = Path(path)
        if not path.exists():
            raise ModelArtefactError(
                f"model artefact missing at {path}. Train one with "
                "notebooks/05_train.py, or build the placeholder with "
                "python models/build_placeholder_model.py"
            )
        return cls(joblib.load(path), population_shift=population_shift,
                   pd_floor=pd_floor, pd_ceiling=pd_ceiling)

    @classmethod
    def from_config(cls, cfg: dict) -> "PDModel":
        """Load the artefact named in config, with the configured
        population shift applied. This is the only loader the app uses."""
        from engine.config import resolve

        cal = cfg.get("calibration", {})
        return cls.load(
            resolve(cfg, "model_artefact"),
            population_shift=float(cal.get("population_shift_log_odds", 0.0)),
            pd_floor=float(cal.get("pd_floor", 0.0)),
            pd_ceiling=float(cal.get("pd_ceiling", 1.0)),
        )

    # -- transform ----------------------------------------------------------
    def _standardise(self, features: dict[str, float]) -> np.ndarray:
        std = self._a["standardisation"]
        row = []
        for name in FEATURE_NAMES:
            center, scale = std[name]
            scale = scale or 1.0
            row.append((float(features[name]) - float(center)) / float(scale))
        return np.asarray([row], dtype=float)

    # -- predict ------------------------------------------------------------
    def predict(self, features: dict[str, float]) -> Prediction:
        validate_features(features)
        z = self._standardise(features)
        est = self._a["estimator"]

        unshifted = float(est.predict_proba(z)[0, 1])
        unshifted = min(max(unshifted, 1e-9), 1 - 1e-9)

        # Intercept recalibration for the population transfer. A constant in
        # log-odds space, so ranking and monotonicity are untouched.
        log_odds = math.log(unshifted / (1 - unshifted)) + self.population_shift
        raw = 1.0 / (1.0 + math.exp(-log_odds))
        raw = min(max(raw, 1e-6), 1 - 1e-6)

        cal = self._a.get("calibrator")
        if cal is not None:
            pd_hat = float(np.asarray(cal.predict([raw])).ravel()[0])
        else:
            pd_hat = raw
        # Clamp to the range the calibration data supports. Isotonic hands
        # back a hard 0 in its lowest bin; displaying that as "0.0%" claims
        # certainty the data cannot carry.
        pd_hat = min(max(pd_hat, self.pd_floor, 1e-6),
                     self.pd_ceiling, 1 - 1e-6)

        contribs, base_lo, method = self._contributions(z, est, features)
        return Prediction(
            pd=pd_hat,
            raw_pd=raw,
            log_odds=math.log(raw / (1 - raw)),
            base_log_odds=base_lo,
            contributions=contribs,
            contribution_method=method,
        )

    @staticmethod
    def _final_step(est: Any) -> Any:
        """Unwrap an sklearn Pipeline to its final estimator.

        The training script ships a Pipeline (imputer + model); the
        placeholder ships a bare estimator. Attribution needs the actual
        model in both cases.
        """
        steps = getattr(est, "steps", None)
        return steps[-1][1] if steps else est

    def _contributions(
        self, z: np.ndarray, est: Any, features: dict[str, float]
    ) -> tuple[dict[str, float], float, str]:
        """Per-applicant attribution in log-odds.

        For the logistic baseline this is exact: coefficient × standardised
        value. No approximation, no SHAP dependency. For the boosted model
        we use TreeSHAP when available.
        """
        inner = self._final_step(est)

        if self.kind == "logistic" and hasattr(inner, "coef_"):
            coef = np.asarray(inner.coef_).ravel()
            contribs = {
                name: float(coef[i] * z[0, i]) for i, name in enumerate(FEATURE_NAMES)
            }
            base = float(np.asarray(inner.intercept_).ravel()[0])
            base += self.population_shift
            return contribs, base, "exact logistic coefficient × standardised value"

        try:  # pragma: no cover - depends on optional dependency
            import shap

            explainer = shap.TreeExplainer(inner)
            values = explainer(z)
            arr = np.asarray(values.values).reshape(-1)
            base = float(np.asarray(values.base_values).ravel()[0])
            contribs = {
                name: float(arr[i]) for i, name in enumerate(FEATURE_NAMES)
            }
            return contribs, base + self.population_shift, "TreeSHAP"
        except Exception:  # pragma: no cover
            return {}, 0.0, "unavailable (install shap for boosted-model attribution)"

    # -- introspection ------------------------------------------------------
    def card(self) -> dict[str, Any]:
        return {
            "model_version": self.version,
            "kind": self.kind,
            "dataset": self.dataset,
            "trained_at": self.trained_at,
            "is_placeholder": self.is_placeholder,
            "metrics": self.metrics,
            "features": list(FEATURE_NAMES),
        }


def plain_language(feature: str, value: float) -> str:
    """Render a feature name as something a human reads, per the spec's
    'plain language rather than raw feature names'."""
    spec = SPEC_BY_NAME.get(feature)
    label = spec.plain if spec else feature
    if spec and spec.is_indicator:
        return label if value >= 0.5 else f"not {label[0].lower() + label[1:]}"
    return label
