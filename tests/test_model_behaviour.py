"""Monotonicity and feature-schema tests.

The monotonicity test is the one that matters most. A model that raises PD
when income rises is broken in a way no AUC will reveal, and a reviewer who
knows credit will try exactly this.
"""

from __future__ import annotations

import dataclasses

from tests.context import cfg, model, profiles, score

from engine.features import (
    FEATURE_NAMES,
    FeatureValidationError,
    bridge_bureau_score,
    clamp_to_training_support,
    validate_features,
)


def _pd_for(applicant):
    r = score(applicant, with_suggestions=False)
    assert r.pd is not None
    return r.pd


def test_raising_income_never_increases_pd():
    base = profiles()["credit_hungry"]
    previous = _pd_for(base)
    for multiplier in (1.2, 1.5, 2.0, 3.0):
        raised = dataclasses.replace(
            base,
            income=dataclasses.replace(
                base.income,
                gross_monthly=base.income.gross_monthly * multiplier,
                net_monthly=base.income.net_monthly * multiplier,
            ),
        )
        current = _pd_for(raised)
        assert current <= previous + 1e-9, (
            f"PD rose from {previous:.4%} to {current:.4%} when income was "
            f"multiplied by {multiplier}"
        )
        previous = current


def test_lowering_obligations_never_increases_pd():
    """Lower DTI, same everything else."""
    base = profiles()["high_foir_clean"]
    previous = _pd_for(base)
    for factor in (0.75, 0.5, 0.25, 0.0):
        lines = tuple(
            dataclasses.replace(t, emi=t.emi * factor) for t in base.trade_lines
        )
        current = _pd_for(dataclasses.replace(base, trade_lines=lines))
        assert current <= previous + 1e-9, (
            f"PD rose from {previous:.4%} to {current:.4%} when EMIs were "
            f"scaled by {factor}"
        )
        previous = current


def test_raising_bureau_score_never_increases_pd():
    base = profiles()["recovering_delinquency"]
    previous = _pd_for(base)
    for target in (700, 750, 800, 850):
        raised = dataclasses.replace(
            base, bureau=dataclasses.replace(base.bureau, score=target)
        )
        current = _pd_for(raised)
        assert current <= previous + 1e-9, (
            f"PD rose from {previous:.4%} to {current:.4%} at score {target}"
        )
        previous = current


def test_missing_feature_raises_rather_than_imputing():
    good = {name: 0.0 for name in FEATURE_NAMES}
    good["fico_avg"] = 700
    good["log_annual_inc"] = 11.0
    good["log_loan_amnt"] = 9.5
    validate_features(good)

    for dropped in ("fico_avg", "dti", "loan_to_income"):
        broken = {k: v for k, v in good.items() if k != dropped}
        try:
            validate_features(broken)
        except FeatureValidationError:
            continue
        raise AssertionError(f"missing {dropped} was silently accepted")


def test_out_of_range_feature_raises():
    good = {name: 0.0 for name in FEATURE_NAMES}
    good["fico_avg"] = 700
    good["log_annual_inc"] = 11.0
    good["log_loan_amnt"] = 9.5

    for name, bad in (("fico_avg", 1200), ("revol_util", -5), ("dti", 9999)):
        broken = dict(good)
        broken[name] = bad
        try:
            validate_features(broken)
        except FeatureValidationError:
            continue
        raise AssertionError(f"{name}={bad} was accepted")


def test_unexpected_feature_raises():
    good = {name: 0.0 for name in FEATURE_NAMES}
    good["fico_avg"] = 700
    good["log_annual_inc"] = 11.0
    good["log_loan_amnt"] = 9.5
    good["some_new_idea"] = 1.0
    try:
        validate_features(good)
    except FeatureValidationError:
        return
    raise AssertionError("an unknown feature was accepted into the vector")


def test_clamping_to_training_support_is_reported():
    features = {name: 0.0 for name in FEATURE_NAMES}
    features["fico_avg"] = 700
    features["log_annual_inc"] = 11.0
    features["log_loan_amnt"] = 14.0  # far above Lending Club's loan range
    clamped, notes = clamp_to_training_support(features)
    assert clamped["log_loan_amnt"] < 14.0
    assert notes, "a clamp must produce a caveat, never happen silently"
    assert any("above the training range" in n for n in notes)


def test_bureau_score_bridge_is_monotone_and_bounded():
    c = cfg()
    lo, hi = c["bureau_bridge"]["target_range"]
    assert bridge_bureau_score(300, c) == lo
    assert bridge_bureau_score(900, c) == hi
    assert bridge_bureau_score(None, c) is None
    previous = -1.0
    for score_value in range(300, 901, 25):
        mapped = bridge_bureau_score(score_value, c)
        assert mapped > previous
        previous = mapped

    try:
        bridge_bureau_score(950, c)
    except FeatureValidationError:
        return
    raise AssertionError("a score above 900 was accepted")


def test_population_shift_does_not_change_ranking():
    """The shift is a constant in log-odds, so it must preserve order."""
    shifted = model()
    assert shifted.population_shift != 0.0, "expected a configured shift"
    keys = ["clean_prime", "credit_hungry", "recovering_delinquency"]
    pds = [_pd_for(profiles()[k]) for k in keys]
    assert pds == sorted(pds), f"ranking changed: {dict(zip(keys, pds))}"
