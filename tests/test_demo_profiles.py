"""One test per demo profile.

Each asserts the band AND the mechanism that produced it. Asserting the band
alone would pass for the wrong reason — a profile can land in E because the
model scored it badly or because a knockout fired, and those are different
claims about the engine.

Band expectations are tied to the config version and the model artefact. When
either changes, re-derive them rather than loosening the assertions.
"""

from __future__ import annotations

from tests.context import band_rank, profiles, score


def test_clean_prime_lands_in_the_best_band():
    r = score(profiles()["clean_prime"])
    assert r.band == "A"
    assert r.pd_is_meaningful and r.pd < 0.05
    assert not r.knockouts
    assert not r.policy_flags
    # bureau score should be the single largest downward contribution
    top = r.contributions[0]
    assert top.feature == "fico_avg" and top.log_odds_delta < 0


def test_thin_file_takes_the_fallback_path_and_is_capped():
    r = score(profiles()["thin_file"])
    assert r.thin_file
    assert r.pd is None and not r.pd_is_meaningful
    assert r.thin_file_view is not None
    assert not r.model_features, "the model must not run on a thin file"
    # never the best band: absence of adverse history is not good history
    assert band_rank(r.band) >= band_rank("C")
    assert any("not used" in c for c in r.caveats)


def test_high_foir_clean_is_caught_by_the_expense_floor_not_by_credit():
    """The profile the whole expense model exists for: spotless repayment
    record, but a household of five in a Metro on one income."""
    r = score(profiles()["high_foir_clean"])
    assert not r.knockouts

    # Asserted as RELATIVE movement, not an absolute band. Which band this
    # applicant lands in depends on the model's calibration, which changes
    # legitimately every time the model is retrained. What must not change
    # is that the credit model sees nothing wrong and the expense floor
    # overrides it.
    assert r.band_raised_by_policy, "policy should have raised the model band"
    assert band_rank(r.band) > band_rank(r.model_band)

    codes = {f.code for f in r.policy_flags}
    assert "negative_surplus" in codes

    surplus = next(m for m in r.metrics if m.key == "surplus")
    foir = next(m for m in r.metrics if m.key == "foir")
    assert surplus.value < 0
    # the point: FOIR alone would not have condemned this applicant as hard
    assert foir.value is not None


def test_recovering_delinquency_is_graded_not_knocked_out():
    r = score(profiles()["recovering_delinquency"])
    assert not r.knockouts, "a cured delinquency is not a knockout"
    codes = {f.code for f in r.policy_flags}
    assert "recovered_delinquency" in codes or "high_utilisation" in codes
    assert band_rank(r.band) >= band_rank("C")


def test_active_delinquency_hits_hard_knockouts():
    r = score(profiles()["active_delinquency"])
    assert r.knockouts, "a live 90+ DPD written-off account must knock out"
    codes = {k.code for k in r.knockouts}
    assert "severe_current_dpd" in codes
    assert "active_impaired_account" in codes or "write_off_12m" in codes
    # knockouts force the worst band regardless of the model's estimate
    assert r.band == "E"
    # and they are reported separately from graded flags
    assert all(k.is_knockout for k in r.knockouts)
    assert all(not f.is_knockout for f in r.policy_flags)


def test_credit_hungry_is_caught_by_enquiry_velocity_not_utilisation():
    r = score(profiles()["credit_hungry"])
    assert not r.knockouts
    codes = {f.code for f in r.policy_flags}
    assert "enquiry_burst" in codes
    assert "high_utilisation" not in codes, "utilisation is deliberately low here"

    # No absolute band assertion here either. The credit model has little to
    # go on for this applicant — low utilisation, nothing adverse, a decent
    # score — which is precisely why the enquiry-velocity rule exists. The
    # claim under test is that the rule fires and moves the band, not where
    # the band ends up.
    assert band_rank(r.band) > band_rank(r.model_band)
    enquiry_flag = next(f for f in r.policy_flags if f.code == "enquiry_burst")
    assert "enquiries in 3 months" in enquiry_flag.triggered_by


def test_every_profile_scores_without_raising():
    for key, applicant in profiles().items():
        r = score(applicant)
        assert r.band in {"A", "B", "C", "D", "E"}, key
        assert r.config_version and r.model_version, key
        assert r.assessed_at, key
