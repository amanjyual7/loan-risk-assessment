"""Pincode reference data tests."""

from __future__ import annotations

from tests.context import cfg, reference, score

from engine.pincode import is_well_formed


def test_known_pincodes_resolve_to_expected_tier():
    ref = reference()
    c = cfg()["pincode"]
    expected = {400001: "Metro", 110001: "Metro", 504273: "Tier 3"}
    for pin, tier in expected.items():
        loc = ref.lookup(pin, default_tier=c["default_tier_on_miss"],
                         miss_label=c["miss_label"])
        assert loc.tier == tier, f"{pin} resolved to {loc.tier}, expected {tier}"
        assert not loc.assumed
        assert loc.state and loc.city and loc.region


def test_malformed_pincodes_are_rejected_not_looked_up():
    # 999999: first digit 9 is Army Postal Service, absent from civilian data.
    # 012345: leading zero is unassigned.
    for bad in ("999999", "012345", "12345", "1234567", "abcdef", ""):
        assert not is_well_formed(bad), f"{bad!r} should be malformed"

    ref = reference()
    c = cfg()["pincode"]
    for bad in ("999999", "012345"):
        try:
            ref.lookup(bad, default_tier=c["default_tier_on_miss"],
                       miss_label=c["miss_label"])
        except ValueError:
            continue
        raise AssertionError(f"{bad} was looked up instead of rejected")


def test_valid_but_absent_pincode_takes_the_manual_path_and_is_flagged():
    ref = reference()
    c = cfg()["pincode"]
    absent = 123456
    assert absent not in {400001, 110001, 504273}

    loc = ref.lookup(absent, default_tier=c["default_tier_on_miss"],
                     miss_label=c["miss_label"], manual_state="Karnataka")
    assert loc.assumed
    assert loc.tier == c["default_tier_on_miss"]
    assert loc.assumption_note == c["miss_label"]
    assert loc.state == "Karnataka"   # the manually picked state survives
    assert loc.city is None           # never guessed

    # and the assumption reaches the result screen
    from engine.profiles import record_from_dict
    import dataclasses

    from tests.context import profiles

    base = profiles()["clean_prime"]
    moved = dataclasses.replace(
        base, identity=dataclasses.replace(base.identity, pincode=absent),
        manual_state="Karnataka",
    )
    result = score(moved)
    assert result.location.assumed
    assert any("not in the reference data" in cav for cav in result.caveats)


def test_unique_pincode_index_holds_after_load():
    ref = reference()
    ref.assert_unique_index()
    assert len(ref) > 0


def test_tier_is_an_ordered_categorical_not_a_string_match():
    scale = reference().scale
    assert scale.rank("Metro") < scale.rank("Tier 1") < scale.rank("Tier 3")
    assert scale.is_at_least("Metro", "Tier 2")
    assert not scale.is_at_least("Tier 3", "Tier 1")
    # normalisation folds the usual spelling variants
    assert scale.normalise("tier 2") == "Tier 2"
    assert scale.normalise("METRO") == "Metro"
