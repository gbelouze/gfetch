import pytest

from gfetch.profiles import PROFILES, get_profile


def test_get_profile_known() -> None:
    profile = get_profile("sentinel-2")
    assert profile.name == "sentinel-2"
    assert profile.default_bands
    assert profile.cloud_mask_band == "scl"
    assert 9 in profile.cloud_mask_out  # cloud high probability


def test_get_profile_sentinel1() -> None:
    profile = get_profile("sentinel-1")
    assert profile.name == "sentinel-1"
    assert profile.default_bands == ("vv", "vh")
    assert profile.cloud_mask_band is None


def test_get_profile_unknown() -> None:
    with pytest.raises(ValueError, match="Unknown satellite"):
        get_profile("not-a-real-satellite")


def test_all_profiles_have_a_matching_source_collection() -> None:
    from gfetch.sources import SOURCES

    for profile in PROFILES.values():
        for source in SOURCES.values():
            # Should not raise.
            source.collection(profile.name)
