"""Satellite profiles: the good-defaults layer bundling a satellite's default bands
and cloud-mask configuration, so the common case is "give me an AOI" while every
default stays overridable.
"""

from dataclasses import dataclass

__all__ = ["PROFILES", "SatelliteProfile", "get_profile"]


@dataclass(frozen=True)
class SatelliteProfile:
    """
    Default bands and cloud-masking configuration for one satellite.

    Attributes
    ----------
    name : str
        Satellite/profile name (e.g. 'sentinel-2'), also used to resolve a
        `gfetch.sources.StacSource`'s collection id.
    default_bands : tuple[str, ...]
        Asset keys downloaded/loaded when no explicit band list is given.
    cloud_mask_band : str | None
        Asset key of the classification band used for cloud masking, or None if this
        satellite has no such band.
    cloud_mask_out : frozenset[int]
        Classification values to mask out as invalid/cloudy. Unused if
        `cloud_mask_band` is None.
    """

    name: str
    default_bands: tuple[str, ...]
    cloud_mask_band: str | None
    cloud_mask_out: frozenset[int]


# Sentinel-2 L2A Scene Classification (SCL) values: 0 no-data, 1 saturated/defective,
# 3 cloud shadow, 8/9 cloud medium/high probability, 10 thin cirrus.
_SENTINEL2_SCL_MASK_OUT = frozenset({0, 1, 3, 8, 9, 10})

PROFILES: dict[str, SatelliteProfile] = {
    "sentinel-2": SatelliteProfile(
        name="sentinel-2",
        default_bands=("red", "green", "blue"),
        cloud_mask_band="scl",
        cloud_mask_out=_SENTINEL2_SCL_MASK_OUT,
    ),
    # SAR isn't affected by clouds, so there's no cloud-mask-equivalent band.
    "sentinel-1": SatelliteProfile(
        name="sentinel-1",
        default_bands=("vv", "vh"),
        cloud_mask_band=None,
        cloud_mask_out=frozenset(),
    ),
}


def get_profile(satellite: str) -> SatelliteProfile:
    """
    Look up a registered satellite profile by name.

    Parameters
    ----------
    satellite : str
        Satellite name, one of the keys in `PROFILES`.

    Returns
    -------
    SatelliteProfile
        The corresponding profile.
    """
    try:
        return PROFILES[satellite]
    except KeyError as e:
        raise ValueError(
            f"Unknown satellite {satellite!r}; known satellites: {sorted(PROFILES)}"
        ) from e
