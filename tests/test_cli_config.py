from pathlib import Path

import pytest
import shapely
import yaml
from odc.geo.crs import CRS

from gfetch import countries as countries_module
from gfetch.cli.config import (
    load,
    resolve_bands,
    resolve_cloud_mask,
    resolve_compute_workers,
    resolve_output_variables,
    resolve_source,
)


def _write_config(path: Path, **overrides: object) -> Path:
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
        "time_range": {"start": "2024-01-01", "end": "2024-06-01"},
        "output_dir": str(path.parent),
        **overrides,
    }
    path.write_text(yaml.dump(config_dict))
    return path


def test_load_minimal_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    cfg = load(config_path, "s2")

    assert cfg.resolved_aoi.bbox == (2.2, 48.7, 2.5, 49.0)
    assert cfg.time_range.datetime == "2024-01-01/2024-06-01"
    assert cfg.satellite == "sentinel-2"
    assert cfg.source is None
    assert resolve_source(cfg) == "earthsearch"
    assert cfg.bands is None
    assert cfg.resampling == {}
    assert cfg.orbit_state is None
    assert cfg.output_dir == (tmp_path / "s2").expanduser().absolute()
    assert cfg.shard_factor is None


def test_config_derived_paths(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    cfg = load(config_path, "s2")

    assert cfg.cache_dir == cfg.output_dir / "cache"
    assert cfg.items_path == cfg.output_dir / "items.json"
    assert cfg.cached_items_path == cfg.output_dir / "cached_items.json"
    assert cfg.zarr_path(CRS("EPSG:32736")) == cfg.output_dir / "mosaic_epsg32736.zarr"


def test_satellite_section_overrides_generic(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        n_workers=4,
        s2={
            "source": "planetary-computer",
            "bands": ["red", "green"],
            "max_cloud_cover": 20.0,
            "n_workers": 8,
            "resampling": {"*": "bilinear", "scl": "nearest"},
            "orbit_state": "ascending",
            "shard_factor": 16,
        },
    )
    cfg = load(config_path, "s2")

    assert cfg.source == "planetary-computer"
    assert cfg.bands == ["red", "green"]
    assert cfg.max_cloud_cover == 20.0
    assert cfg.n_workers == 8
    assert cfg.resampling == {"*": "bilinear", "scl": "nearest"}
    assert cfg.orbit_state == "ascending"
    assert cfg.shard_factor == 16


def test_generic_fields_apply_when_section_does_not_override(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", n_workers=16, s2={})
    cfg = load(config_path, "s2")

    assert cfg.n_workers == 16


def test_satellite_field_is_force_set_for_builtins(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", s1={"satellite": "not-a-real-satellite"})
    cfg = load(config_path, "s1")

    assert cfg.satellite == "sentinel-1"


def test_output_dir_defaults_to_satellite_subfolder(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", s2={})
    cfg = load(config_path, "s2")

    assert cfg.output_dir == (tmp_path / "s2").expanduser().absolute()


def test_output_dir_explicit_override_wins(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit_s2_dir"
    config_path = _write_config(tmp_path / "config.yaml", s2={"output_dir": str(explicit)})
    cfg = load(config_path, "s2")

    assert cfg.output_dir == explicit.expanduser().absolute()


def test_custom_satellite_requires_collection_and_bands(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml", custom={"landsat8": {"source": "earthsearch"}}
    )

    with pytest.raises(ValueError, match="collection"):
        load(config_path, "landsat8")


def test_custom_satellite_loads_with_defaults(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        custom={
            "landsat8": {
                "source": "earthsearch",
                "collection": "landsat-c2-l2",
                "bands": ["red", "green", "blue"],
            }
        },
    )
    cfg = load(config_path, "landsat8")

    assert cfg.satellite == "landsat8"
    assert cfg.source == "earthsearch"
    assert cfg.collection == "landsat-c2-l2"
    assert cfg.bands == ["red", "green", "blue"]
    assert cfg.output_dir == (tmp_path / "landsat8").expanduser().absolute()


def test_custom_satellite_can_override_satellite_name(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        custom={
            "landsat8": {
                "satellite": "landsat-8",
                "source": "earthsearch",
                "collection": "landsat-c2-l2",
                "bands": ["red"],
            }
        },
    )
    cfg = load(config_path, "landsat8")

    assert cfg.satellite == "landsat-8"


def test_unknown_satellite_key_raises(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")

    with pytest.raises(ValueError, match="s3"):
        load(config_path, "s3")


def test_gedi_only_generic_fields_are_ignored_not_an_error(tmp_path: Path) -> None:
    """A unified config's generic section legitimately carries GEDI-only fields
    (e.g. `rh_percentiles`) for the `gedi:` section to use - `Config`'s loader must
    ignore them rather than crash on its own strict structured-merge schema.
    """
    config_path = _write_config(
        tmp_path / "config.yaml", rh_percentiles=[0, 50, 100], anc_fields=["quality_flag"]
    )

    cfg = load(config_path, "s2")

    assert cfg.satellite == "sentinel-2"


def test_countries_resolves_aoi_to_union_bbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        countries_module,
        "resolve_country_polygon",
        lambda names: shapely.box(29.0, -27.0, 41.0, -1.0),
    )
    config_path = _write_config(
        tmp_path / "config.yaml",
        aoi=None,
        countries=["Mozambique", "United Republic of Tanzania"],
    )

    cfg = load(config_path, "s2")

    assert cfg.resolved_aoi.bbox == (29.0, -27.0, 41.0, -1.0)


def test_aoi_geometry_is_the_country_polygon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    polygon = shapely.Polygon([(29.0, -27.0), (41.0, -27.0), (29.0, -1.0)])
    monkeypatch.setattr(countries_module, "resolve_country_polygon", lambda names: polygon)
    config_path = _write_config(tmp_path / "config.yaml", aoi=None, countries=["Mozambique"])

    cfg = load(config_path, "s2")

    assert cfg.aoi_geometry is polygon


def test_aoi_geometry_is_the_aoi_box(tmp_path: Path) -> None:
    cfg = load(_write_config(tmp_path / "config.yaml"), "s2")

    assert cfg.aoi_geometry.equals(shapely.box(2.2, 48.7, 2.5, 49.0))


def test_both_aoi_and_countries_raises(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", countries=["Mozambique"])

    with pytest.raises(ValueError, match="exactly one"):
        load(config_path, "s2")


def test_neither_aoi_nor_countries_raises(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", aoi=None)

    with pytest.raises(ValueError, match="exactly one"):
        load(config_path, "s2")


def test_resolve_bands_and_cloud_mask_fall_back_to_profile(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    cfg = load(config_path, "s2")

    assert resolve_bands(cfg) == ["red", "green", "blue"]
    assert resolve_cloud_mask(cfg) == ("scl", frozenset({0, 1, 3, 7, 8, 9, 10}))


def test_resolve_bands_and_cloud_mask_config_overrides_profile(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        s2={"bands": ["red"], "cloud_mask_band": "custom_scl", "cloud_mask_out": [1, 2]},
    )
    cfg = load(config_path, "s2")

    assert resolve_bands(cfg) == ["red"]
    assert resolve_cloud_mask(cfg) == ("custom_scl", frozenset({1, 2}))


def test_resolve_bands_and_cloud_mask_custom_satellite_has_no_profile_fallback(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        custom={
            "landsat8": {
                "source": "earthsearch",
                "collection": "landsat-c2-l2",
                "bands": ["red", "green"],
            }
        },
    )
    cfg = load(config_path, "landsat8")

    assert resolve_bands(cfg) == ["red", "green"]
    assert resolve_cloud_mask(cfg) == (None, frozenset())


@pytest.mark.parametrize(("value", "expected"), [(8, (8, 8)), ([4, 16], (4, 16)), ([4, 4], (4, 4))])
def test_resolve_compute_workers(tmp_path: Path, value: object, expected: tuple) -> None:
    cfg = load(_write_config(tmp_path / "config.yaml", n_compute_workers=value), "s2")

    assert resolve_compute_workers(cfg) == expected


def test_resolve_compute_workers_defaults_to_available_cpus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("gfetch.cli.config.available_cpus", lambda: 7)
    cfg = load(_write_config(tmp_path / "config.yaml"), "s2")

    assert resolve_compute_workers(cfg) == (7, 7)


@pytest.mark.parametrize("value", [0, [16, 4], [1, 2, 3], "8"])
def test_resolve_compute_workers_rejects_invalid(tmp_path: Path, value: object) -> None:
    cfg = load(_write_config(tmp_path / "config.yaml", n_compute_workers=value), "s2")

    with pytest.raises(ValueError, match="n_compute_workers"):
        resolve_compute_workers(cfg)


def test_resolve_output_variables_defaults_to_bands(tmp_path: Path) -> None:
    cfg = load(_write_config(tmp_path / "config.yaml", orbit_state="ascending"), "s1")
    assert resolve_output_variables(cfg) == ["vv", "vh"]


def test_resolve_output_variables_as_bands_splits_by_orbit_state(tmp_path: Path) -> None:
    cfg = load(_write_config(tmp_path / "config.yaml", orbit_state="as_bands"), "s1")
    assert resolve_bands(cfg) == ["vv", "vh"]
    assert resolve_output_variables(cfg) == [
        "vv_ascending",
        "vv_descending",
        "vh_ascending",
        "vh_descending",
    ]


@pytest.mark.parametrize(
    ("satellite_key", "overrides", "expected"),
    [
        ("s1", {}, "planetary-computer"),
        ("s2", {}, "earthsearch"),
        ("s1", {"source": "earthsearch"}, "earthsearch"),
        ("s1", {"s1": {"source": "earthsearch"}}, "earthsearch"),
        ("x", {"custom": {"x": {"collection": "c", "bands": ["b"]}}}, "earthsearch"),
    ],
)
def test_resolve_source_falls_back_to_profile_default(
    tmp_path: Path, satellite_key: str, overrides: dict, expected: str
) -> None:
    cfg = load(_write_config(tmp_path / "config.yaml", **overrides), satellite_key)

    assert resolve_source(cfg) == expected
