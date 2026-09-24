from pathlib import Path

import pytest
import shapely
import yaml
from omegaconf.errors import ConfigKeyError

from gfetch import countries as countries_module
from gfetch.cli.gedi_config import GediL2AConfig, load


def _write_config(path: Path, **overrides: object) -> Path:
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
        "output": str(path.parent / "l2a.parquet"),
        **overrides,
    }
    path.write_text(yaml.dump(config_dict))
    return path


def test_load_minimal_config(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml")
    cfg = load(config_path, "l2a")

    assert isinstance(cfg, GediL2AConfig)
    assert cfg.resolved_aoi.bbox == (2.2, 48.7, 2.5, 49.0)
    assert cfg.output == (tmp_path / "l2a.parquet").expanduser().absolute()
    assert cfg.time_range is None
    assert cfg.fields is None
    assert cfg.anc_fields is None
    assert cfg.rh_percentiles is None


def test_load_overrides(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path / "config.yaml",
        time_range={"start": "2020-01-01", "end": "2020-06-01"},
        fields=["beam", "orbit"],
        anc_fields=["quality_flag", "rh"],
        rh_percentiles=[0, 50, 100],
    )
    cfg = load(config_path, "l2a")

    assert isinstance(cfg, GediL2AConfig)
    assert cfg.time_range is not None
    assert cfg.time_range.start == "2020-01-01"
    assert cfg.time_range.end == "2020-06-01"
    assert cfg.fields == ["beam", "orbit"]
    assert cfg.anc_fields == ["quality_flag", "rh"]
    assert cfg.rh_percentiles == [0, 50, 100]


def test_output_defaults_from_generic_output_dir(tmp_path: Path) -> None:
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
        "output_dir": str(tmp_path),
        "gedi_l2a": {"fields": ["beam"]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    cfg = load(config_path, "l2a")

    assert cfg.output == (tmp_path / "gedi" / "l2a.parquet").expanduser().absolute()
    assert cfg.fields == ["beam"]


def test_raster_only_generic_fields_are_ignored_not_an_error(tmp_path: Path) -> None:
    """A unified config's generic section legitimately carries raster-only fields
    (e.g. `n_workers`) for `s1`/`s2` sections to use - `gedi:`'s loader must ignore
    them rather than crash on `GediConfig`'s strict structured-merge schema.
    """
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
        "output_dir": str(tmp_path),
        "n_workers": 8,
        "resolution": 10.0,
        "bands": ["red"],
        "gedi_l2a": {"fields": ["beam"]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    cfg = load(config_path, "l2a")

    assert cfg.fields == ["beam"]
    assert cfg.output == (tmp_path / "gedi" / "l2a.parquet").expanduser().absolute()


def test_output_explicit_in_gedi_section_wins_over_output_dir_default(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.parquet"
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
        "output_dir": str(tmp_path),
        "gedi_l2a": {"output": str(explicit)},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    cfg = load(config_path, "l2a")

    assert cfg.output == explicit.expanduser().absolute()


def test_countries_resolves_aoi_to_union_bbox(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        countries_module,
        "resolve_country_polygon",
        lambda names: shapely.box(29.0, -27.0, 41.0, -1.0),
    )
    config_path = _write_config(tmp_path / "config.yaml", aoi=None, countries=["Mozambique"])

    cfg = load(config_path, "l2a")

    assert cfg.resolved_aoi.bbox == (29.0, -27.0, 41.0, -1.0)


def test_both_aoi_and_countries_raises(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", countries=["Mozambique"])

    with pytest.raises(ValueError, match="exactly one"):
        load(config_path, "l2a")


def test_l4a_reads_its_own_section_and_ignores_l2a_only_fields(tmp_path: Path) -> None:
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.5, "top": 49.0},
        "output_dir": str(tmp_path),
        "rh_percentiles": [98],
        "gedi_l2a": {"anc_fields": ["quality_flag"]},
        "gedi_l4a": {"anc_fields": ["agbd_se"]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))

    cfg = load(config_path, "l4a")

    assert not isinstance(cfg, GediL2AConfig)
    assert cfg.anc_fields == ["agbd_se"]
    assert cfg.output == (tmp_path / "gedi" / "l4a.parquet").expanduser().absolute()


def test_l4a_section_rejects_rh_percentiles(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", gedi_l4a={"rh_percentiles": [98]})

    with pytest.raises(ConfigKeyError, match="rh_percentiles"):
        load(config_path, "l4a")


def test_bare_gedi_section_raises(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "config.yaml", gedi={"fields": ["beam"]})

    with pytest.raises(ValueError, match="gedi_l2a"):
        load(config_path, "l2a")
