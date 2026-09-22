from pathlib import Path

import yaml

from gfetch.cli.gedi_config import load


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
    cfg = load(config_path)

    assert cfg.aoi.bbox == (2.2, 48.7, 2.5, 49.0)
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
    cfg = load(config_path)

    assert cfg.time_range is not None
    assert cfg.time_range.start == "2020-01-01"
    assert cfg.time_range.end == "2020-06-01"
    assert cfg.fields == ["beam", "orbit"]
    assert cfg.anc_fields == ["quality_flag", "rh"]
    assert cfg.rh_percentiles == [0, 50, 100]
