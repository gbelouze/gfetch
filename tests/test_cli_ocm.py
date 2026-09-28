import datetime
from pathlib import Path

import pystac
import yaml

from gfetch.cli.config import Config, load
from gfetch.cli.ocm import ocm as ocm_cmd
from gfetch.ocm import ocm_path


def _write_config(path: Path, **s2: object) -> Path:
    config_dict = {
        "aoi": {"left": 2.2, "bottom": 48.7, "right": 2.21, "top": 48.71},
        "time_range": {"start": "2024-01-01", "end": "2024-06-01"},
        "output_dir": str(path.parent),
        "s2": {"ocm": True, "bands": ["red", "green", "nir"], **s2},
    }
    path.write_text(yaml.dump(config_dict))
    return path


def _write_cached_items(cfg: Config, ids: list[str]) -> None:
    items = [
        pystac.Item(
            id=item_id,
            geometry=None,
            bbox=None,
            datetime=datetime.datetime(2024, 3, 1, tzinfo=datetime.UTC),
            properties={},
        )
        for item_id in ids
    ]
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pystac.ItemCollection(items).save_object(str(cfg.cached_items_path))


def _fake_write_ocms(written: list[str]):
    def write(jobs: list[tuple[pystac.Item, Path]], **kwargs: object):
        for item, path in jobs:
            written.append(item.id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            yield item

    return write


def _patch_ocm(monkeypatch, written: list[str]) -> None:
    monkeypatch.setattr("gfetch.cli.ocm.write_ocms", _fake_write_ocms(written))
    monkeypatch.setattr("gfetch.cli.ocm.load_models", lambda *a, **k: [])
    monkeypatch.setattr("gfetch.cli.ocm.resolve_device", lambda device: "cpu")
    monkeypatch.setattr("gfetch.cli.ocm.set_num_threads", lambda n: None)


def test_ocm_skips_existing_masks_and_splits_items_across_tasks(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml")
    cfg = load(cfg_path, "s2")
    _write_cached_items(cfg, ["a", "b", "c", "d"])
    done = ocm_path(cfg.cache_dir, "c")
    done.parent.mkdir(parents=True)
    done.touch()
    written: list[str] = []
    _patch_ocm(monkeypatch, written)

    ocm_cmd(cfg_path, "s2", task_id=0, n_tasks=2)
    assert written == ["a"]

    ocm_cmd(cfg_path, "s2", task_id=1, n_tasks=2)
    assert written == ["a", "b", "d"]


def test_ocm_does_nothing_unless_enabled(tmp_path: Path, monkeypatch) -> None:
    cfg_path = _write_config(tmp_path / "config.yaml", ocm=False)
    _write_cached_items(load(cfg_path, "s2"), ["a"])
    written: list[str] = []
    _patch_ocm(monkeypatch, written)

    ocm_cmd(cfg_path, "s2")

    assert written == []
