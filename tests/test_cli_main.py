from pathlib import Path

import pytest

from gfetch.cli.main import app


def test_builtin_satellite_commands_dispatch_with_correct_satellite_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []

    def fake_search(config_path: Path, satellite_key: str) -> None:
        calls.append(("search", config_path, satellite_key))

    def fake_download(config_path: Path, satellite_key: str) -> None:
        calls.append(("download", config_path, satellite_key))

    def fake_mosaic(
        config_path: Path, satellite_key: str, *, task_id: int = 0, n_tasks: int = 1
    ) -> None:
        calls.append(("mosaic", config_path, satellite_key, task_id, n_tasks))

    monkeypatch.setattr("gfetch.cli.search.search", fake_search)
    monkeypatch.setattr("gfetch.cli.download.download", fake_download)
    monkeypatch.setattr("gfetch.cli.mosaic.mosaic", fake_mosaic)

    app(["s1", "search", "config.yaml"], result_action="return_value")
    app(["s2", "download", "config.yaml"], result_action="return_value")
    app(
        ["s2", "mosaic", "config.yaml", "--task-id", "1", "--n-tasks", "2"],
        result_action="return_value",
    )

    assert calls == [
        ("search", Path("config.yaml"), "s1"),
        ("download", Path("config.yaml"), "s2"),
        ("mosaic", Path("config.yaml"), "s2", 1, 2),
    ]


def test_custom_satellite_commands_pass_through_name(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def fake_search(config_path: Path, satellite_key: str) -> None:
        calls.append(("search", config_path, satellite_key))

    def fake_download(config_path: Path, satellite_key: str) -> None:
        calls.append(("download", config_path, satellite_key))

    def fake_mosaic(
        config_path: Path, satellite_key: str, *, task_id: int = 0, n_tasks: int = 1
    ) -> None:
        calls.append(("mosaic", config_path, satellite_key, task_id, n_tasks))

    monkeypatch.setattr("gfetch.cli.search.search", fake_search)
    monkeypatch.setattr("gfetch.cli.download.download", fake_download)
    monkeypatch.setattr("gfetch.cli.mosaic.mosaic", fake_mosaic)

    app(["custom", "search", "landsat8", "config.yaml"], result_action="return_value")
    app(["custom", "download", "landsat8", "config.yaml"], result_action="return_value")
    app(["custom", "mosaic", "landsat8", "config.yaml"], result_action="return_value")

    assert calls == [
        ("search", Path("config.yaml"), "landsat8"),
        ("download", Path("config.yaml"), "landsat8"),
        ("mosaic", Path("config.yaml"), "landsat8", 0, 1),
    ]


def test_gedi_commands_dispatch_with_correct_product(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, str]] = []

    def fake_gedi(config_path: Path, product: str) -> None:
        calls.append((config_path, product))

    monkeypatch.setattr("gfetch.cli.gedi.gedi", fake_gedi)

    app(["gedi", "l2a", "config.yaml"], result_action="return_value")
    app(["gedi", "l4a", "config.yaml"], result_action="return_value")

    assert calls == [(Path("config.yaml"), "l2a"), (Path("config.yaml"), "l4a")]


def test_vrt_clean_and_coverage_dispatch_with_correct_satellite_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []

    def fake_vrt(config_path: Path, satellite_key: str) -> None:
        calls.append(("vrt", config_path, satellite_key))

    def fake_clean(config_path: Path, satellite_key: str) -> None:
        calls.append(("clean", config_path, satellite_key))

    def fake_coverage(config_path: Path, satellite_key: str) -> None:
        calls.append(("coverage", config_path, satellite_key))

    monkeypatch.setattr("gfetch.cli.finalize.vrt", fake_vrt)
    monkeypatch.setattr("gfetch.cli.finalize.clean", fake_clean)
    monkeypatch.setattr("gfetch.cli.coverage.coverage", fake_coverage)

    app(["s2", "vrt", "config.yaml"], result_action="return_value")
    app(["s1", "clean", "config.yaml"], result_action="return_value")
    app(["s1", "coverage", "config.yaml"], result_action="return_value")
    app(["custom", "vrt", "landsat8", "config.yaml"], result_action="return_value")
    app(["custom", "clean", "landsat8", "config.yaml"], result_action="return_value")
    app(["custom", "coverage", "landsat8", "config.yaml"], result_action="return_value")

    assert calls == [
        ("vrt", Path("config.yaml"), "s2"),
        ("clean", Path("config.yaml"), "s1"),
        ("coverage", Path("config.yaml"), "s1"),
        ("vrt", Path("config.yaml"), "landsat8"),
        ("clean", Path("config.yaml"), "landsat8"),
        ("coverage", Path("config.yaml"), "landsat8"),
    ]


def test_ocm_is_an_s2_only_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def fake_ocm(
        config_path: Path,
        satellite_key: str,
        *,
        task_id: int = 0,
        n_tasks: int = 1,
        device: str | None = None,
        dtype: str | None = None,
        batch_size: int = 8,
    ) -> None:
        calls.append((config_path, satellite_key, task_id, n_tasks, device, dtype, batch_size))

    monkeypatch.setattr("gfetch.cli.ocm.ocm", fake_ocm)

    app(
        [
            *("s2", "ocm", "config.yaml", "--task-id", "1", "--n-tasks", "2"),
            *("--device", "cuda", "--dtype", "float16", "--batch-size", "8"),
        ],
        result_action="return_value",
    )

    assert calls == [(Path("config.yaml"), "s2", 1, 2, "cuda", "float16", 8)]
    with pytest.raises(SystemExit):
        app(["s1", "ocm", "config.yaml"], result_action="return_value", exit_on_error=True)


def test_utils_rechunk_separates_bands_from_stores(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple] = []

    def fake_rechunk(stores: list[Path], **kwargs: object) -> None:
        calls.append((stores, kwargs["bands"], kwargs["chunk"]))

    monkeypatch.setattr("gfetch.cli.utils.rechunk", fake_rechunk)

    app(
        ["utils", "rechunk", "a.zarr", "b.zarr", "--bands", "red", "green", "--chunk", "128"],
        result_action="return_value",
    )

    assert calls == [([Path("a.zarr"), Path("b.zarr")], ["red", "green"], 128)]
