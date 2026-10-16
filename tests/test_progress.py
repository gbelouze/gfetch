import dask
from dask.threaded import get
from rich.progress import Progress

from gfetch.utils.progress import _RichDaskCallback


def test_dask_callback_skips_odc_open_tasks() -> None:
    dsk = {
        ("open-red-abc123", 0): (int, "1"),
        ("open-red-abc123", 1): (int, "2"),
        ("red-abc123", 0): (sum, [("open-red-abc123", 0), ("open-red-abc123", 1)]),
        "median-def456": (abs, ("red-abc123", 0)),
    }
    progress = Progress(disable=True)
    task = progress.add_task("compute", total=None)

    with dask.config.set(scheduler="threads"), _RichDaskCallback(progress, task):
        assert get(dsk, "median-def456") == 3

    (state,) = progress.tasks
    assert state.total == 2
    assert state.completed == 2
