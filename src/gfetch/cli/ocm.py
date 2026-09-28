import logging
from pathlib import Path

import pystac

from gfetch.cli.config import load
from gfetch.ocm import load_models, ocm_path, resolve_device, set_num_threads, write_ocms
from gfetch.utils.progress import count_bar, temporary_task
from gfetch.utils.system import available_cpus

log = logging.getLogger(__name__)


def ocm(
    config_path: Path,
    satellite_key: str,
    *,
    task_id: int = 0,
    n_tasks: int = 1,
    device: str | None = None,
    dtype: str | None = None,
    batch_size: int = 8,
) -> None:
    """
    Compute an OmniCloudMask mask for each of this job's cached items.

    Skips items whose mask already exists, so it's safe to resume, and safe for
    several concurrent invocations to split the items via `task_id`/`n_tasks`.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    satellite_key : str
        Which satellite to load from `config_path`, see `gfetch.cli.config.load`.
    task_id : int
        This invocation's index among `n_tasks` concurrent invocations (e.g. a SLURM
        job array's `$SLURM_ARRAY_TASK_ID`). Defaults to 0.
    n_tasks : int
        Total number of concurrent invocations splitting this job's items between
        them. Defaults to 1: no splitting, this invocation does everything.
    device : str | None
        Torch device, e.g. 'cuda' or 'cpu'. Defaults to None, which uses a GPU if
        one is available.
    dtype : str | None
        Inference dtype, e.g. 'float32' or 'float16'. Defaults to None, which uses
        'float16' on a GPU and 'float32' otherwise.
    batch_size : int
        Number of 1000x1000 patches per forward pass, bounded by GPU memory.
        Defaults to 8.
    """
    cfg = load(config_path, satellite_key)
    if not cfg.ocm:
        log.error(f"`ocm` isn't enabled in {config_path}, set `ocm: true` under `s2:`.")
        return
    if not cfg.cached_items_path.exists():
        log.error(f"{cfg.cached_items_path} not found, run `gfetch s2 download` first.")
        return

    set_num_threads(available_cpus())

    items = list(pystac.ItemCollection.from_file(cfg.cached_items_path))
    my_items = items[task_id::n_tasks]
    log.info(f"Task {task_id}/{n_tasks}: {len(my_items)}/{len(items)} item(s) assigned")
    jobs = [(item, ocm_path(cfg.cache_dir, item.id)) for item in my_items]
    pending = [(item, path) for item, path in jobs if not path.exists()]
    log.info(f"{len(jobs) - len(pending)} mask(s) already computed, {len(pending)} to go")

    if pending:
        device = resolve_device(device)
        if dtype is None:
            dtype = "float16" if device.startswith("cuda") else "float32"
        log.info(f"Running OmniCloudMask on {device} in {dtype}, batch size {batch_size}")
        models = load_models(cfg.ocm_model_path, device=device, dtype=dtype)
        with (
            count_bar() as progress,
            temporary_task(progress, "Computing OmniCloudMask", total=len(pending)) as task,
        ):
            for item in write_ocms(
                pending, models=models, device=device, dtype=dtype, batch_size=batch_size
            ):
                log.debug(f"{item.id}: mask written")
                progress.advance(task)
    log.info(f"OmniCloudMask masks ready for {len(my_items)} item(s)")
