import logging
from pathlib import Path

import pystac

from gfetch.cli.config import load
from gfetch.ocm import ocm_path, set_num_threads, write_ocm
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

    with (
        count_bar() as progress,
        temporary_task(progress, "Computing OmniCloudMask", total=len(my_items)) as task,
    ):
        for item in my_items:
            path = ocm_path(cfg.cache_dir, item.id)
            if path.exists():
                log.debug(f"{item.id}: mask already computed, skipping")
            else:
                write_ocm(item, path, model_dir=cfg.ocm_model_path, device=device)
            progress.advance(task)
    log.info(f"OmniCloudMask masks ready for {len(my_items)} item(s)")
