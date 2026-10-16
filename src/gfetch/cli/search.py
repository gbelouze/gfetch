import logging
import sys
from pathlib import Path

import pystac
import shapely
from rich.prompt import Confirm

from gfetch.cli.config import ORBIT_STATE_AS_BANDS, Config, load, resolve_source
from gfetch.mosaic import ORBIT_STATES
from gfetch.search import search as search_items
from gfetch.sources import HeterogeneousFallbackError, get_source

log = logging.getLogger(__name__)


def _build_query(cfg: Config) -> dict | None:
    """
    Translate a config's optional search filters into a STAC query-extension dict.

    Parameters
    ----------
    cfg : Config
        Job configuration.

    Returns
    -------
    dict | None
        Query-extension filters, or None if none are set.

    Raises
    ------
    ValueError
        If `cfg.orbit_state` is set to anything other than 'ascending'/'descending'/
        'as_bands'.
    """
    query: dict = {}
    if cfg.max_cloud_cover is not None:
        query["eo:cloud_cover"] = {"lt": cfg.max_cloud_cover}
    if cfg.orbit_state is not None and cfg.orbit_state != ORBIT_STATE_AS_BANDS:
        if cfg.orbit_state not in ORBIT_STATES:
            valid = [*ORBIT_STATES, ORBIT_STATE_AS_BANDS]
            raise ValueError(f"orbit_state must be one of {valid}, got {cfg.orbit_state!r}")
        query["sat:orbit_state"] = {"eq": cfg.orbit_state}
    return query or None


def search(
    config_path: Path, satellite_key: str, *, accept_heterogeneous_baselines: bool = False
) -> None:
    """
    Search a STAC source for items matching a configuration's AOI/time range, and
    write the results as this job's `search` -> `download` hand-off file.

    Does nothing if the search results already exist, so a resubmitted job doesn't
    search again. Delete `items.json` in the output directory to rerun, e.g. after
    changing the config. A time range needing a fallback collection with heterogeneous
    processing (see `gfetch.sources.HeterogeneousFallbackError`) is searched only if
    `accept_heterogeneous_baselines` is set, or if confirmed at a prompt when stdin is
    a terminal.

    Parameters
    ----------
    config_path : Path
        Path to the configuration YAML file.
    satellite_key : str
        Which satellite to load from `config_path` - `'s1'`, `'s2'`, or a name under
        its `custom:` section. See `gfetch.cli.config.load`.
    accept_heterogeneous_baselines : bool
        Search such a fallback collection without asking. Defaults to False.

    Raises
    ------
    HeterogeneousFallbackError
        If the time range needs such a fallback, `accept_heterogeneous_baselines` is
        False and stdin is not a terminal.
    """
    cfg = load(config_path, satellite_key)
    if cfg.items_path.exists():
        log.info(f"Search results already at {cfg.items_path}, skipping search")
        return
    source = get_source(resolve_source(cfg))
    query = _build_query(cfg)
    log.debug(f"query={query}")

    def run(accept: bool) -> list[pystac.Item]:
        return search_items(
            source,
            cfg.satellite,
            cfg.resolved_aoi.bbox,
            cfg.time_range.datetime,
            query=query,
            collection=cfg.collection,
            intersects=shapely.geometry.mapping(cfg.aoi_geometry),
            accept_heterogeneous=accept,
        )

    try:
        items = run(accept_heterogeneous_baselines)
    except HeterogeneousFallbackError as e:
        log.warning(e)
        if not sys.stdin.isatty():
            log.error(
                "Not asking for confirmation without a terminal: pass "
                "--accept-heterogeneous-baselines to search it anyway."
            )
            raise
        if not Confirm.ask("Search it anyway?", default=False):
            log.error("Search cancelled, nothing written.")
            return
        items = run(True)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    pystac.ItemCollection(items).save_object(str(cfg.items_path))
    log.info(f"Wrote {len(items)} items to {cfg.items_path}")
