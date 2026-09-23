"""Online tuning of the dask worker count from observed processing times."""

import logging

import numpy as np

__all__ = ["WorkerTuner"]

log = logging.getLogger(__name__)


class WorkerTuner:
    """
    Pick a worker count per unit of work, learning from the cost of previous units.

    The first three picks are forced to `min_workers`, `max_workers` and their
    middle. After that, the cost (seconds per byte) is modelled as
    `a + b / w + c * w`: `b / w` is the part that parallelizes, `c * w` the
    contention that grows with workers. The model is fitted by least squares to
    every observation so far, and the pick is the integer in range with the lowest
    predicted cost, plus Gaussian exploration noise.

    Parameters
    ----------
    min_workers : int
        Smallest worker count to try.
    max_workers : int
        Largest worker count to try.
    noise : float | None
        Standard deviation of the exploration noise added to the predicted best
        worker count. Defaults to None, which uses a tenth of the range, at least 1.
    seed : int | None
        Seed for the exploration noise. Defaults to None (unseeded).
    """

    def __init__(
        self,
        min_workers: int,
        max_workers: int,
        noise: float | None = None,
        seed: int | None = None,
    ) -> None:
        if not 1 <= min_workers <= max_workers:
            raise ValueError(
                f"Need 1 <= min_workers <= max_workers, got {min_workers}, {max_workers}"
            )
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.noise = noise if noise is not None else max(1.0, (max_workers - min_workers) / 10)
        self._rng = np.random.default_rng(seed)
        forced = [min_workers, max_workers, (min_workers + max_workers) // 2]
        self._forced = list(dict.fromkeys(forced))
        self._workers: list[int] = []
        self._costs: list[float] = []

    @property
    def fixed(self) -> bool:
        """
        Returns
        -------
        bool
            True if the range holds a single worker count, so nothing is tuned.
        """
        return self.min_workers == self.max_workers

    def next(self) -> int:
        """
        Pick the worker count for the next unit of work.

        Returns
        -------
        int
            Worker count within `[min_workers, max_workers]`.
        """
        if self.fixed:
            return self.min_workers
        if len(self._workers) < len(self._forced):
            return self._forced[len(self._workers)]

        candidates = np.arange(self.min_workers, self.max_workers + 1)
        predicted = self._predict(candidates)
        best = int(candidates[np.argmin(predicted)])
        pick = round(best + self._rng.normal(0.0, self.noise))
        pick = min(max(pick, self.min_workers), self.max_workers)
        log.info(f"Worker tuning: predicted best {best}, trying {pick}")
        return pick

    def observe(self, workers: int, seconds: float, nbytes: int) -> None:
        """
        Record how long a unit of work took with a given worker count.

        Parameters
        ----------
        workers : int
            Worker count used.
        seconds : float
            Processing time of the unit.
        nbytes : int
            Bytes the unit wrote (its uncompressed output size).
        """
        if self.fixed or nbytes <= 0:
            return
        cost = seconds / nbytes
        self._workers.append(workers)
        self._costs.append(cost)
        log.info(
            f"Worker tuning: {workers} worker(s) -> {seconds:.1f} s for "
            f"{nbytes / 1e6:.0f} MB ({cost * 1e9:.2f} s/GB)"
        )

    def _predict(self, workers: np.ndarray) -> np.ndarray:
        """
        Predicted cost per worker count, from the model fitted to observations.

        Falls back to each count's mean observed cost (infinite if unobserved) when
        fewer than three distinct counts have been observed, too few for the model's
        three parameters.

        Parameters
        ----------
        workers : np.ndarray
            Worker counts to predict for.

        Returns
        -------
        np.ndarray
            Predicted cost (seconds per byte) for each of `workers`.
        """
        observed = np.array(self._workers, dtype=float)
        costs = np.array(self._costs)
        if len(np.unique(observed)) < 3:
            means = {w: costs[observed == w].mean() for w in np.unique(observed)}
            return np.array([means.get(float(w), np.inf) for w in workers])
        design = np.column_stack([np.ones_like(observed), 1 / observed, observed])
        a, b, c = np.linalg.lstsq(design, costs, rcond=None)[0]
        w = workers.astype(float)
        return a + b / w + c * w
