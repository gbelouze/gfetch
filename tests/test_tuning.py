import pytest

from gfetch.utils.tuning import WorkerTuner


def _cost(workers: int) -> float:
    """Seconds per byte with a minimum at 6 workers (a + b/w + c*w, sqrt(b/c) = 6)."""
    return 1.0 + 36.0 / workers + 1.0 * workers


def test_first_three_picks_are_min_max_middle() -> None:
    tuner = WorkerTuner(2, 16, noise=0.0)
    picks = []
    for _ in range(3):
        picks.append(tuner.next())
        tuner.observe(picks[-1], _cost(picks[-1]), 1)

    assert picks == [2, 16, 9]


def test_converges_to_the_best_worker_count() -> None:
    tuner = WorkerTuner(1, 32, noise=0.0)
    for _ in range(6):
        workers = tuner.next()
        tuner.observe(workers, _cost(workers), 1)

    assert tuner.next() == 6


def test_cost_is_normalized_by_bytes() -> None:
    tuner = WorkerTuner(1, 32, noise=0.0)
    for nbytes in (10, 1000, 7):
        workers = tuner.next()
        tuner.observe(workers, _cost(workers) * nbytes, nbytes)

    assert tuner.next() == 6


def test_fixed_range_always_returns_its_value() -> None:
    tuner = WorkerTuner(4, 4)

    assert [tuner.next() for _ in range(5)] == [4] * 5


def test_small_range_falls_back_to_best_observed() -> None:
    tuner = WorkerTuner(3, 4, noise=0.0)
    tuner.observe(tuner.next(), 2.0, 1)
    tuner.observe(tuner.next(), 1.0, 1)

    assert tuner.next() == 4


def test_noisy_picks_stay_in_range() -> None:
    tuner = WorkerTuner(2, 5, noise=10.0, seed=0)
    for _ in range(30):
        workers = tuner.next()
        assert 2 <= workers <= 5
        tuner.observe(workers, _cost(workers), 1)


def test_rejects_invalid_range() -> None:
    with pytest.raises(ValueError, match="min_workers"):
        WorkerTuner(8, 4)
