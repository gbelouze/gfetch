"""System resource helpers."""

import os

__all__ = ["available_cpus"]


def available_cpus() -> int:
    """
    Number of CPUs actually available to this process.

    Prefers `os.sched_getaffinity`, which reflects any cpuset cgroup restriction placed
    on this process (e.g. a SLURM job's `--cpus-per-task` reservation) - unlike
    `os.cpu_count()`, which reports the whole machine regardless of any per-job
    reservation. Falls back to `os.cpu_count()` on platforms without
    `sched_getaffinity` (e.g. macOS).

    Returns
    -------
    int
        Number of CPUs available to this process.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1
