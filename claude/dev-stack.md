# gfetch — dev stack

Tooling and repo conventions for `gfetch`, carried over from two sibling projects with
the same author: `/Users/gbelouze/Documents/phd/src/lsatfetch` (Landsat/AWS tile
fetching, the closer architectural analogue) and
`/Users/gbelouze/Documents/phd/src/geefetch` (the GEE-based predecessor gfetch
supersedes — see `tech-stack.md`'s "Prior art" section for why it's relevant despite
being GEE-based). Where the two disagree, both are noted with the reasoning for which one
gfetch follows.

## Environment & packaging

- **uv-first**, same as lsatfetch: `uv sync`, `uv add <pkg>`, `uv run <cmd>`. Never call
  `pip` directly. Build backend `uv_build` (matches the current `pyproject.toml` scaffold
  already in the repo). `requires-python = ">=3.12"`, pinned via `.python-version` (also
  already in place, set to `3.12`).
- **GDAL risk**: GDAL/rasterio's binary wheels are known to fight with uv's resolver
  (ABI/version pinning issues across GDAL, rasterio, pyproj, fiona). Default plan: try to
  keep the whole stack uv-managed first. **Fallback**: a conda environment named
  `gfetch` already exists locally (`/opt/homebrew/Caskroom/miniconda/base/envs/gfetch`)
  for exactly this contingency — reach for it only if/when a real uv+GDAL conflict shows
  up, not preemptively.
  - Noted mismatch to fix before relying on it: the conda env currently has **Python
    3.14.7**, while the project targets **3.12** — align this (recreate the env pinned to
    3.12) before actually using it, so behavior matches the uv-managed environment.
- HPC deployment target: **SLURM** (confirmed 2026-09-19). No specific filesystem
  confirmed yet, but see `tech-stack.md`'s HPC section — assume shared POSIX storage
  (Lustre/GPFS/NFS-class), not S3, unless told otherwise for a given cluster.

## Linting & formatting — ruff

Mirror lsatfetch's `[tool.ruff]` config (line-length 100, target py312):

```toml
[tool.ruff]
target-version = "py312"
line-length = 100
[tool.ruff.lint]
select = [
    "E",    # pycodestyle
    "F",    # Pyflakes
    "UP",   # pyupgrade
    "B",    # flake8-bugbear
    "SIM",  # flake8-simplify
    "I",    # isort
    "DOC",  # pydocstyle/pydoclint (docstring presence/shape)
    "PTH",  # flake8-use-pathlib
    "LOG",  # flake8-logging
    "INP",  # flake8-no-pep420
    "PIE",
    "T20",  # no stray print()
    "PT",   # pytest style
]
ignore = ["DOC501", "DOC201"]

[tool.ruff.lint.pydocstyle]
convention = "numpy"

[tool.ruff.lint.extend-per-file-ignores]
"tests/*.py" = ["INP001"]
```

`isort` profile "black" is set too in lsatfetch (`multi_line_output = 3`,
`line_length = 88`) but is redundant once ruff's `I` rule + `ruff format` are both
active — carry it over only if we hit a case ruff's isort emulation doesn't handle the
way we want; don't add it speculatively.

Use `ruff format` for formatting (not black directly).

## Type checking — pyrefly

**Deviation from lsatfetch**: lsatfetch uses `ty` (Astral) with `mypy` also run in
pre-commit as a second check. For gfetch, use **pyrefly** (Meta's Rust-based type
checker) as the sole type checker, per explicit instruction (2026-09-19). This is a
deliberate per-project choice, not a correction to lsatfetch's setup — don't backport it
there without being asked.

Open question: lsatfetch runs two type checkers (`ty` + `mypy`) for extra coverage during
the tool's early-adopter period. Decide whether gfetch does the same (pyrefly + mypy) or
trusts pyrefly alone — leaning toward pyrefly-only unless it proves to miss things Mypy
catches, to keep the pre-commit loop fast.

## Docstrings — numpy style, enforced via pydoclint

Same as lsatfetch: numpy-style docstrings, checked by `pydoclint` (both ruff's `DOC`
rules and the standalone `pydoclint-flake8` pre-commit hook, since ruff's docstring
checks and pydoclint's don't fully overlap). `.flake8` exists purely to scope flake8 to
`select = DOC` so it doesn't duplicate ruff's other lint rules:

```ini
[flake8]
show-source = True
builtins = unicode
select = DOC
extend-ignore = DOC5
```

```toml
[tool.pydoclint]
style = "numpy"
skip-checking-raises = true
check-style-mismatch = true
ignore = ["DOC201", "DOC202", "DOC402", "DOC403", "DOC501"]  # ruff already covers these
```

Docstring content rules (from lsatfetch's `AGENTS.md`, carry over as-is):
- Docstring on every public function, especially large/core/user-facing ones.
- Document every meaningful parameter, with its type written out verbatim as in the
  signature, and "Defaults to ..." as the last sentence when the parameter has a default.

## pre-commit

Mirror lsatfetch's hook set, swapping the type-checker hook for pyrefly:

```yaml
repos:
  - repo: https://github.com/Lucas-C/pre-commit-hooks
    hooks: [remove-crlf, forbid-tabs, remove-tabs]
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: trailing-whitespace
        args: [--markdown-linebreak-ext=md]
      - id: check-merge-conflict
      - id: check-yaml
        args: [--unsafe]
  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix, --preview]
      - id: ruff-format
  - repo: local
    hooks:
      - id: pyrefly
        name: pyrefly check
        entry: uv run --active pyrefly check
        language: python
        files: '.*\.py'
  - repo: https://github.com/jsh9/pydoclint
    hooks:
      - id: pydoclint-flake8
        args: [--style=numpy, --check-return-types=False, --check-style-mismatch=True]
```

Pin actual `rev:` values to current releases at scaffold time rather than copying
lsatfetch's (possibly stale) pins.

## Logging

Carry over lsatfetch's `rich`-based setup almost verbatim
(`src/lsatfetch/utils/log.py`): `RichHandler` for console output with markup, optional
second `RichHandler` writing to a logfile, `rich.traceback.install()` for readable
tracebacks, module-level `log = logging.getLogger(__name__)` everywhere, `logging`
never `print` (enforced by ruff's `T20`).

**Multiprocess logging** — carry over `src/lsatfetch/utils/log_multiprocessing.py`'s
pattern as the starting point: workers log normally via `logging.getLogger(__name__)`;
a `PicklableQueueHandler` (stringifies exceptions before they cross the process
boundary, since tracebacks aren't picklable) forwards every record from worker processes
into a `multiprocessing.Manager().Queue()`; the parent runs a `LogQueueConsumer` context
manager (background thread draining the queue) that replays records through its own
handlers, prefixed with `[PID=...]`.

**HPC caveat to resolve before reuse**: this pattern is single-node (a
`multiprocessing.Queue` doesn't cross node boundaries). It works unmodified for gfetch's
within-a-node parallelism (e.g. many download workers in one job), but doesn't by itself
give aggregated logs across a SLURM job array spanning multiple nodes. Default plan for
now: each SLURM task/node logs independently (to its own file, same as any normal SLURM
job's stdout/stderr capture) — no cross-node log aggregation attempted until there's a
concrete need for it. Flagged as an open question, not solved yet.

## Progress bars — rich

Carry over `src/lsatfetch/utils/progress.py`'s `default_bar()` pattern: a
`rich.progress.Progress` with project-specific columns (bytes transferred, ETA, custom
stats), built once per script/CLI command and passed down to whatever needs to report
progress, rather than each function creating its own `Progress`. Includes a
`temporary_task` context manager (adds a task, guarantees its removal from the display
on exit even on exception) — reuse as-is.

**Disabling in non-interactive contexts** — lsatfetch disables via an explicit env var
(`LSATFETCH_DEBUG=1`/`true`). For gfetch, keep that escape hatch (`GFETCH_DEBUG`) *and*
also auto-disable when not attached to a TTY (`rich.console.Console().is_terminal`),
since SLURM batch job stdout/stderr is captured to a file, not a terminal — an animated
progress bar there just spams the log file with redraw noise. This is a genuine
improvement over lsatfetch's setup worth carrying back there too, at some point, but out
of scope for gfetch's own build right now.

Use a shared `Lock` (`global_console_lock`, from
`src/lsatfetch/utils/multiprocessing.py`) when multiple threads/processes might write to
the same Rich console concurrently, to avoid interleaved/garbled output.

**Thread-based vs. process-based parallelism changes which pattern applies.** lsatfetch's
`default_bar()`/`Progress` works unmodified across threads (shared memory — a
`ThreadPoolExecutor` worker can update the same `Progress` object directly). It does
*not* work across a `ProcessPoolExecutor` (separate memory). geefetch needed real
multi-process parallelism (GEE/geedim auth is per-process) and solved it with
`utils/progress_multiprocessing.py`: a `QueuedProgress` proxy implementing the same
`add_task`/`update`/`advance`/`remove_task` shape, which workers call instead of the real
`Progress`; it just enqueues commands onto a `multiprocessing.Manager().Queue()`, and a
`ProgressQueueConsumer` in the main process drains the queue on a background thread and
replays the calls against the real `Progress` (paired with the equivalent
`LogQueueConsumer` for logs, both consumers started together — see
`data/get.py::download`'s `with mp.Manager() as manager: ... LogQueueConsumer(...),
ProgressQueueConsumer(...), ProcessPoolExecutor(...)`). Pick per gfetch stage: the
download stage is I/O-bound and fits `ThreadPoolExecutor` (per the Parallelization
section below), so lsatfetch's direct-sharing approach is enough; if the compute/mosaic
stage ends up process-parallelized (CPU-bound odc-stac loads across cores), reuse
geefetch's queue-proxy pattern instead.

## Parallelization

From lsatfetch's `AGENTS.md`, carry over as-is:
- `joblib` for simple parallelization (no need to track individual future state).
- `ThreadPoolExecutor` when `as_completed` is needed — e.g. driving a progress bar as
  results land, rather than waiting on a blocking `.map()`.
- `SequentialExecutor` (from `src/lsatfetch/utils/multiprocessing.py`) — a drop-in
  `Executor` that runs everything synchronously in the calling thread, useful for tests
  and debugging without changing calling code (`with SequentialExecutor() as ex: ...`
  behaves like a real pool but is trivial to step through).

## Retries & atomic writes

Directly relevant to `tech-stack.md`'s HPC download-durability design — lsatfetch
already implements almost exactly what we scoped there
(`src/lsatfetch/core.py::download_tile`):
- `@retry(tries=5, backoff=1.7, logger=log)` from the `retry` package around the
  network call.
- Atomic write: download into a `tempfile.TemporaryDirectory()`, then `shutil.move()`
  into the final path only once the download completes fully — a killed process leaves
  no half-written file at the destination path.

Gap versus what `tech-stack.md` calls for: lsatfetch's atomicity covers a single file,
not the "many workers, one shared cache dir, no coordination" case gfetch needs (its
sentinel-file-per-completed-asset design). Reuse the temp-dir-then-move mechanics
directly; add the sentinel-file completion marker on top when building gfetch's download
stage.

This covers the **download** stage only (discrete COG files). geefetch's alternative —
validating file integrity on resume instead of atomic writes — was considered and
explicitly rejected for gfetch, not carried over. Whether/how atomicity needs to be
re-derived for the **write** stage (Zarr chunks, not discrete files) is a separate, still
open question — see `tech-stack.md`'s "HPC / distributed execution" section, under
active research as of 2026-09-19. Don't assume the download-stage pattern transfers
unchanged.

## Config — omegaconf + declarative YAML

Both sibling projects converge on the same pattern: `omegaconf`-backed YAML
configuration as the primary way to describe a dataset-acquisition job (AOI, time range,
satellite, output dir, worker counts, ...), plus a CLI `init` subcommand that writes out
a template config to start from (see lsatfetch's `cli/init.py` +
`cli/config_spec.yaml`). This fits gfetch's "give an AOI, get good defaults" goal well —
most parameters should have sane defaults in the config schema — and a config file is
also the natural unit to hand to a SLURM job (`sbatch` script points at a config path
rather than a long argument list), which matters given the HPC stage-separation design
in `tech-stack.md`: the same config file can be reused unchanged across the `download`
and `load`/`mosaic` SLURM jobs. Adopt this from the start rather than bolting on config
support later.

## CLI — cyclopts

Follow lsatfetch (`cyclopts.App`, one `@app.command` per subcommand, docstring becomes
the `--help` text via numpy-style parsing, heavy imports deferred inside each command
function body so `--help` stays fast), not geefetch, which uses `click`. `cyclopts` is
the newer choice between the two sibling projects and is what lsatfetch's `AGENTS.md`
explicitly directs ("Avoid hydra, jsonargparse unless specifically requested") — treat
`click` as geefetch's older pattern, superseded, not a second option to weigh. Avoid
hydra/jsonargparse unless a specific need arises. Given `tech-stack.md`'s stage
separation (`search`/`download`/`load`/`write` each independently invocable for SLURM job
splitting), each stage should map to its own `gfetch` subcommand from the start, not be
bolted on later — plus an `init` subcommand for scaffolding a template config (see
"Config" above).

## Testing

`pytest`, tests live in `tests/`, mirrored package structure. lsatfetch's
`[tool.pytest.ini_options]`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["slow"]
log_cli = true
log_cli_level = "DEBUG"
log_level = "DEBUG"
```

`moto` for mocking S3 in tests (lsatfetch dev dependency) — relevant to gfetch too once
we're testing against S3-backed sources (Earth Search) or S3-based caches.

## Repo conventions

- `CHANGELOG.md`, `CONTRIBUTING.md`, `README.md` with a feature checklist (lsatfetch's
  README uses a `- [ ]`/`- [x]` checklist of planned features as a lightweight roadmap —
  worth adopting the same pattern for gfetch's README once the feature set from
  `tech-stack.md` is finalized).
- `docs/` for longer-form design notes (e.g. lsatfetch's `docs/output-format.md`) —
  distinct from `claude/`, which is our own working/tracking space (this file and
  `tech-stack.md`) rather than user-facing documentation.
- lsatfetch keeps a `claude/tasks.md` as a running, dated log of confirmed bugs/design
  issues discovered during development (each entry: what was confirmed, why it matters,
  candidate fixes ranked). Worth starting a similar `claude/tasks.md` for gfetch once
  implementation begins and real issues start turning up — nothing to log yet.
- `AGENTS.md` at the repo root, summarizing all of the above for coding-agent
  consumption (build/lint/test commands, style rules) — lsatfetch has one; gfetch should
  get an equivalent once the stack settled here is actually scaffolded, so it can name
  real, working commands rather than aspirational ones.

## Open questions

- pyrefly vs. pyrefly+mypy (belt-and-suspenders like lsatfetch) — leaning pyrefly-only,
  revisit if it misses real bugs.
- Cross-node log aggregation on SLURM multi-node jobs — no plan yet, not blocking.
- Whether `isort`'s explicit profile config is worth carrying over given ruff's `I` rule
  + `ruff format` already cover most of it.
- Conda `gfetch` env's Python version (3.14.7) needs realigning to 3.12 before it's
  actually usable as the GDAL fallback.
