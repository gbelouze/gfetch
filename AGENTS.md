# AGENTS.md

This document provides guidelines for agents working on this codebase.

See `claude/tech-stack.md` (architecture/library decisions) and `claude/dev-stack.md`
(tooling conventions, incl. deviations from sibling projects) for the full reasoning
behind everything below — this file summarizes only what's needed to build/lint/test.

## Build, Lint, Test Commands

### Installation

```bash
uv sync                     # Sync dependencies from pyproject.toml
uv run pre-commit install   # Install pre-commit hooks
```

### Dependency Management

```bash
uv add <package>            # Add a dependency
uv add <package> --dev      # Add a dev dependency
uv remove <package>         # Remove a dependency
```

### Running Tests

```bash
uv run pytest                        # Run all tests
uv run pytest tests/                 # Run tests from specific directory
uv run pytest -k "test_name"         # Run tests matching pattern
uv run pytest -m slow                # Run tests marked as slow
```

### Type Checking

```bash
uv run pyrefly check                 # Type-check the project (pyrefly, not ty/mypy)
```

### Linting and Formatting

```bash
uv run ruff check --preview src      # Run ruff linter
uv run ruff check --fix --preview src  # Run ruff with auto-fix
uv run ruff format src               # Format code with ruff
uv run flake8                        # Run flake8 (pydoclint only, see .flake8)
uv run pre-commit run --all-files    # Run all pre-commit hooks
```

Note: Never call pip directly. Use `uv run <command>`.

## Code Style Guidelines

### General Principles

- Write clear, maintainable code
- Use type hints for function signatures
- Add docstrings to all public functions, especially large/core/user-facing ones
- Prefer explicit over implicit

### Python-Specific Guidelines

#### Type Hinting

- Use built-in generics: `list`, `dict`, `set`, `tuple` instead of
  `List`, `Dict`, `Set`, `Tuple` from `typing`. `Any` is an exception to this rule.
- Use `x | None` instead of `Optional[x]`
- Use `x | None` instead of `Union[x, None]`
- All function signatures should have type hints

#### Docstrings

- numpy-style, checked by `pydoclint` (ruff's `DOC` rules + the standalone
  `pydoclint-flake8` pre-commit hook — they don't fully overlap)
- Document every meaningful parameter, with its type written out verbatim as in the
  signature, and "Defaults to ..." as the last sentence when the parameter has a default

#### Logging

- Always use `logging` instead of `print` (enforced by ruff's `T20`)
- Module-level `log = logging.getLogger(__name__)`
- Multiprocess workers: forward records via a `multiprocessing.Manager().Queue()` to a
  consumer running in the parent (single-node only — doesn't cross SLURM node
  boundaries; each node logs independently for now)

#### Imports

- isort via ruff's `I` rule
- Absolute imports

#### Parallelization

- `joblib` for simple parallelization (no need to track individual future state)
- `ThreadPoolExecutor` when `as_completed` is needed (e.g. progress bars) — the download
  stage is I/O-bound and thread-parallel
- If a stage needs process-based parallelism (e.g. a CPU-bound compute/mosaic stage),
  use a queue-proxy pattern for progress/logging across the process boundary, not direct
  object sharing

#### CLI

- `cyclopts`, one `@app.command` per pipeline stage (`search`/`download`/`load`/`write`),
  plus an `init` subcommand to scaffold a config file. Defer heavy imports inside each
  command body so `--help` stays fast.
- Avoid hydra, jsonargparse unless specifically requested

#### Config

- `omegaconf`-backed declarative YAML as the primary way to describe a job (AOI, time
  range, satellite, output dir, worker counts). Every pipeline stage should be able to
  read the same config file (important for HPC: `download` and `load`/`write` run as
  separate SLURM jobs but share one config).

#### Path Handling

- When filtering paths, always skip files starting with `.`:
  ```python
  files = path.glob("[!.]*.parquet")
  ```

#### Progress Bars

- `rich.progress.Progress`, built once per CLI command and passed down. Auto-disable
  when not attached to a TTY (`GFETCH_DEBUG` env var also forces it off) — SLURM batch
  stdout/stderr is captured to a file, not a terminal.

#### String Formatting

- Always use f-strings, never `%` formatting or `.format()`

#### Comments

- Avoid section comments (##, ###, etc.)
- Only add comments for complex, non-obvious code (rare)

#### Ternary Operator

- Never use `or` as ternary
- Always use explicit `x if condition else y`

### Naming Conventions

- `snake_case` for variables, functions, methods
- `PascalCase` for classes
- `SCREAMING_SNAKE_CASE` for constants
- `_leading_underscore` for private/internal members

### Error Handling

- Use specific exceptions (not bare `except:`)
- Add context to exceptions when helpful
- Log errors with appropriate level before raising

### File Organization

- `src/gfetch/` — package source, mirrored by `tests/`
- `claude/` — design/tracking notes (not user-facing docs; see `docs/` for those)
- Configuration in `pyproject.toml`

### HPC-specific conventions

- Every pipeline stage must be independently callable/CLI-invokable, so an external
  orchestrator (SLURM/Snakemake/Nextflow/Parsl) can pin it to the right node class.
- Download-stage writes: temp path + atomic `rename`, plus one sentinel completion file
  per unit of work (e.g. `<item_id>/<asset_key>.complete`) — never a central manifest
  database (avoids distributed-locking issues on shared HPC filesystems).
- Write-stage (Zarr) chunk regions must be pre-planned so each worker writes into a
  strictly disjoint region; per-chunk atomicity (zarr-python v3 `LocalStore`) does not
  cover two workers racing on the *same* chunk.

## Pre-commit Hooks

The repository uses pre-commit hooks that run:

- `remove-crlf`, `remove-tabs`, `forbid-tabs`
- `trailing-whitespace`
- `check-merge-conflict`, `check-yaml`
- `ruff` (lint + format)
- `pyrefly` on the whole project
- `pydoclint-flake8`

Run `uv run pre-commit run --all-files` to verify code quality before committing.

## Key Tools and Versions

- Python: >=3.12
- Linter: ruff
- Formatter: ruff-format
- Type checker: pyrefly (not `ty`/`mypy` — deliberate per-project deviation from
  sibling projects, see `claude/dev-stack.md`)
- Docstring style: numpy (pydoclint)
- Test runner: pytest
