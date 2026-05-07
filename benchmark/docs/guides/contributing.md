# Contributing

This guide covers the developer tooling that rides along with every
change. For *what* to add (tasks, metrics, models), see the other
guides in this section. This page is about *how* to land changes that
pass CI cleanly.

## Development install

```bash
cd benchmark
make install          # editable install + pre-commit hooks
```

`make install` is a shortcut for:

```bash
pip install -e '.[dev,hub,hf-datasets,schemas,docs]'
pre-commit install
```

The `dev` extra bundles everything the lint/format/type pipeline
needs: `ruff`, `black`, `mypy`, `pytest`, `pytest-cov`, `pre-commit`.

## One-shot commands

| Command | What it runs |
|---|---|
| `make test` | `pytest -q` |
| `make cov` | tests + coverage report, fails under 70% |
| `make lint` | ruff check (hard) + black/ruff format check (advisory) |
| `make fmt` | ruff format + black + ruff check --fix |
| `make type` | mypy with the pyproject strict overlay |
| `make check` | lint + type + cov — everything CI runs |
| `make docs` / `make serve` | build / live-serve the MkDocs site |
| `make smoke` | run `examples/run_depth.py` (end-to-end synthetic pipeline) |

`make help` lists every target.

## Tooling slate

| Tool | What it enforces | Command |
|---|---|---|
| [ruff](https://docs.astral.sh/ruff/) | Lint (F, E, I import sort, B bugbear) + format | `ruff check .` / `ruff format .` |
| [black](https://black.readthedocs.io/) | Formatting (100 char lines) | `black rpx_benchmark tests` |
| [mypy](https://mypy.readthedocs.io/) | Types on the stable-surface modules (strict) | `mypy rpx_benchmark` |
| [pytest](https://docs.pytest.org/) | Unit + integration + golden tests | `pytest` |
| [pre-commit](https://pre-commit.com/) | Runs the above on every commit | `pre-commit run --all-files` |

## Type-checking policy

`rpx_benchmark` ships a `py.typed` marker (PEP 561) so downstream
users get accurate type information when they install it.

Inside this repo, `mypy` runs strict against a curated set of modules
on `main`:

- `rpx_benchmark.api` — task enums + GT / Prediction / `Sample`
  dataclasses. The public data contract.
- `rpx_benchmark.schemas` — Pydantic manifest models.
- `rpx_benchmark._schemas_lazy` — lazy facade.
- `rpx_benchmark.exceptions` — error hierarchy.

The rest of the package runs with permissive mypy until each
milestone migrates it (see the `CHANGELOG.md` *Typing* section as
modules land in strict mode). New modules introduced from now on
should be strict-clean from their first commit.

## Running tests

```bash
pytest -q                              # full suite
pytest tests/test_schemas.py -q        # single file
pytest -k monocular_depth              # by keyword
pytest --cov=rpx_benchmark --cov-report=term-missing
```

## Commit workflow

1. Make your change and add/update docs alongside — docs and code
   land together, not in a follow-up.
2. Update `CHANGELOG.md` under `## [Unreleased]`.
3. `pre-commit run --all-files` to catch lint/format/type issues.
4. `pytest -q` to confirm nothing regressed.
5. Commit. Pre-commit will run automatically on staged files.

## CI matrix

`.github/workflows/tests.yml` runs on every push and PR:

| Job | What it enforces |
|---|---|
| `pytest` × Python 3.10 / 3.11 / 3.12 | Full offline test suite, coverage floor 70% |
| `ruff (lint + format)` | `ruff check` fails hard; `ruff format --check` + `black --check` advisory until the tree is reformatted once |
| `mypy (strict modules)` | Strict overlay on the stable-surface modules configured in `[tool.mypy]` |

CUDA and MPS matrix legs aren't wired yet — they need self-hosted
runners. The profiler backends already degrade gracefully on
CPU-only runners, so every test passes without them.

## Releasing

Tag a version on `main` and push it:

```bash
git tag v0.2.0
git push origin v0.2.0
```

`.github/workflows/release.yml` then:

1. builds a wheel + sdist,
2. smoke-imports the wheel,
3. publishes to PyPI via Trusted Publishing (OIDC — no API token),
4. cuts a GitHub release whose notes are extracted from the matching
   `CHANGELOG.md` section (or `[Unreleased]` as a fallback).

Before the first release, add this repo as a trusted publisher on
PyPI for the `rpx-benchmark` project (PyPI → Manage → Publishing →
Add).

## Determinism

For reproducible benchmark runs, seed with
`rpx_benchmark.seed_all(int)` or the `rpx_benchmark.deterministic(int)`
context manager. Both cover Python `random`, NumPy, PyTorch (CPU /
CUDA / MPS), and HuggingFace `transformers.set_seed` where available,
and both are safe to call when the corresponding library isn't
installed.
