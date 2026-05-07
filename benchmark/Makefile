# rpx-benchmark developer shortcuts. Run `make help` for the menu.
#
# Everything routes through the dev extras in pyproject.toml — no
# tools assumed globally installed. Run `make install` once.

.DEFAULT_GOAL := help
PKG := rpx_benchmark
PY  := python

.PHONY: help install test cov lint fmt type check docs serve smoke visualize visualize-phases visualize-save visualize-lite visualize-headless clean

help:  ## List every target
	@awk 'BEGIN{FS=":.*##"; printf "\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Editable install with every dev + runtime extra
	$(PY) -m pip install -e '.[dev,hub,hf-datasets,schemas,docs]'
	pre-commit install

test:  ## Run the test suite
	pytest -q

cov:  ## Tests with coverage report (70% floor)
	pytest -q --cov=$(PKG) --cov-report=term-missing --cov-fail-under=70

lint:  ## Ruff check (hard fail) + black --check + ruff format --check (advisory)
	ruff check $(PKG) tests scripts
	- ruff format --check $(PKG) tests scripts
	- black --check $(PKG) tests scripts

fmt:  ## Auto-format with ruff + black
	ruff format $(PKG) tests scripts
	black $(PKG) tests scripts
	ruff check --fix $(PKG) tests scripts

type:  ## Mypy with the pyproject-configured strict overlay
	mypy $(PKG)

check: lint type cov  ## Everything CI runs, locally (lint + type + coverage)

docs:  ## Build the MkDocs site under ./site/
	mkdocs build --clean

serve:  ## Serve the docs with live reload at localhost:8000
	mkdocs serve

smoke:  ## Run the minimal BYO-model example at examples/run_depth.py
	$(PY) examples/run_depth.py

visualize:  ## Stream RPX into Rerun (no disk artifact; HF cache only)
	PYTHONPATH=. $(PY) scripts/visualize_rerun.py

visualize-phases:  ## Compare all 3 phases (clutter / interaction / clean) side-by-side
	PYTHONPATH=. $(PY) scripts/visualize_rerun.py --all-phases

visualize-save:  ## Same as `visualize` but also write a portable .rrd to site/
	PYTHONPATH=. $(PY) scripts/visualize_rerun.py --save

visualize-lite:  ## Resource-constrained preset (Pi 4 / Jetson Nano)
	PYTHONPATH=. $(PY) scripts/visualize_rerun.py --lite

visualize-headless:  ## Build a .rrd without launching the viewer (CI / SSH)
	PYTHONPATH=. $(PY) scripts/visualize_rerun.py --no-spawn --save

clean:  ## Remove build + cache artefacts
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache site
	find . -type d -name __pycache__ -exec rm -rf {} +
