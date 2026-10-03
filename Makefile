# Common tasks. `make install` uses uv.lock (CPU-only PyTorch, as in CI).
# On a GPU machine use `make install-gpu`: same packages, CUDA PyTorch from PyPI.
PY := .venv/bin/python
EXTRAS := --extra dev --extra bench --extra serve

.PHONY: install install-gpu hooks test coverage test-all lint format bench bench-quick profile experiments experiments-hybrid hybrid serve loadtest docker legacy

install:
	uv sync --locked $(EXTRAS)

install-gpu:
	uv venv --python 3.12 .venv
	uv pip install --python $(PY) --no-sources -e ".[dev,bench,serve]"

hooks:
	uv run pre-commit install

test:
	$(PY) -m pytest -m "not slow"

coverage:
	$(PY) -m pytest --cov --cov-report=term --cov-report=html

test-all:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
	$(PY) -m mypy

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

bench:            ## all suites on the default device; results/ <suite>/<device>.json
	$(PY) -m qnnbench.bench all

bench-quick:
	$(PY) -m qnnbench.bench all --quick

profile:          ## torch.profiler trace of a training step -> runs/profile/trace.json
	$(PY) -m qnnbench.profile

experiments:      ## 5-seed accuracy comparison (paper protocol)
	$(PY) -m qnnbench.experiments --seeds 5 --epochs 3

experiments-hybrid: ## quantum filter vs classical controls, 5 seeds
	$(PY) -m qnnbench.experiments --suite hybrid --seeds 5 --epochs 5

hybrid:           ## quanvolution vs classical controls on full MNIST, one seed
	for f in quanv random learned; do \
		$(PY) -m qnnbench.hybrid --features $$f --epochs 5 --out results/hybrid/$$f.json; \
	done

serve:
	$(PY) -m uvicorn qnnbench.serve:app --port 8000

loadtest:         ## batching off vs on, against freshly spawned servers
	$(PY) -m qnnbench.loadtest --spawn --max-batch 1 64 --concurrency 1 16 64 --out results/serve/loadtest.json

docker:
	docker build -t qnnbench .

legacy:           ## original TFQ pipeline (Python 3.11)
	cd legacy/tfq && uv venv --python 3.11 .venv-tfq && \
		uv pip install --python .venv-tfq/bin/python -r requirements.txt && \
		.venv-tfq/bin/python -m analysis.nn_qnn_compare
