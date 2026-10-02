# Common tasks. CPU-only torch keeps the local install small; on a GPU box,
# install torch from PyPI (CUDA wheels) instead: `make install TORCH_INDEX=`.
TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
PY := .venv/bin/python

.PHONY: install test test-all lint format bench bench-quick experiments hybrid serve loadtest docker legacy

install:
	uv venv --python-preference only-managed --python 3.12 .venv
	uv pip install --python $(PY) torch $(if $(TORCH_INDEX),--index-url $(TORCH_INDEX))
	uv pip install --python $(PY) -e ".[dev,bench,serve]"

test:
	$(PY) -m pytest -m "not slow"

test-all:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests

format:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

bench:            ## all suites on the default device; results/ <suite>/<device>.json
	$(PY) -m qnnbench.bench all

bench-quick:
	$(PY) -m qnnbench.bench all --quick

experiments:      ## 5-seed accuracy comparison (paper protocol)
	$(PY) -m qnnbench.experiments --seeds 5 --epochs 3

hybrid:           ## quanvolution vs classical controls on full MNIST
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
