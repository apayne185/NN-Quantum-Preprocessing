import json

import pytest
import torch

from qnnbench.bench.compare import compare
from qnnbench.bench.compare import main as compare_main
from qnnbench.bench.harness import Measurement, measure, save_results, saved_tensor_bytes
from qnnbench.bench.suites import SUITES


def test_percentiles_and_throughput():
    m = Measurement("s", {}, times_ms=[float(t) for t in range(1, 21)], items_per_call=10)
    assert m.p50_ms == 10.5
    assert m.p95_ms == 19
    assert m.throughput == pytest.approx(10 / 0.0105)


def test_measure_counts_repeats_after_warmup():
    calls = []
    m = measure(lambda: calls.append(1), suite="s", case={}, device=torch.device("cpu"),
                warmup=3, repeats=7)  # fmt: skip
    assert len(calls) == 10 and len(m.times_ms) == 7


def test_saved_tensor_bytes_sees_activations():
    w = torch.randn(256, 256, requires_grad=True)
    x = torch.randn(8, 256)
    assert saved_tensor_bytes(lambda: (x @ w).relu().sum()) >= x.numel() * 4


def test_save_results_writes_env_and_derived_fields(tmp_path):
    m = Measurement("demo", {"n": 1}, [1.0, 2.0, 3.0])
    path = save_results("demo", [m], torch.device("cpu"), tmp_path)
    payload = json.loads(path.read_text())
    assert payload["env"]["torch"] == torch.__version__
    assert payload["measurements"][0]["p50_ms"] == 2.0


def _result(p50s: dict[int, float]) -> dict:
    return {"env": {}, "measurements": [{"case": {"n": n}, "p50_ms": t} for n, t in p50s.items()]}


def test_compare_flags_only_regressions_beyond_tolerance():
    base = _result({1: 10.0, 2: 10.0, 3: 10.0})
    lines, regressed = compare(base, _result({1: 11.0, 2: 8.0, 3: 10.5}), tolerance=0.15)
    assert not regressed
    lines, regressed = compare(base, _result({1: 12.0, 2: 10.0, 3: 10.0}), tolerance=0.15)
    assert regressed and any("REGRESSED" in line for line in lines)


def test_compare_cli_exit_code(tmp_path):
    base, cur = tmp_path / "a.json", tmp_path / "b.json"
    base.write_text(json.dumps(_result({1: 10.0})))
    cur.write_text(json.dumps(_result({1: 20.0})))
    assert compare_main([str(base), str(cur)]) == 1
    assert compare_main([str(base), str(base)]) == 0


@pytest.mark.slow
@pytest.mark.parametrize("suite", ["qubits", "grad", "dtype"])
def test_quick_suites_run(suite):
    assert SUITES[suite](torch.device("cpu"), quick=True)
