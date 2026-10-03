"""Tests for the reporting and operations tooling: experiments, plots, load test, logging."""

import asyncio
import json
import logging

import pytest

from qnnbench.experiments import summarize
from qnnbench.logs import JsonFormatter, TextFormatter, configure_logging


def _run(acc: float, n_params: int = 32) -> dict:
    return {"n_params": n_params, "history": [{"test_acc": acc, "samples_per_s": 100.0}]}


def test_summarize_reports_mean_std_range_and_ceiling():
    runs = {"QNN": [_run(0.8), _run(0.9)], "MLP": [_run(0.91, 37)]}
    ceiling = {"n_patterns": 193, "test_acc": 0.914}
    summary, table = summarize(runs, ceiling, seeds=2, epochs=3)
    assert summary["QNN"]["test_acc_mean"] == pytest.approx(0.85)
    assert summary["QNN"]["test_acc_std"] > 0
    assert summary["MLP"]["test_acc_std"] == 0.0  # single run
    assert "| QNN | 32 | 85.0% ± 7.1 | 80.0–90.0% |" in table
    assert "Bayes ceiling" in table and "91.4%" in table


def _result_file(tmp_path, suite: str, measurements: list[dict]):
    env = {"device": "cpu", "device_name": "Intel(R) Core(TM) i7-1065G7 CPU @ 1.30GHz"}
    path = tmp_path / suite / "cpu.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"suite": suite, "env": env, "measurements": measurements}))


def test_plot_renders_every_suite(tmp_path):
    pytest.importorskip("matplotlib")
    from qnnbench.bench.plot import main

    def m(case, p50, **extra):
        return {"case": case, "p50_ms": p50, "throughput": 32 / p50, "extra": extra}

    _result_file(tmp_path, "qubits", [m({"n_qubits": n, "fuse": f}, n * (2 - f / 5))
                                      for n in (7, 9) for f in (0, 5)])  # fmt: skip
    _result_file(tmp_path, "fusion", [m({"n_qubits": 9, "fuse": k, "pass": p}, 10 - k)
                                      for k in (0, 2) for p in ("forward", "fwd+bwd")])  # fmt: skip
    grad = [m({"n_qubits": n, "method": meth, "fuse": 5}, n, saved_MB=n)
            for n in (9, 13) for meth in ("autograd", "adjoint")]  # fmt: skip
    _result_file(tmp_path, "grad", grad)
    _result_file(tmp_path, "frameworks", [m({"n_qubits": n, "framework": "pennylane"}, n)
                                          for n in (9, 13)])  # fmt: skip
    _result_file(tmp_path, "batch", [m({"n_qubits": 9, "batch": b}, b) for b in (1, 2, 4)])
    main(["--results", str(tmp_path), "--out", str(tmp_path / "figs")])
    names = sorted(p.name for p in (tmp_path / "figs").glob("*.png"))
    assert names == [f"{s}-intel-i7-1065g7-cpu.png"
                     for s in ("batch", "frameworks", "fusion", "grad", "qubits")]  # fmt: skip


def test_load_test_against_in_process_app(monkeypatch):
    pytest.importorskip("fastapi")
    import httpx

    from qnnbench.loadtest import run_load
    from qnnbench.serve import app

    monkeypatch.setenv("QNNBENCH_DEVICE", "cpu")

    async def scenario():
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            return await run_load("http://test", concurrency=4, n_requests=12, transport=transport)

    r = asyncio.run(scenario())
    assert r["requests"] == 12 and r["errors"] == 0 and r["status_counts"] == {200: 12}
    assert r["p50_ms"] > 0 and "qnnbench_batch_size_bucket" in r["server_metrics"]


def _record(**extra) -> logging.LogRecord:
    record = logging.makeLogRecord({"name": "qnnbench.t", "levelname": "INFO", "msg": "done"})
    record.__dict__.update(extra)
    return record


def test_json_formatter_lifts_extra_fields():
    out = json.loads(JsonFormatter().format(_record(epoch=3, test_acc=0.91)))
    assert out["msg"] == "done" and out["level"] == "info"
    assert out["epoch"] == 3 and out["test_acc"] == 0.91


def test_text_formatter_and_configure():
    assert TextFormatter().format(_record(epoch=3)) == "I qnnbench.t: done  epoch=3"
    configure_logging("json", "DEBUG")
    assert isinstance(logging.getLogger("qnnbench").handlers[0].formatter, JsonFormatter)
    with pytest.raises(ValueError):
        configure_logging("xml")
    configure_logging("text", "INFO")
