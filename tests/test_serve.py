import asyncio
import time

import pytest
import torch

pytest.importorskip("fastapi")
pytest.importorskip("prometheus_client")
from fastapi.testclient import TestClient  # noqa: E402

from qnnbench.serve import MicroBatcher, Overloaded, Settings, app  # noqa: E402


def test_micro_batcher_groups_concurrent_requests():
    seen = []

    def fn(xs):
        seen.append(len(xs))
        return xs.sum(dim=1)

    async def scenario():
        batcher = MicroBatcher(fn, max_batch=8, max_wait_ms=50)
        batcher.start()
        results = await asyncio.gather(
            *(batcher.submit(torch.full((2,), float(i))) for i in range(20))
        )
        await batcher.stop()
        return results

    results = asyncio.run(scenario())
    assert results == [2.0 * i for i in range(20)]  # each caller gets its own answer
    assert max(seen) == 8 and sum(seen) == 20  # batched, capped at max_batch


def test_micro_batcher_propagates_errors_and_keeps_serving():
    calls = []

    def fn(xs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return xs.sum(dim=1)

    async def scenario():
        batcher = MicroBatcher(fn, max_batch=4, max_wait_ms=1)
        batcher.start()
        with pytest.raises(RuntimeError):
            await batcher.submit(torch.ones(2))
        ok = await batcher.submit(torch.ones(2))
        await batcher.stop()
        return ok

    assert asyncio.run(scenario()) == 2.0


def test_full_queue_sheds_instead_of_waiting():
    def slow(xs):
        time.sleep(0.2)
        return xs.sum(dim=1)

    async def scenario():
        batcher = MicroBatcher(slow, max_batch=1, max_wait_ms=0, max_queue=2)
        batcher.start()
        first = asyncio.ensure_future(batcher.submit(torch.ones(2)))
        await asyncio.sleep(0.05)  # worker is now busy with `first`
        queued = [asyncio.ensure_future(batcher.submit(torch.ones(2))) for _ in range(2)]
        await asyncio.sleep(0)
        with pytest.raises(Overloaded):
            await batcher.submit(torch.ones(2))
        results = await asyncio.gather(first, *queued)
        await batcher.stop()
        return results

    assert asyncio.run(scenario()) == [2.0, 2.0, 2.0]


def test_expired_requests_are_dropped_before_inference():
    ran = []

    def slow(xs):
        ran.append(len(xs))
        time.sleep(0.2)
        return xs.sum(dim=1)

    async def scenario():
        batcher = MicroBatcher(slow, max_batch=1, max_wait_ms=0)
        batcher.start()
        busy = asyncio.ensure_future(batcher.submit(torch.ones(2)))
        await asyncio.sleep(0.05)
        with pytest.raises(asyncio.TimeoutError):
            await batcher.submit(torch.ones(2), timeout=0.05)  # expires while queued
        await busy
        await asyncio.sleep(0.05)
        await batcher.stop()
        return batcher.metrics.expired._value.get()

    expired = asyncio.run(scenario())
    assert ran == [1] and expired == 1  # the expired request never reached the model


def test_settings_from_env_parses_and_validates():
    env = {
        "QNNBENCH_MAX_BATCH": "8",
        "QNNBENCH_MAX_WAIT_MS": "2.5",
        "QNNBENCH_TORCH_THREADS": "3",
        "QNNBENCH_CHECKPOINT": "w.pt",
    }
    s = Settings.from_env(env)
    assert (s.max_batch, s.max_wait_ms, s.torch_threads, s.checkpoint) == (8, 2.5, 3, "w.pt")
    with pytest.raises(ValueError):
        Settings.from_env({"QNNBENCH_MAX_BATCH": "0"})


def test_endpoints_metrics_and_draining(monkeypatch):
    monkeypatch.setenv("QNNBENCH_DEVICE", "cpu")
    with TestClient(app) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        assert client.get("/readyz").json()["model_version"] == "untrained"
        r = client.post("/predict", json={"pixels": [1, 0] * 8})
        assert r.status_code == 200 and r.headers["X-Model-Version"] == "untrained"
        body = r.json()
        assert body["digit"] in (3, 6) and -1 <= body["score"] <= 1
        assert client.post("/predict", json={"pixels": [0.5] * 16}).status_code == 422
        assert client.post("/predict", json={"pixels": [1] * 15}).status_code == 422
        text = client.get("/metrics").text
        assert 'qnnbench_requests_total{outcome="ok"} 1.0' in text
        assert 'qnnbench_requests_total{outcome="invalid"} 1.0' in text
        assert "qnnbench_batch_size_bucket" in text

        app.state.draining = True
        assert client.get("/readyz").status_code == 503
        r = client.post("/predict", json={"pixels": [1, 0] * 8})
        assert r.status_code == 503 and r.headers["Retry-After"] == "1"
        app.state.draining = False
