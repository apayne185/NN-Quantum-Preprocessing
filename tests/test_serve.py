import asyncio

import pytest
import torch

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from qnnbench.serve import MicroBatcher, app  # noqa: E402


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


def test_predict_endpoint_and_metrics(monkeypatch):
    monkeypatch.setenv("QNNBENCH_DEVICE", "cpu")
    with TestClient(app) as client:
        assert client.get("/healthz").json()["status"] == "ok"
        r = client.post("/predict", json={"pixels": [1, 0] * 8})
        assert r.status_code == 200
        body = r.json()
        assert body["digit"] in (3, 6) and -1 <= body["score"] <= 1
        assert client.post("/predict", json={"pixels": [0.5] * 16}).status_code == 422
        assert client.post("/predict", json={"pixels": [1] * 15}).status_code == 422
        assert "qnnbench_requests_total 1" in client.get("/metrics").text
