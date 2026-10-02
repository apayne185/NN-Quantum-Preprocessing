"""Inference service for the QNN with dynamic micro-batching.

    QNNBENCH_CHECKPOINT=runs/qnn.pt uvicorn qnnbench.serve:app --port 8000

A state-vector simulation costs nearly the same for 1 sample as for dozens
(see the ``batch`` benchmark), so serving one request per forward pass wastes
most of the hardware. Requests are queued; a single worker collects up to
``MAX_BATCH`` of them, or whatever arrived within ``MAX_WAIT_MS`` of the
first, and runs them as one batch. ``MAX_BATCH=1`` disables batching for
comparison (see :mod:`qnnbench.loadtest`).

Configuration (environment): QNNBENCH_CHECKPOINT, QNNBENCH_DEVICE,
QNNBENCH_MAX_BATCH (64), QNNBENCH_MAX_WAIT_MS (5), QNNBENCH_FUSE (5).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from qnnbench.env import resolve_device
from qnnbench.models import QNN


class MicroBatcher:
    """Collects concurrent requests into batches for one inference worker thread."""

    def __init__(
        self, fn: Callable[[torch.Tensor], torch.Tensor], max_batch: int, max_wait_ms: float
    ):
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1e3
        self.queue: asyncio.Queue = asyncio.Queue()
        # One thread: batches run back to back, and torch releases the GIL inside
        # kernels, so the event loop keeps accepting requests meanwhile.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self.batch_sizes: deque[int] = deque(maxlen=10_000)
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def stop(self):
        if self._task:
            self._task.cancel()
        self.executor.shutdown(wait=True)

    async def submit(self, x: torch.Tensor) -> float:
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put((x, fut))
        return await fut

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = [await self.queue.get()]
            deadline = loop.time() + self.max_wait
            while len(batch) < self.max_batch:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break
            xs = torch.stack([x for x, _ in batch])
            try:
                out = await loop.run_in_executor(self.executor, self.fn, xs)
            except Exception as e:  # fail this batch's requests, keep serving
                for _, fut in batch:
                    if not fut.done():
                        fut.set_exception(e)
                continue
            self.batch_sizes.append(len(batch))
            for (_, fut), y in zip(batch, out.tolist(), strict=True):
                if not fut.done():  # client may have disconnected
                    fut.set_result(y)


class PredictRequest(BaseModel):
    pixels: list[float] = Field(min_length=16, max_length=16, description="binarized 4x4 image")


class PredictResponse(BaseModel):
    digit: int
    score: float  # <Z> in [-1, 1]; positive means "3"


def load_model(checkpoint: str | None, device: torch.device, fuse: int) -> QNN:
    model = QNN(fuse=fuse, grad_method="autograd")
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["state_dict"])
    return model.to(device).eval()


class Stats:
    def __init__(self):
        self.latencies_ms: deque[float] = deque(maxlen=10_000)
        self.requests = 0
        self.errors = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = resolve_device(os.environ.get("QNNBENCH_DEVICE", "auto"))
    model = load_model(
        os.environ.get("QNNBENCH_CHECKPOINT"), device, int(os.environ.get("QNNBENCH_FUSE", 5))
    )

    @torch.no_grad()
    def infer(xs: torch.Tensor) -> torch.Tensor:
        return model(xs.to(device)).cpu()

    batcher = MicroBatcher(
        infer,
        max_batch=int(os.environ.get("QNNBENCH_MAX_BATCH", 64)),
        max_wait_ms=float(os.environ.get("QNNBENCH_MAX_WAIT_MS", 5)),
    )
    batcher.start()
    app.state.batcher, app.state.stats, app.state.device = batcher, Stats(), device
    yield
    await batcher.stop()


app = FastAPI(title="qnnbench", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "device": str(app.state.device)}


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if any(p not in (0.0, 1.0) for p in req.pixels):
        raise HTTPException(422, "pixels must be binary (0 or 1)")
    stats: Stats = app.state.stats
    t0 = time.perf_counter()
    try:
        score = await app.state.batcher.submit(torch.tensor(req.pixels))
    except Exception as e:
        stats.errors += 1
        raise HTTPException(500, f"inference failed: {type(e).__name__}") from e
    stats.requests += 1
    stats.latencies_ms.append((time.perf_counter() - t0) * 1e3)
    return PredictResponse(digit=3 if score > 0 else 6, score=score)


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """Prometheus text format: request counts, latency and batch-size quantiles."""
    stats: Stats = app.state.stats
    sizes = sorted(app.state.batcher.batch_sizes)
    lat = sorted(stats.latencies_ms)
    lines = [
        f"qnnbench_requests_total {stats.requests}",
        f"qnnbench_errors_total {stats.errors}",
        f"qnnbench_batches_total {len(sizes)}",
    ]
    for q in (0.5, 0.95, 0.99):
        if lat:
            lines.append(
                f'qnnbench_latency_ms{{quantile="{q}"}} {lat[int(q * (len(lat) - 1))]:.3f}'
            )
        if sizes:
            lines.append(
                f'qnnbench_batch_size{{quantile="{q}"}} {sizes[int(q * (len(sizes) - 1))]}'
            )
    return "\n".join(lines) + "\n"
