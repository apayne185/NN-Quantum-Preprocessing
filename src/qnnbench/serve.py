"""Inference service for the QNN with dynamic micro-batching and load shedding.

    QNNBENCH_CHECKPOINT=runs/qnn.pt uvicorn qnnbench.serve:app --port 8000

A state-vector simulation costs nearly the same for 1 sample as for dozens
(see the ``batch`` benchmark), so serving one request per forward pass wastes
most of the hardware. Requests are queued; a single worker collects up to
``max_batch`` of them, or whatever arrived within ``max_wait_ms`` of the
first, and runs them as one batch. ``max_batch=1`` disables batching for
comparison (see :mod:`qnnbench.loadtest`).

Overload behaviour, which an unbounded queue gets wrong: past saturation every
extra client only adds queueing delay, so latency grows without bound while
throughput stays flat. Instead:

* The queue is bounded (``max_queue``). When it is full the request is
  rejected at once with 503 + ``Retry-After``, so clients back off or retry
  another replica rather than wait.
* Every request has a deadline (``request_timeout_ms``). Requests that expire
  while queued are dropped before inference, so no compute is spent on
  answers nobody is waiting for; the client gets 504.
* On shutdown the server stops admitting work (``/readyz`` turns 503 so a load
  balancer stops routing to it), finishes what is queued, then exits.

Configuration is read from ``QNNBENCH_*`` environment variables; see
:class:`Settings`.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields
from typing import Any

import torch
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from qnnbench.env import resolve_device
from qnnbench.logs import configure_logging
from qnnbench.models import QNN

log = logging.getLogger("qnnbench.serve")


@dataclass(frozen=True)
class Settings:
    """Server configuration; each field is read from ``QNNBENCH_<NAME>``."""

    checkpoint: str | None = None
    device: str = "auto"
    fuse: int = 5
    max_batch: int = 64
    max_wait_ms: float = 5.0
    max_queue: int = 256
    request_timeout_ms: float = 2000.0
    drain_timeout_s: float = 10.0
    # CPU only. Leave a core for the event loop: if torch's threads take every
    # core, HTTP parsing stalls, requests reach the batcher late, and batches
    # stay small (measured in the README's serving results).
    torch_threads: int | None = None

    @classmethod
    def from_env(cls, env=os.environ) -> Settings:
        values: dict[str, Any] = {}
        for f in fields(cls):
            raw = env.get(f"QNNBENCH_{f.name.upper()}")
            if raw is None or raw == "":
                continue
            kind = str(f.type)
            values[f.name] = (
                int(raw) if "int" in kind else float(raw) if "float" in kind else raw
            )  # fmt: skip
        s = cls(**values)
        if s.max_batch < 1 or s.max_queue < 1 or s.max_wait_ms < 0 or s.request_timeout_ms <= 0:
            raise ValueError(f"invalid server settings: {s}")
        return s


class Metrics:
    """Prometheus metrics on a per-app registry (so tests can build several apps)."""

    def __init__(self):
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "qnnbench_requests", "Prediction requests by outcome", ["outcome"],
            registry=self.registry,
        )  # fmt: skip
        self.latency = Histogram(
            "qnnbench_request_latency_seconds", "End-to-end latency of successful requests",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
            registry=self.registry,
        )  # fmt: skip
        self.batch_size = Histogram(
            "qnnbench_batch_size", "Requests per inference batch",
            buckets=(1, 2, 4, 8, 16, 32, 64, 128), registry=self.registry,
        )  # fmt: skip
        self.inference = Histogram(
            "qnnbench_inference_seconds", "Model time per batch",
            buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
            registry=self.registry,
        )  # fmt: skip
        self.queue_depth = Gauge(
            "qnnbench_queue_depth", "Requests waiting for a batch", registry=self.registry
        )
        self.expired = Counter(
            "qnnbench_expired", "Requests dropped before inference because their deadline passed",
            registry=self.registry,
        )  # fmt: skip


class Overloaded(Exception):
    """The queue is full; the caller should shed the request."""


class MicroBatcher:
    """Collects concurrent requests into batches for one inference worker thread."""

    def __init__(
        self,
        fn: Callable[[torch.Tensor], torch.Tensor],
        max_batch: int,
        max_wait_ms: float,
        max_queue: int = 0,  # 0 = unbounded
        metrics: Metrics | None = None,
    ):
        self.fn, self.max_batch, self.max_wait = fn, max_batch, max_wait_ms / 1e3
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        self.metrics = metrics or Metrics()
        # One thread: batches run back to back, and torch releases the GIL inside
        # kernels, so the event loop keeps accepting requests meanwhile.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
        self._task: asyncio.Task | None = None

    def start(self):
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def drain(self, timeout: float):
        """Wait until queued work is finished, up to ``timeout`` seconds."""
        deadline = time.monotonic() + timeout
        while not self.queue.empty() and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

    async def stop(self):
        if self._task:
            self._task.cancel()
        self.executor.shutdown(wait=True)

    async def submit(self, x: torch.Tensor, timeout: float | None = None) -> float:
        """Queue one input; raises Overloaded if full, TimeoutError past ``timeout`` s."""
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        deadline = loop.time() + timeout if timeout else float("inf")
        try:
            self.queue.put_nowait((x, fut, deadline))
        except asyncio.QueueFull:
            raise Overloaded from None
        self.metrics.queue_depth.set(self.queue.qsize())
        # shield: a timeout cancels our wait, not the queued future; the
        # worker sees the expired deadline and skips it.
        return await asyncio.wait_for(asyncio.shield(fut), timeout)

    async def _next_batch(self, loop) -> list:
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
        self.metrics.queue_depth.set(self.queue.qsize())
        return batch

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            batch = await self._next_batch(loop)
            now = loop.time()
            live = [(x, fut) for x, fut, deadline in batch if deadline > now and not fut.done()]
            if expired := len(batch) - len(live):
                self.metrics.expired.inc(expired)
                for _, fut, deadline in batch:
                    if deadline <= now and not fut.done():
                        fut.set_exception(TimeoutError())
            if not live:
                continue
            xs = torch.stack([x for x, _ in live])
            t0 = time.perf_counter()
            try:
                out = await loop.run_in_executor(self.executor, self.fn, xs)
            except Exception as e:  # fail this batch's requests, keep serving
                log.exception("inference batch failed", extra={"batch_size": len(live)})
                for _, fut in live:
                    if not fut.done():
                        fut.set_exception(e)
                continue
            self.metrics.inference.observe(time.perf_counter() - t0)
            self.metrics.batch_size.observe(len(live))
            for (_, fut), y in zip(live, out.tolist(), strict=True):
                if not fut.done():  # client may have disconnected
                    fut.set_result(y)


class PredictRequest(BaseModel):
    pixels: list[float] = Field(min_length=16, max_length=16, description="binarized 4x4 image")


class PredictResponse(BaseModel):
    digit: int
    score: float  # <Z> in [-1, 1]; positive means "3"
    model_version: str


def load_model(checkpoint: str | None, device: torch.device, fuse: int) -> tuple[QNN, str]:
    """Load weights; the version is a content hash, so replicas can be compared."""
    model = QNN(fuse=fuse, grad_method="autograd")
    version = "untrained"
    if checkpoint:
        with open(checkpoint, "rb") as f:
            version = hashlib.sha256(f.read()).hexdigest()[:12]
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["state_dict"])
    return model.to(device).eval(), version


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    settings = Settings.from_env()
    device = resolve_device(settings.device)
    if settings.torch_threads:
        torch.set_num_threads(settings.torch_threads)
    model, version = load_model(settings.checkpoint, device, settings.fuse)

    @torch.no_grad()
    def infer(xs: torch.Tensor) -> torch.Tensor:
        return model(xs.to(device)).cpu()

    metrics = Metrics()
    batcher = MicroBatcher(
        infer, settings.max_batch, settings.max_wait_ms, settings.max_queue, metrics
    )
    batcher.start()
    app.state.batcher, app.state.metrics, app.state.settings = batcher, metrics, settings
    app.state.device, app.state.model_version, app.state.draining = device, version, False
    configured = {k: v for k, v in vars(settings).items() if v is not None}
    log.info("server ready", extra={**configured, "device": str(device), "model_version": version})
    yield
    app.state.draining = True
    log.info("draining", extra={"queued": batcher.queue.qsize()})
    await batcher.drain(settings.drain_timeout_s)
    await batcher.stop()
    log.info("stopped")


app = FastAPI(title="qnnbench", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    """Liveness: the process is up and the event loop responds."""
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(response: Response):
    """Readiness: accepting traffic. 503 while draining, so load balancers route away."""
    if app.state.draining:
        response.status_code = 503
        return {"status": "draining"}
    return {
        "status": "ready",
        "device": str(app.state.device),
        "model_version": app.state.model_version,
        "queue_depth": app.state.batcher.queue.qsize(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest, response: Response):
    m: Metrics = app.state.metrics
    if any(p not in (0.0, 1.0) for p in req.pixels):
        m.requests.labels("invalid").inc()
        raise HTTPException(422, "pixels must be binary (0 or 1)")
    if app.state.draining:
        m.requests.labels("draining").inc()
        raise HTTPException(503, "server is shutting down", headers={"Retry-After": "1"})
    t0 = time.perf_counter()
    try:
        score = await app.state.batcher.submit(
            torch.tensor(req.pixels), timeout=app.state.settings.request_timeout_ms / 1e3
        )
    except Overloaded:
        m.requests.labels("shed").inc()
        raise HTTPException(503, "overloaded, retry later", headers={"Retry-After": "1"}) from None
    except (asyncio.TimeoutError, TimeoutError):
        m.requests.labels("timeout").inc()
        raise HTTPException(504, "request deadline exceeded") from None
    except Exception as e:
        m.requests.labels("error").inc()
        raise HTTPException(500, f"inference failed: {type(e).__name__}") from e
    m.requests.labels("ok").inc()
    m.latency.observe(time.perf_counter() - t0)
    response.headers["X-Model-Version"] = app.state.model_version
    return PredictResponse(
        digit=3 if score > 0 else 6, score=score, model_version=app.state.model_version
    )


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(app.state.metrics.registry), media_type=CONTENT_TYPE_LATEST)
