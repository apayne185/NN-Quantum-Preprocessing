"""Closed-loop load test for the inference service.

    python -m qnnbench.loadtest --spawn --max-batch 1 64 --concurrency 64

With ``--spawn`` it starts one uvicorn server per ``--max-batch`` value, so
batching on/off is compared on identical hardware; otherwise it targets
``--url``. ``concurrency`` clients each send their next request as soon as
the previous one returns.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx


async def run_load(
    url: str, concurrency: int, n_requests: int, transport: httpx.AsyncBaseTransport | None = None
) -> dict:
    """Closed loop: ``concurrency`` clients, each sending its next request on a response."""
    latencies: list[float] = []
    errors = 0
    status_counts: dict[int, int] = {}
    remaining = n_requests
    rng = random.Random(0)

    async def client(http: httpx.AsyncClient):
        nonlocal remaining, errors
        while remaining > 0:
            remaining -= 1
            pixels = [float(rng.random() > 0.5) for _ in range(16)]
            t0 = time.perf_counter()
            try:
                r = await http.post(f"{url}/predict", json={"pixels": pixels})
            except httpx.HTTPError:
                errors += 1
                continue
            status_counts[r.status_code] = status_counts.get(r.status_code, 0) + 1
            if r.status_code == 200:
                latencies.append((time.perf_counter() - t0) * 1e3)
            elif r.status_code not in (503, 504):  # shed / deadline: expected under overload
                errors += 1

    limits = httpx.Limits(max_connections=concurrency)
    async with httpx.AsyncClient(timeout=60, limits=limits, transport=transport) as http:
        t0 = time.perf_counter()
        await asyncio.gather(*(client(http) for _ in range(concurrency)))
        elapsed = time.perf_counter() - t0
        server_metrics = (await http.get(f"{url}/metrics")).text

    lat = sorted(latencies)
    pct = lambda q: lat[int(q * (len(lat) - 1))] if lat else float("nan")  # noqa: E731
    return {
        "concurrency": concurrency,
        "requests": len(lat),
        "errors": errors,
        "status_counts": status_counts,
        "shed_503": status_counts.get(503, 0),
        "timeout_504": status_counts.get(504, 0),
        "throughput_rps": len(lat) / elapsed,
        "p50_ms": statistics.median(lat) if lat else float("nan"),
        "p95_ms": pct(0.95),
        "p99_ms": pct(0.99),
        "server_metrics": server_metrics,
    }


def _wait_ready(url: str, proc: subprocess.Popen, timeout: float = 60):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("server exited during startup")
        try:
            if httpx.get(f"{url}/readyz", timeout=1).status_code == 200:
                return
        except httpx.HTTPError:
            time.sleep(0.25)
    raise TimeoutError("server did not become ready")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--spawn", action="store_true", help="start a server per --max-batch value")
    p.add_argument("--max-batch", type=int, nargs="+", default=[1, 64])
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 16, 64])
    p.add_argument("--requests", type=int, default=1000)
    p.add_argument("--torch-threads", type=int, help="QNNBENCH_TORCH_THREADS for spawned servers")
    p.add_argument("--max-queue", type=int, help="QNNBENCH_MAX_QUEUE for spawned servers")
    p.add_argument(
        "--limit-concurrency",
        type=int,
        help="uvicorn --limit-concurrency: reject with 503 at the connection layer, before parsing",
    )
    p.add_argument("--out", type=Path)
    args = p.parse_args(argv)

    rows = []
    for max_batch in args.max_batch if args.spawn else [None]:
        proc = None
        if args.spawn:
            port = args.url.rsplit(":", 1)[1]
            env = {**os.environ, "QNNBENCH_MAX_BATCH": str(max_batch)}
            if args.torch_threads:
                env["QNNBENCH_TORCH_THREADS"] = str(args.torch_threads)
            if args.max_queue:
                env["QNNBENCH_MAX_QUEUE"] = str(args.max_queue)
            proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "qnnbench.serve:app", "--port", port,
                 "--log-level", "warning",
                 *(["--limit-concurrency", str(args.limit_concurrency)]
                   if args.limit_concurrency else [])],
                env=env,
            )  # fmt: skip
            _wait_ready(args.url, proc)
        try:
            for c in args.concurrency:
                r = asyncio.run(run_load(args.url, c, args.requests))
                r["max_batch"] = max_batch
                rows.append(r)
                print(
                    f"max_batch={max_batch} concurrency={c:>3}: "
                    f"{r['throughput_rps']:7.1f} req/s  p50={r['p50_ms']:7.1f} ms  "
                    f"p95={r['p95_ms']:7.1f} ms  p99={r['p99_ms']:7.1f} ms  "
                    f"shed={r['shed_503']} timeout={r['timeout_504']} errors={r['errors']} "
                    f"statuses={r['status_counts']}"
                )
        finally:
            if proc:
                proc.terminate()
                proc.wait(timeout=30)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
