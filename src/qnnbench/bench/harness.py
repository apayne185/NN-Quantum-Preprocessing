"""Timing and memory measurement shared by every benchmark suite.

Rules this harness enforces, because each one is a classic way to get wrong
GPU numbers:

* Warm up first, so one-off costs (CUDA context, cuBLAS handles, allocator
  growth, torch.compile) are not timed.
* On CUDA, time with CUDA events recorded on the stream, not with
  ``time.perf_counter()``: kernel launches are asynchronous, so host timers
  measure launch overhead unless you synchronise after every call.
* Report percentiles over many repeats, not one run.
* Record peak device memory relative to what was allocated before the call.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from qnnbench.env import env_info


@dataclass
class Measurement:
    suite: str
    case: dict
    times_ms: list[float]
    items_per_call: int = 1
    peak_mem_bytes: int | None = None
    extra: dict = field(default_factory=dict)

    @property
    def p50_ms(self) -> float:
        return statistics.median(self.times_ms)

    @property
    def p95_ms(self) -> float:
        ordered = sorted(self.times_ms)
        return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]

    @property
    def throughput(self) -> float:
        """Items per second at the median latency."""
        return self.items_per_call / (self.p50_ms / 1e3)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(p50_ms=self.p50_ms, p95_ms=self.p95_ms, throughput=self.throughput)
        if len(self.times_ms) > 1:
            d["std_ms"] = statistics.stdev(self.times_ms)
        return d


def measure(
    fn: Callable[[], object],
    *,
    suite: str,
    case: dict,
    device: torch.device,
    items_per_call: int = 1,
    warmup: int = 2,
    repeats: int = 10,
    max_seconds: float = 30.0,
) -> Measurement:
    """Time ``fn`` after ``warmup`` calls; stops early after ``max_seconds`` of repeats."""
    cuda = device.type == "cuda"
    for _ in range(warmup):
        fn()
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)

    times: list[float] = []
    budget_start = time.perf_counter()
    if cuda:
        events = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            events.append((start, end))
            if time.perf_counter() - budget_start > max_seconds:
                break
        torch.cuda.synchronize(device)
        times = [s.elapsed_time(e) for s, e in events]
        peak = torch.cuda.max_memory_allocated(device) - base
    else:
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1e3)
            if time.perf_counter() - budget_start > max_seconds:
                break
        peak = None
    return Measurement(suite, case, times, items_per_call, peak)


def saved_tensor_bytes(fn: Callable[[], torch.Tensor]) -> int:
    """Bytes autograd keeps alive for the backward pass of ``fn()`` (deduplicated by storage).

    Works on any device, so activation-memory comparisons do not need a GPU.
    """
    storages: dict[int, int] = {}

    def pack(t):
        storages[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        fn()
    return sum(storages.values())


def fits_on_device(bytes_needed: int, device: torch.device, headroom: float = 0.6) -> bool:
    """Skip cases that would OOM instead of crashing the whole suite."""
    if device.type != "cuda":
        return bytes_needed < 8 * 2**30  # keep CPU runs well under laptop RAM
    free, _ = torch.cuda.mem_get_info(device)
    return bytes_needed < headroom * free


def device_slug(env: dict) -> str:
    name = env["device_name"] if env["device"].startswith("cuda") else f"cpu-{env['device_name']}"
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def save_results(
    suite: str, measurements: list[Measurement], device: torch.device, out_dir: Path
) -> Path:
    env = env_info(device)
    path = Path(out_dir) / suite / f"{device_slug(env)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"suite": suite, "env": env, "measurements": [m.to_dict() for m in measurements]}
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def format_table(measurements: list[Measurement]) -> str:
    if not measurements:
        return "(no measurements)"
    keys = list(measurements[0].case)
    extra_keys = sorted({k for m in measurements for k in m.extra})
    header = [*keys, "p50_ms", "p95_ms", "items/s", "peak_MB", *extra_keys]
    rows = []
    for m in measurements:
        peak = "-" if m.peak_mem_bytes is None else f"{m.peak_mem_bytes / 2**20:.1f}"
        extras = [_fmt(m.extra.get(k, "")) for k in extra_keys]
        rows.append(
            [*(str(m.case.get(k)) for k in keys), f"{m.p50_ms:.3f}", f"{m.p95_ms:.3f}",
             f"{m.throughput:,.0f}", peak, *extras]
        )  # fmt: skip
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    lines = ["  ".join(h.rjust(w) for h, w in zip(header, widths, strict=True))]
    lines += ["  ".join(c.rjust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    return "\n".join(lines)


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)
