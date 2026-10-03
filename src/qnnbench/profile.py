"""Profile QNN training steps with torch.profiler and export a timeline trace.

    python -m qnnbench.profile --n-qubits 17 --fuse 5 --out runs/profile
    # open runs/profile/trace.json in https://ui.perfetto.dev or chrome://tracing

Each gate and fused block is a named range ("fused_dense:k=5,4gates",
"fused_diag:16gates", ...), so the trace shows where a step's time goes per
circuit block, with the aten kernels underneath. On CUDA the trace also has
the GPU stream: gaps between kernels there are launch overhead or host
synchronisation. ``--nvtx`` additionally emits NVTX ranges for Nsight Systems:

    nsys profile -t cuda,nvtx python -m qnnbench.profile --device cuda --nvtx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, schedule

from qnnbench.env import env_info, resolve_device
from qnnbench.models import QNN
from qnnbench.sim import statevector


def run(
    n_qubits: int = 17,
    batch: int = 32,
    fuse: int = 5,
    grad_method: str = "autograd",
    device: str = "auto",
    steps: int = 3,
    nvtx: bool = False,
    out: Path = Path("runs/profile"),
    row_limit: int = 15,
) -> dict:
    dev = resolve_device(device)
    torch.manual_seed(0)
    model = QNN(n_data=n_qubits - 1, fuse=fuse, grad_method=grad_method).to(dev)
    x = (torch.rand(batch, n_qubits - 1) > 0.5).float().to(dev)

    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    statevector.LABELS.update(enabled=True, nvtx=nvtx and dev.type == "cuda")
    try:
        # wait 1 step, warm up 1 (profiler running, results discarded), record `steps`.
        with profile(
            activities=activities,
            schedule=schedule(wait=1, warmup=1, active=steps),
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            for _ in range(steps + 2):
                with torch.profiler.record_function("train_step"):
                    model.zero_grad(set_to_none=True)
                    model(x).sum().backward()
                prof.step()
    finally:
        statevector.LABELS.update(enabled=False, nvtx=False)

    out.mkdir(parents=True, exist_ok=True)
    trace = out / "trace.json"
    prof.export_chrome_trace(str(trace))
    sort = "self_cuda_time_total" if dev.type == "cuda" else "self_cpu_time_total"
    table = prof.key_averages().table(sort_by=sort, row_limit=row_limit)
    (out / "summary.txt").write_text(table)

    config = dict(n_qubits=n_qubits, batch=batch, fuse=fuse, grad_method=grad_method,
                  steps=steps)  # fmt: skip
    meta = {"config": config, "env": env_info(dev), "trace": str(trace)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(table)
    print(f"-> {trace} (open in https://ui.perfetto.dev)")
    return meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--n-qubits", type=int, default=17)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--fuse", type=int, default=5)
    p.add_argument("--grad-method", default="autograd")
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--nvtx", action="store_true", help="emit NVTX ranges (CUDA only)")
    p.add_argument("--out", type=Path, default=Path("runs/profile"))
    args = p.parse_args(argv)
    run(**vars(args))


if __name__ == "__main__":
    main()
