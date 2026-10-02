"""Benchmark the original TFQ QNN in the same JSON format as ``qnnbench.bench``.

Runs in the legacy environment (Python 3.11, TFQ 0.7.3), so it does not import
qnnbench. From legacy/tfq:

    .venv-tfq/bin/python bench_tfq.py --out ../../results/frameworks

Measures a forward pass and a training step (forward + adjoint gradient,
TFQ's default differentiator) for the Farhi QNN at batch 32, so it lines up
with the ``frameworks`` and ``grad`` suites.
"""

import argparse
import json
import platform
import re
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import cirq
import numpy as np
import sympy
import tensorflow as tf
import tensorflow_quantum as tfq

BATCH = 32


def qnn(n_data):
    data = cirq.LineQubit.range(1, n_data + 1)
    readout = cirq.LineQubit(0)
    c = cirq.Circuit([cirq.X(readout), cirq.H(readout)])
    for prefix, gate in (("xx", cirq.XX), ("zz", cirq.ZZ)):
        for i, q in enumerate(data):
            c.append(gate(q, readout) ** sympy.Symbol(f"{prefix}-{i}"))
    c.append(cirq.H(readout))
    return c, cirq.Z(readout), data


def inputs(n_data, data_qubits, rng):
    bits = rng.random((BATCH, n_data)) > 0.5
    circuits = [cirq.Circuit(cirq.X(q) for q, b in zip(data_qubits, row) if b) for row in bits]
    return tfq.convert_to_tensor(circuits)


def time_ms(fn, warmup=2, repeats=10, max_seconds=30):
    for _ in range(warmup):
        fn()
    times, start = [], time.perf_counter()
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1e3)
        if time.perf_counter() - start > max_seconds:
            break
    return times


def measurement(suite, case, times):
    p50 = statistics.median(times)
    return {"suite": suite, "case": case, "times_ms": times, "items_per_call": BATCH,
            "peak_mem_bytes": None, "extra": {}, "p50_ms": p50,
            "p95_ms": sorted(times)[min(len(times) - 1, int(np.ceil(0.95 * len(times))) - 1)],
            "throughput": BATCH / (p50 / 1e3)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sizes", type=int, nargs="+", default=[9, 13, 17])
    p.add_argument("--out", type=Path, default=Path("../../results/frameworks"))
    args = p.parse_args()

    rng = np.random.default_rng(0)
    fwd, step = [], []
    for n in args.sizes:
        circuit, readout, data_qubits = qnn(n - 1)
        layer = tfq.layers.PQC(circuit, readout)  # default: adjoint differentiator
        x = inputs(n - 1, data_qubits, rng)
        y = tf.ones((BATCH, 1))

        def train_step():
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.square(layer(x) - y))
            return tape.gradient(loss, layer.trainable_variables)

        fwd.append(measurement("frameworks", {"n_qubits": n, "framework": "tensorflow-quantum"},
                               time_ms(lambda: layer(x))))
        step.append(measurement("grad", {"n_qubits": n, "method": "tfq adjoint", "fuse": "qsim"},
                                time_ms(train_step)))
        print(f"n={n}: forward p50 {fwd[-1]['p50_ms']:.2f} ms, "
              f"train step p50 {step[-1]['p50_ms']:.2f} ms")

    cpu = platform.processor() or platform.machine()
    env = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "python": platform.python_version(), "tensorflow": tf.__version__,
           "tfq": tfq.__version__, "device": "cpu", "device_name": cpu,
           "hostname": platform.node()}
    slug = "tfq-cpu-" + re.sub(r"[^a-z0-9]+", "-", cpu.lower()).strip("-")
    for suite, ms in (("frameworks", fwd), ("grad", step)):
        path = args.out.parent / suite / f"{slug}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"suite": suite, "env": env, "measurements": ms}, indent=2))
        print(f"-> {path}")


if __name__ == "__main__":
    main()
