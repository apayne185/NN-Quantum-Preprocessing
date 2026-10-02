"""Benchmark suites. Each isolates one trade-off and returns a list of Measurements.

All suites run on CPU or CUDA; sizes scale down with ``quick=True`` (used by CI).
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable

import torch

from qnnbench.bench.harness import Measurement, fits_on_device, measure, saved_tensor_bytes
from qnnbench.models import QNN
from qnnbench.sim import GRAD_METHODS, basis_state, expectation

BATCH = 32
DEFAULT_FUSE = 5


def _inputs(batch: int, n_qubits: int, device, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(batch, n_qubits - 1, generator=g) > 0.5).float().to(device)


def _qnn(n_qubits: int, device, fuse: int = DEFAULT_FUSE, **kw) -> QNN:
    torch.manual_seed(0)
    return QNN(n_data=n_qubits - 1, fuse=fuse, **kw).to(device)


def _passes(model: QNN) -> int:
    return len(model.plan) if model.plan is not None else len(model.circuit.ops)


def _eff_bandwidth(model: QNN, batch: int, n: int, ms: float, itemsize: int = 8) -> float:
    """GB/s under a minimum-traffic model: each pass reads and writes the state once."""
    bytes_moved = _passes(model) * 2 * batch * 2**n * itemsize
    return bytes_moved / (ms / 1e3) / 1e9


def _state_bytes(batch: int, n: int, itemsize: int = 8) -> int:
    return batch * 2**n * itemsize


# --------------------------------------------------------------------------- suites


def suite_qubits(device, quick: bool) -> list[Measurement]:
    """Forward latency vs qubit count: exponential state, launch-bound vs bandwidth-bound."""
    top = 13 if quick else (25 if device.type == "cuda" else 21)
    out = []
    for n in range(7, top + 1, 2):
        if not fits_on_device(4 * _state_bytes(BATCH, n), device):
            break
        x = _inputs(BATCH, n, device)
        for fuse in (0, DEFAULT_FUSE):
            model = _qnn(n, device, fuse=fuse)
            with torch.no_grad():
                m = measure(lambda: model(x), suite="qubits", case={"n_qubits": n, "fuse": fuse},
                            device=device, items_per_call=BATCH)  # fmt: skip
            m.extra["passes"] = _passes(model)
            m.extra["eff_GBps"] = _eff_bandwidth(model, BATCH, n, m.p50_ms)
            out.append(m)
    return out


def suite_batch(device, quick: bool) -> list[Measurement]:
    """Latency vs throughput as the batch grows: where batching stops paying off."""
    n = 13 if quick else 17
    top = 64 if quick else (1024 if device.type == "cuda" else 256)
    model = _qnn(n, device)
    out = []
    b = 1
    while b <= top and fits_on_device(4 * _state_bytes(b, n), device):
        x = _inputs(b, n, device)
        with torch.no_grad():
            m = measure(lambda: model(x), suite="batch", case={"n_qubits": n, "batch": b},
                        device=device, items_per_call=b)  # fmt: skip
        m.extra["eff_GBps"] = _eff_bandwidth(model, b, n, m.p50_ms)
        out.append(m)
        b *= 2
    return out


def suite_fusion(device, quick: bool) -> list[Measurement]:
    """Max fused block width k: fewer passes over memory vs 2^k FLOPs per amplitude."""
    n = 13 if quick else 17
    x = _inputs(BATCH, n, device)
    out = []
    for k in (0, 2, 3, 4, 5, 6, 7):
        model = _qnn(n, device, fuse=k)
        with torch.no_grad():
            fwd = measure(lambda: model(x), suite="fusion",
                          case={"n_qubits": n, "fuse": k, "pass": "forward"},
                          device=device, items_per_call=BATCH)  # fmt: skip
        train = measure(lambda: model(x).sum().backward(), suite="fusion",
                        case={"n_qubits": n, "fuse": k, "pass": "fwd+bwd"},
                        device=device, items_per_call=BATCH)  # fmt: skip
        for m in (fwd, train):
            m.extra["passes"] = _passes(model)
        out += [fwd, train]
    return out


def suite_grad(device, quick: bool) -> list[Measurement]:
    """Gradient methods: time per training step vs memory kept for backward."""
    sizes = (9, 11) if quick else (9, 13, 17)
    out = []
    for n in sizes:
        x = _inputs(BATCH, n, device)
        for method in GRAD_METHODS:
            for fuse in (0, DEFAULT_FUSE):
                model = _qnn(n, device, fuse=fuse, grad_method=method)
                saved = saved_tensor_bytes(lambda: model(x)) - model.theta.numel() * 4
                m = measure(lambda: model(x).sum().backward(), suite="grad",
                            case={"n_qubits": n, "method": method, "fuse": fuse},
                            device=device, items_per_call=BATCH, warmup=1,
                            repeats=5 if method == "param_shift" else 10)  # fmt: skip
                m.extra["saved_MB"] = saved / 2**20
                out.append(m)
    return out


def suite_dtype(device, quick: bool) -> list[Measurement]:
    """complex64 vs complex128: speed and memory against numerical error."""
    sizes = (9, 13) if quick else (13, 17, 19)
    out = []
    for n in sizes:
        x = _inputs(BATCH, n, device)
        ref = _qnn(n, device, dtype=torch.complex128)
        with torch.no_grad():
            expected = ref(x)
        for dtype in (torch.complex64, torch.complex128):
            model = _qnn(n, device, dtype=dtype)
            with torch.no_grad():
                model.theta.copy_(ref.theta.to(model.theta.dtype))
                err = (model(x).double() - expected).abs().max().item()
                m = measure(lambda: model(x), suite="dtype",
                            case={"n_qubits": n, "dtype": str(dtype).removeprefix("torch.")},
                            device=device, items_per_call=BATCH)  # fmt: skip
            m.extra["max_abs_err"] = err
            m.extra["state_MB"] = _state_bytes(BATCH, n, dtype.itemsize) / 2**20
            out.append(m)
    return out


def suite_exec(device, quick: bool) -> list[Measurement]:
    """Eager vs torch.compile vs CUDA graphs. Small circuits are launch-bound."""
    sizes = (5, 9) if quick else (5, 9, 13, 17)
    modes = ["eager", "compile"]
    if device.type == "cuda":
        modes.append("reduce-overhead")  # inductor + CUDA graphs: one launch per replay
    out = []
    for n in sizes:
        x = _inputs(BATCH, n, device)
        for mode in modes:
            model = _qnn(n, device)
            torch._dynamo.reset()
            fn = (
                model
                if mode == "eager"
                else torch.compile(model, mode=None if mode == "compile" else mode)
            )
            try:
                with torch.no_grad(), warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t0 = time.perf_counter()
                    fn(x)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    first_call_s = time.perf_counter() - t0
                    m = measure(lambda: fn(x), suite="exec", case={"n_qubits": n, "mode": mode},
                                device=device, items_per_call=BATCH)  # fmt: skip
            except Exception as e:  # report, don't abort the suite
                print(f"  exec n={n} mode={mode} failed: {type(e).__name__}: {str(e)[:120]}")
                continue
            m.extra["first_call_s"] = first_call_s
            out.append(m)
    return out


def suite_frameworks(device, quick: bool) -> list[Measurement]:
    """Same QNN forward pass in this simulator, PennyLane and Cirq."""
    sizes = (9,) if quick else (9, 13, 17)
    out = []
    for n in sizes:
        x = _inputs(BATCH, n, device)
        model = _qnn(n, device)
        with torch.no_grad():
            ours = model(x)
            impls: dict[str, Callable[[], object]] = {
                "qnnbench (fused)": lambda: model(x),
                "qnnbench (unfused)": lambda: expectation(
                    model.circuit,
                    model.theta,
                    basis_state(x, n, 1),
                    0,
                ),
            }
            for name, factory in (("pennylane", _pennylane_qnn), ("cirq", _cirq_qnn)):
                if name == "cirq" and n > 13 and not quick:
                    continue  # per-sample loop; minutes per call at 17 qubits
                try:
                    fn, check = factory(model, x)
                except ImportError:
                    print(f"  {name} not installed; skipping")
                    continue
                err = (check().to(ours.device, ours.dtype) - ours).abs().max().item()
                if err > 1e-4:
                    raise AssertionError(f"{name} disagrees with qnnbench by {err}")
                impls[name] = fn
            for name, fn in impls.items():
                m = measure(fn, suite="frameworks", case={"n_qubits": n, "framework": name},
                            device=device, items_per_call=BATCH, warmup=1, repeats=5)  # fmt: skip
                out.append(m)
    return out


def _pennylane_qnn(model: QNN, x: torch.Tensor):
    import pennylane as qml

    n = model.circuit.n_qubits
    dev = qml.device("default.qubit", wires=n)
    theta = model.theta.detach()

    @qml.qnode(dev, interface="torch")
    def circuit(state):
        qml.StatePrep(state, wires=range(n))  # broadcast over the batch dimension
        qml.PauliX(0)
        qml.Hadamard(0)
        for op in model.circuit.ops:
            if op.param is None:
                continue
            gate = qml.IsingXX if op.gate == "xx" else qml.IsingZZ
            # Cirq's XX**t equals IsingXX(pi * t) up to a global phase.
            gate(math.pi * theta[op.param], wires=list(op.qubits))
        qml.Hadamard(0)
        return qml.expval(qml.PauliZ(0))

    states = basis_state(x.cpu(), n, offset=1)
    fn = lambda: circuit(states)  # noqa: E731
    return fn, lambda: torch.as_tensor(fn())


def _cirq_qnn(model: QNN, x: torch.Tensor):
    import cirq
    import numpy as np

    n = model.circuit.n_qubits
    qs = cirq.LineQubit.range(n)
    circuit = model.circuit.to_cirq(model.theta)
    sim = cirq.Simulator(dtype=np.complex64)
    bits = x.cpu().int().tolist()

    def fn():
        out = []
        for row in bits:
            initial = int("0" + "".join(map(str, row)), 2)
            psi = sim.simulate(circuit, qubit_order=qs, initial_state=initial).final_state_vector
            out.append(
                cirq.Z(qs[0])
                .expectation_from_state_vector(psi, {q: i for i, q in enumerate(qs)})
                .real
            )  # noqa: E501
        return out

    return fn, lambda: torch.tensor(fn())


def suite_kernels(device, quick: bool) -> list[Measurement]:
    """One XX gate three ways. Shows why TorchInductor can't help complex tensors."""
    from qnnbench.sim.gates import GATES
    from qnnbench.sim.statevector import apply_2q

    n = 13 if quick else 17
    a, b = 0, n // 2
    state = torch.randn(BATCH, 2**n, dtype=torch.complex64, device=device)
    mat = GATES["xx"].matrix(torch.tensor(0.3, device=device), torch.complex64, device)
    e = torch.polar(torch.ones((), device=device), torch.tensor(math.pi * 0.3, device=device))
    ca, cb = (1 + e) / 2, (1 - e) / 2
    shape = (-1, 2, 2 ** (b - a - 1), 2, 2 ** (n - b - 1))

    def flip_complex(s):
        v = s.reshape(shape)
        return (ca * v + cb * v.flip(1, 3)).reshape(s.shape)

    def flip_split_real(sr):
        v = sr.reshape(*shape, 2)
        w = v.flip(1, 3)
        re = ca.real * v[..., 0] - ca.imag * v[..., 1] + cb.real * w[..., 0] - cb.imag * w[..., 1]
        im = ca.real * v[..., 1] + ca.imag * v[..., 0] + cb.real * w[..., 1] + cb.imag * w[..., 0]
        return torch.stack([re, im], -1).reshape(sr.shape)

    split = torch.view_as_real(state).contiguous()
    impls = {
        "einsum (complex, eager)": lambda: apply_2q(state, mat, (a, b), n, False),
        "flip (complex, eager)": lambda: flip_complex(state),
        "flip (complex, compiled)": torch.compile(lambda: flip_complex(state)),
        "flip (split real, eager)": lambda: flip_split_real(split),
        "flip (split real, compiled)": torch.compile(lambda: flip_split_real(split)),
    }
    ref = apply_2q(state, mat, (a, b), n, False)
    out = []
    for name, fn in impls.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = fn()
        if result.dtype != torch.complex64:
            result = torch.view_as_complex(result)
        assert torch.allclose(result, ref, atol=1e-5), name
        m = measure(fn, suite="kernels", case={"n_qubits": n, "kernel": name},
                    device=device, items_per_call=BATCH)  # fmt: skip
        m.extra["eff_GBps"] = 2 * _state_bytes(BATCH, n) / (m.p50_ms / 1e3) / 1e9
        out.append(m)
    return out


def suite_quanv(device, quick: bool) -> list[Measurement]:
    """Quanvolution throughput: gate-by-gate vs fused unitary, chunk size, copy overlap."""
    import numpy as np

    from qnnbench.quanv import QuanvConfig, Quanvolution, quanvolve_dataset

    n_images = 256 if quick else 4096
    images = np.random.default_rng(0).integers(0, 256, size=(n_images, 28, 28), dtype=np.uint8)
    cases = [("gates", 1024, False), *(("fused", c, False) for c in (256, 1024, 4096))]
    if device.type == "cuda":
        cases += [("fused", c, True) for c in (256, 1024, 4096)]
    out = []
    for mode, chunk, overlap in cases:
        filt = Quanvolution(QuanvConfig(), mode, device)
        m = measure(lambda: quanvolve_dataset(images, filt, chunk, overlap), suite="quanv",
                    case={"mode": mode, "chunk": chunk, "overlap": overlap},
                    device=device, items_per_call=n_images, warmup=1, repeats=5)  # fmt: skip
        m.extra["circuits_per_s"] = m.throughput * 196
        out.append(m)
    return out


SUITES: dict[str, Callable[[torch.device, bool], list[Measurement]]] = {
    "qubits": suite_qubits,
    "batch": suite_batch,
    "fusion": suite_fusion,
    "grad": suite_grad,
    "dtype": suite_dtype,
    "exec": suite_exec,
    "frameworks": suite_frameworks,
    "kernels": suite_kernels,
    "quanv": suite_quanv,
}
