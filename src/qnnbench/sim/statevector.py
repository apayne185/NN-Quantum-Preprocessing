"""Batched state-vector simulation in plain PyTorch.

A batch of states is a ``(B, 2**n)`` complex tensor. Applying a gate to
qubits ``q1 < q2`` reshapes the state *as a view* to
``(B * 2**q1, 2, 2**(q2-q1-1), 2, rest)`` and contracts the gate along the two
size-2 axes, so each gate is one kernel over the whole batch with no Python
loop over samples.

Cost model (used by the bandwidth numbers in the benchmarks): every gate
reads and writes the full state, so a gate is memory-bound at roughly
``2 * B * 2**n * itemsize`` bytes. Diagonal gates (Z, CZ, ZZ, RZ) take a
cheaper path: a broadcast multiply with no matmul or transpose.
"""

from __future__ import annotations

import contextlib

import torch

from qnnbench.sim.circuit import Circuit, Op
from qnnbench.sim.fusion import Block
from qnnbench.sim.gates import GATES

Tensor = torch.Tensor

# Named profiler ranges per gate / fused block. Off by default so normal runs
# and torch.compile graphs are untouched; `qnnbench.profile` switches it on.
# With NVTX on, the same names appear in Nsight Systems timelines on CUDA.
LABELS = {"enabled": False, "nvtx": False}


@contextlib.contextmanager
def _labelled(name: str):
    with torch.profiler.record_function(name):
        if LABELS["nvtx"]:
            torch.cuda.nvtx.range_push(name)
            try:
                yield
            finally:
                torch.cuda.nvtx.range_pop()
        else:
            yield


def region(name: str):
    return _labelled(name) if LABELS["enabled"] else contextlib.nullcontext()


# --------------------------------------------------------------------------- state prep


def zero_state(batch: int, n: int, dtype=torch.complex64, device=None) -> Tensor:
    state = torch.zeros(batch, 2**n, dtype=dtype, device=device)
    state[:, 0] = 1
    return state


def basis_state(bits: Tensor, n: int, offset: int = 0, dtype=torch.complex64) -> Tensor:
    """Computational basis states |bits> on qubits ``offset .. offset+k-1``.

    Equivalent to starting from |0...0> and applying X to every qubit whose bit
    is 1, but costs one scatter instead of up to k full-state gate passes.
    """
    bits = bits.to(torch.int64)
    k = bits.shape[1]
    weights = 2 ** torch.arange(n - 1 - offset, n - 1 - offset - k, -1, device=bits.device)
    index = (bits * weights).sum(dim=1)
    state = torch.zeros(bits.shape[0], 2**n, dtype=dtype, device=bits.device)
    state.scatter_(1, index[:, None], 1)
    return state


def product_state(qubit_states: Tensor) -> Tensor:
    """Tensor product of per-qubit states: ``(B, n, 2)`` -> ``(B, 2**n)``."""
    state = qubit_states[:, 0]
    for q in range(1, qubit_states.shape[1]):
        state = (state[:, :, None] * qubit_states[:, q, None, :]).reshape(state.shape[0], -1)
    return state


# --------------------------------------------------------------------------- gate kernels


def apply_1q(state: Tensor, mat: Tensor, q: int, n: int, diagonal: bool) -> Tensor:
    s = state.reshape(-1, 2, 2 ** (n - q - 1))
    out = s * mat.view(1, 2, 1) if diagonal else torch.matmul(mat, s)
    return out.reshape(state.shape)


def apply_2q(state: Tensor, mat: Tensor, qubits: tuple[int, int], n: int, diagonal: bool) -> Tensor:
    a, b = qubits
    if a > b:  # reorder the gate's qubit axes so the lower index comes first
        a, b = b, a
        mat = mat.view(2, 2).T if diagonal else mat.view(2, 2, 2, 2).permute(1, 0, 3, 2)
    s = state.reshape(-1, 2, 2 ** (b - a - 1), 2, 2 ** (n - b - 1))
    if diagonal:
        out = s * mat.reshape(1, 2, 1, 2, 1)
    else:
        out = torch.einsum("xyij,aimjr->axmyr", mat.reshape(2, 2, 2, 2), s)
    return out.reshape(state.shape)


def apply_kq(state: Tensor, mat: Tensor, qubits: tuple[int, ...], n: int) -> Tensor:
    """Dense k-qubit gate (``qubits`` sorted): move target axes last, one GEMM, move back."""
    k = len(qubits)
    batch = state.shape[0]
    axes = [q + 1 for q in qubits]
    last = list(range(n + 1 - k, n + 1))
    s = state.reshape(batch, *([2] * n)).movedim(axes, last)
    shape = s.shape
    out = (s.reshape(-1, 2**k) @ mat.transpose(0, 1)).reshape(shape).movedim(last, axes)
    return out.reshape(batch, -1)


def apply(state: Tensor, mat: Tensor, op: Op, n: int) -> Tensor:
    diagonal = GATES[op.gate].diagonal
    if len(op.qubits) == 1:
        return apply_1q(state, mat, op.qubits[0], n, diagonal)
    return apply_2q(state, mat, (op.qubits[0], op.qubits[1]), n, diagonal)


def dagger(mat: Tensor, diagonal: bool) -> Tensor:
    return mat.conj() if diagonal else mat.conj().transpose(-1, -2)


# --------------------------------------------------------------------------- forward


def _param(op: Op, params: Tensor | None, real_dtype) -> Tensor | None:
    if op.param is not None:
        if params is None:
            raise ValueError(
                f"{op.gate} on {op.qubits} needs parameter {op.param}, got params=None"
            )
        return params[op.param]
    if op.value is not None:
        return torch.tensor(op.value, dtype=real_dtype)
    return None


def gate_matrices(circuit: Circuit, params: Tensor | None, dtype, device) -> list[Tensor]:
    real = torch.float64 if dtype == torch.complex128 else torch.float32
    mats = []
    for op in circuit.ops:
        p = _param(op, params, real)
        if p is not None:
            p = p.to(device)
        mats.append(GATES[op.gate].matrix(p, dtype, device))
    return mats


def simulate(
    circuit: Circuit, params: Tensor | None, state: Tensor, plan: list[Block] | None = None
) -> Tensor:
    """Apply every gate of ``circuit`` to a batch of states. Differentiable via autograd.

    With a ``plan`` from :func:`qnnbench.sim.fusion.plan_fusion`, gates are
    applied fused block by block; the result is the same up to rounding.
    """
    mats = gate_matrices(circuit, params, state.dtype, state.device)
    n = circuit.n_qubits
    if plan is None:
        for op, mat in zip(circuit.ops, mats, strict=True):
            with region(f"gate:{op.gate}{list(op.qubits)}"):
                state = apply(state, mat, op, n)
        return state
    for block in plan:
        if len(block.ops) == 1:
            i = block.ops[0]
            with region(f"gate:{circuit.ops[i].gate}{list(circuit.ops[i].qubits)}"):
                state = apply(state, mats[i], circuit.ops[i], n)
        elif block.diagonal:
            with region(f"fused_diag:{len(block.ops)}gates"):
                state = state * _fused_diagonal(circuit, mats, block, state)
        else:
            with region(f"fused_dense:k={len(block.qubits)},{len(block.ops)}gates"):
                state = apply_kq(state, _fused_unitary(circuit, mats, block), block.qubits, n)
    return state


def _fused_diagonal(circuit: Circuit, mats: list[Tensor], block: Block, like: Tensor) -> Tensor:
    """Product of a run of diagonal gates as one (1, 2**n) phase vector."""
    n = circuit.n_qubits
    d = torch.ones(1, 2**n, dtype=like.dtype, device=like.device)
    for i in block.ops:
        d = apply(d, mats[i], circuit.ops[i], n)
    return d


def _fused_unitary(circuit: Circuit, mats: list[Tensor], block: Block) -> Tensor:
    """Product of a block's gates as one 2**k x 2**k unitary on ``block.qubits``."""
    k = len(block.qubits)
    local = {q: j for j, q in enumerate(block.qubits)}
    u = torch.eye(2**k, dtype=mats[0].dtype, device=mats[0].device)  # row j = |j>
    for i in block.ops:
        op = circuit.ops[i]
        u = apply(
            u, mats[i], Op(op.gate, tuple(local[q] for q in op.qubits), op.param, op.value), k
        )
    return u.transpose(0, 1)  # column j = U |j>


def expect_z(state: Tensor, qubit: int, n: int) -> Tensor:
    """<Z> on one qubit for each state in the batch: P(0) - P(1)."""
    probs = (state.real**2 + state.imag**2).reshape(state.shape[0], 2**qubit, 2, -1)
    p = probs.sum(dim=(1, 3))
    return p[:, 0] - p[:, 1]


def expect_z_all(state: Tensor, n: int) -> Tensor:
    """<Z_q> for every qubit: ``(B, n)``."""
    return torch.stack([expect_z(state, q, n) for q in range(n)], dim=1)


# --------------------------------------------------------------------------- gradients


class _AdjointExpectation(torch.autograd.Function):
    """<Z_readout> with gradients by adjoint differentiation (Jones & Gacon, 2020).

    Autograd keeps every intermediate state alive for the backward pass, so its
    memory grows with circuit depth. Here the forward pass keeps only the final
    state; the backward pass walks the circuit in reverse, un-applying each
    gate, which keeps memory at a constant ~3 states at the price of roughly
    twice the gate applications.
    """

    @staticmethod
    def forward(ctx, params, state0, circuit, readout, plan):
        with torch.no_grad():
            final = simulate(circuit, params, state0, plan)
        ctx.save_for_backward(params, final)
        ctx.circuit, ctx.readout = circuit, readout
        return expect_z(final, readout, circuit.n_qubits)

    @staticmethod
    def backward(ctx, grad_out):
        params, final = ctx.saved_tensors
        circuit, n = ctx.circuit, ctx.circuit.n_qubits
        dtype, device = final.dtype, final.device
        real = torch.float64 if dtype == torch.complex128 else torch.float32
        z = torch.tensor([1, -1], dtype=dtype, device=device)

        with torch.no_grad():
            mats = gate_matrices(circuit, params, dtype, device)
            psi = final
            lam = apply_1q(final, z, ctx.readout, n, diagonal=True)  # O |psi>
            grad = torch.zeros(final.shape[0], params.numel(), dtype=real, device=device)
            for op, mat in zip(reversed(circuit.ops), reversed(mats), strict=True):
                gate = GATES[op.gate]
                psi = apply(psi, dagger(mat, gate.diagonal), op, n)
                if op.param is not None:
                    assert gate.derivative is not None  # Op() rejects params on fixed gates
                    dmat = gate.derivative(params[op.param], dtype, device)
                    mu = apply(psi, dmat, op, n)
                    grad[:, op.param] += 2 * (lam.conj() * mu).real.sum(dim=1)
                lam = apply(lam, dagger(mat, gate.diagonal), op, n)
            grad_params = (grad_out[:, None].to(real) * grad).sum(dim=0)
        return grad_params, None, None, None, None


class _ParameterShiftExpectation(torch.autograd.Function):
    """<Z_readout> with gradients by the parameter-shift rule.

    Needs only forward evaluations (2 per parameterised gate), which is what
    real quantum hardware supports, so it is the most expensive method on a
    simulator but the only one that transfers to a QPU.
    """

    @staticmethod
    def forward(ctx, params, state0, circuit, readout, plan):
        with torch.no_grad():
            out = expect_z(simulate(circuit, params, state0, plan), readout, circuit.n_qubits)
        ctx.save_for_backward(params, state0)
        ctx.circuit, ctx.readout, ctx.plan = circuit, readout, plan
        return out

    @staticmethod
    def backward(ctx, grad_out):
        params, state0 = ctx.saved_tensors
        circuit, n = ctx.circuit, ctx.circuit.n_qubits
        grad = torch.zeros(
            state0.shape[0], params.numel(), dtype=params.dtype, device=params.device
        )
        with torch.no_grad():
            for i, op in enumerate(circuit.ops):
                if op.param is None:
                    continue
                gate = GATES[op.gate]
                assert gate.shift is not None and gate.shift_coeff is not None
                evals = []
                for sign in (1, -1):
                    # Shift only this occurrence, so shared parameters sum correctly.
                    ops = list(circuit.ops)
                    shifted = Op(
                        op.gate, op.qubits, value=float(params[op.param]) + sign * gate.shift
                    )
                    ops[i] = shifted
                    c = Circuit(n, ops)
                    # Shifting a value keeps the circuit structure, so the plan still applies.
                    evals.append(expect_z(simulate(c, params, state0, ctx.plan), ctx.readout, n))
                grad[:, op.param] += gate.shift_coeff * (evals[0] - evals[1]).to(params.dtype)
        return (grad_out[:, None].to(params.dtype) * grad).sum(dim=0), None, None, None, None


GRAD_METHODS = ("autograd", "adjoint", "param_shift")


def expectation(
    circuit: Circuit,
    params: Tensor,
    state0: Tensor,
    readout: int,
    method: str = "autograd",
    plan: list[Block] | None = None,
) -> Tensor:
    """<Z_readout> after running ``circuit`` on ``state0``, differentiable w.r.t. ``params``.

    ``plan`` (gate fusion) speeds up every forward simulation, including the
    forward passes inside the adjoint and parameter-shift methods. The
    adjoint backward pass always walks the individual gates.
    """
    if method == "autograd":
        return expect_z(simulate(circuit, params, state0, plan), readout, circuit.n_qubits)
    if method == "adjoint":
        return _AdjointExpectation.apply(params, state0, circuit, readout, plan)
    if method == "param_shift":
        return _ParameterShiftExpectation.apply(params, state0, circuit, readout, plan)
    raise ValueError(f"unknown gradient method {method!r}; choose from {GRAD_METHODS}")


# --------------------------------------------------------------------------- utilities


def circuit_unitary(
    circuit: Circuit, params: Tensor | None = None, dtype=torch.complex64
) -> Tensor:
    """Full 2^n x 2^n unitary, by simulating every basis state as one batch.

    Lets a small fixed circuit be folded into a single matrix (one GEMM per
    batch instead of one kernel per gate). Only sensible for small n.
    """
    dim = 2**circuit.n_qubits
    basis = torch.eye(dim, dtype=dtype)
    return simulate(circuit, params, basis).T  # column j = U |j>
