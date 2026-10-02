"""Correctness of the PyTorch simulator against Cirq, and of all gradient methods."""

import math

import numpy as np
import pytest
import torch

from qnnbench.sim import (
    GRAD_METHODS,
    Circuit,
    basis_state,
    circuit_unitary,
    expect_z,
    expect_z_all,
    expectation,
    plan_fusion,
    product_state,
    qnn_circuit,
    simulate,
    zero_state,
)
from qnnbench.sim.gates import GATES

cirq = pytest.importorskip("cirq")


def random_circuit(n: int, depth: int, seed: int) -> tuple[Circuit, torch.Tensor]:
    """Random circuit over every gate type, on random (including reversed) qubit pairs."""
    rng = np.random.default_rng(seed)
    c = Circuit(n)
    p = 0
    for _ in range(depth):
        name = rng.choice(list(GATES))
        g = GATES[name]
        qubits = rng.choice(n, size=g.n_qubits, replace=False).tolist()
        if g.parametric:
            c.add(name, *qubits, param=p)
            p += 1
        else:
            c.add(name, *qubits)
    params = torch.tensor(rng.uniform(-math.pi, math.pi, size=p), dtype=torch.float64)
    return c, params


def cirq_state(c: Circuit, params, initial: int = 0) -> np.ndarray:
    qs = cirq.LineQubit.range(c.n_qubits)
    result = cirq.Simulator(dtype=np.complex128).simulate(
        c.to_cirq(params), qubit_order=qs, initial_state=initial
    )
    return result.final_state_vector


@pytest.mark.parametrize("gate", sorted(GATES))
def test_each_gate_matches_cirq_unitary(gate):
    g = GATES[gate]
    c = Circuit(g.n_qubits)
    params = None
    if g.parametric:
        c.add(gate, *range(g.n_qubits), param=0)
        params = torch.tensor([0.37], dtype=torch.float64)
    else:
        c.add(gate, *range(g.n_qubits))
    ours = circuit_unitary(c, params, dtype=torch.complex128).numpy()
    np.testing.assert_allclose(ours, cirq.unitary(c.to_cirq(params)), atol=1e-12)


@pytest.mark.parametrize("seed", range(8))
def test_random_circuit_state_matches_cirq(seed):
    n = 5
    c, params = random_circuit(n, depth=40, seed=seed)
    initial = seed % 2**n
    state0 = torch.zeros(1, 2**n, dtype=torch.complex128)
    state0[0, initial] = 1
    ours = simulate(c, params, state0)[0].numpy()
    np.testing.assert_allclose(ours, cirq_state(c, params, initial), atol=1e-10)


def test_complex64_agrees_with_complex128_to_single_precision():
    c, params = random_circuit(6, depth=60, seed=1)
    s64 = simulate(c, params.float(), zero_state(1, 6, torch.complex64))
    s128 = simulate(c, params, zero_state(1, 6, torch.complex128))
    np.testing.assert_allclose(s64.numpy(), s128.numpy(), atol=1e-5)


def test_qnn_circuit_matches_original_model_and_cirq():
    c = qnn_circuit()
    assert c.n_qubits == 17 and c.n_params == 32  # same as the TFQ PQC layer
    params = torch.rand(32, dtype=torch.float64) * 2 * math.pi
    bits = torch.tensor([[1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1]])
    ours = simulate(c, params, basis_state(bits, 17, offset=1, dtype=torch.complex128))
    initial = int("0" + "".join(map(str, bits[0].tolist())), 2)
    np.testing.assert_allclose(ours[0].numpy(), cirq_state(c, params, initial), atol=1e-10)


def test_basis_state_equals_x_gates():
    bits = torch.tensor([[1, 0, 1], [0, 1, 1]])
    c = Circuit(4)
    expected = []
    for row in bits.tolist():
        c = Circuit(4)
        for q, b in enumerate(row):
            if b:
                c.add("x", q + 1)
        expected.append(simulate(c, None, zero_state(1, 4))[0])
    torch.testing.assert_close(basis_state(bits, 4, offset=1), torch.stack(expected))


def test_product_state_matches_ry_encoding():
    angles = torch.tensor([[0.3, 1.2, 2.5]], dtype=torch.float64)
    per_qubit = torch.stack([torch.cos(angles / 2), torch.sin(angles / 2)], dim=-1)
    ours = product_state(per_qubit.to(torch.complex128))
    c = Circuit(3)
    for q in range(3):
        c.add("ry", q, value=float(angles[0, q]))
    ref = simulate(c, None, zero_state(1, 3, torch.complex128))
    torch.testing.assert_close(ours, ref)


def test_expect_z_all_matches_cirq():
    c, params = random_circuit(4, depth=30, seed=3)
    state = simulate(c, params, zero_state(1, 4, torch.complex128))
    qs = cirq.LineQubit.range(4)
    psi = cirq_state(c, params)
    for q in range(4):
        ref = cirq.Z(qs[q]).expectation_from_state_vector(psi, {x: i for i, x in enumerate(qs)})
        assert expect_z(state, q, 4).item() == pytest.approx(ref.real, abs=1e-10)
    assert expect_z_all(state, 4).shape == (1, 4)


@pytest.mark.parametrize("method", GRAD_METHODS)
def test_gradients_match_finite_differences(method):
    n = 5
    c, params = random_circuit(n, depth=30, seed=7)
    # Share one parameter across two gates to exercise gradient accumulation.
    c.add("zz", 0, 3, param=0).add("rx", 2, param=0)
    params = params.clone().requires_grad_(True)
    state0 = basis_state(
        torch.tensor([[1, 0, 1, 1, 0], [0, 1, 1, 0, 1]]), n, dtype=torch.complex128
    )
    weights = torch.tensor([0.7, -1.3], dtype=torch.float64)

    loss = (weights * expectation(c, params, state0, readout=0, method=method)).sum()
    (grad,) = torch.autograd.grad(loss, params)

    eps = 1e-6
    fd = torch.zeros_like(params)
    with torch.no_grad():
        for i in range(params.numel()):
            plus, minus = params.clone(), params.clone()
            plus[i] += eps
            minus[i] -= eps
            f = lambda p: (weights * expect_z(simulate(c, p, state0), 0, n)).sum()  # noqa: E731
            fd[i] = (f(plus) - f(minus)) / (2 * eps)
    torch.testing.assert_close(grad, fd, atol=1e-6, rtol=1e-5)


def test_adjoint_saves_constant_memory_independent_of_depth():
    """Autograd saves activations per gate; adjoint saves only params + final state."""

    def saved_bytes(method, depth):
        c, params = random_circuit(8, depth=depth, seed=0)
        params = params.float().requires_grad_(True)
        state0 = zero_state(16, 8)
        seen = {}

        def pack(t):
            seen[t.untyped_storage().data_ptr()] = t.untyped_storage().nbytes()
            return t

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            expectation(c, params, state0, 0, method).sum()
        # Exclude the parameter vector itself, which grows with depth either way.
        return sum(seen.values()) - params.numel() * params.element_size()

    state_bytes = 16 * 2**8 * 8
    assert saved_bytes("autograd", 80) > 2 * saved_bytes("autograd", 20)
    assert saved_bytes("adjoint", 80) == saved_bytes("adjoint", 20) == state_bytes


@pytest.mark.parametrize("max_qubits", [2, 3, 4, 5])
@pytest.mark.parametrize("seed", range(4))
def test_fused_simulation_matches_unfused(max_qubits, seed):
    n = 6
    c, params = random_circuit(n, depth=50, seed=seed)
    plan = plan_fusion(c, max_qubits)
    assert sorted(i for b in plan for i in b.ops) == list(range(len(c.ops)))
    assert all(b.diagonal or len(b.qubits) <= max_qubits for b in plan)
    state0 = basis_state(torch.tensor([[1, 0, 1, 1, 0, 0]]), n, dtype=torch.complex128)
    torch.testing.assert_close(simulate(c, params, state0, plan), simulate(c, params, state0))


def test_fusion_merges_the_qnn_zz_layer_into_one_block():
    c = qnn_circuit()
    plan = plan_fusion(c, 5)
    diag_blocks = [b for b in plan if b.diagonal]
    assert len(diag_blocks) == 1 and len(diag_blocks[0].ops) == 16
    assert len(plan) < len(c.ops) / 4


@pytest.mark.parametrize("method", GRAD_METHODS)
def test_gradients_with_fusion_match_unfused(method):
    c = qnn_circuit(n_data=6)
    params = torch.rand(c.n_params, dtype=torch.float64, requires_grad=True)
    state0 = basis_state(torch.tensor([[1, 0, 1, 1, 0, 1]]), 7, offset=1, dtype=torch.complex128)
    (g_ref,) = torch.autograd.grad(expectation(c, params, state0, 0).sum(), params)
    out = expectation(c, params, state0, 0, method=method, plan=plan_fusion(c, 3))
    (g,) = torch.autograd.grad(out.sum(), params)
    torch.testing.assert_close(g, g_ref)
