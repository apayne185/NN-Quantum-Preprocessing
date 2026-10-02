import math

import numpy as np
import pytest
import torch

from qnnbench.quanv import (
    QuanvConfig,
    Quanvolution,
    encode,
    extract_patches,
    quanvolve_dataset,
    random_circuit,
)
from qnnbench.sim import expect_z_all, simulate, zero_state

cirq = pytest.importorskip("cirq")


def test_patches_are_row_major_2x2_blocks():
    img = torch.arange(16.0).reshape(1, 4, 4)
    patches = extract_patches(img)
    assert patches.shape == (4, 4)
    assert patches[0].tolist() == [0, 1, 4, 5]  # top-left block
    assert patches[1].tolist() == [2, 3, 6, 7]  # top-right block


def test_encoding_matches_ry_gates():
    x = torch.tensor([[0.1, 0.5, 0.9, 0.0]])
    circuit = random_circuit(QuanvConfig(n_layers=0))
    for q, v in enumerate(x[0].tolist()):
        circuit.add("ry", q, value=math.pi * v)
    torch.testing.assert_close(encode(x), simulate(circuit, None, zero_state(1, 4)))


def test_fused_and_gate_modes_agree_and_match_cirq():
    torch.manual_seed(0)
    images = torch.rand(3, 28, 28)
    cfg = QuanvConfig(n_layers=2, seed=4)
    fused = Quanvolution(cfg, "fused")(images)
    gates = Quanvolution(cfg, "gates")(images)
    assert fused.shape == (3, 4, 14, 14)
    torch.testing.assert_close(fused, gates, atol=1e-5, rtol=0)

    # One patch end to end through Cirq: encoding + random circuit + <Z_q>.
    patch = extract_patches(images[:1])[5]
    qs = cirq.LineQubit.range(4)
    c = cirq.Circuit(cirq.ry(math.pi * float(v)).on(qs[i]) for i, v in enumerate(patch))
    c += random_circuit(cfg).to_cirq()
    psi = cirq.Simulator(dtype=np.complex128).simulate(c, qubit_order=qs).final_state_vector
    ref = [cirq.Z(q).expectation_from_state_vector(psi, {q: i for i, q in enumerate(qs)}).real
           for q in qs]  # fmt: skip
    row, col = divmod(5, 14)
    ref = torch.tensor(ref, dtype=torch.float64)
    torch.testing.assert_close(fused[0, :, row, col].double(), ref, atol=1e-5, rtol=0)


def test_dataset_pipeline_matches_direct_call_across_chunks():
    rng = np.random.default_rng(0)
    images = rng.integers(0, 256, size=(10, 28, 28), dtype=np.uint8)
    filt = Quanvolution(QuanvConfig())
    feats = quanvolve_dataset(images, filt, chunk=3, out_dtype=torch.float32)
    direct = filt(torch.from_numpy(images).float() / 255)
    torch.testing.assert_close(feats, direct)


def test_expectations_are_bounded():
    feats = Quanvolution()(torch.rand(2, 28, 28))
    assert feats.abs().max() <= 1 + 1e-5
    # Different patches give different features (the filter is not constant).
    assert feats.std() > 0.05
    assert expect_z_all(zero_state(1, 4), 4).tolist() == [[1.0, 1.0, 1.0, 1.0]]
