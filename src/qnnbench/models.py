"""The QNN and the classical baselines it is compared against."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

from qnnbench.sim import basis_state, expectation, qnn_circuit


class QNN(nn.Module):
    """Farhi et al. QNN on binarized 4x4 images; output is <Z_readout> in [-1, 1].

    Pixels are basis-encoded on qubits 1..16 and coupled to readout qubit 0 by
    one parameterised layer per entry of ``layers`` (XX then ZZ by default:
    32 parameters, identical to the original TFQ model).
    """

    def __init__(
        self,
        n_data: int = 16,
        layers: tuple[str, ...] = ("xx", "zz"),
        grad_method: str = "autograd",
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()
        self.circuit = qnn_circuit(n_data, layers)
        self.grad_method = grad_method
        self.dtype = dtype
        real = torch.float64 if dtype == torch.complex128 else torch.float32
        # Same initialisation as tfq.layers.PQC: uniform in [0, 2*pi).
        self.theta = nn.Parameter(torch.rand(self.circuit.n_params, dtype=real) * 2 * math.pi)

    def forward(self, bits: torch.Tensor) -> torch.Tensor:
        state = basis_state(bits, self.circuit.n_qubits, offset=1, dtype=self.dtype)
        return expectation(self.circuit, self.theta, state, readout=0, method=self.grad_method)


class FairMLP(nn.Module):
    """The 37-parameter classical baseline on the same 16 binary pixels."""

    def __init__(self, n_in: int = 16, hidden: int = 2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class LookupTable:
    """Majority label per distinct input pattern: the Bayes-optimal classifier.

    Binarized 4x4 images take only ~200 distinct values in training, so for
    this input representation no model can beat a per-pattern majority vote
    on the training distribution. It is the ceiling any QNN or NN result
    should be read against. Unseen patterns get the overall majority label.
    """

    def fit(self, x: np.ndarray, y: np.ndarray) -> LookupTable:
        keys = [row.tobytes() for row in x.astype(np.uint8)]
        votes: dict[bytes, list[int]] = {}
        for k, label in zip(keys, y.astype(int), strict=True):
            votes.setdefault(k, [0, 0])[label] += 1
        self.table = {k: int(v[1] >= v[0]) for k, v in votes.items()}
        self.default = int(y.mean() >= 0.5)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        rows = x.astype(np.uint8)
        return np.array([self.table.get(r.tobytes(), self.default) for r in rows], dtype=bool)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
