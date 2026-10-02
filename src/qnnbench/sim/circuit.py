"""A minimal circuit description: an ordered list of gates on integer qubits.

Qubit 0 is the most significant bit of the state index (Cirq's convention for
sorted qubits), so ``to_cirq`` produces a circuit whose final state vector
matches ``qnnbench.sim.statevector.simulate`` element for element.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from qnnbench.sim.gates import GATES


@dataclass(frozen=True)
class Op:
    gate: str
    qubits: tuple[int, ...]
    param: int | None = None  # index into the trainable parameter vector
    value: float | None = None  # fixed angle/exponent for a non-trainable parametric gate

    def __post_init__(self):
        g = GATES[self.gate]
        if len(self.qubits) != g.n_qubits:
            raise ValueError(f"{self.gate} acts on {g.n_qubits} qubits, got {self.qubits}")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"repeated qubit in {self.qubits}")
        if g.parametric and (self.param is None) == (self.value is None):
            raise ValueError(f"{self.gate} needs exactly one of param= or value=")
        if not g.parametric and (self.param is not None or self.value is not None):
            raise ValueError(f"{self.gate} takes no parameter")


@dataclass
class Circuit:
    n_qubits: int
    ops: list[Op] = field(default_factory=list)

    def add(self, gate: str, *qubits: int, param: int | None = None, value: float | None = None):
        if any(q < 0 or q >= self.n_qubits for q in qubits):
            raise ValueError(f"qubit out of range in {qubits} for {self.n_qubits} qubits")
        self.ops.append(Op(gate, tuple(qubits), param, value))
        return self

    @property
    def n_params(self) -> int:
        return 1 + max((op.param for op in self.ops if op.param is not None), default=-1)

    def to_cirq(self, params: torch.Tensor | None = None):
        """Build the equivalent ``cirq.Circuit`` (used only by tests and benchmarks)."""
        import cirq

        qs = cirq.LineQubit.range(self.n_qubits)
        values = [] if params is None else params.detach().cpu().double().tolist()
        out = cirq.Circuit()
        for op in self.ops:
            p = values[op.param] if op.param is not None else op.value
            targets = [qs[q] for q in op.qubits]
            fixed = {"x": cirq.X, "h": cirq.H, "z": cirq.Z, "cnot": cirq.CNOT, "cz": cirq.CZ}
            if op.gate in fixed:
                gate = fixed[op.gate]
            elif op.gate in ("xx", "zz"):
                gate = (cirq.XX if op.gate == "xx" else cirq.ZZ) ** p
            else:
                gate = {"rx": cirq.rx, "ry": cirq.ry, "rz": cirq.rz}[op.gate](p)
            out.append(gate.on(*targets), strategy=cirq.InsertStrategy.EARLIEST)
        return out


def qnn_circuit(n_data: int = 16, layers: tuple[str, ...] = ("xx", "zz")) -> Circuit:
    """Farhi et al. QNN: every data qubit couples to one readout qubit, per layer.

    Qubit 0 is the readout (TFQ placed it at GridQubit(-1, -1), which also sorts
    first); data qubits are 1..n_data in row-major pixel order. With the
    defaults this is exactly the 32-parameter circuit of the original model.
    """
    c = Circuit(n_data + 1)
    c.add("x", 0).add("h", 0)
    p = 0
    for gate in layers:
        for q in range(1, n_data + 1):
            c.add(gate, q, 0, param=p)
            p += 1
    c.add("h", 0)
    return c
