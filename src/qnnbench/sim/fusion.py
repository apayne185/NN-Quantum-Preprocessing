"""Gate fusion: group a circuit's gates so the state is swept fewer times.

The simulator is memory-bound: every gate reads and writes the whole
``B * 2**n`` state, while doing only a handful of FLOPs per amplitude.
Fusing gates trades extra FLOPs for fewer passes over memory:

* A run of consecutive diagonal gates (e.g. a whole ZZ layer) becomes one
  ``2**n`` phase vector. Building it does not depend on the batch, so it
  costs about 1/B of a state pass per gate, and the batch is swept once.
* Consecutive non-diagonal gates whose combined support has at most
  ``max_qubits`` qubits become one dense ``2**k x 2**k`` unitary, applied as
  a single GEMM. Work per amplitude grows as ``2**k``, so k is a
  compute/bandwidth trade-off (see the ``fusion`` benchmark suite).

The plan depends only on circuit structure, never on parameter values, so it
is computed once and reused every step.
"""

from __future__ import annotations

from dataclasses import dataclass

from qnnbench.sim.circuit import Circuit
from qnnbench.sim.gates import GATES


@dataclass(frozen=True)
class Block:
    diagonal: bool
    ops: tuple[int, ...]  # indices into circuit.ops, in application order
    qubits: tuple[int, ...]  # sorted union of the ops' qubits


def plan_fusion(circuit: Circuit, max_qubits: int = 4) -> list[Block]:
    """Greedy left-to-right fusion of adjacent gates."""
    if max_qubits < 2:
        raise ValueError("max_qubits must be >= 2 (two-qubit gates cannot be split)")
    blocks: list[Block] = []
    ops: list[int] = []
    qubits: set[int] = set()
    diagonal = False

    def flush():
        if ops:
            blocks.append(Block(diagonal, tuple(ops), tuple(sorted(qubits))))

    for i, op in enumerate(circuit.ops):
        is_diag = GATES[op.gate].diagonal
        if ops and is_diag and diagonal:
            fits = True  # a diagonal run is never limited in width
        elif ops and not is_diag and not diagonal:
            fits = len(qubits | set(op.qubits)) <= max_qubits
        else:
            fits = False
        if not fits:
            flush()
            ops, qubits, diagonal = [], set(), is_diag
        ops.append(i)
        qubits |= set(op.qubits)
    flush()
    return blocks
