"""Gate definitions: matrices, analytic derivatives and parameter-shift rules.

Parametric gates follow Cirq conventions so circuits can be cross-checked
against ``cirq.Simulator``:

* ``XX**t``, ``ZZ**t`` (exponent ``t``): eigenvalue ``-1`` picks up ``e^{i pi t}``.
* ``rx/ry/rz(theta)`` = ``exp(-i theta P / 2)``.

Diagonal gates return their diagonal only, so the simulator can apply them as
a broadcast multiply instead of a matrix contraction.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

Tensor = torch.Tensor


@dataclass(frozen=True)
class Gate:
    name: str
    n_qubits: int
    diagonal: bool
    # param -> (2^k, 2^k) matrix, or (2^k,) diagonal if ``diagonal``
    matrix: Callable[[Tensor | None, torch.dtype, torch.device], Tensor]
    # d(matrix)/d(param); None for fixed gates
    derivative: Callable[[Tensor, torch.dtype, torch.device], Tensor] | None = None
    # Parameter-shift rule: df/dp = coeff * (f(p + shift) - f(p - shift))
    shift: float | None = None
    shift_coeff: float | None = None

    @property
    def parametric(self) -> bool:
        return self.derivative is not None


def _const(values, dtype, device) -> Tensor:
    return torch.tensor(values, dtype=dtype, device=device)


def _phase(angle: Tensor, dtype: torch.dtype) -> Tensor:
    """e^{i * angle} as a complex tensor that stays differentiable in ``angle``."""
    real = angle.to(torch.float64 if dtype == torch.complex128 else torch.float32)
    return torch.polar(torch.ones_like(real), real)


_X = [[0, 1], [1, 0]]
_H = [[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]]
_I4 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
_XX = [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
_CNOT = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]


def _xx_pow(t, dtype, device):
    e = _phase(math.pi * t, dtype)
    return (1 + e) / 2 * _const(_I4, dtype, device) + (1 - e) / 2 * _const(_XX, dtype, device)


def _xx_pow_d(t, dtype, device):
    de = 1j * math.pi * _phase(math.pi * t, dtype)
    return de / 2 * (_const(_I4, dtype, device) - _const(_XX, dtype, device))


def _zz_pow(t, dtype, device):
    e = _phase(math.pi * t, dtype)
    one = torch.ones_like(e)
    return torch.stack([one, e, e, one])


def _zz_pow_d(t, dtype, device):
    de = 1j * math.pi * _phase(math.pi * t, dtype)
    zero = torch.zeros_like(de)
    return torch.stack([zero, de, de, zero])


def _rx(theta, dtype, device):
    c, s = torch.cos(theta / 2).to(dtype), torch.sin(theta / 2).to(dtype)
    return torch.stack([torch.stack([c, -1j * s]), torch.stack([-1j * s, c])])


def _rx_d(theta, dtype, device):
    c, s = torch.cos(theta / 2).to(dtype), torch.sin(theta / 2).to(dtype)
    return 0.5 * torch.stack([torch.stack([-s, -1j * c]), torch.stack([-1j * c, -s])])


def _ry(theta, dtype, device):
    c, s = torch.cos(theta / 2).to(dtype), torch.sin(theta / 2).to(dtype)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])])


def _ry_d(theta, dtype, device):
    c, s = torch.cos(theta / 2).to(dtype), torch.sin(theta / 2).to(dtype)
    return 0.5 * torch.stack([torch.stack([-s, -c]), torch.stack([c, -s])])


def _rz(theta, dtype, device):
    e = _phase(theta / 2, dtype)
    return torch.stack([e.conj(), e])


def _rz_d(theta, dtype, device):
    e = _phase(theta / 2, dtype)
    return torch.stack([-0.5j * e.conj(), 0.5j * e])


def _fixed(values):
    return lambda _p, dtype, device: _const(values, dtype, device)


# exponent-t gates: f(t) = A + B cos(pi t) + C sin(pi t), so the shift is 1/2 in t
# and the rule picks up a factor pi from d(pi t)/dt.
_EXP_SHIFT = dict(shift=0.5, shift_coeff=math.pi / 2)
# rotation gates exp(-i theta P / 2): the standard pi/2 shift with coefficient 1/2.
_ROT_SHIFT = dict(shift=math.pi / 2, shift_coeff=0.5)

GATES: dict[str, Gate] = {
    "x": Gate("x", 1, False, _fixed(_X)),
    "h": Gate("h", 1, False, _fixed(_H)),
    "z": Gate("z", 1, True, _fixed([1, -1])),
    "cnot": Gate("cnot", 2, False, _fixed(_CNOT)),
    "cz": Gate("cz", 2, True, _fixed([1, 1, 1, -1])),
    "xx": Gate("xx", 2, False, _xx_pow, _xx_pow_d, **_EXP_SHIFT),
    "zz": Gate("zz", 2, True, _zz_pow, _zz_pow_d, **_EXP_SHIFT),
    "rx": Gate("rx", 1, False, _rx, _rx_d, **_ROT_SHIFT),
    "ry": Gate("ry", 1, False, _ry, _ry_d, **_ROT_SHIFT),
    "rz": Gate("rz", 1, True, _rz, _rz_d, **_ROT_SHIFT),
}
