"""Quanvolutional preprocessing (Henderson et al., 2019, arXiv:1904.04767).

Every 2x2 patch of a 28x28 image is angle-encoded onto 4 qubits
(``RY(pi * pixel)``), passed through a fixed random circuit, and measured:
the four ``<Z>`` values become four output channels, so an image becomes a
``(4, 14, 14)`` feature map. The circuit is not trained; it is a fixed,
quantum feature extractor in front of a classical CNN.

At dataset scale this is a throughput problem: 60,000 images x 196 patches
is about 12 million 4-qubit circuits. Two execution strategies:

* ``gates``: simulate the random circuit gate by gate (one kernel per gate).
* ``fused``: the circuit is fixed and acts on only 4 qubits, so fold it into
  one 16x16 unitary once. Each patch is then a single GEMM row, and the
  whole pipeline (encode, GEMM, |amp|^2, GEMM with the Z sign table) is a
  handful of kernels per chunk.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from qnnbench.sim import Circuit, circuit_unitary, expect_z_all, product_state, simulate

N_QUBITS = 4  # one per pixel of a 2x2 patch


@dataclass(frozen=True)
class QuanvConfig:
    n_layers: int = 1
    gates_per_layer: int = 8
    cnot_fraction: float = 0.3
    seed: int = 0

    def cache_key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:12]


def random_circuit(cfg: QuanvConfig) -> Circuit:
    """Fixed random layers in the style of PennyLane's ``RandomLayers``."""
    rng = np.random.default_rng(cfg.seed)
    c = Circuit(N_QUBITS)
    for _ in range(cfg.n_layers * cfg.gates_per_layer):
        if rng.random() < cfg.cnot_fraction:
            a, b = rng.choice(N_QUBITS, size=2, replace=False).tolist()
            c.add("cnot", a, b)
        else:
            gate = str(rng.choice(["rx", "ry", "rz"]))
            c.add(gate, int(rng.integers(N_QUBITS)), value=float(rng.uniform(0, 2 * math.pi)))
    return c


def extract_patches(images: torch.Tensor) -> torch.Tensor:
    """(N, 28, 28) in [0, 1] -> (N * 196, 4) non-overlapping 2x2 patches, row-major."""
    n = images.shape[0]
    p = F.unfold(images[:, None], kernel_size=2, stride=2)  # (N, 4, 196)
    return p.transpose(1, 2).reshape(n * p.shape[2], N_QUBITS)


def encode(patches: torch.Tensor, dtype=torch.complex64) -> torch.Tensor:
    """Angle encoding RY(pi * x) on |0>: each qubit is (cos(pi x / 2), sin(pi x / 2))."""
    half = math.pi * patches / 2
    per_qubit = torch.stack([torch.cos(half), torch.sin(half)], dim=-1).to(dtype)
    return product_state(per_qubit)


class Quanvolution:
    """Apply the quanvolutional filter to batches of images on one device."""

    def __init__(self, cfg: QuanvConfig | None = None, mode: str = "fused", device="cpu"):
        cfg = cfg or QuanvConfig()
        if mode not in ("fused", "gates"):
            raise ValueError(f"mode must be 'fused' or 'gates', got {mode!r}")
        self.cfg, self.mode, self.device = cfg, mode, torch.device(device)
        self.circuit = random_circuit(cfg)
        # Columns of U are U|j>; states are rows, so new_state = state @ U^T.
        self.u_t = circuit_unitary(self.circuit).transpose(0, 1).contiguous().to(self.device)
        # z_signs[j, q] = <j|Z_q|j>, so <Z_q> = probs @ z_signs (a second tiny GEMM).
        bits = (torch.arange(2**N_QUBITS)[:, None] >> torch.arange(N_QUBITS - 1, -1, -1)) & 1
        self.z_signs = (1 - 2 * bits).float().to(self.device)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """(N, 28, 28) float in [0, 1] on ``self.device`` -> (N, 4, 14, 14) float32."""
        n, h, w = images.shape
        state = encode(extract_patches(images))
        if self.mode == "fused":
            amps = state @ self.u_t
            z = (amps.real**2 + amps.imag**2) @ self.z_signs
        else:
            z = expect_z_all(simulate(self.circuit, None, state), N_QUBITS)
        return z.reshape(n, h // 2, w // 2, N_QUBITS).permute(0, 3, 1, 2).contiguous()


@torch.no_grad()
def quanvolve_dataset(
    images_u8: np.ndarray,
    filt: Quanvolution,
    chunk: int = 2048,
    overlap: bool = True,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Run the filter over a whole uint8 dataset in chunks; returns CPU features.

    Images cross the bus as uint8 (4x fewer bytes than float32) and are
    converted on the device. On CUDA with ``overlap=True`` the next chunk's
    host-to-device copy runs on a side stream while the current chunk
    computes, and results come back with non-blocking copies into pinned
    memory. Features are stored as float16: values lie in [-1, 1], so the
    ~1e-3 precision costs nothing and halves memory and disk.
    """
    device = filt.device
    n = len(images_u8)
    src = torch.from_numpy(np.ascontiguousarray(images_u8))
    cuda = device.type == "cuda"
    out = torch.empty((n, N_QUBITS, 14, 14), dtype=out_dtype, pin_memory=cuda)
    starts = list(range(0, n, chunk))

    if not cuda:
        for s in starts:
            x = src[s : s + chunk].to(device).float() / 255
            out[s : s + chunk] = filt(x).to(out_dtype)
        return out

    src = src.pin_memory()  # page-locked memory: required for truly async H2D copies
    compute = torch.cuda.current_stream(device)
    copy = torch.cuda.Stream(device) if overlap else compute

    def prefetch(s):
        with torch.cuda.stream(copy):
            return src[s : s + chunk].to(device, non_blocking=True)

    nxt = prefetch(starts[0])
    for i, s in enumerate(starts):
        compute.wait_stream(copy)  # chunk i's copy must finish before we read it
        cur = nxt
        cur.record_stream(compute)  # allocated on `copy`, used on `compute`: tell the allocator
        if i + 1 < len(starts):
            nxt = prefetch(starts[i + 1])  # overlaps with the compute below
        feats = filt(cur.float() / 255).to(out_dtype)
        out[s : s + len(cur)].copy_(feats, non_blocking=True)
    torch.cuda.synchronize(device)
    return out


def cached_features(
    images_u8: np.ndarray, split: str, cfg: QuanvConfig, device, cache_dir: Path = Path("data")
) -> torch.Tensor:
    """Quanvolved features for a dataset split, computed once and cached on disk."""
    path = Path(cache_dir) / "quanv" / f"{split}-{len(images_u8)}-{cfg.cache_key()}.pt"
    if path.exists():
        return torch.load(path)
    feats = quanvolve_dataset(images_u8, Quanvolution(cfg, "fused", device))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, path)
    return feats
