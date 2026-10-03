"""Train a CNN on quanvolutional features, with controls, on full 10-class MNIST.

    python -m qnnbench.hybrid --features quanv --epochs 5
    torchrun --nproc-per-node 2 -m qnnbench.hybrid --features quanv   # DDP

Three feature extractors share one CNN head, so only the 2x2 patch encoder
differs:

* ``quanv``: fixed random quantum circuit (see :mod:`qnnbench.quanv`).
* ``random``: fixed random classical 2x2/stride-2 conv + tanh. The control:
  does the quantum filter beat a random classical one with the same shape?
* ``learned``: the same 2x2/stride-2 conv, but trained end to end.

Training-loop features: automatic mixed precision (fp16 with loss scaling,
or bf16), gradient accumulation, torch.compile, pinned-memory loading,
DistributedDataParallel via torchrun, and post-training int8 quantization
of the classifier for CPU inference.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

from qnnbench.data import load_mnist
from qnnbench.env import env_info
from qnnbench.quanv import QuanvConfig, cached_features


@dataclass
class HybridConfig:
    features: str = "quanv"  # quanv | random | learned
    epochs: int = 5
    batch_size: int = 128  # effective batch: micro_batch * accum_steps * world_size
    accum_steps: int = 1
    lr: float = 2e-3
    train_size: int = 60000
    amp: str = "auto"  # auto | off | fp16 | bf16
    compile: bool = False
    num_workers: int = 2
    seed: int = 0
    quantize: bool = True


class PatchEncoder(nn.Module):
    """2x2/stride-2 conv to 4 channels + tanh; frozen for the random control."""

    def __init__(self, trainable: bool):
        super().__init__()
        self.conv = nn.Conv2d(1, 4, kernel_size=2, stride=2)
        self.conv.requires_grad_(trainable)

    def forward(self, x):
        return torch.tanh(self.conv(x))


class Head(nn.Module):
    """Small CNN on (4, 14, 14) feature maps."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 14 -> 7
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 7 -> 3
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


def build_model(features: str) -> nn.Module:
    if features == "quanv":
        return Head()  # inputs are precomputed quantum features
    if features in ("random", "learned"):
        return nn.Sequential(PatchEncoder(trainable=features == "learned"), Head())
    raise ValueError(f"unknown features {features!r}")


# --------------------------------------------------------------------------- distributed


def setup_distributed() -> tuple[int, int, torch.device]:
    """Initialise DDP when launched by torchrun; otherwise run single-process."""
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        device = torch.device("cuda", local)
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if world > 1:
        # NCCL for GPU collectives; Gloo lets the same code path run on CPU.
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return rank, world, device


def all_reduce_sum(t: torch.Tensor, world: int) -> torch.Tensor:
    if world > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def amp_dtype(setting: str, device: torch.device) -> torch.dtype | None:
    if setting == "off":
        return None
    if setting == "auto":
        if device.type != "cuda":
            return None
        # bf16 needs Ampere (sm_80)+; Turing cards like the GTX 1650 / T4 get fp16.
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return {"fp16": torch.float16, "bf16": torch.bfloat16}[setting]


# --------------------------------------------------------------------------- data


def load_split(cfg: HybridConfig, split: str, device: torch.device) -> TensorDataset:
    raw = load_mnist()
    x_u8, y = raw[f"x_{split}"], raw[f"y_{split}"]
    if split == "train":
        x_u8, y = x_u8[: cfg.train_size], y[: cfg.train_size]
    labels = torch.from_numpy(y).long()
    if cfg.features == "quanv":
        feats = cached_features(x_u8, split, QuanvConfig(), device)
        return TensorDataset(feats, labels)
    return TensorDataset(torch.from_numpy(x_u8)[:, None], labels)


def to_input(x: torch.Tensor, features: str) -> torch.Tensor:
    # uint8 images travel to the device and are scaled there; features are float16.
    return x.float() / 255 if features != "quanv" else x.float()


# --------------------------------------------------------------------------- train / eval


def train(cfg: HybridConfig, log=print) -> dict:
    rank, world, device = setup_distributed()
    torch.manual_seed(cfg.seed)
    is_main = rank == 0
    say = log if is_main else (lambda *_: None)

    if cfg.batch_size % (cfg.accum_steps * world):
        raise ValueError("batch_size must be divisible by accum_steps * world_size")
    micro = cfg.batch_size // (cfg.accum_steps * world)

    # Rank 0 builds the feature cache first so ranks don't race on the same file.
    if world > 1 and not is_main:
        dist.barrier()
    train_ds = load_split(cfg, "train", device)
    test_ds = load_split(cfg, "test", device)
    if world > 1 and is_main:
        dist.barrier()

    sampler: DistributedSampler | None = (
        DistributedSampler(train_ds, seed=cfg.seed) if world > 1 else None
    )
    pin = device.type == "cuda"
    loader = DataLoader(
        train_ds,
        batch_size=micro,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
    )

    net = build_model(cfg.features).to(device)
    ddp = (
        nn.parallel.DistributedDataParallel(
            net, device_ids=[device.index] if device.type == "cuda" else None
        )
        if world > 1
        else None
    )
    model: nn.Module = ddp or net  # what we train; `net` stays the unwrapped module
    forward = torch.compile(model) if cfg.compile else model
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    dtype = amp_dtype(cfg.amp, device)
    # Loss scaling only for fp16: its 5-bit exponent underflows small gradients.
    scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16)

    history = []
    for epoch in range(1, cfg.epochs + 1):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        seen = 0
        opt.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(loader):
            xb = to_input(xb.to(device, non_blocking=pin), cfg.features)
            yb = yb.to(device, non_blocking=pin)
            last_micro = (step + 1) % cfg.accum_steps == 0
            # Skip DDP's gradient all-reduce on all but the last micro-batch.
            sync = ddp.no_sync() if ddp is not None and not last_micro else contextlib.nullcontext()
            with sync:
                with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=bool(dtype)):
                    loss = F.cross_entropy(forward(xb), yb)
                scaler.scale(loss / cfg.accum_steps).backward()
            if last_micro:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
            loss_sum += loss.detach() * len(yb)
            seen += len(yb)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

        totals = all_reduce_sum(torch.tensor([loss_sum.item(), seen], device=device), world)
        test_acc = evaluate(model, test_ds, cfg, device, dtype, world, rank)
        stats = {
            "epoch": epoch,
            "train_loss": (totals[0] / totals[1]).item(),
            "test_acc": test_acc,
            "epoch_time_s": elapsed,
            "samples_per_s": totals[1].item() / elapsed,
            "peak_mem_MB": torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else None,
        }
        history.append(stats)
        say(
            f"[{cfg.features}] epoch {epoch}: loss={stats['train_loss']:.4f} "
            f"test_acc={test_acc:.4f} ({elapsed:.1f}s, {stats['samples_per_s']:,.0f} samples/s)"
        )

    result = {
        "config": asdict(cfg),
        "world_size": world,
        "amp_dtype": str(dtype),
        "history": history,
        "env": env_info(device),
    }
    if is_main and cfg.quantize:
        result["quantization"] = quantize_and_compare(net, test_ds, cfg)
        say(f"[{cfg.features}] int8: {json.dumps(result['quantization'])}")
    if world > 1:
        dist.destroy_process_group()
    return result


@torch.no_grad()
def evaluate(model, ds, cfg, device, dtype, world, rank) -> float:
    model.eval()
    # Each rank scores a strided shard; the counts are summed across ranks.
    x, y = ds.tensors[0][rank::world], ds.tensors[1][rank::world]
    correct = torch.zeros((), device=device)
    for i in range(0, len(x), 1024):
        xb = to_input(x[i : i + 1024].to(device), cfg.features)
        with torch.autocast(device.type, dtype=dtype or torch.float32, enabled=bool(dtype)):
            pred = model(xb).argmax(1)
        correct += (pred == y[i : i + 1024].to(device)).sum()
    totals = all_reduce_sum(torch.stack([correct, torch.tensor(len(x), device=device)]), world)
    return (totals[0] / totals[1]).item()


def quantize_and_compare(model: nn.Module, ds: TensorDataset, cfg: HybridConfig) -> dict:
    """Dynamic int8 quantization of the Linear layers, measured on CPU.

    Weights are stored as int8 and activations quantized on the fly, which
    suits batch-1 CPU inference where Linear layers are weight-bandwidth
    bound. Uses torch.ao (deprecated upstream in favour of torchao, but still
    the dependency-free option).
    """
    from torch.ao.quantization import quantize_dynamic

    fp32 = build_model(cfg.features)
    fp32.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
    fp32.eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        int8 = quantize_dynamic(fp32, {nn.Linear}, dtype=torch.qint8)

    x = to_input(ds.tensors[0][:2000], cfg.features)
    y = ds.tensors[1][:2000]

    def size_mb(m):
        buf = io.BytesIO()
        torch.save(m.state_dict(), buf)
        return buf.tell() / 2**20

    def latency_ms(m, batch):
        xb = x[:batch]
        with torch.no_grad():
            for _ in range(3):
                m(xb)
            t0 = time.perf_counter()
            for _ in range(20):
                m(xb)
        return (time.perf_counter() - t0) / 20 * 1e3

    out = {}
    with torch.no_grad():
        for name, m in (("fp32", fp32), ("int8", int8)):
            out[name] = {
                "acc_2k": (m(x).argmax(1) == y).float().mean().item(),
                "size_MB": size_mb(m),
                "cpu_latency_ms_b1": latency_ms(m, 1),
                "cpu_latency_ms_b256": latency_ms(m, 256),
            }
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for f in HybridConfig.__dataclass_fields__.values():
        flag = f"--{f.name.replace('_', '-')}"
        if str(f.type) == "bool":
            p.add_argument(flag, action=argparse.BooleanOptionalAction, default=f.default)
        else:
            p.add_argument(
                flag, type={"int": int, "float": float}.get(str(f.type), str), default=f.default
            )
    p.add_argument("--out", type=Path, help="write the run as JSON (rank 0)")
    args = vars(p.parse_args(argv))
    out = args.pop("out")
    result = train(HybridConfig(**args))
    if out and int(os.environ.get("RANK", 0)) == 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
