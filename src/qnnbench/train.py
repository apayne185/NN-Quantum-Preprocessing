"""Training loop for the QNN and the fair MLP baseline.

The loop keeps loss/accuracy accumulators on the device and synchronises once
per epoch. Calling ``.item()`` every step would force a CPU/GPU sync per batch
and serialise the host with the device.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from qnnbench.data import prepare_binary_mnist
from qnnbench.env import env_info, resolve_device
from qnnbench.models import QNN, FairMLP, LookupTable, n_params


@dataclass
class TrainConfig:
    model: str = "qnn"  # qnn | mlp
    epochs: int = 3
    batch_size: int = 32
    lr: float = 1e-3  # Keras Adam default, as in the original
    seed: int = 0
    num_train: int | None = None  # train on the first N examples only
    device: str = "auto"
    grad_method: str = "autograd"
    dtype: str = "complex64"
    fuse: int = 5  # max qubits per fused gate block; 0 disables fusion
    compile: bool = False


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    train_acc: float
    test_loss: float
    test_acc: float
    epoch_time_s: float
    samples_per_s: float


@dataclass
class RunResult:
    config: dict
    n_params: int
    history: list[EpochStats] = field(default_factory=list)
    env: dict = field(default_factory=dict)

    @property
    def final_test_acc(self) -> float:
        return self.history[-1].test_acc


def hinge_loss(pred: torch.Tensor, y_pm1: torch.Tensor) -> torch.Tensor:
    return torch.clamp(1 - pred * y_pm1, min=0).mean()


LossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def build_model(cfg: TrainConfig) -> tuple[nn.Module, LossFn]:
    """Return the model and its loss; both models take labels in {-1, +1}."""
    if cfg.model == "qnn":
        model = QNN(grad_method=cfg.grad_method, dtype=getattr(torch, cfg.dtype), fuse=cfg.fuse)
        return model, hinge_loss
    if cfg.model == "mlp":
        bce = nn.BCEWithLogitsLoss()
        return FairMLP(), lambda logits, y: bce(logits, (y > 0).to(logits.dtype))
    raise ValueError(f"unknown model {cfg.model!r}")


def train(cfg: TrainConfig, data=None, log=print, save_path: Path | None = None) -> RunResult:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = resolve_device(cfg.device)
    data = data or prepare_binary_mnist()

    x_train = torch.from_numpy(data.x_train[: cfg.num_train]).to(device)
    y_train = torch.from_numpy(data.y_train[: cfg.num_train] * 2.0 - 1.0).float().to(device)
    x_test = torch.from_numpy(data.x_test).to(device)
    y_test = torch.from_numpy(data.y_test * 2.0 - 1.0).float().to(device)

    model, loss_fn = build_model(cfg)
    model.to(device)
    forward = torch.compile(model) if cfg.compile else model
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    result = RunResult(asdict(cfg), n_params(model), env=env_info(device))
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        t0 = time.perf_counter()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device)
        perm = torch.randperm(len(x_train), generator=gen).to(device)
        for i in range(0, len(x_train), cfg.batch_size):
            idx = perm[i : i + cfg.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            out = forward(xb).float()
            loss = loss_fn(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            loss_sum += loss.detach() * len(idx)
            correct += ((out.detach() > 0) == (yb > 0)).sum()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

        test_loss, test_acc = evaluate(forward, loss_fn, x_test, y_test, cfg.batch_size * 8)
        stats = EpochStats(
            epoch=epoch,
            train_loss=loss_sum.item() / len(x_train),
            train_acc=correct.item() / len(x_train),
            test_loss=test_loss,
            test_acc=test_acc,
            epoch_time_s=elapsed,
            samples_per_s=len(x_train) / elapsed,
        )
        result.history.append(stats)
        log(
            f"[{cfg.model} seed={cfg.seed}] epoch {epoch}: "
            f"train_loss={stats.train_loss:.4f} train_acc={stats.train_acc:.4f} "
            f"test_loss={stats.test_loss:.4f} test_acc={stats.test_acc:.4f} "
            f"({elapsed:.1f}s, {stats.samples_per_s:.0f} samples/s)"
        )
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(cfg), "state_dict": model.state_dict()}, save_path)
    return result


@torch.no_grad()
def evaluate(forward, loss_fn, x, y, batch_size) -> tuple[float, float]:
    loss_sum = torch.zeros((), device=x.device)
    correct = torch.zeros((), device=x.device)
    for i in range(0, len(x), batch_size):
        out = forward(x[i : i + batch_size]).float()
        yb = y[i : i + batch_size]
        loss_sum += loss_fn(out, yb) * len(yb)
        correct += ((out > 0) == (yb > 0)).sum()
    return loss_sum.item() / len(x), correct.item() / len(x)


def lookup_table_accuracy(data=None) -> dict:
    data = data or prepare_binary_mnist()
    table = LookupTable().fit(data.x_train, data.y_train)
    return {
        "train_acc": float((table.predict(data.x_train) == data.y_train).mean()),
        "test_acc": float((table.predict(data.x_test) == data.y_test).mean()),
        "n_patterns": len(table.table),
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Train the QNN or the fair MLP baseline.")
    for f in TrainConfig.__dataclass_fields__.values():
        if str(f.type) == "bool":
            p.add_argument(f"--{f.name.replace('_', '-')}", action="store_true")
        else:
            caster = {"int": int, "float": float, "int | None": int}.get(str(f.type), str)
            p.add_argument(f"--{f.name.replace('_', '-')}", type=caster, default=f.default)
    p.add_argument("--out", type=Path, help="write the run (config, history, env) as JSON")
    p.add_argument("--save", type=Path, help="save the trained weights (state_dict)")
    args = vars(p.parse_args(argv))
    out, save = args.pop("out"), args.pop("save")
    cfg = TrainConfig(**args)
    result = train(cfg, save_path=save)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(result), indent=2) + "\n")
    return result


if __name__ == "__main__":
    main()
