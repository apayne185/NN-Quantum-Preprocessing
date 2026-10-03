"""Multi-seed accuracy comparison of the QNN, the fair MLP and the lookup-table ceiling.

    python -m qnnbench.experiments --seeds 5 --epochs 3
    python -m qnnbench.experiments --suite hybrid --seeds 5 --epochs 5

The default suite reproduces the paper's protocol (3 epochs, batch 32, Adam 1e-3, the 500-example
"short" QNN) but with independent models, fixed seeds and mean +- std, and
with the Bayes ceiling for the binarized 4x4 encoding reported alongside.

The ``hybrid`` suite compares the quantum patch filter against its classical
controls (random and learned 2x2 filters) on full MNIST over several seeds;
a single seed cannot separate differences of a few tenths of a percent.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any

from qnnbench.data import prepare_binary_mnist
from qnnbench.train import TrainConfig, lookup_table_accuracy, train

RUNS: dict[str, dict[str, Any]] = {
    "QNN (500 examples)": dict(model="qnn", num_train=500),
    "QNN": dict(model="qnn"),
    "Fair MLP": dict(model="mlp"),
}


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--suite", choices=["paper", "hybrid"], default="paper")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("results/experiments"))
    args = p.parse_args(argv)
    if args.suite == "hybrid":
        return run_hybrid(args.seeds, args.epochs, args.out)

    data = prepare_binary_mnist()
    ceiling = lookup_table_accuracy(data)
    runs: dict[str, list[dict]] = {}
    for name, overrides in RUNS.items():
        runs[name] = []
        for seed in range(args.seeds):
            cfg = TrainConfig(epochs=args.epochs, seed=seed, device=args.device, **overrides)
            runs[name].append(asdict(train(cfg, data)))

    summary, table = summarize(runs, ceiling, args.seeds, args.epochs)
    print("\n" + table)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"accuracy_{args.epochs}ep.json").write_text(
        json.dumps({"summary": summary, "ceiling": ceiling, "runs": runs}, indent=2) + "\n"
    )
    (args.out / f"accuracy_{args.epochs}ep.md").write_text(table + "\n")


def summarize(
    runs: dict[str, list[dict]], ceiling: dict, seeds: int, epochs: int
) -> tuple[dict, str]:
    """Per-model accuracy statistics and the markdown table for the README."""
    summary = {}
    for name, results in runs.items():
        accs = [r["history"][-1]["test_acc"] for r in results]
        throughput = [statistics.mean(h["samples_per_s"] for h in r["history"]) for r in results]
        mean, std = _mean_std(accs)
        summary[name] = {
            "test_acc_mean": mean,
            "test_acc_std": std,
            "test_acc_min": min(accs),
            "test_acc_max": max(accs),
            "train_samples_per_s": statistics.mean(throughput),
            "n_params": results[0]["n_params"],
        }

    lines = [
        f"| Model | Params | Test accuracy ({seeds} seeds, {epochs} epochs) | Range |",
        "|---|---|---|---|",
    ]
    for name, s in summary.items():
        lines.append(
            f"| {name} | {s['n_params']} | {100 * s['test_acc_mean']:.1f}% "
            f"± {100 * s['test_acc_std']:.1f} | {100 * s['test_acc_min']:.1f}–"
            f"{100 * s['test_acc_max']:.1f}% |"
        )
    lines.append(
        f"| Lookup table (Bayes ceiling) | {ceiling['n_patterns']} patterns | "
        f"{100 * ceiling['test_acc']:.1f}% | deterministic |"
    )
    return summary, "\n".join(lines)


HYBRID_FEATURES = {
    "Quantum filter (quanvolution)": "quanv",
    "Random classical 2x2 filter": "random",
    "Learned 2x2 filter": "learned",
}


def run_hybrid(seeds: int, epochs: int, out: Path) -> dict:
    from qnnbench.hybrid import HybridConfig
    from qnnbench.hybrid import train as train_hybrid

    runs: dict[str, list[dict]] = {}
    for name, features in HYBRID_FEATURES.items():
        runs[name] = [
            train_hybrid(
                HybridConfig(
                    features=features, epochs=epochs, seed=seed, num_workers=0, quantize=False
                )
            )  # fmt: skip
            for seed in range(seeds)
        ]
    summary, table = summarize_hybrid(runs, seeds, epochs)
    print("\n" + table)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"hybrid_{epochs}ep.json").write_text(
        json.dumps({"summary": summary, "runs": runs}, indent=2) + "\n"
    )
    (out / f"hybrid_{epochs}ep.md").write_text(table + "\n")
    return summary


def summarize_hybrid(runs: dict[str, list[dict]], seeds: int, epochs: int) -> tuple[dict, str]:
    summary = {}
    lines = [
        f"| Patch encoder | Test accuracy ({seeds} seeds, {epochs} epochs) | Range |",
        "|---|---|---|",
    ]
    for name, results in runs.items():
        accs = [r["history"][-1]["test_acc"] for r in results]
        mean, std = _mean_std(accs)
        summary[name] = {"test_acc_mean": mean, "test_acc_std": std,
                         "test_acc_min": min(accs), "test_acc_max": max(accs)}  # fmt: skip
        lines.append(
            f"| {name} | {100 * mean:.2f}% ± {100 * std:.2f} | "
            f"{100 * min(accs):.2f}–{100 * max(accs):.2f}% |"
        )
    return summary, "\n".join(lines)


if __name__ == "__main__":
    main()
