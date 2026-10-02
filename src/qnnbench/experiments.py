"""Multi-seed accuracy comparison of the QNN, the fair MLP and the lookup-table ceiling.

    python -m qnnbench.experiments --seeds 5 --epochs 3

Reproduces the paper's protocol (3 epochs, batch 32, Adam 1e-3, the 500-example
"short" QNN) but with independent models, fixed seeds and mean +- std, and
with the Bayes ceiling for the binarized 4x4 encoding reported alongside.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict
from pathlib import Path

from qnnbench.data import prepare_binary_mnist
from qnnbench.train import TrainConfig, lookup_table_accuracy, train

RUNS = {
    "QNN (500 examples)": dict(model="qnn", num_train=500),
    "QNN": dict(model="qnn"),
    "Fair MLP": dict(model="mlp"),
}


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", type=Path, default=Path("results/experiments"))
    args = p.parse_args(argv)

    data = prepare_binary_mnist()
    ceiling = lookup_table_accuracy(data)
    runs: dict[str, list[dict]] = {}
    for name, overrides in RUNS.items():
        runs[name] = []
        for seed in range(args.seeds):
            cfg = TrainConfig(epochs=args.epochs, seed=seed, device=args.device, **overrides)
            runs[name].append(asdict(train(cfg, data)))

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
        f"| Model | Params | Test accuracy ({args.seeds} seeds, {args.epochs} epochs) | Range |",
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
    table = "\n".join(lines)
    print("\n" + table)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"accuracy_{args.epochs}ep.json").write_text(
        json.dumps({"summary": summary, "ceiling": ceiling, "runs": runs}, indent=2)
    )
    (args.out / f"accuracy_{args.epochs}ep.md").write_text(table + "\n")


if __name__ == "__main__":
    main()
