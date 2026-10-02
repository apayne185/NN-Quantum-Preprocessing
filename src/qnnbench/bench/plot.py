"""Render benchmark results as PNG figures for the README.

    python -m qnnbench.bench.plot --results results --out docs/figures

One figure per suite and device. Every panel has a single y-axis, series are
direct-labelled at their last point as well as listed in a legend, and colors
follow a fixed categorical order so a series keeps its color across figures.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Validated categorical order (light surface); assigned in order, never cycled.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="semibold")
    ax.set_xlabel(xlabel, color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def _lines(ax, series: dict[str, list[tuple[float, float]]], order: list[str] | None = None):
    names = order or list(series)
    for i, name in enumerate(names):
        pts = sorted(series[name])
        if not pts:
            continue
        xs, ys = zip(*pts, strict=True)
        color = SERIES[i]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=name)  # fmt: skip
        if len(names) <= 4:  # direct label at the line end, in text ink
            ax.annotate(name, (xs[-1], ys[-1]), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK_2)  # fmt: skip
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_2)


def _figure(n_panels: int = 1):
    fig, axes = plt.subplots(1, n_panels, figsize=(6.4 * n_panels, 4.0), facecolor=SURFACE)
    return fig, [axes] if n_panels == 1 else list(axes)


def _save(fig, path: Path):
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"-> {path}")


def _device(payload: dict) -> str:
    env = payload["env"]
    return env["device_name"] if env["device"].startswith("cuda") else f"CPU: {env['device_name']}"


# --------------------------------------------------------------------------- per suite


def plot_qubits(ms, device, out):
    series = defaultdict(list)
    for m in ms:
        name = "fused (k=5)" if m["case"]["fuse"] else "unfused"
        series[name].append((m["case"]["n_qubits"], m["p50_ms"]))
    fig, (ax,) = _figure()
    _lines(ax, series, ["unfused", "fused (k=5)"])
    ax.set_yscale("log")
    _style(ax, f"QNN forward latency vs qubits, batch 32 - {device}", "qubits", "p50 latency (ms)")
    _save(fig, out)


def plot_fusion(ms, device, out):
    series = defaultdict(list)
    for m in ms:
        series[m["case"]["pass"]].append((m["case"]["fuse"], m["p50_ms"]))
    n = ms[0]["case"]["n_qubits"]
    fig, (ax,) = _figure()
    _lines(ax, series, ["forward", "fwd+bwd"])
    ax.set_xticks(sorted({m["case"]["fuse"] for m in ms}))
    _style(ax, f"Gate fusion width, {n} qubits, batch 32 - {device}",
           "max qubits per fused block (0 = off)", "p50 latency (ms)")  # fmt: skip
    _save(fig, out)


def plot_grad(ms, device, out):
    time_s, mem_s = defaultdict(list), defaultdict(list)
    for m in ms:
        if m["case"].get("fuse") == 0:
            continue  # plot the fused variants; unfused ones are in the JSON
        name = m["case"]["method"]
        time_s[name].append((m["case"]["n_qubits"], m["p50_ms"]))
        if "saved_MB" in m["extra"]:
            mem_s[name].append((m["case"]["n_qubits"], m["extra"]["saved_MB"]))
    order = [k for k in ("autograd", "adjoint", "param_shift", "tfq adjoint") if k in time_s]
    fig, (ax_t, ax_m) = _figure(2)
    _lines(ax_t, time_s, order)
    ax_t.set_yscale("log")
    _style(ax_t, f"Training step time - {device}", "qubits", "p50 fwd+bwd (ms)")
    _lines(ax_m, mem_s, [k for k in order if k in mem_s])
    ax_m.set_yscale("log")
    _style(ax_m, "Memory kept for backward", "qubits", "saved tensors (MB)")
    _save(fig, out)


def plot_frameworks(ms, device, out):
    series = defaultdict(list)
    for m in ms:
        series[m["case"]["framework"]].append((m["case"]["n_qubits"], m["p50_ms"]))
    order = [k for k in ("qnnbench (fused)", "qnnbench (unfused)", "pennylane",
                         "tensorflow-quantum", "cirq") if k in series]  # fmt: skip
    fig, (ax,) = _figure()
    _lines(ax, series, order)
    ax.set_yscale("log")
    _style(ax, f"QNN forward pass by framework, batch 32 - {device}", "qubits", "p50 latency (ms)")
    _save(fig, out)


def plot_batch(ms, device, out):
    series = {"throughput": [(m["case"]["batch"], m["throughput"]) for m in ms]}
    n = ms[0]["case"]["n_qubits"]
    fig, (ax,) = _figure()
    _lines(ax, series)
    ax.get_legend().remove()  # single series: the title names it
    ax.set_xscale("log", base=2)
    _style(ax, f"Throughput vs batch size, {n} qubits - {device}", "batch size", "samples / s")
    _save(fig, out)


PLOTS = {
    "qubits": plot_qubits,
    "fusion": plot_fusion,
    "grad": plot_grad,
    "frameworks": plot_frameworks,
    "batch": plot_batch,
}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("docs/figures"))
    args = p.parse_args(argv)

    for suite, fn in PLOTS.items():
        # Group files by device; CPU results from different tools (e.g. TFQ) on the
        # same host are merged into one figure.
        by_device: dict[str, list[dict]] = defaultdict(list)
        for f in sorted((args.results / suite).glob("*.json")):
            payload = json.loads(f.read_text())
            by_device[_device(payload)] += payload["measurements"]
        for device, ms in by_device.items():
            slug = "".join(c if c.isalnum() else "-" for c in device.lower()).strip("-")
            fn(ms, device, args.out / f"{suite}-{slug}.png")


if __name__ == "__main__":
    main()
