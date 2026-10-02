"""Run benchmark suites: ``python -m qnnbench.bench qubits grad --device cuda``."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from qnnbench.bench.harness import format_table, save_results
from qnnbench.bench.suites import SUITES
from qnnbench.env import resolve_device


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("suites", nargs="*", default=["all"], help=f"any of {list(SUITES)} or 'all'")
    p.add_argument("--device", default="auto")
    p.add_argument("--quick", action="store_true", help="small sizes, for CI smoke tests")
    p.add_argument("--out", type=Path, default=Path("results"))
    p.add_argument("--threads", type=int, help="torch CPU threads (default: torch's choice)")
    args = p.parse_args(argv)

    names = list(SUITES) if args.suites == ["all"] else args.suites
    unknown = set(names) - set(SUITES)
    if unknown:
        p.error(f"unknown suites {sorted(unknown)}; choose from {list(SUITES)}")
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)

    for name in names:
        print(f"\n== {name} on {device} ==")
        measurements = SUITES[name](device, args.quick)
        print(format_table(measurements))
        if not args.quick:
            print(f"-> {save_results(name, measurements, device, args.out)}")


if __name__ == "__main__":
    main()
