"""Compare two benchmark result files and fail on latency regressions.

    python -m qnnbench.bench.compare results/grad/baseline.json results/grad/new.json

Cases are matched on their parameters. A case regresses when its median
latency grows by more than ``--tolerance`` (relative). Use it on a dedicated
machine: shared CI runners are too noisy for tight tolerances.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _key(m: dict) -> str:
    return json.dumps(m["case"], sort_keys=True)


def compare(baseline: dict, current: dict, tolerance: float) -> tuple[list[str], bool]:
    base = {_key(m): m for m in baseline["measurements"]}
    lines, regressed = [], False
    for m in current["measurements"]:
        b = base.get(_key(m))
        if b is None:
            lines.append(f"  new      {_key(m)}  {m['p50_ms']:.3f} ms")
            continue
        ratio = m["p50_ms"] / b["p50_ms"]
        status = "ok"
        if ratio > 1 + tolerance:
            status, regressed = "REGRESSED", True
        elif ratio < 1 - tolerance:
            status = "improved"
        lines.append(
            f"  {status:9s}{_key(m)}  {b['p50_ms']:.3f} -> {m['p50_ms']:.3f} ms ({ratio:.2f}x)"
        )
    for k in base.keys() - {_key(m) for m in current["measurements"]}:
        lines.append(f"  missing  {k}")
    return lines, regressed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("baseline", type=Path)
    p.add_argument("current", type=Path)
    p.add_argument("--tolerance", type=float, default=0.15)
    args = p.parse_args(argv)
    baseline, current = (json.loads(f.read_text()) for f in (args.baseline, args.current))

    for field in ("device_name", "torch"):
        if baseline["env"].get(field) != current["env"].get(field):
            print(f"warning: {field} differs: {baseline['env'].get(field)} vs "
                  f"{current['env'].get(field)}")  # fmt: skip
    lines, regressed = compare(baseline, current, args.tolerance)
    print("\n".join(lines))
    return 1 if regressed else 0


if __name__ == "__main__":
    sys.exit(main())
