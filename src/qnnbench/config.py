"""Run configuration from TOML files, with command-line overrides.

    python -m qnnbench.hybrid --config configs/hybrid_quanv.toml --epochs 2

Precedence: dataclass defaults < config file < explicit command-line flags.
Unknown keys in the file are an error, so a typo can't silently fall back to
a default.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib


def load_toml(path: Path | str, cls: type) -> dict[str, Any]:
    """Read ``path`` and check every key is a field of the dataclass ``cls``."""
    with open(path, "rb") as f:
        values = tomllib.load(f)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"{path}: unknown keys {unknown}; valid keys are {sorted(known)}")
    return values


def add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
    """One --flag per dataclass field, plus --config. Defaults are left unset
    (None) so we can tell which flags were actually given."""
    parser.add_argument("--config", type=Path, help="TOML file with any of the options below")
    for f in dataclasses.fields(cls):
        flag = f"--{f.name.replace('_', '-')}"
        kind = str(f.type)
        if kind == "bool":
            parser.add_argument(flag, action=argparse.BooleanOptionalAction, default=None,
                                help=f"default: {f.default}")  # fmt: skip
        else:
            caster = int if "int" in kind else float if "float" in kind else str
            parser.add_argument(flag, type=caster, default=None, help=f"default: {f.default}")


def build_config(cls: type, args: argparse.Namespace):
    """Defaults, then the --config file, then flags that were given explicitly."""
    values: dict[str, Any] = {}
    if getattr(args, "config", None):
        values.update(load_toml(args.config, cls))
    for f in dataclasses.fields(cls):
        given = getattr(args, f.name, None)
        if given is not None:
            values[f.name] = given
    return cls(**values)
