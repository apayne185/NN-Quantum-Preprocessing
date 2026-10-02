"""Device selection and the environment metadata recorded with every result."""

from __future__ import annotations

import os
import platform
import subprocess
from datetime import datetime, timezone

import torch


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine()


def env_info(device: torch.device) -> dict:
    """Everything needed to tell whether two benchmark results are comparable."""
    info = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(device),
        "device_name": device_name(device),
        "cpu_threads": torch.get_num_threads(),
        "hostname": platform.node(),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            cuda=torch.version.cuda,
            gpu_memory_gb=round(props.total_memory / 1e9, 1),
            compute_capability=f"{props.major}.{props.minor}",
            sm_count=props.multi_processor_count,
            driver_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        )
    return info
