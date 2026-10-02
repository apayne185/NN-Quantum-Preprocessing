"""MNIST loading and the Farhi et al. / TFQ preprocessing pipeline, without TensorFlow.

The steps mirror the original TFQ code so results are comparable:
filter to digits 3 vs 6, bilinear downsample 28x28 -> 4x4, drop images whose
pixels map to both labels, then binarize at 0.5 for basis-state encoding.
"""

from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
MNIST_SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
DEFAULT_CACHE = Path("data")


def load_mnist(cache_dir: Path | str = DEFAULT_CACHE) -> dict[str, np.ndarray]:
    """Download (once) and load MNIST as uint8 images and int labels.

    Uses the same mnist.npz file as ``tf.keras.datasets.mnist`` and verifies
    its checksum, so a corrupted or tampered download fails loudly.
    """
    path = Path(cache_dir) / "mnist.npz"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(MNIST_URL, tmp)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != MNIST_SHA256:
        raise RuntimeError(f"{path} has sha256 {digest}, expected {MNIST_SHA256}")
    with np.load(path) as f:
        return {k: f[k] for k in ("x_train", "y_train", "x_test", "y_test")}


def filter_digits(
    x: np.ndarray, y: np.ndarray, positive: int = 3, negative: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Keep two digits; label is True for ``positive`` and False for ``negative``."""
    keep = (y == positive) | (y == negative)
    return x[keep], y[keep] == positive


def downsample(x: np.ndarray, size: int) -> np.ndarray:
    """Bilinear resize of (N, H, W) images in [0, 1] to (N, size, size), float32.

    ``align_corners=False`` without antialiasing matches TF2's default
    ``tf.image.resize`` (half-pixel centers), which the original pipeline used.
    """
    t = torch.from_numpy(np.asarray(x, dtype=np.float32))[:, None]
    out = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)
    return out[:, 0].numpy()


@dataclass(frozen=True)
class DedupStats:
    n_input: int
    n_unique: int
    n_contradicting: int
    n_kept: int


def remove_contradicting(
    xs: np.ndarray, ys: np.ndarray
) -> tuple[np.ndarray, np.ndarray, DedupStats]:
    """Drop every image whose exact pixel pattern appears with both labels.

    Duplicates with a single consistent label are kept once, in first-seen
    order, matching the TFQ tutorial's dict-based implementation.
    """
    flat = np.ascontiguousarray(xs.reshape(len(xs), -1))
    keys = flat.view(np.dtype((np.void, flat.dtype.itemsize * flat.shape[1]))).ravel()
    _, first_idx, inverse = np.unique(keys, return_index=True, return_inverse=True)
    inverse = inverse.ravel()
    n_groups = len(first_idx)
    pos = np.bincount(inverse, weights=ys.astype(np.float64), minlength=n_groups)
    total = np.bincount(inverse, minlength=n_groups)
    consistent = (pos == 0) | (pos == total)

    order = np.argsort(first_idx)  # first-seen order
    kept_groups = order[consistent[order]]
    idx = first_idx[kept_groups]
    stats = DedupStats(
        n_input=len(xs),
        n_unique=n_groups,
        n_contradicting=int((~consistent).sum()),
        n_kept=len(idx),
    )
    return xs[idx], ys[idx], stats


@dataclass(frozen=True)
class BinaryMNIST:
    """3-vs-6 MNIST at 4x4, binarized. Images are (N, 16) float32 in {0, 1}."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    dedup_small: DedupStats
    dedup_binary: DedupStats


def prepare_binary_mnist(
    cache_dir: Path | str = DEFAULT_CACHE, size: int = 4, threshold: float = 0.5
) -> BinaryMNIST:
    raw = load_mnist(cache_dir)
    x_train, y_train = filter_digits(raw["x_train"] / 255.0, raw["y_train"])
    x_test, y_test = filter_digits(raw["x_test"] / 255.0, raw["y_test"])

    x_train_small = downsample(x_train, size)
    x_test_small = downsample(x_test, size)
    x_train_small, y_train, dedup_small = remove_contradicting(x_train_small, y_train)

    x_train_bin = (x_train_small > threshold).astype(np.float32)
    x_test_bin = (x_test_small > threshold).astype(np.float32)
    # Only reported, as in the original: deduplicating after binarization would
    # leave ~150 training images, too few to train on.
    _, _, dedup_binary = remove_contradicting(x_train_bin, y_train)

    return BinaryMNIST(
        x_train=x_train_bin.reshape(len(x_train_bin), -1),
        y_train=y_train,
        x_test=x_test_bin.reshape(len(x_test_bin), -1),
        y_test=y_test,
        dedup_small=dedup_small,
        dedup_binary=dedup_binary,
    )
