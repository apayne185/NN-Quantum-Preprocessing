import numpy as np
import pytest

from qnnbench.data import DedupStats, downsample, filter_digits, remove_contradicting


def test_filter_digits_labels_positive_class_true():
    x = np.arange(5)[:, None]
    y = np.array([3, 6, 1, 3, 6])
    xf, yf = filter_digits(x, y)
    assert xf.ravel().tolist() == [0, 1, 3, 4]
    assert yf.tolist() == [True, False, True, False]


def test_remove_contradicting_drops_conflicts_and_keeps_first_seen_order():
    xs = np.array([[1, 0], [0, 1], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    ys = np.array([True, False, True, True, True])
    x, y, stats = remove_contradicting(xs, ys)
    # [0, 1] appears with both labels and is removed; [1, 0] is kept once.
    assert x.tolist() == [[1, 0], [1, 1]]
    assert y.tolist() == [True, True]
    assert stats == DedupStats(n_input=5, n_unique=3, n_contradicting=1, n_kept=2)


def test_downsample_constant_image_is_constant():
    x = np.full((2, 28, 28), 0.7)
    out = downsample(x, 4)
    assert out.shape == (2, 4, 4)
    np.testing.assert_allclose(out, 0.7, rtol=1e-6)


@pytest.mark.network
def test_pipeline_reproduces_original_tfq_counts():
    """Counts recorded in the original TFQ notebook (docs, cells 28 and 32)."""
    from qnnbench.data import prepare_binary_mnist

    try:
        d = prepare_binary_mnist()
    except OSError as e:  # offline
        pytest.skip(f"MNIST download unavailable: {e}")
    assert d.dedup_small == DedupStats(12049, 10387, 49, 10338)
    assert d.dedup_binary == DedupStats(10338, 193, 44, 149)
    assert d.x_train.shape == (10338, 16)
    assert d.x_test.shape == (1968, 16)
    assert set(np.unique(d.x_train)) <= {0.0, 1.0}
