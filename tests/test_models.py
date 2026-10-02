import numpy as np
import torch

from qnnbench.data import BinaryMNIST, DedupStats
from qnnbench.models import QNN, FairMLP, LookupTable, n_params
from qnnbench.train import TrainConfig, train


def test_parameter_counts_match_original_models():
    assert n_params(QNN()) == 32
    assert n_params(FairMLP()) == 37


def test_qnn_output_is_an_expectation_and_methods_agree():
    torch.manual_seed(0)
    bits = (torch.rand(4, 16) > 0.5).float()
    model = QNN(grad_method="autograd")
    out = model(bits)
    assert out.shape == (4,) and out.abs().max() <= 1 + 1e-5
    model.grad_method = "adjoint"
    torch.testing.assert_close(model(bits), out)


def test_lookup_table_is_majority_vote():
    x = np.array([[0, 1], [0, 1], [0, 1], [1, 1], [1, 1]], dtype=np.float32)
    y = np.array([True, True, False, False, False])  # unseen inputs -> majority (False)
    table = LookupTable().fit(x, y)
    assert table.predict(np.array([[0, 1], [1, 1], [0, 0]])).tolist() == [True, False, False]


def _toy_data(n=256, seed=0) -> BinaryMNIST:
    """Linearly separable stand-in for MNIST: label = first pixel."""
    rng = np.random.default_rng(seed)
    x = (rng.random((n, 16)) > 0.5).astype(np.float32)
    y = x[:, 0] > 0.5
    stats = DedupStats(n, n, 0, n)
    return BinaryMNIST(x, y, x[:64], y[:64], stats, stats)


def test_mlp_learns_toy_task():
    # With 2 hidden ReLUs some seeds kill both units (seed 0 does); seed 1 trains.
    cfg = TrainConfig(model="mlp", epochs=30, batch_size=32, lr=0.05, device="cpu", seed=1)
    result = train(cfg, data=_toy_data(), log=lambda *_: None)
    assert result.final_test_acc > 0.95


def test_training_is_reproducible_with_seed():
    cfg = TrainConfig(model="mlp", epochs=2, device="cpu", seed=3)
    a = train(cfg, data=_toy_data(), log=lambda *_: None)
    b = train(cfg, data=_toy_data(), log=lambda *_: None)
    assert a.history[-1].train_loss == b.history[-1].train_loss


def test_qnn_trains_on_a_tiny_batch():
    data = _toy_data(n=64)
    cfg = TrainConfig(model="qnn", epochs=1, batch_size=32, device="cpu", grad_method="adjoint")
    result = train(cfg, data=data, log=lambda *_: None)
    assert np.isfinite(result.history[0].train_loss)
