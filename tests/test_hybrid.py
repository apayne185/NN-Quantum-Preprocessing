import pytest
import torch

from qnnbench.hybrid import HybridConfig, amp_dtype, build_model, train


@pytest.mark.parametrize("features", ["quanv", "random", "learned"])
def test_models_produce_class_logits(features):
    model = build_model(features)
    x = torch.rand(2, 4, 14, 14) if features == "quanv" else torch.rand(2, 1, 28, 28)
    assert model(x).shape == (2, 10)


def test_random_patch_encoder_is_frozen_and_learned_is_not():
    frozen = build_model("random")[0].conv.weight
    learned = build_model("learned")[0].conv.weight
    assert not frozen.requires_grad and learned.requires_grad


def test_amp_dtype_selection():
    cpu = torch.device("cpu")
    assert amp_dtype("auto", cpu) is None  # CPU autocast only when asked for
    assert amp_dtype("bf16", cpu) is torch.bfloat16
    assert amp_dtype("off", cpu) is None


@pytest.mark.network
@pytest.mark.slow
def test_hybrid_training_runs_with_accumulation_and_quantization():
    cfg = HybridConfig(features="learned", epochs=1, train_size=1024, batch_size=64,
                       accum_steps=2, num_workers=0)  # fmt: skip
    try:
        result = train(cfg, log=lambda *_: None)
    except OSError as e:
        pytest.skip(f"MNIST download unavailable: {e}")
    assert result["history"][0]["test_acc"] > 0.3
    q = result["quantization"]
    assert q["int8"]["size_MB"] < q["fp32"]["size_MB"]
    assert abs(q["int8"]["acc_2k"] - q["fp32"]["acc_2k"]) < 0.05
