import argparse
from pathlib import Path

import pytest

from qnnbench.config import add_dataclass_args, build_config, load_toml
from qnnbench.hybrid import HybridConfig
from qnnbench.train import TrainConfig

CONFIGS = Path(__file__).parent.parent / "configs"


def _parse(cls, argv):
    p = argparse.ArgumentParser()
    add_dataclass_args(p, cls)
    return build_config(cls, p.parse_args(argv))


def test_precedence_defaults_then_file_then_flags(tmp_path):
    cfg_file = tmp_path / "run.toml"
    cfg_file.write_text('model = "mlp"\nepochs = 7\nlr = 0.01\n')
    cfg = _parse(TrainConfig, ["--config", str(cfg_file), "--epochs", "2"])
    assert cfg.model == "mlp"  # from file
    assert cfg.epochs == 2  # flag beats file
    assert cfg.lr == 0.01  # from file
    assert cfg.batch_size == TrainConfig().batch_size  # default


def test_flags_without_file_and_bool_options():
    cfg = _parse(HybridConfig, ["--features", "random", "--no-quantize", "--compile"])
    assert (cfg.features, cfg.quantize, cfg.compile) == ("random", False, True)
    assert _parse(TrainConfig, ["--num-train", "500"]).num_train == 500


def test_unknown_keys_are_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text("epochz = 3\n")
    with pytest.raises(ValueError, match="epochz"):
        load_toml(bad, TrainConfig)


@pytest.mark.parametrize("path", sorted(CONFIGS.glob("*.toml")), ids=lambda p: p.name)
def test_shipped_configs_are_valid(path):
    cls = HybridConfig if path.name.startswith("hybrid") else TrainConfig
    cfg = cls(**load_toml(path, cls))
    if isinstance(cfg, HybridConfig):
        assert cfg.batch_size % cfg.accum_steps == 0
