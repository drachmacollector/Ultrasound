"""train.py refuses to write into the frozen checkpoints/ folder and rejects unimplemented options."""
from __future__ import annotations

import pytest

from src.train.train import assert_not_frozen, assert_supported_options, resolve_output_dirs


def test_frozen_checkpoint_dir_refused() -> None:
    with pytest.raises(SystemExit, match="frozen"):
        assert_not_frozen({"backbone": "x", "checkpoint_dir": "checkpoints/convnext_tiny", "log_dir": "runs/x"})


def test_frozen_subdir_refused() -> None:
    with pytest.raises(SystemExit, match="frozen"):
        assert_not_frozen({"backbone": "x", "checkpoint_dir": "checkpoints/convnext_tiny/smoke_test",
                           "log_dir": "runs/x"})


def test_out_dir_redirects() -> None:
    cfg = resolve_output_dirs({"backbone": "x", "checkpoint_dir": "checkpoints/convnext_tiny",
                               "log_dir": "logs/convnext_tiny"}, "runs/_smoke/a")
    assert cfg["checkpoint_dir"].replace("\\", "/") == "runs/_smoke/a"
    assert cfg["log_dir"].replace("\\", "/") == "runs/_smoke/a/tb"


@pytest.mark.parametrize("opt", [{"preprocess_mode": "sector_letterbox"}, {"sampler": "acquisition_balanced"},
                                 {"style_aug": True}, {"caliper_aug_p": 0.3}, {"use_acam": True}])
def test_unimplemented_options_rejected(opt: dict) -> None:
    with pytest.raises(NotImplementedError):
        assert_supported_options(opt)


def test_b0_options_accepted() -> None:
    assert_supported_options({"preprocess_mode": "resize", "sampler": "none", "style_aug": False,
                              "caliper_aug_p": 0.0, "use_class_weights": True})
