"""src/models/loading.py: inadmissible checkpoints are refused; a valid one loads with config-driven settings."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.models.backbone import build_model
from src.models.loading import RefusedCheckpointError, check_admissible, load_model

BASE_CFG = {
    "backbone": "resnet10t",
    "num_classes": 8,
    "image_size": 64,
    "normalize_mean": [0.5, 0.5, 0.5],
    "normalize_std": [0.5, 0.5, 0.5],
    "train_csv": "data/splits/train.csv",
    "val_csv": "data/splits/val.csv",
}


def test_refuses_multitask_state_dict() -> None:
    with pytest.raises(RefusedCheckpointError, match="multitask"):
        check_admissible(["retinanet.head.weight", "backbone.x"], BASE_CFG)


def test_refuses_acam() -> None:
    with pytest.raises(RefusedCheckpointError, match="ACAM"):
        check_admissible(["stem.weight"], {**BASE_CFG, "use_acam": True})


@pytest.mark.parametrize("bad", ["data/splits/multitask_train.csv", "data/processed/external/hc18.csv",
                                 "data\\splits\\test.csv"])
def test_refuses_contaminated_training_csv(bad: str) -> None:
    with pytest.raises(RefusedCheckpointError, match="forbidden token"):
        check_admissible(["stem.weight"], {**BASE_CFG, "train_csv": bad})


def test_loads_valid_checkpoint(tmp_path: Path) -> None:
    model = build_model("resnet10t", num_classes=8, pretrained=False)
    ckpt = tmp_path / "best.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": BASE_CFG}, ckpt)
    loaded = load_model(ckpt, device=torch.device("cpu"))
    assert loaded.img_size == 64 and loaded.preprocess_mode == "resize"
    assert loaded.normalize_mean == (0.5, 0.5, 0.5)
    x = loaded.eval_transform()(image=torch.zeros(80, 100, 3, dtype=torch.uint8).numpy())["image"]
    assert tuple(x.shape) == (3, 64, 64)
    with torch.no_grad():
        assert loaded.model(x[None]).shape == (1, 8)
