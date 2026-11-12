"""
tests/test_transforms_parity.py

Train/eval preprocessing parity for the B0 (`resize`) pipeline.

* the image loader is deterministic, returns HxWx3 uint8 with R=G=B, and converts
  RGB sources (e.g. Egypt images in the African set) to grey;
* with every random augmentation disabled, the training pipeline is geometrically
  identical to the eval pipeline;
* the eval pipeline is bit-identical to the legacy definition (A.Resize -> Normalize
  -> ToTensorV2) on real FPDB images, i.e. frozen checkpoints see exactly what they
  were trained/evaluated on.
R5 extends this file to the sector_letterbox modes.
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from albumentations.pytorch import ToTensorV2

from src.data.transforms import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_eval_transform,
    get_train_transform,
    load_and_prep_grayscale_to_rgb,
)


def test_loader_grey_png(tmp_path: Path) -> None:
    gray = np.random.default_rng(0).integers(0, 256, (37, 53), dtype=np.uint8)
    path = tmp_path / "g.png"
    cv2.imwrite(str(path), gray)
    out = load_and_prep_grayscale_to_rgb(path)
    assert out.shape == (37, 53, 3) and out.dtype == np.uint8
    for c in range(3):
        np.testing.assert_array_equal(out[..., c], gray)


def test_loader_rgb_source_becomes_grey(tmp_path: Path) -> None:
    bgr = np.random.default_rng(1).integers(0, 256, (20, 30, 3), dtype=np.uint8)
    path = tmp_path / "c.png"
    cv2.imwrite(str(path), bgr)
    out = load_and_prep_grayscale_to_rgb(str(path))
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])


def test_loader_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_and_prep_grayscale_to_rgb(tmp_path / "nope.png")


def _without_random(compose: A.Compose) -> A.Compose:
    """Drop every probabilistic augmentation, keeping deterministic steps."""
    keep: list[A.BasicTransform | A.BaseCompose] = [
        t for t in compose.transforms if isinstance(t, (A.Resize, A.Normalize, ToTensorV2))
    ]
    return A.Compose(keep)


@pytest.mark.parametrize("img_size,mean,std", [(224, IMAGENET_MEAN, IMAGENET_STD), (300, (0.5,) * 3, (0.5,) * 3)])
def test_train_eval_geometric_parity(img_size: int, mean: tuple, std: tuple) -> None:
    img = np.random.default_rng(2).integers(0, 256, (381, 647, 3), dtype=np.uint8)
    train_det = _without_random(get_train_transform(img_size, mean, std))
    eval_t = get_eval_transform(img_size, mean, std)
    a = train_det(image=img)["image"]
    b = eval_t(image=img)["image"]
    assert isinstance(a, torch.Tensor) and a.shape == (3, img_size, img_size)
    assert torch.equal(a, b)


@pytest.mark.data
def test_resize_mode_bit_identical_to_legacy() -> None:
    legacy = A.Compose([A.Resize(224, 224), A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()])
    current = get_eval_transform(224, IMAGENET_MEAN, IMAGENET_STD)
    df = pd.read_csv("data/splits/test.csv").groupby("source_machine").head(2)
    for p in df["image_path"]:
        img = load_and_prep_grayscale_to_rgb(Path(p.replace("\\", "/")))
        assert torch.equal(legacy(image=img)["image"], current(image=img)["image"])
