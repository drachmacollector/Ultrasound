"""R5.1 preprocessing modes: letterbox geometry, inpainting, train/eval parity, cache key, style augmentation."""
from __future__ import annotations

import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pytest
import torch
from albumentations.pytorch import ToTensorV2

from src.data.transforms import get_eval_transform, get_train_transform, style_augmentations
from src.preprocess import pipeline as P


def toy(h: int = 300, w: int = 420) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    gray = rng.integers(10, 200, (h, w), dtype=np.uint8)
    sector = np.zeros((h, w), np.uint8)
    sector[40:260, 100:380] = 1
    marker = np.zeros((h, w), np.uint8)
    marker[100:103, 150:170] = 1
    gray[marker > 0] = 255
    ui = np.zeros((h, w), np.uint8)
    ui[5:15, 5:60] = 1
    return gray, {"sector": sector, "sector_clean": sector.copy(), "ui": ui, "marker": marker}


@pytest.mark.parametrize("mode", P.LETTERBOX_MODES)
def test_letterbox_prepare_crops_and_zeroes(mode: str) -> None:
    gray, m = toy()
    img, sec = P.prepare_native(gray, m, mode)
    assert img.shape == sec.shape == (220, 280)               # sector bounding box
    assert sec.all()                                          # rectangular toy sector fills its box
    if mode == "sector_letterbox":
        assert np.array_equal(img, gray[40:260, 100:380])
    else:                                                     # markers inpainted before cropping
        assert img[60:63, 50:70].max() < 255
        outside_marker = np.ones_like(sec, bool)
        outside_marker[55:68, 45:75] = False
        assert np.array_equal(img[outside_marker], gray[40:260, 100:380][outside_marker])


def test_letterbox_zeroes_outside_a_non_rectangular_sector() -> None:
    gray, m = toy()
    m["sector_clean"][40:60, 100:120] = 0
    img, sec = P.prepare_native(gray, m, "sector_letterbox")
    assert not img[:20, :20].any() and not sec[:20, :20].any()


@pytest.mark.parametrize("size", [224, 300])
def test_letterbox_keeps_aspect_ratio(size: int) -> None:
    """A drawn circle stays a circle (axis ratio within 2%) after the letterbox geometry."""
    img = np.zeros((300, 520), np.uint8)
    cv2.circle(img, (260, 150), 120, 255, -1)
    out = P.to_input_geometry(img, size, "sector_letterbox")
    assert out.shape == (size, size)
    ys, xs = np.nonzero(out > 127)
    ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    assert abs(ratio - 1) < 0.02
    assert not out[:5].any() and not out[-5:].any()           # centred zero padding top and bottom


def test_letterbox_padding_uses_fill_zero_argument() -> None:
    """albumentations 2.0.8 names the pad value ``fill``; constructing it must work."""
    pad = [t for t in P.geometric(224, "sector_letterbox") if isinstance(t, A.PadIfNeeded)][0]
    assert pad.fill == 0 and pad.border_mode == cv2.BORDER_CONSTANT


@pytest.mark.parametrize("mode", ["resize", *P.LETTERBOX_MODES])
@pytest.mark.parametrize("style", [False, True])
def test_train_without_random_ops_equals_eval(mode: str, style: bool) -> None:
    img = np.random.default_rng(3).integers(0, 256, (231, 377, 3), dtype=np.uint8)
    a = get_train_transform(224, mode=mode, style_aug=style, augment=False)(image=img)["image"]
    b = get_eval_transform(224, mode=mode)(image=img)["image"]
    assert isinstance(a, torch.Tensor) and torch.equal(a, b)


def test_resize_eval_is_the_legacy_pipeline() -> None:
    img = np.random.default_rng(4).integers(0, 256, (381, 647, 3), dtype=np.uint8)
    legacy = A.Compose([A.Resize(224, 224), A.Normalize(), ToTensorV2()])
    assert torch.equal(legacy(image=img)["image"], get_eval_transform(224)(image=img)["image"])


def test_style_augmentation_list_constructs_and_runs() -> None:
    names = [type(t).__name__ for t in style_augmentations()]
    assert names == ["RandomGamma", "RandomBrightnessContrast", "GaussianBlur", "MultiplicativeNoise",
                     "Downscale", "CLAHE"]
    img = np.random.default_rng(5).integers(0, 256, (240, 300, 3), dtype=np.uint8)
    t = get_train_transform(224, mode="sector_letterbox", style_aug=True)
    for _ in range(5):
        out = t(image=img)["image"]
        assert out.shape == (3, 224, 224) and out.dtype == torch.float32


def test_cache_key_and_stale_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    params = tmp_path / "overlay_params.json"
    params.write_text(json.dumps({"a": 1}), encoding="utf-8")
    monkeypatch.setattr(P, "OVERLAY_PARAMS", params)
    monkeypatch.setattr(P, "CACHE_ROOT", tmp_path / "cache")
    with pytest.raises(FileNotFoundError):
        P.check_cache("sector_letterbox")
    P.write_cache_manifest("sector_letterbox")
    P.check_cache("sector_letterbox")
    params.write_text(json.dumps({"a": 2}), encoding="utf-8")       # detector parameters changed
    with pytest.raises(RuntimeError):
        P.check_cache("sector_letterbox")


def test_load_prepared_letterbox_reads_image_and_sector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(P, "CACHE_ROOT", tmp_path)
    gray, m = toy()
    img, sec = P.prepare_native(gray, m, "sector_letterbox")
    ip = P.cache_path("sector_letterbox", None, "fpdb", "img1")
    ip.parent.mkdir(parents=True)
    cv2.imwrite(str(ip), img)
    cv2.imwrite(str(P.sector_cache_path("sector_letterbox", "fpdb", "img1")), sec * 255)
    g2, s2 = P.load_prepared("sector_letterbox", 224, "fpdb", "img1", "unused.png")
    assert np.array_equal(g2, img) and np.array_equal(s2, sec)
