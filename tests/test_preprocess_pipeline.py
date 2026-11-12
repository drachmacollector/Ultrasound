"""src/preprocess/pipeline.py: mode rendering and cache parity with the on-the-fly pipeline."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.data.transforms import get_eval_transform, get_train_transform, gray_to_rgb, load_grayscale
from src.preprocess.pipeline import (
    ALL_MODES,
    CACHED_MODES,
    LETTERBOX_MODES,
    is_cached,
    is_letterbox,
    render,
    resize_for_cache,
)


def toy() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    gray = rng.integers(0, 256, (120, 160), dtype=np.uint8)
    sector = np.zeros_like(gray)
    sector[20:110, 40:140] = 1
    ui = np.zeros_like(gray)
    ui[2:10, 2:30] = 1
    marker = np.zeros_like(gray)
    marker[50:53, 60:63] = 1
    return gray, {"sector": sector, "sector_clean": sector.copy(), "ui": ui, "marker": marker}


def test_nonanat_only_keeps_outside_and_markers() -> None:
    gray, m = toy()
    out = render(gray, m, "nonanat_only")
    s, mk = m["sector_clean"].astype(bool), m["marker"].astype(bool)
    assert np.array_equal(out[~s], gray[~s])                 # outside the sector untouched
    assert np.array_equal(out[mk], gray[mk])                 # markers pasted back
    fill = s & ~mk
    assert (out[fill] == np.uint8(np.median(gray[s]))).all()  # one grey level inside


def test_overlay_only_and_shape_only() -> None:
    gray, m = toy()
    ov = render(gray, m, "overlay_only")
    keep = (m["ui"] | m["marker"]).astype(bool)
    assert np.array_equal(ov[keep], gray[keep]) and not ov[~keep].any()
    sh = render(gray, m, "sector_shape_only")
    assert set(np.unique(sh)) == {0, 255} and np.array_equal(sh == 255, m["sector"].astype(bool))
    assert render(gray, m, "resize") is gray


def test_mode_registry() -> None:
    assert not is_cached("resize") and all(is_cached(m) for m in CACHED_MODES)
    assert all(is_letterbox(m) and not is_cached(m) for m in LETTERBOX_MODES)
    assert set(ALL_MODES) == {"resize", *CACHED_MODES, *LETTERBOX_MODES}
    with pytest.raises(ValueError):
        is_cached("sector_crop")


@pytest.mark.parametrize("size", [224, 300])
def test_cached_image_matches_on_the_fly_pipeline(tmp_path: Path, size: int) -> None:
    """Pre-resized cache + standard eval transform == standard eval transform on the native image."""
    gray, m = toy()
    native = render(gray, m, "nonanat_only")
    p = tmp_path / "c.png"
    cv2.imwrite(str(p), resize_for_cache(native, size))
    t = get_eval_transform(img_size=size)
    cached = t(image=gray_to_rgb(load_grayscale(p)))["image"]
    direct = t(image=gray_to_rgb(native))["image"]
    assert np.array_equal(cached.numpy(), direct.numpy())
    # the train pipeline starts with the same resize, so it is an identity on the cached image
    assert get_train_transform(img_size=size).transforms[0].__class__.__name__ == "Resize"
