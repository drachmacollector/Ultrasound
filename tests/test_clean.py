"""R7 E11: the reusable input cleaning reproduces the study's B2 input exactly."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.preprocess.clean import clean_image
from src.preprocess.pipeline import cache_path


def _fan(h: int = 240, w: int = 320) -> np.ndarray:
    """Synthetic frame: dark background, a bright textured fan sector, one caliper cross inside it."""
    rng = np.random.default_rng(0)
    img = np.zeros((h, w), np.uint8)
    mask = np.zeros((h, w), np.uint8)
    cv2.ellipse(mask, (w // 2, 10), (200, 200), 0, 50, 130, 1, -1)
    img[mask > 0] = rng.integers(60, 140, int(mask.sum()), dtype=np.uint8)
    cv2.line(img, (155, 120), (165, 120), 255, 1)
    cv2.line(img, (160, 115), (160, 125), 255, 1)
    return img


def test_synthetic_shapes_and_outside_zero():
    c = clean_image(_fan(), size=224)
    assert c.letterbox.shape == (224, 224) and c.letterbox.dtype == np.uint8
    assert c.native.shape == c.sector.shape
    assert (c.native[c.sector == 0] == 0).all()                      # nothing survives outside the sector
    assert set(c.masks) == {"sector", "sector_clean", "ui", "marker"}


def test_native_crop_letterbox_keeps_aspect():
    c = clean_image(_fan(), size=224)
    h, w = c.native.shape
    rows = np.nonzero(c.letterbox.any(axis=1))[0]
    # the longer side fills 224 px; the shorter side is scaled by the same factor (+-1 px rounding)
    if w >= h:
        assert abs((rows.max() - rows.min() + 1) - round(h * 224 / w)) <= 2


@pytest.mark.parametrize("ds", ["hc18", "ucl", "african"])
def test_matches_cached_b2_input(ds):
    man = Path(f"data/processed/external/{ds}.csv")
    if not man.exists():
        pytest.skip("external manifests not available")
    rows = pd.read_csv(man).sample(3, random_state=1)
    for _, r in rows.iterrows():
        ref = cv2.imread(str(cache_path("sector_letterbox_inpaint", None, ds, r.image_id)), cv2.IMREAD_GRAYSCALE)
        if ref is None or not Path(r.image_path).exists():
            pytest.skip("B2 cache or raw images not available")
        got = clean_image(cv2.imread(r.image_path, cv2.IMREAD_GRAYSCALE), size=None).native
        assert got.shape == ref.shape and np.array_equal(got, ref), r.image_id


def test_cli(tmp_path):
    src = tmp_path / "in" / "sub"
    src.mkdir(parents=True)
    cv2.imwrite(str(src / "a.png"), _fan())
    out = tmp_path / "out"
    rc = subprocess.run([sys.executable, "scripts/clean_ultrasound.py", "--in", str(tmp_path / "in"), "--out", str(out),
                         "--size", "128", "--save-masks"], capture_output=True, text=True)
    assert rc.returncode == 0, rc.stderr
    assert cv2.imread(str(out / "sub" / "a.png"), cv2.IMREAD_GRAYSCALE).shape == (128, 128)
    assert (out / "sub" / "a_masks.png").exists()
    log = pd.read_csv(out / "clean_log.csv")
    assert list(log["status"]) == ["ok"]
