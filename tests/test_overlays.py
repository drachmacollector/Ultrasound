"""Overlay detection and inpainting on synthetic frames with known overlays."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.preprocess.overlays import (
    DEFAULT_PARAMS,
    OverlayParams,
    cross_template,
    detect_glyphs,
    detect_overlays,
    inpaint_overlays,
)
from src.preprocess.sector import segment_sector


def _frame(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    h, w = 480, 640
    fan = np.zeros((h, w), np.uint8)
    cv2.ellipse(fan, (w // 2, -60), (520, 520), 90, -35, 35, 1, -1)
    rng = np.random.default_rng(seed)
    tissue = cv2.GaussianBlur(rng.gamma(2.0, 30.0, (h, w)), (5, 5), 0)   # smooth, never saturated
    img = np.where(fan > 0, np.clip(tissue, 10, 150), 0).astype(np.uint8)
    return img, fan


def _draw_cross(img: np.ndarray, x: int, y: int, size: int, kind: str = "+") -> None:
    t = (cross_template(size, kind) > 0).astype(np.uint8)
    r = size // 2
    region = img[y - r:y + r + 1, x - r:x + r + 1]
    region[t > 0] = 255


def test_crosses_found_and_speckle_clean() -> None:
    img, _ = _frame()
    for (x, y), s in zip([(260, 200), (360, 260), (320, 330)], [9, 13, 17]):
        _draw_cross(img, x, y, s)
    ov = detect_overlays(img)
    assert ov.n_crosses == 3 and ov.caliper_score == 3
    clean, _ = _frame(seed=1)
    assert detect_overlays(clean).caliper_score == 0


def test_text_outside_and_glyph_inside_hull() -> None:
    img, fan = _frame()
    cv2.putText(img, "Voluson E6", (470, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 220, 1)
    ov = detect_overlays(img)
    text_region = np.zeros_like(img)
    text_region[10:40, 465:600] = 1
    assert ov.ui_mask[text_region > 0].sum() > 20
    assert ov.ui_mask[fan > 0].mean() < 0.01               # tissue is not UI
    assert ov.n_ui_components > 0


def test_glyph_on_black_detected() -> None:
    img = np.zeros((480, 640), np.uint8)                      # realistic frame size (area cap is relative)
    cv2.putText(img, "A2", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 230, 2)
    assert detect_glyphs(img).sum() > 50


def test_inpaint_restores_background_and_leaves_rest() -> None:
    img, _ = _frame()
    marked = img.copy()
    _draw_cross(marked, 320, 250, 15)
    mask = (marked != img).astype(np.uint8)
    out = inpaint_overlays(marked, mask, dilate_px=2, radius=3)
    local = img[235:266, 305:336]
    assert abs(float(out[mask > 0].mean()) - float(np.median(local))) <= 10
    far = cv2.dilate(mask, np.ones((15, 15), np.uint8)) == 0
    assert np.array_equal(out[far], marked[far])


def test_params_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "params.json"
    DEFAULT_PARAMS.to_json(p)
    assert OverlayParams.from_json(p) == DEFAULT_PARAMS


def test_frozen_params_match_code() -> None:
    frozen = Path("data/processed/audit/overlay_params.json")
    if frozen.exists():
        assert OverlayParams.from_json(frozen) == DEFAULT_PARAMS, "code defaults drifted from frozen params"


def test_sector_clean_excludes_glyphs() -> None:
    img, _ = _frame()
    cv2.putText(img, "E6", (300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 220, 1)   # label inside the fan's hull
    sec = segment_sector(img)
    ov = detect_overlays(img, sec)
    assert ov.sector_clean.sum() <= sec.mask.sum()
