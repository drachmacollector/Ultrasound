"""Sector segmentation on synthetic frames with known geometry."""
from __future__ import annotations

import cv2
import numpy as np

import pytest

from src.preprocess.sector import (
    background_level,
    build_consensus,
    refine_with_consensus,
    sector_from_mask,
    segment_sector,
)


def _speckle(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(rng.gamma(2.0, 45.0, shape), 15, 230).astype(np.uint8)


def _fan(h: int, w: int, apex_y: int = -60, radius: int = 520, half_angle: float = 35.0) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, (w // 2, apex_y), (radius, radius), 90, -half_angle, half_angle, 1, -1)
    return m


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    return float((a & b).sum() / max((a | b).sum(), 1))


def test_fan_with_fluid_text_and_ruler() -> None:
    h, w = 480, 640
    fan = _fan(h, w)
    img = np.where(fan > 0, _speckle((h, w)), 0).astype(np.uint8)
    cv2.ellipse(img, (330, 250), (70, 45), 0, 0, 360, 4, -1)            # dark fluid inside the fan
    cv2.putText(img, "Voluson E6", (450, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 220, 1)
    for y in range(40, 460, 20):                                         # depth ruler, 1-px ticks
        cv2.line(img, (10, y), (16, y), 255, 1)
    sec = segment_sector(img)
    assert _iou(sec.mask, fan) >= 0.95
    assert sec.shape_class == "fan"


def test_fan_clipped_by_frame_edges() -> None:
    h, w = 400, 500
    fan = _fan(h, w, apex_y=-150, radius=700, half_angle=40.0)          # runs off bottom and sides
    img = np.where(fan > 0, _speckle((h, w), 1), 0).astype(np.uint8)
    sec = segment_sector(img)
    assert _iou(sec.mask, fan) >= 0.95
    assert sec.touches_border[2]                                         # bottom


def test_full_frame_image() -> None:
    sec = segment_sector(_speckle((300, 300), 2))
    assert sec.shape_class == "full" and sec.area_frac > 0.97


def test_linear_sector() -> None:
    img = np.zeros((300, 400), np.uint8)
    img[40:280, 60:340] = _speckle((240, 280), 3)
    sec = segment_sector(img)
    rect = np.zeros_like(img)
    rect[40:280, 60:340] = 1
    assert _iou(sec.mask, rect) >= 0.95 and sec.shape_class == "linear"


def test_nonzero_background() -> None:
    h, w = 300, 400
    fan = _fan(h, w, apex_y=-40, radius=330)
    img = np.where(fan > 0, _speckle((h, w), 4), 2).astype(np.uint8)    # background grey level 2
    assert background_level(img) == 2
    assert _iou(segment_sector(img).mask, fan) >= 0.95


def test_consensus_core_threshold() -> None:
    a = np.zeros((10, 10), np.uint8)
    a[2:8, 2:8] = 1
    b = a.copy()
    b[0:2, :] = 1                                   # extra region in 1 of 20 masks only
    masks = [a] * 19 + [b]
    core = build_consensus(masks)                   # 0.95 share
    assert np.array_equal(core, a)
    assert build_consensus(masks, min_share=0.05)[0:2].all()
    with pytest.raises(ValueError):
        build_consensus([])


def test_refine_with_consensus_is_union() -> None:
    img_mask = np.zeros((20, 20), np.uint8)
    img_mask[5:15, 5:12] = 1                        # dark right part of the fan was cut off
    core = np.zeros_like(img_mask)
    core[6:14, 6:15] = 1
    sec = refine_with_consensus(sector_from_mask(img_mask), core)
    assert np.array_equal(sec.mask, img_mask | core)
    assert sec.bbox == (5, 5, 15, 15)
    with pytest.raises(ValueError):
        refine_with_consensus(sector_from_mask(img_mask), np.zeros((5, 5), np.uint8))
