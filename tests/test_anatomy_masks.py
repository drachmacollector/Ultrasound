"""src/localisation/anatomy_masks.py: landmark masks match analytic shapes and the model geometry."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transform
from src.localisation import anatomy_masks as AM

H, W = 400, 560


def dice(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return 2 * (a & b).sum() / (a.sum() + b.sum())


def analytic_ellipse(cx, cy, a, b, theta_deg, shape=(H, W)) -> np.ndarray:
    yy, xx = np.mgrid[: shape[0], : shape[1]].astype(np.float64)
    t = math.radians(theta_deg)
    u = (xx - cx) * math.cos(t) + (yy - cy) * math.sin(t)
    v = -(xx - cx) * math.sin(t) + (yy - cy) * math.cos(t)
    return ((u / a) ** 2 + (v / b) ** 2 <= 1).astype(np.uint8)


def axis_endpoints(cx, cy, a, b, theta_deg):
    t = math.radians(theta_deg)
    ux, uy = math.cos(t), math.sin(t)
    vx, vy = -math.sin(t), math.cos(t)
    return ((cx - a * ux, cy - a * uy), (cx + a * ux, cy + a * uy),
            (cx - b * vx, cy - b * vy), (cx + b * vx, cy + b * vy))


@pytest.mark.parametrize("theta", [0.0, 30.0, 90.0])
def test_head_ellipse_matches_drawn_ellipse(theta: float) -> None:
    cx, cy, a, b = 281.3, 197.6, 150.0, 110.0
    ref = analytic_ellipse(cx, cy, a, b, theta)
    m = AM.head_ellipse(*axis_endpoints(cx, cy, a, b, theta), (H, W))
    assert m.dtype == np.uint8 and m.shape == (H, W)
    assert dice(m, ref) >= 0.98


@pytest.mark.parametrize("theta", [0.0, 30.0, 90.0])
def test_abdomen_uses_longer_axis_as_major(theta: float) -> None:
    cx, cy, a, b = 270.0, 200.0, 140.0, 95.0
    p1, p2, q1, q2 = axis_endpoints(cx, cy, a, b, theta)
    ref = analytic_ellipse(cx, cy, a, b, theta)
    m1 = AM.abdomen_ellipse(p1, p2, q1, q2, (H, W))      # TAD = major
    m2 = AM.abdomen_ellipse(q1, q2, p1, p2, (H, W))      # TAD = minor: swapped internally
    assert dice(m1, ref) >= 0.98
    assert np.array_equal(m1, m2)


@pytest.mark.parametrize("rel", AM.FEMUR_REL_WIDTHS)
def test_femur_capsule_area(rel: float) -> None:
    p1, p2 = (100.0, 120.0), (420.0, 260.0)
    fl = math.dist(p1, p2)
    w = AM.femur_width(p1, p2, rel)
    assert w == pytest.approx(max(3.0, rel * fl))
    m = AM.femur_capsule(p1, p2, (H, W), rel)
    expected = fl * w + math.pi * (w / 2) ** 2
    assert abs(m.sum() - expected) / expected < 0.05


def test_femur_minimum_width() -> None:
    assert AM.femur_width((0, 0), (10, 0), 0.10) == 3.0


def test_clipped_landmarks_give_valid_mask() -> None:
    m = AM.head_ellipse((-60.0, 200.0), (300.0, 200.0), (120.0, 60.0), (120.0, 340.0), (H, W))
    assert m.shape == (H, W) and m.dtype == np.uint8 and set(np.unique(m)) <= {0, 1}
    assert m[:, 0].any() and m.sum() > 0                  # touches the clipped border
    box = AM.landmark_box(m)
    assert box is not None and box[0] == 0


def test_landmark_box_margin() -> None:
    m = np.zeros((100, 100), np.uint8)
    m[20:40, 30:80] = 1                                   # 50 wide, 20 high
    assert AM.landmark_box(m, margin=0.0) == (30, 20, 80, 40)
    assert AM.landmark_box(m, margin=0.10) == (25, 18, 85, 42)
    assert AM.landmark_box(np.zeros((5, 5))) is None


def test_mask_for_row_dispatch() -> None:
    row = pd.Series({"anatomy": "Femur", "height": H, "width": W, "fl_1_x": 100.0, "fl_1_y": 100.0,
                     "fl_2_x": 300.0, "fl_2_y": 100.0})
    m = AM.anatomy_mask_for_row(row)
    assert m.shape == (H, W) and m[100, 200] == 1 and m[150, 200] == 0
    with pytest.raises(ValueError):
        AM.anatomy_mask_for_row(pd.Series({"anatomy": "Thorax", "height": 5, "width": 5}))


@pytest.mark.parametrize("size", [224, 300])
def test_to_model_space_matches_eval_transform(size: int) -> None:
    """The projected mask agrees with an image carrying the same ellipse, sent through the real eval transform."""
    m = AM.head_ellipse(*axis_endpoints(300.0, 180.0, 170.0, 120.0, 30.0), (H, W))
    img = np.repeat((m * 255).astype(np.uint8)[..., None], 3, axis=2)
    t = get_eval_transform(img_size=size)(image=img)["image"].numpy()
    ch0 = t[0] * IMAGENET_STD[0] + IMAGENET_MEAN[0]       # undo normalisation -> [0, 1]
    proj = AM.to_model_space(m, preprocess_mode="resize", img_size=size)
    assert proj.shape == (size, size) and proj.dtype == np.float32
    assert dice(proj > 0.5, ch0 > 0.5) >= 0.98


def test_to_model_space_cached_modes_and_unknown() -> None:
    m = np.ones((10, 20), np.uint8)
    assert AM.to_model_space(m, preprocess_mode="nonanat_only", img_size=8).shape == (8, 8)
    with pytest.raises(NotImplementedError):
        AM.to_model_space(m, preprocess_mode="sector_letterbox", img_size=8)
