"""
src/localisation/anatomy_masks.py  (R4.2)

Expert-landmark anatomy masks in **native image pixel space** (uint8, shape (H, W)), and their
projection into model-input space.

* Head: ellipse with OFD as the major axis and BPD as the minor axis.
* Abdomen: ellipse on TAD / APAD; the longer of the two is the major axis (the FP release does
  not order them consistently, plans/00 §3.2).
* Femur: capsule around the FL segment, thickness ``max(3, rel_width * |FL|)``, round caps.

Ellipses: centre = midpoint of the major-axis landmarks, semi-axes |major|/2 and |minor|/2,
angle = atan2 of the major vector. Shapes are rasterised exactly (a pixel is in the mask if its
centre is inside the shape), so mask areas carry no outward line-drawing bias; landmarks
outside the image are fine (the window is clipped). The plan's ``cv2.ellipse`` drawing was
replaced because its boundary pixels inflated thin femur capsules by > 5%.

``to_model_space`` applies the geometric transform the model saw, as a soft mask in [0, 1]
(``cv2.INTER_LINEAR``, the interpolation ``A.Resize`` uses); threshold at 0.5 for binary metrics.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pandas as pd

from src.preprocess.pipeline import ALL_MODES, is_cached

FEMUR_REL_WIDTHS = (0.05, 0.10, 0.15)
FEMUR_MIN_WIDTH_PX = 3.0
BOX_MARGIN = 0.10


def _pt(p) -> np.ndarray:
    return np.asarray(p, dtype=np.float64).reshape(2)


def _grid(shape: tuple[int, int], lo: np.ndarray, hi: np.ndarray):
    """Pixel-centre coordinates of the shape's bounding window (clipped), plus the window slices."""
    h, w = shape[:2]
    x0, y0 = max(0, int(math.floor(lo[0]))), max(0, int(math.floor(lo[1])))
    x1, y1 = min(w, int(math.ceil(hi[0])) + 1), min(h, int(math.ceil(hi[1])) + 1)
    if x0 >= x1 or y0 >= y1:
        return None
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    return xx, yy, (slice(y0, y1), slice(x0, x1))


def _ellipse(major1, major2, minor1, minor2, shape: tuple[int, int]) -> np.ndarray:
    """Pixels whose centre lies inside the ellipse (exact rasterisation, no outward bias)."""
    p1, p2, q1, q2 = map(_pt, (major1, major2, minor1, minor2))
    c = (p1 + p2) / 2
    a = np.linalg.norm(p2 - p1) / 2
    b = np.linalg.norm(q2 - q1) / 2
    m = np.zeros(shape[:2], np.uint8)
    if a <= 0 or b <= 0:
        return m
    t = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
    g = _grid(shape, c - a, c + a)
    if g is None:
        return m
    xx, yy, win = g
    u = (xx - c[0]) * math.cos(t) + (yy - c[1]) * math.sin(t)
    v = -(xx - c[0]) * math.sin(t) + (yy - c[1]) * math.cos(t)
    m[win] = (u / a) ** 2 + (v / b) ** 2 <= 1.0
    return m


def head_ellipse(ofd1, ofd2, bpd1, bpd2, shape: tuple[int, int]) -> np.ndarray:
    return _ellipse(ofd1, ofd2, bpd1, bpd2, shape)


def abdomen_ellipse(tad1, tad2, apad1, apad2, shape: tuple[int, int]) -> np.ndarray:
    """Major axis = the longer of TAD and APAD."""
    t1, t2, a1, a2 = map(_pt, (tad1, tad2, apad1, apad2))
    if np.linalg.norm(t2 - t1) >= np.linalg.norm(a2 - a1):
        return _ellipse(t1, t2, a1, a2, shape)
    return _ellipse(a1, a2, t1, t2, shape)


def femur_width(fl1, fl2, rel_width: float = 0.10) -> float:
    return max(FEMUR_MIN_WIDTH_PX, rel_width * float(np.linalg.norm(_pt(fl2) - _pt(fl1))))


def femur_capsule(fl1, fl2, shape: tuple[int, int], rel_width: float = 0.10) -> np.ndarray:
    """Pixels whose centre lies within w/2 of the FL segment (rectangle plus round caps)."""
    p1, p2 = _pt(fl1), _pt(fl2)
    r = femur_width(p1, p2, rel_width) / 2
    m = np.zeros(shape[:2], np.uint8)
    g = _grid(shape, np.minimum(p1, p2) - r, np.maximum(p1, p2) + r)
    if g is None:
        return m
    xx, yy, win = g
    d = p2 - p1
    L2 = float(d @ d)
    t = np.zeros_like(xx) if L2 == 0 else np.clip(((xx - p1[0]) * d[0] + (yy - p1[1]) * d[1]) / L2, 0, 1)
    m[win] = (xx - p1[0] - t * d[0]) ** 2 + (yy - p1[1] - t * d[1]) ** 2 <= r * r
    return m


def landmark_box(mask: np.ndarray, margin: float = BOX_MARGIN) -> tuple[int, int, int, int] | None:
    """Bounding box (x1, y1, x2, y2; x2/y2 exclusive) of the mask, grown by ``margin`` x size, clipped."""
    ys, xs = np.nonzero(np.asarray(mask) > 0.5)
    if xs.size == 0:
        return None
    h, w = mask.shape[:2]
    x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    mx, my = margin * (x2 - x1), margin * (y2 - y1)
    return (int(max(0, math.floor(x1 - mx))), int(max(0, math.floor(y1 - my))),
            int(min(w, math.ceil(x2 + mx))), int(min(h, math.ceil(y2 + my))))


def _xy(row: pd.Series, name: str) -> tuple[float, float]:
    return float(row[f"{name}_x"]), float(row[f"{name}_y"])


def anatomy_mask_for_row(row: pd.Series, shape: tuple[int, int] | None = None,
                         femur_rel_width: float = 0.10) -> np.ndarray:
    """Mask for one landmark-table row (``fpdb_landmarks`` / ``external_landmarks`` schema)."""
    shape = shape or (int(row["height"]), int(row["width"]))
    anat = row["anatomy"]
    if anat == "Head":
        return head_ellipse(_xy(row, "ofd_1"), _xy(row, "ofd_2"), _xy(row, "bpd_1"), _xy(row, "bpd_2"), shape)
    if anat == "Abdomen":
        return abdomen_ellipse(_xy(row, "tad_1"), _xy(row, "tad_2"), _xy(row, "apad_1"), _xy(row, "apad_2"), shape)
    if anat == "Femur":
        return femur_capsule(_xy(row, "fl_1"), _xy(row, "fl_2"), shape, femur_rel_width)
    raise ValueError(f"no landmark mask for anatomy {anat!r}")


def to_model_space(mask: np.ndarray, *, preprocess_mode: str, img_size: int, sector=None) -> np.ndarray:
    """Native-space mask -> (img_size, img_size) float32 soft mask in the geometry the model saw.

    ``resize`` and the cached R3 modes (stored pre-resized) share the square resize. The
    letterbox modes (R5) must call the image pipeline's own geometry function; until they
    exist this raises.
    """
    if preprocess_mode == "resize" or (preprocess_mode in ALL_MODES and is_cached(preprocess_mode)):
        return cv2.resize(np.asarray(mask, np.float32), (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    raise NotImplementedError(f"to_model_space: preprocess_mode={preprocess_mode!r} not implemented yet")
