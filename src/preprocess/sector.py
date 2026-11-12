"""
src/preprocess/sector.py

Segment the ultrasound field of view (the "sector": fan, linear rectangle, or the
whole frame for pre-cropped images) from the surrounding UI.

FETAL_PLANES_DB frames carry burned-in text (device name, settings), a grey-level
ramp and a depth ruler beside the fan, and the fan often runs off the frame
edges. The algorithm therefore:

1. estimates the background grey level (mode of the darkest pixels; usually 0,
   but 1-2 on some Voluson E6 / Mindray frames) and thresholds just above it.
   Fan interiors contain near-black fluid, so any Otsu-style threshold would
   cut them out;
2. opens with a small kernel to remove thin text strokes and ruler ticks;
3. closes with a larger kernel to merge speckle fragments and bridge fluid;
4. keeps the largest connected component and fills its convex hull.

Everything is label-free; there are no learned parameters.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

FULL_FRAME_FRAC = 0.97
FAN_WIDTH_RATIO = 0.85


@dataclass(frozen=True)
class Sector:
    mask: np.ndarray                         # uint8 {0,1}, HxW
    bbox: tuple[int, int, int, int]          # x0, y0, x1, y1 (exclusive)
    area_frac: float
    width_ratio: float                       # top width / bottom width of the sector
    shape_class: str                         # "fan" | "linear" | "full"
    touches_border: tuple[bool, bool, bool, bool]   # top, right, bottom, left


def _ellipse(k: int) -> np.ndarray:
    k = max(1, int(k)) | 1   # odd
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def width_ratio(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> float:
    """Sector width 10% below its top / width 10% above its bottom (fan << 1, linear ~ 1)."""
    x0, y0, x1, y1 = bbox
    h = y1 - y0
    top_w = int(mask[y0 + max(1, h // 10)].sum())
    bot_w = int(mask[y1 - 1 - max(1, h // 10)].sum())
    return top_w / bot_w if bot_w > 0 else 1.0


def _shape_class(ratio: float, area_frac: float) -> str:
    if area_frac >= FULL_FRAME_FRAC:
        return "full"
    return "fan" if ratio < FAN_WIDTH_RATIO else "linear"


def background_level(gray: np.ndarray) -> int:
    """Most common grey value among the darkest pixels (<= 20): the UI background."""
    dark = gray[gray <= 20]
    if dark.size == 0:
        return 0
    return int(np.bincount(dark.ravel(), minlength=21).argmax())


def segment_sector(gray: np.ndarray, thr: int | None = None, open_k: int = 3) -> Sector:
    """Segment the ultrasound sector of a greyscale uint8 image.

    Args:
        gray: HxW uint8 (an HxWx3 image is converted to grey).
        thr: fixed foreground threshold; default ``background_level + 3``.
        open_k: opening kernel removing thin text/ruler strokes.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        raise TypeError(f"expected uint8 image, got {gray.dtype}")
    h, w = gray.shape
    g = cv2.medianBlur(gray, 3)
    if thr is None:
        thr = background_level(gray) + 3
    fg = (g > thr).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _ellipse(open_k))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, _ellipse(max(15, round(0.03 * min(h, w)))))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    mask = np.zeros((h, w), np.uint8)
    if n > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        pts = cv2.findNonZero((labels == largest).astype(np.uint8))
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 1)
    if not mask.any():                      # blank image: treat the whole frame as sector
        mask[:] = 1

    return sector_from_mask(mask)


def sector_from_mask(mask: np.ndarray) -> Sector:
    """Build a Sector (bbox, area, shape class, border contacts) from a {0,1} mask."""
    mask = (mask > 0).astype(np.uint8)
    ys, xs = np.nonzero(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    area_frac = float(mask.mean())
    touches = (bool(mask[0].any()), bool(mask[:, -1].any()), bool(mask[-1].any()), bool(mask[:, 0].any()))
    ratio = width_ratio(mask, bbox)
    return Sector(mask=mask, bbox=bbox, area_frac=area_frac, width_ratio=ratio,
                  shape_class=_shape_class(ratio, area_frac), touches_border=touches)


CONSENSUS_MIN_SHARE = 0.95


def build_consensus(masks: list[np.ndarray], min_share: float = CONSENSUS_MIN_SHARE) -> np.ndarray:
    """"Core" sector of a same-device, same-size group of *training* images.

    Pixels inside the per-image sector in >= ``min_share`` of the group. A majority vote
    (0.5) covers 92-95% of the frame for the main Voluson E6 / Aloka geometries (zoomed
    frames fill the screen) and would swallow the UI strips; the 0.95 core is the
    always-imaged fan interior and stays clear of them (checked visually, R1).
    """
    if not masks:
        raise ValueError("no masks")
    acc = np.zeros(masks[0].shape, np.float32)
    for m in masks:
        acc += m > 0
    return (acc / len(masks) >= min_share).astype(np.uint8)


def refine_with_consensus(sector: Sector, consensus: np.ndarray) -> Sector:
    """Union of the per-image sector with the device-geometry consensus.

    The per-image segmentation's dominant failure (R1 sector check, GATE_G1) is
    *under*-inclusion: dark, fluid- or shadow-filled parts of the fan fall below the
    background threshold and are cut off. Including non-fan background is harmless
    (it is black), excluding anatomy is not, so the union is the conservative choice.
    """
    if consensus.shape != sector.mask.shape:
        raise ValueError(f"consensus shape {consensus.shape} != image {sector.mask.shape}")
    return sector_from_mask(sector.mask | consensus)
