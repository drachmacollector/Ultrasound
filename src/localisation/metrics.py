"""
src/localisation/metrics.py  (R4.3)

Saliency-vs-anatomy metrics. ``sal`` is any (H, W) saliency map in model-input space; only
its positive part counts (``relu``). Masks/regions are (H, W) arrays in the same space; masks
are binarised at 0.5, ``fraction_in`` regions may be soft weights in [0, 1].

* Energy pointing game (EPG) = sum(relu(sal) * mask) / sum(relu(sal)). A uniform map scores
  the mask's area fraction, so EPG is always reported with that baseline:
  ``epg_minus_area`` (primary; 0 = no better than uniform) and ``epg_over_area``.
* Sector-aware EPG restricts both energy and area to the fan sector:
  sum(sal * mask * sector) / sum(sal * sector), baseline area(mask & sector) / area(sector).
  It separates "looks at the anatomy" from "does not look outside the fan".
"""
from __future__ import annotations

import cv2
import numpy as np

from src.localisation.anatomy_masks import landmark_box
from src.models.cam_localisation import cam_to_bbox

NAN = float("nan")


def _relu(sal: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(sal, dtype=np.float64), 0.0)


def _bin(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask, dtype=np.float64) > 0.5


def energy_pointing_game(sal: np.ndarray, mask: np.ndarray) -> float:
    s = _relu(sal)
    tot = s.sum()
    return float(s[_bin(mask)].sum() / tot) if tot > 0 else NAN


def epg_relative(sal: np.ndarray, mask: np.ndarray, sector: np.ndarray | None = None) -> dict[str, float]:
    m = _bin(mask)
    area = float(m.mean())
    epg = energy_pointing_game(sal, m)
    out = {"epg": epg, "area_frac": area, "epg_minus_area": epg - area,
           "epg_over_area": epg / area if area > 0 else NAN}
    if sector is not None:
        sec = _bin(sector)
        s = _relu(sal) * sec
        tot, n_sec = s.sum(), sec.sum()
        e = float(s[m].sum() / tot) if tot > 0 else NAN
        a = float((m & sec).sum() / n_sec) if n_sec > 0 else NAN
        out.update({"epg_in_sector": e, "area_frac_in_sector": a, "epg_in_sector_minus_area": e - a})
    return out


def pointing_hit(sal: np.ndarray, mask: np.ndarray, tol_px: int = 0) -> bool:
    """True if any maximum of ``sal`` lies inside the mask (dilated by ``tol_px``)."""
    s = np.asarray(sal, dtype=np.float64)
    m = _bin(mask)
    if tol_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * tol_px + 1,) * 2)
        m = cv2.dilate(m.astype(np.uint8), k).astype(bool)
    return bool(m[s == s.max()].any())


def box_iou(a: tuple[int, int, int, int] | None, b: tuple[int, int, int, int] | None) -> float:
    """IoU of two (x1, y1, x2, y2) boxes; 0 if either is missing or empty."""
    if a is None or b is None:
        return 0.0
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def cam_box_iou(sal: np.ndarray, mask: np.ndarray) -> float:
    """IoU of the saliency box (``cam_to_bbox``, Otsu) and the landmark box; 0 if no CAM box."""
    s = _relu(sal)
    mx = s.max()
    cam_box = cam_to_bbox((s / mx).astype(np.float32)) if mx > 0 else None
    return box_iou(cam_box, landmark_box(_bin(mask)))


def fraction_in(sal: np.ndarray, region: np.ndarray) -> float:
    """Share of positive saliency energy inside ``region`` (soft weights allowed)."""
    s = _relu(sal)
    tot = s.sum()
    return float((s * np.asarray(region, dtype=np.float64)).sum() / tot) if tot > 0 else NAN
