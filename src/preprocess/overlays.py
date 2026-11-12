"""
src/preprocess/overlays.py

Classical detection and inpainting of on-screen overlays.

* **UI / text**: (a) every bright pixel outside the (dilated) sector, plus
  (b) small bright components *anywhere* that sit on the black background
  (device name, settings, labels, ruler ticks), i.e. glyph-like components whose
  surrounding ring is mostly background. (b) catches labels that fall inside the
  convex hull of a fan that touches the top of the frame (e.g. "Voluson E6").
* **Markers** (inside the sector): measurement calipers ('+' / 'x') via
  normalised template matching on a white top-hat map, plus collinear runs of
  small bright dots (dotted measurement lines). Calipers are rendered differently
  per device (white or dark, solid or dotted), so this detector is only validated
  at image level against the binary audit (R1.7, GATE_G1.md).

Thresholds live in ``OverlayParams``. Tuning uses only the *train* split audit
images; the frozen parameters are saved to ``data/processed/audit/overlay_params.json``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from src.preprocess.sector import Sector, background_level, segment_sector


@dataclass(frozen=True)
class OverlayParams:
    # --- UI / text ---
    ui_min_intensity: int = 40            # UI pixels are brighter than this
    sector_dilate_px: int = 5             # margin around the sector before "outside" starts
    glyph_max_area_frac: float = 0.004    # glyph components are small...
    glyph_min_area: int = 8               # ...but not single speckle grains
    glyph_ring_px: int = 3
    glyph_ring_bg_frac: float = 0.6       # ...and mostly surrounded by background
    # --- markers ---
    tophat_k: int = 9
    marker_min_intensity: int = 200       # tuned on train audit (R1.7)
    marker_min_tophat: int = 60
    cross_sizes: tuple[int, ...] = (7, 9, 11, 13, 17, 21)
    cross_ncc: float = 0.78               # tuned on train audit (R1.7)
    max_crosses_per_image: int = 12       # more "crosses" than this = textured speckle
    dot_min_area: int = 2
    dot_max_area: int = 30
    dotted_min_dots: int = 5
    dotted_max_gap: int = 18
    dotted_angle_tol_deg: float = 6.0
    dotted_spacing_cv: float = 0.35
    score_dotted_lines: bool = False      # dotted runs fired on speckle in the train audit -> excluded
    caliper_score_threshold: int = 1      # image-level "calipers present" iff caliper_score >= this

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "OverlayParams":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        d["cross_sizes"] = tuple(d["cross_sizes"])
        return cls(**d)


DEFAULT_PARAMS = OverlayParams()


@dataclass(frozen=True)
class Overlays:
    ui_mask: np.ndarray                 # uint8 {0,1}: text, labels, ramps, rulers (anywhere)
    marker_mask: np.ndarray             # uint8 {0,1}: calipers / dotted lines inside the sector
    sector_clean: np.ndarray            # sector mask with in-hull UI glyphs removed
    n_ui_components: int
    n_crosses: int
    n_dotted_lines: int

    score_dotted_lines: bool = False

    @property
    def caliper_score(self) -> int:
        return self.n_crosses + (self.n_dotted_lines if self.score_dotted_lines else 0)

    @property
    def any_overlay_mask(self) -> np.ndarray:
        return (self.ui_mask | self.marker_mask).astype(np.uint8)


def cross_template(size: int, kind: str, stroke: int = 1) -> np.ndarray:
    """Synthetic '+' or 'x' marker of odd ``size`` px on a black canvas (float32 0/1)."""
    size |= 1
    t = np.zeros((size, size), np.uint8)
    c = size // 2
    if kind == "+":
        cv2.line(t, (c, 0), (c, size - 1), 1, stroke)
        cv2.line(t, (0, c), (size - 1, c), 1, stroke)
    elif kind == "x":
        cv2.line(t, (0, 0), (size - 1, size - 1), 1, stroke)
        cv2.line(t, (0, size - 1), (size - 1, 0), 1, stroke)
    else:
        raise ValueError(kind)
    return t.astype(np.float32)


def _ellipse(k: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def detect_glyphs(gray: np.ndarray, p: OverlayParams = DEFAULT_PARAMS,
                  protect: np.ndarray | None = None) -> np.ndarray:
    """Small bright components surrounded mostly by background (text, labels, ticks).

    Components whose centroid falls inside ``protect`` (the eroded device-geometry
    consensus sector) are treated as tissue fragments, not UI.
    """
    bg = background_level(gray)
    bright = (gray > max(p.ui_min_intensity, bg + 3)).astype(np.uint8)
    background = (gray <= bg + 3).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bright, connectivity=8)
    max_area = p.glyph_max_area_frac * gray.size
    out = np.zeros_like(bright)
    ring_k = _ellipse(2 * p.glyph_ring_px + 1)
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < p.glyph_min_area or area > max_area:
            continue
        if protect is not None and protect[int(round(cents[i][1])), int(round(cents[i][0]))]:
            continue
        x0, y0 = max(0, x - p.glyph_ring_px - 1), max(0, y - p.glyph_ring_px - 1)
        x1, y1 = min(gray.shape[1], x + w + p.glyph_ring_px + 1), min(gray.shape[0], y + h + p.glyph_ring_px + 1)
        comp = (labels[y0:y1, x0:x1] == i).astype(np.uint8)
        ring = cv2.dilate(comp, ring_k) & (1 - comp)
        if ring.sum() and background[y0:y1, x0:x1][ring > 0].mean() >= p.glyph_ring_bg_frac:
            out[y0:y1, x0:x1] |= comp
    return out


def _bright_thin(gray: np.ndarray, region: np.ndarray, p: OverlayParams) -> np.ndarray:
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, _ellipse(p.tophat_k))
    m = (tophat >= p.marker_min_tophat) & (gray >= p.marker_min_intensity) & (region > 0)
    return m.astype(np.uint8)


def _detect_crosses(bright: np.ndarray, p: OverlayParams) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    """Accepted cross centres (x, y, size) and their footprint mask."""
    img = bright.astype(np.float32)
    found: list[tuple[int, int, int, float]] = []
    for size in p.cross_sizes:
        for kind in ("+", "x"):
            t = cross_template(size, kind)
            if img.shape[0] < t.shape[0] or img.shape[1] < t.shape[1]:
                continue
            r = cv2.matchTemplate(img, t, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.nonzero(r >= p.cross_ncc)
            found.extend((int(x + size // 2), int(y + size // 2), size, float(r[y, x])) for y, x in zip(ys, xs))
    found.sort(key=lambda f: -f[3])
    kept: list[tuple[int, int, int]] = []
    for x, y, s, _ in found:                       # greedy non-max suppression
        if all((x - kx) ** 2 + (y - ky) ** 2 > max(s, ks) ** 2 for kx, ky, ks in kept):
            kept.append((x, y, s))
    footprint = np.zeros_like(bright)
    if len(kept) > p.max_crosses_per_image:
        return [], footprint
    for x, y, s in kept:
        h = s // 2 + 1
        sl = (slice(max(0, y - h), y + h + 1), slice(max(0, x - h), x + h + 1))
        footprint[sl] = bright[sl]
    return kept, footprint


def _detect_dotted(bright: np.ndarray, p: OverlayParams) -> tuple[int, np.ndarray]:
    """Count collinear, evenly spaced runs of small bright dots; return (n_runs, mask)."""
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bright, connectivity=8)
    area = stats[1:, cv2.CC_STAT_AREA]
    extent = np.maximum(stats[1:, cv2.CC_STAT_WIDTH], stats[1:, cv2.CC_STAT_HEIGHT])
    idx = np.nonzero((area >= p.dot_min_area) & (area <= p.dot_max_area) & (extent <= 7))[0]
    mask = np.zeros_like(bright)
    if len(idx) < p.dotted_min_dots or len(idx) > 2000:
        return 0, mask
    pts = cents[1:][idx]
    used = np.zeros(len(pts), bool)
    runs = 0
    tol = p.dotted_angle_tol_deg
    for i in range(len(pts)):
        if used[i]:
            continue
        d = pts - pts[i]
        dist = np.hypot(d[:, 0], d[:, 1])
        near = np.nonzero((dist > 0) & (dist <= p.dotted_max_gap * (p.dotted_min_dots - 1)) & ~used)[0]
        if len(near) < p.dotted_min_dots - 1:
            continue
        ang = np.degrees(np.arctan2(d[near, 1], d[near, 0])) % 180.0
        for a0 in np.unique(np.round(ang / tol)) * tol:
            sel = near[np.abs(((ang - a0) + 90) % 180 - 90) <= tol]
            members = np.concatenate([[i], sel])
            if len(members) < p.dotted_min_dots:
                continue
            u = np.array([np.cos(np.radians(a0)), np.sin(np.radians(a0))])
            gaps = np.diff(np.sort((pts[members] - pts[i]) @ u))
            if gaps.min() < 2 or gaps.max() > p.dotted_max_gap:
                continue
            if gaps.std() / max(gaps.mean(), 1e-6) > p.dotted_spacing_cv:
                continue
            runs += 1
            used[members] = True
            mask[np.isin(labels, idx[members] + 1)] = 1
            break
    return runs, mask


def detect_overlays(gray: np.ndarray, sector: Sector | None = None,
                    params: OverlayParams = DEFAULT_PARAMS, protect: np.ndarray | None = None) -> Overlays:
    """Detect UI/text (anywhere) and measurement markers (inside the sector).

    ``protect``: optional {0,1} mask (eroded device-geometry consensus sector); glyph-like
    components centred inside it are tissue fragments, not UI.
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    sector = sector or segment_sector(gray)
    inside = cv2.dilate(sector.mask, _ellipse(2 * params.sector_dilate_px + 1))
    outside_ui = ((gray > params.ui_min_intensity) & (inside == 0)).astype(np.uint8)
    glyphs = detect_glyphs(gray, params, protect)
    ui_mask = (outside_ui | glyphs).astype(np.uint8)
    n_ui = int(cv2.connectedComponents(ui_mask, connectivity=8)[0] - 1)

    sector_clean = (sector.mask & (1 - cv2.dilate(glyphs, _ellipse(5)))).astype(np.uint8)
    inner = cv2.erode(sector_clean, _ellipse(5))
    bright = _bright_thin(gray, inner, params)
    crosses, cross_mask = _detect_crosses(bright, params)
    n_dotted, dotted_mask = _detect_dotted(bright, params)
    return Overlays(ui_mask=ui_mask, marker_mask=(cross_mask | dotted_mask).astype(np.uint8),
                    sector_clean=sector_clean, n_ui_components=n_ui,
                    n_crosses=len(crosses), n_dotted_lines=n_dotted,
                    score_dotted_lines=params.score_dotted_lines)


def inpaint_overlays(gray: np.ndarray, mask: np.ndarray, dilate_px: int = 2, radius: int = 3) -> np.ndarray:
    """Telea inpainting of ``mask`` (dilated by ``dilate_px``) on a uint8 grey image."""
    if not mask.any():
        return gray.copy()
    m = cv2.dilate(mask.astype(np.uint8), _ellipse(2 * dilate_px + 1))
    return cv2.inpaint(gray, m, radius, cv2.INPAINT_TELEA)
