"""
from __future__ import annotations

src/models/cam_localisation.py

CAM-to-bounding-box extraction for weakly-supervised anatomical localisation.

§ Cross-reference: Phase 8, Stage 3, Task 3.1
  (docs/instructions/08_FINAL_EVALUATION_AND_POLISH.md §4, Task 3.1)
  Rationale: SonoNet (Baumgartner et al.) demonstrates that a classifier's
  saliency maps can be thresholded into a bounding box without any detection
  annotations.  This module applies the same principle to the Grad-CAM
  heatmaps already produced by src/models/gradcam.py, requiring zero new
  training and no detection annotations.

Design (per Task 3.1 specification):
--------------------------------------
1. INPUT: HxW float32 CAM heatmap in [0, 1] — the same array returned by
   GradCAM()(input_tensor, targets)[0] (or cam_resized after resize to frame).

2. THRESHOLD (two selectable methods):
   - "otsu"       (default): cv2.threshold with cv2.THRESH_OTSU on a uint8
     copy of the normalized heatmap.  Standard, parameter-free, well-
     understood.  May produce an empty mask on very diffuse/flat CAMs — this
     is the expected and correct behaviour (returns None).
   - "percentile" (fallback): threshold at the P-th percentile of all pixel
     values.  Selectable via method="percentile"; uses the `percentile`
     argument (default 80.0).  Useful when Otsu is too permissive/restrictive
     on a specific class during Task 3.3 spot-check.

3. LARGEST CONNECTED COMPONENT (not raw bbox of all thresholded pixels):
   cv2.connectedComponentsWithStats — selects the single component with the
   largest area and takes its bounding rectangle.
   Rationale: avoids a box spanning the whole image when the CAM has multiple
   small disconnected hot regions — a failure mode directly documented in
   docs/EXPERIMENTS.md Grad-CAM spot-check notes ("Heatmap was split across
   two disconnected regions... indicating diffuse uncertainty").

4. MINIMUM AREA FILTER: suppress boxes smaller than `min_area_frac` of the
   frame area (default 0.02 = 2%).  Guards against single-pixel cluster boxes.

5. OUTPUT: (x1, y1, x2, y2) in pixel coordinates (top-left, bottom-right),
   or None when:
     - The CAM is all-zero or uniform (Otsu → empty mask)
     - No connected component survives the minimum-area filter
     - Any other degenerate input

   None is a valid, expected outcome — callers must handle it gracefully.
   This function NEVER raises on any syntactically-valid numpy array input.

Important framing (from Task 3.2 honesty requirement):
  The box produced here is a "weakly-supervised approximate region" derived
  from classifier saliency, NOT a precisely localised anatomical boundary.
  It must be displayed with a dashed/semi-transparent style and labelled
  "approx. region (saliency-derived)" — see src/realtime/app.py for the
  rendering implementation.

Usage:
    conda run -n fetalplane python -m src.models.cam_localisation
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants (callers may read these for UI display)
# ---------------------------------------------------------------------------

#: Default thresholding method
DEFAULT_METHOD: str = "otsu"

#: Default percentile cutoff (used only when method="percentile")
DEFAULT_PERCENTILE: float = 80.0

#: Default minimum box area as a fraction of the full frame area
DEFAULT_MIN_AREA_FRAC: float = 0.02


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def cam_to_bbox(
    cam: np.ndarray,
    method: str = DEFAULT_METHOD,
    percentile: float = DEFAULT_PERCENTILE,
    min_area_frac: float = DEFAULT_MIN_AREA_FRAC,
) -> tuple[int, int, int, int] | None:
    """Extract a bounding box from a Grad-CAM heatmap.

    Uses thresholding → largest connected component → bounding rectangle.
    Returns None when the CAM is too diffuse or all-zero; never raises.

    Args:
        cam:           HxW float32 heatmap in [0, 1].  If values are outside
                       this range, they are clipped and normalised before use.
        method:        Thresholding method: "otsu" (default) or "percentile".
        percentile:    Percentile cutoff value (0–100).  Used only when
                       method="percentile".  Default 80.0.
        min_area_frac: Minimum accepted box area as a fraction of HxW.
                       Default 0.02 (2%).  Boxes smaller than this are
                       treated as degenerate and suppressed (returns None).

    Returns:
        (x1, y1, x2, y2) integer pixel coordinates (top-left, bottom-right),
        or None if no meaningful box can be extracted.
    """
    # ------------------------------------------------------------------
    # 0. Input validation + normalisation
    # ------------------------------------------------------------------
    if cam is None:
        return None

    cam = np.asarray(cam, dtype=np.float32)
    if cam.ndim != 2:
        log.warning("cam_to_bbox: expected 2-D array, got shape %s — returning None.", cam.shape)
        return None

    h, w = cam.shape
    frame_area = h * w
    if frame_area == 0:
        return None

    # Clip and normalise to [0, 1] defensively
    cam = np.clip(cam, 0.0, 1.0)
    cam_min, cam_max = float(cam.min()), float(cam.max())
    if cam_max - cam_min < 1e-6:
        # Flat / all-zero CAM — no discriminative region
        log.debug("cam_to_bbox: CAM is flat (range=%.2e) — returning None.", cam_max - cam_min)
        return None

    # Normalize to full [0,1] range so thresholding is consistent
    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    # ------------------------------------------------------------------
    # 1. Convert to uint8 for cv2 thresholding
    # ------------------------------------------------------------------
    cam_u8 = (cam_norm * 255.0).astype(np.uint8)

    # ------------------------------------------------------------------
    # 2. Threshold → binary mask
    # ------------------------------------------------------------------
    try:
        if method == "otsu":
            # cv2.THRESH_OTSU finds the optimal threshold automatically.
            # thresh_val is ignored; we only care about the binary mask.
            _, mask = cv2.threshold(
                cam_u8, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
        elif method == "percentile":
            thresh_val = float(np.percentile(cam_u8, percentile))
            _, mask = cv2.threshold(cam_u8, thresh_val, 255, cv2.THRESH_BINARY)
        else:
            log.warning("cam_to_bbox: unknown method %r — falling back to otsu.", method)
            _, mask = cv2.threshold(
                cam_u8, 0, 255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
    except Exception as exc:  # noqa: BLE001
        log.debug("cam_to_bbox: threshold failed (%s) — returning None.", exc)
        return None

    if mask is None or mask.max() == 0:
        log.debug("cam_to_bbox: threshold produced empty mask — returning None.")
        return None

    # ------------------------------------------------------------------
    # 3. Largest connected component
    # ------------------------------------------------------------------
    try:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("cam_to_bbox: connectedComponentsWithStats failed (%s).", exc)
        return None

    # Label 0 is the background — skip it
    if num_labels <= 1:
        # Only background found
        return None

    # Find the label with the largest area (excluding background=0)
    # stats[:, cv2.CC_STAT_AREA] gives per-label pixel counts
    component_areas = stats[1:, cv2.CC_STAT_AREA]  # skip background
    best_component = int(np.argmax(component_areas)) + 1  # +1 for background offset

    # ------------------------------------------------------------------
    # 4. Extract bounding box and apply minimum-area filter
    # ------------------------------------------------------------------
    cx = int(stats[best_component, cv2.CC_STAT_LEFT])
    cy = int(stats[best_component, cv2.CC_STAT_TOP])
    cw = int(stats[best_component, cv2.CC_STAT_WIDTH])
    ch = int(stats[best_component, cv2.CC_STAT_HEIGHT])
    box_area = cw * ch

    min_area_px = min_area_frac * frame_area
    if box_area < min_area_px:
        log.debug(
            "cam_to_bbox: box area %d < min threshold %.0f px (%.1f%% of frame) — returning None.",
            box_area, min_area_px, min_area_frac * 100.0,
        )
        return None

    x1 = max(0, cx)
    y1 = max(0, cy)
    x2 = min(w - 1, cx + cw)
    y2 = min(h - 1, cy + ch)

    log.debug(
        "cam_to_bbox: box=(%d,%d,%d,%d)  area=%d px  method=%s",
        x1, y1, x2, y2, box_area, method,
    )
    return (x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("cam_localisation smoke test")

    # Single hot blob
    cam_blob = np.zeros((64, 64), dtype=np.float32)
    cam_blob[20:45, 15:50] = 1.0
    box = cam_to_bbox(cam_blob)
    assert box is not None, "Expected a box for a clear hot blob"
    log.info("Single blob → box=%s  ✓", box)

    # Flat CAM → None
    cam_flat = np.full((64, 64), 0.5, dtype=np.float32)
    box_flat = cam_to_bbox(cam_flat)
    assert box_flat is None, "Expected None for flat CAM"
    log.info("Flat CAM → None  ✓")

    # All-zero CAM → None
    cam_zero = np.zeros((64, 64), dtype=np.float32)
    box_zero = cam_to_bbox(cam_zero)
    assert box_zero is None, "Expected None for all-zero CAM"
    log.info("Zero CAM → None  ✓")

    log.info("All smoke tests passed.")
    sys.exit(0)
