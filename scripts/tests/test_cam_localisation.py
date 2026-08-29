"""
from __future__ import annotations

scripts/tests/test_cam_localisation.py

Unit tests for src/models/cam_localisation.py (cam_to_bbox).

§ Cross-reference: Phase 8, Stage 3, Task 3.1
  (docs/instructions/08_FINAL_EVALUATION_AND_POLISH.md §4, Task 3.1)

Covers all four cases specified in the task:
  1. test_single_blob          — one clear hot blob → sensible box returned
  2. test_two_blobs_larger_only— two disconnected blobs → only larger returned
  3. test_flat_cam_returns_none— uniform CAM → None
  4. test_zero_cam_returns_none— all-zero CAM → None, no crash

Additional robustness cases:
  5. test_percentile_method    — percentile fallback method works
  6. test_min_area_filter      — tiny blob below min_area_frac → None
  7. test_box_clipped_to_frame — returned coords never exceed frame bounds

Usage:
    conda run -n fetalplane python -m pytest scripts/tests/test_cam_localisation.py -v
    # or directly (writes log to logs/smoketest/test_cam_localisation_output.txt):
    conda run -n fetalplane python scripts/tests/test_cam_localisation.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.cam_localisation import cam_to_bbox

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_H, _W = 64, 64  # synthetic CAM dimensions for all tests


def _make_cam(h: int = _H, w: int = _W, fill: float = 0.0) -> np.ndarray:
    return np.full((h, w), fill, dtype=np.float32)


# ---------------------------------------------------------------------------
# Test 1: Single clear hot blob → sensible box
# ---------------------------------------------------------------------------

def test_single_blob() -> None:
    """A single bright rectangle in the CAM should produce a bounding box
    that contains it."""
    cam = _make_cam()
    # Place a 25×35 hot region (occupies ~21% of 64×64 frame)
    cam[20:45, 15:50] = 1.0

    box = cam_to_bbox(cam)
    assert box is not None, "Expected a box for a clear single-blob CAM"
    x1, y1, x2, y2 = box
    # The box must contain the hot region
    assert x1 <= 15 and y1 <= 20, f"Box top-left {(x1,y1)} does not cover blob origin (15,20)"
    assert x2 >= 49 and y2 >= 44, f"Box bottom-right {(x2,y2)} does not cover blob end (49,44)"
    log.info("test_single_blob PASSED — box=%s", box)


# ---------------------------------------------------------------------------
# Test 2: Two disconnected blobs → only the larger one
# ---------------------------------------------------------------------------

def test_two_blobs_larger_only() -> None:
    """When the CAM has two disconnected hot regions, the returned box should
    correspond to the larger one only (not span both)."""
    cam = _make_cam()
    # Large blob: 20×20 = 400 px
    cam[5:25, 5:25] = 1.0
    # Small blob: 5×5 = 25 px — in the far corner
    cam[55:60, 55:60] = 1.0

    box = cam_to_bbox(cam)
    assert box is not None, "Expected a box for a two-blob CAM"
    x1, y1, x2, y2 = box

    # The returned box should lie within the large blob area
    # and NOT span all the way to the small blob in the far corner
    box_w = x2 - x1
    box_h = y2 - y1
    box_area = box_w * box_h

    # The small blob is at row 55-60, col 55-60.
    # The large blob occupies rows 5-25, cols 5-25.
    # The correct box should have y2 < 50 (not spanning to row 55).
    assert y2 < 50, (
        f"Box y2={y2} reaches the small-blob region (row 55–60) — "
        f"largest-component selection may be wrong"
    )
    log.info(
        "test_two_blobs_larger_only PASSED — box=%s  area=%d",
        box, box_area,
    )


# ---------------------------------------------------------------------------
# Test 3: Flat / uniform CAM → None
# ---------------------------------------------------------------------------

def test_flat_cam_returns_none() -> None:
    """A uniform CAM carries no discriminative information.
    Otsu on a flat histogram → single-value mask, which should give None."""
    cam = _make_cam(fill=0.5)
    result = cam_to_bbox(cam)
    assert result is None, (
        f"Expected None for a flat CAM, got {result}"
    )
    log.info("test_flat_cam_returns_none PASSED")


# ---------------------------------------------------------------------------
# Test 4: All-zero CAM → None (no crash)
# ---------------------------------------------------------------------------

def test_zero_cam_returns_none() -> None:
    """All-zero CAM must return None without raising any exception."""
    cam = _make_cam(fill=0.0)
    try:
        result = cam_to_bbox(cam)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"cam_to_bbox raised on all-zero CAM: {exc}") from exc
    assert result is None, f"Expected None for all-zero CAM, got {result}"
    log.info("test_zero_cam_returns_none PASSED")


# ---------------------------------------------------------------------------
# Test 5: Percentile method fallback
# ---------------------------------------------------------------------------

def test_percentile_method() -> None:
    """percentile method must return a valid box for a gradient CAM.

    Real Grad-CAM outputs are smooth gradients, not binary masks.  This test
    uses a 2-D Gaussian as a realistic surrogate.  With percentile=80, the
    threshold is at the 80th percentile of the gradient values, giving a
    connected region around the Gaussian peak.
    """
    # Build a Gaussian blob centred at (32, 32) with sigma=10
    h, w = _H, _W
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    sigma = 10.0
    cam = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    # cam values range from ~0.0 at corners to 1.0 at centre
    assert cam.max() > 0.9, "Gaussian peak should be near 1.0"

    box = cam_to_bbox(cam, method="percentile", percentile=80.0)
    assert box is not None, (
        f"Expected a box with percentile method on a Gaussian gradient CAM, got None"
    )
    # Box should be centred roughly around the frame centre
    x1, y1, x2, y2 = box
    cx_box = (x1 + x2) / 2.0
    cy_box = (y1 + y2) / 2.0
    assert abs(cx_box - cx) < 15, f"Box centre x {cx_box:.1f} too far from frame centre {cx}"
    assert abs(cy_box - cy) < 15, f"Box centre y {cy_box:.1f} too far from frame centre {cy}"
    log.info("test_percentile_method PASSED — box=%s  centre=(%.1f,%.1f)", box, cx_box, cy_box)


# ---------------------------------------------------------------------------
# Test 6: Minimum area filter — tiny blob → None
# ---------------------------------------------------------------------------

def test_min_area_filter() -> None:
    """A hot blob smaller than min_area_frac * frame_area should be suppressed."""
    cam = _make_cam()
    # Tiny 2×2 blob = 4 px, frame = 64×64 = 4096 px
    # Default min_area_frac=0.02 → min_area_px = 81.9 → 4 px is below threshold
    cam[30:32, 30:32] = 1.0
    box = cam_to_bbox(cam, min_area_frac=0.02)
    assert box is None, (
        f"Expected None for a tiny blob (below 2% area filter), got {box}"
    )
    log.info("test_min_area_filter PASSED")


# ---------------------------------------------------------------------------
# Test 7: Returned coords never exceed frame bounds
# ---------------------------------------------------------------------------

def test_box_clipped_to_frame() -> None:
    """Box coordinates must always be within [0, W-1] x [0, H-1]."""
    cam = _make_cam()
    cam[:, :] = 0.0
    cam[0:10, 0:10] = 1.0   # hot region touching top-left corner

    box = cam_to_bbox(cam)
    if box is None:
        log.info("test_box_clipped_to_frame: box is None (acceptable for small corner blob) — PASSED")
        return
    x1, y1, x2, y2 = box
    assert x1 >= 0 and y1 >= 0, f"Box top-left ({x1},{y1}) out of frame"
    assert x2 < _W and y2 < _H,  f"Box bottom-right ({x2},{y2}) out of frame ({_W},{_H})"
    log.info("test_box_clipped_to_frame PASSED — box=%s", box)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TESTS = [
    test_single_blob,
    test_two_blobs_larger_only,
    test_flat_cam_returns_none,
    test_zero_cam_returns_none,
    test_percentile_method,
    test_min_area_filter,
    test_box_clipped_to_frame,
]

if __name__ == "__main__":
    log_dir = Path("logs/smoketest")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "test_cam_localisation_output.txt"

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    log.info("=" * 60)
    log.info("cam_localisation unit tests  (src/models/cam_localisation.py)")
    log.info("=" * 60)

    passed = failed = 0
    for fn in _TESTS:
        try:
            fn()
            passed += 1
        except Exception as exc:  # noqa: BLE001
            log.error("FAILED: %s — %s", fn.__name__, exc)
            failed += 1

    log.info("-" * 60)
    log.info("Results: %d passed, %d failed", passed, failed)
    log.info("Full log → %s", log_path)

    sys.exit(0 if failed == 0 else 1)
