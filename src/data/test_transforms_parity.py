"""
src/data/test_transforms_parity.py

Verifies that prep_frame_grayscale_to_rgb() (in-memory BGR input) produces
output that is consistent with load_and_prep_grayscale_to_rgb() (path-based),
as required by PHASE_5_KICKOFF_PROMPT.md §2.

Parity notes (documented as a deliberate design note, not a bug):
-  For grayscale images (R=G=B per pixel, the typical ultrasound frame):
   cv2.cvtColor(BGRframe, COLOR_BGR2GRAY) and cv2.imread(path, IMREAD_GRAYSCALE)
   produce bit-identical output.  This is the relevant case for ultrasound inference.
-  For arbitrary colour images (R≠G≠B), OpenCV's COLOR_BGR2GRAY and the internal
   PNG-reader IMREAD_GRAYSCALE path use slightly different integer-arithmetic
   approximations of the same luminance formula (0.299R + 0.587G + 0.114B),
   resulting in ±1 differences in up to ~50% of pixels.  This is a known OpenCV
   implementation detail, not a correctness issue for grayscale ultrasound input.
-  The previous implementation round-tripped through imencode/imdecode to achieve
   bit-identical parity with imread(GRAYSCALE) even for colour images, but was
   ~10× slower per frame.  Per PHASE_5_KICKOFF_PROMPT.md §2, the spec-mandated
   implementation is COLOR_BGR2GRAY, and strict parity is only required for the
   actual input domain (grayscale ultrasound frames).
"""
import os
import sys

# Ensure the project root is on the path when run directly or via conda run.
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.join(_here, "..", "..")
if _root not in sys.path:
    sys.path.insert(0, _root)

import cv2
import numpy as np

from src.data.transforms import load_and_prep_grayscale_to_rgb, prep_frame_grayscale_to_rgb


def test_grayscale_input_strict_parity() -> None:
    """MAIN PARITY TEST — strict equality for grayscale images (R=G=B per pixel).

    This is the relevant case for ultrasound inference: video frames from
    cv2.VideoCapture on a greyscale-encoded AVI will have R=G=B everywhere,
    and both the path-based and in-memory functions must produce identical output.
    """
    test_path = "_tmp_parity_gray_test.png"
    gray_vals = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    # Produce a 3-channel BGR where all channels are equal (what VideoCapture
    # returns for a truly-grayscale video frame).
    dummy_bgr = np.stack([gray_vals, gray_vals, gray_vals], axis=-1)
    cv2.imwrite(test_path, dummy_bgr)
    try:
        img_rgb_path = load_and_prep_grayscale_to_rgb(test_path)
        frame_bgr = cv2.imread(test_path)
        assert frame_bgr is not None, "cv2.imread returned None — PNG write failed."
        img_rgb_mem = prep_frame_grayscale_to_rgb(frame_bgr)
        np.testing.assert_array_equal(
            img_rgb_path, img_rgb_mem,
            err_msg=(
                "Strict parity FAILED for grayscale input (R=G=B).  "
                "This is the ultrasound use case and must be bit-identical."
            ),
        )
        print("PASS  test_grayscale_input_strict_parity")
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)


def test_color_input_near_parity() -> None:
    """NEAR-PARITY TEST for arbitrary colour images — documents ±1 rounding.

    For random BGR images (R≠G≠B), COLOR_BGR2GRAY and imread(GRAYSCALE) use
    slightly different integer approximations of the same luminance formula,
    producing ±1 differences.  This test asserts max absolute difference ≤ 1,
    documents the known behaviour, and confirms it does NOT worsen for actual inputs.

    Note: this is NOT the primary use case.  Ultrasound frames are grayscale
    and are covered by test_grayscale_input_strict_parity above.
    """
    test_path = "_tmp_parity_color_test.png"
    dummy_bgr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    cv2.imwrite(test_path, dummy_bgr)
    try:
        img_rgb_path = load_and_prep_grayscale_to_rgb(test_path).astype(np.int16)
        frame_bgr = cv2.imread(test_path)
        assert frame_bgr is not None, "cv2.imread returned None — PNG write failed."
        img_rgb_mem = prep_frame_grayscale_to_rgb(frame_bgr).astype(np.int16)
        max_diff = int(np.max(np.abs(img_rgb_path - img_rgb_mem)))
        assert max_diff <= 1, (
            f"Max absolute difference for colour input = {max_diff} (expected ≤ 1).  "
            "This indicates an unexpected divergence beyond the known ±1 rounding."
        )
        print(
            f"PASS  test_color_input_near_parity  "
            f"(max_abs_diff={max_diff}, expected ≤1 for colour images — see docstring)"
        )
    finally:
        if os.path.exists(test_path):
            os.remove(test_path)


def test_output_shape_and_dtype() -> None:
    """Sanity-check: output must be HxWx3 uint8 regardless of input frame size."""
    h, w = 480, 640
    frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    out = prep_frame_grayscale_to_rgb(frame)
    assert out.shape == (h, w, 3), f"Expected shape ({h},{w},3), got {out.shape}"
    assert out.dtype == np.uint8, f"Expected uint8, got {out.dtype}"
    print("PASS  test_output_shape_and_dtype")


if __name__ == "__main__":
    test_grayscale_input_strict_parity()
    test_color_input_near_parity()
    test_output_shape_and_dtype()
    print("\nAll parity tests passed.")
