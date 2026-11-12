"""
src/realtime/capture.py

Single interface over cv2.VideoCapture for either a webcam index (int) or a
video file path (str/Path), so the rest of the pipeline is source-agnostic.

Per PHASE_5_KICKOFF_PROMPT.md §8 (Task 7).
"""
from __future__ import annotations

from pathlib import Path

import cv2


class FrameSource:
    """Unified read-only wrapper around cv2.VideoCapture.

    Accepts both webcam indices (int) and file paths (str or Path).  All
    downstream pipeline code operates on this abstraction, never on raw
    VideoCapture objects, so switching between a live feed and a pre-recorded
    clip requires zero changes to the inference or render layers.

    Args:
        source: Webcam device index (int) or path to a video file (str/Path).
        loop:   When True and source is a file path, rewind to frame 0 on EOF
                and continue reading.  Ignored for webcam sources (loop is
                undefined and potentially harmful there).

    Raises:
        RuntimeError: If the source cannot be opened by OpenCV.
    """

    def __init__(self, source: int | str | Path, loop: bool = False) -> None:
        self.source = source if isinstance(source, int) else str(source)
        self.loop = loop
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open video source: {source!r}.  "
                "For a webcam, check the device index.  "
                "For a file, check the path and codec support."
            )

    # ------------------------------------------------------------------
    # Core read
    # ------------------------------------------------------------------

    def read(self) -> tuple[bool, cv2.typing.MatLike | None]:
        """Read the next frame from the source.

        Returns:
            (ret, frame) where ret is True on success and frame is a BGR uint8
            ndarray.  On EOF from a non-looping file source or a dead webcam,
            returns (False, None).  On EOF from a looping file source, rewinds
            and returns the first frame of the next loop iteration.
        """
        ret, frame = self.cap.read()
        if not ret:
            if self.loop and not self.is_webcam:
                # Rewind and try once more; two consecutive failures → source dead.
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
        return ret, frame

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    def fps(self) -> float:
        """Nominal FPS reported by the source (may be 0 for live webcams)."""
        return self.cap.get(cv2.CAP_PROP_FPS) or 0.0

    def frame_count(self) -> int:
        """Total frames in the file; -1 for live sources that don't know."""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def resolution(self) -> tuple[int, int]:
        """Return (width, height) as reported by the capture device."""
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def release(self) -> None:
        """Release the underlying VideoCapture resource."""
        self.cap.release()

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *args: object) -> None:
        self.release()

    # ------------------------------------------------------------------
    # Type discriminator
    # ------------------------------------------------------------------

    @property
    def is_webcam(self) -> bool:
        """True if the source is a live webcam (integer device index), False if a file.

        Used by Task 9 (app.py) to decide whether to burn the mechanics-test
        watermark into the frame — per PHASE_5_KICKOFF_PROMPT.md §10:
        "burn a persistent, visible caption into the frame whenever
        FrameSource.is_webcam is True."
        """
        return isinstance(self.source, int)
