"""
src/preprocess/caliper_aug.py  (R5.2, arm A2: synthetic caliper augmentation)

With probability ``p``, draw 2-4 caliper markers ('+' or 'x'; arm length 4-10 px, i.e. total size
9-21 px; stroke 1-2 px; intensity U(230, 255)) at positions uniform inside the sector mask eroded by
the marker size, and with probability 0.5 a dotted line of 5-12 dots (spacing 4-8 px) between two of
the markers. Applied to images of every class (Other included), on the **native-scale** prepared
image before the geometric transform, so marker sizes are in real frame pixels.

Calipers are drawn because external sets show them often (UCL, Ghana, Uganda) while FPDB training
images rarely do (Gate G1); the augmentation makes the classifier indifferent to them.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CaliperAug:
    p: float = 0.3
    n_markers: tuple[int, int] = (2, 4)
    arm: tuple[int, int] = (4, 10)
    stroke: tuple[int, int] = (1, 2)
    intensity: tuple[int, int] = (230, 255)
    p_line: float = 0.5
    n_dots: tuple[int, int] = (5, 12)
    dot_spacing: tuple[int, int] = (4, 8)

    def __call__(self, gray: np.ndarray, sector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        out, _ = self.apply(gray, sector, rng)
        return out

    def apply(self, gray: np.ndarray, sector: np.ndarray,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """(augmented image, footprint mask of everything drawn)."""
        footprint = np.zeros(gray.shape, np.uint8)
        if self.p <= 0 or rng.random() >= self.p:
            return gray, footprint
        out = gray.copy()
        arm = int(rng.integers(self.arm[0], self.arm[1] + 1))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * arm + 3,) * 2)
        allowed = cv2.erode((sector > 0).astype(np.uint8), k)
        ys, xs = np.nonzero(allowed)
        if xs.size == 0:
            return gray, footprint
        n = int(rng.integers(self.n_markers[0], self.n_markers[1] + 1))
        pick = rng.choice(xs.size, size=n, replace=xs.size < n)
        centres = [(int(xs[i]), int(ys[i])) for i in pick]
        for cx, cy in centres:
            val = int(rng.integers(self.intensity[0], self.intensity[1] + 1))
            th = int(rng.integers(self.stroke[0], self.stroke[1] + 1))
            layer = np.zeros(gray.shape, np.uint8)
            if rng.random() < 0.5:   # '+'
                segs = [((cx - arm, cy), (cx + arm, cy)), ((cx, cy - arm), (cx, cy + arm))]
            else:                    # 'x'
                d = int(round(arm / np.sqrt(2)))
                segs = [((cx - d, cy - d), (cx + d, cy + d)), ((cx - d, cy + d), (cx + d, cy - d))]
            for a, b in segs:
                cv2.line(layer, a, b, 1, th, lineType=cv2.LINE_8)
            out[layer > 0] = val
            footprint |= layer
        if n >= 2 and rng.random() < self.p_line:
            (x0, y0), (x1, y1) = centres[0], centres[1]
            n_dots = int(rng.integers(self.n_dots[0], self.n_dots[1] + 1))
            spacing = int(rng.integers(self.dot_spacing[0], self.dot_spacing[1] + 1))
            length = float(np.hypot(x1 - x0, y1 - y0))
            if length > 2 * arm:
                ux, uy = (x1 - x0) / length, (y1 - y0) / length
                val = int(rng.integers(self.intensity[0], self.intensity[1] + 1))
                for j in range(n_dots):
                    t = arm + 2 + j * spacing
                    if t > length - arm - 2:
                        break
                    px, py = int(round(x0 + ux * t)), int(round(y0 + uy * t))
                    layer = np.zeros(gray.shape, np.uint8)
                    cv2.circle(layer, (px, py), 1, 1, -1)
                    layer &= (sector > 0).astype(np.uint8)
                    out[layer > 0] = val
                    footprint |= layer
        return out, footprint
