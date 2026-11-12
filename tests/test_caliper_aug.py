"""src/preprocess/caliper_aug.py: markers only inside the sector, only where drawn, deterministic."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.preprocess.caliper_aug import CaliperAug
from src.preprocess.overlays import detect_overlays
from src.preprocess.sector import sector_from_mask


def toy(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = 420, 560
    sector = np.zeros((h, w), np.uint8)
    cv2.ellipse(sector, (w // 2, -40), (430, 430), 90, -38, 38, 1, -1)
    gray = (rng.integers(20, 90, (h, w)) * sector).astype(np.uint8)   # dim speckle inside the fan
    return gray, sector


def test_p_zero_is_identity() -> None:
    gray, sector = toy()
    out, fp = CaliperAug(p=0.0).apply(gray, sector, np.random.default_rng(0))
    assert out is gray and not fp.any()


@pytest.mark.parametrize("seed", range(10))
def test_changes_only_inside_footprint_and_sector(seed: int) -> None:
    gray, sector = toy()
    out, fp = CaliperAug(p=1.0).apply(gray, sector, np.random.default_rng(seed))
    changed = out != gray
    assert changed.any()
    assert not (changed & (fp == 0)).any()                  # nothing outside the drawn footprint
    assert not (fp.astype(bool) & (sector == 0)).any()      # every marker pixel lies in the sector
    assert out[changed].min() >= 230


def test_deterministic_with_seed() -> None:
    gray, sector = toy()
    a = CaliperAug(p=1.0)(gray, sector, np.random.default_rng(5))
    b = CaliperAug(p=1.0)(gray, sector, np.random.default_rng(5))
    assert np.array_equal(a, b)


def test_probability_gate() -> None:
    gray, sector = toy()
    aug = CaliperAug(p=0.3)
    rng = np.random.default_rng(1)
    hits = sum(bool(aug.apply(gray, sector, rng)[1].any()) for _ in range(400))
    assert 0.22 < hits / 400 < 0.38


def test_r1_detector_finds_most_synthetic_markers() -> None:
    """Realism cross-check (plan R5.2): the frozen R1.5 detector finds >= 80% of drawn crosses.

    Checked for 1-px strokes: the R1.5 templates are 1 px wide and find none of the 2-px crosses
    (measured recall 0%), which says more about the detector (it failed Gate G1) than the markers.
    """
    gray, sector = toy()
    found = total = 0
    for seed in range(40):
        out, fp = CaliperAug(p=1.0, p_line=0.0, stroke=(1, 1)).apply(gray, sector, np.random.default_rng(seed))
        n_drawn = cv2.connectedComponents(fp)[0] - 1
        ov = detect_overlays(out, sector_from_mask(sector))
        total += n_drawn
        found += min(ov.n_crosses, n_drawn)
    assert found / total >= 0.8
