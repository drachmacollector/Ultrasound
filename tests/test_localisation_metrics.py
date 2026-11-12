"""src/localisation/metrics.py: hand-computed EPG, pointing, IoU and region fractions."""
from __future__ import annotations

import math

import numpy as np
import pytest

from src.localisation import metrics as LM


def square_mask(n: int = 20, lo: int = 5, hi: int = 10) -> np.ndarray:
    m = np.zeros((n, n), np.uint8)
    m[lo:hi, lo:hi] = 1
    return m


def test_saliency_equal_to_mask() -> None:
    m = square_mask()
    r = LM.epg_relative(m.astype(float), m)
    assert r["epg"] == 1.0
    assert r["area_frac"] == pytest.approx(25 / 400)
    assert r["epg_minus_area"] == pytest.approx(1 - 25 / 400)
    assert r["epg_over_area"] == pytest.approx(400 / 25)
    assert LM.pointing_hit(m.astype(float), m)


def test_uniform_saliency_scores_area_fraction() -> None:
    m = square_mask()
    r = LM.epg_relative(np.full(m.shape, 0.3), m)
    assert r["epg"] == pytest.approx(r["area_frac"])
    assert r["epg_minus_area"] == pytest.approx(0.0, abs=1e-12)
    assert r["epg_over_area"] == pytest.approx(1.0)


def test_saliency_outside_mask_scores_zero() -> None:
    m = square_mask()
    sal = 1.0 - m
    assert LM.energy_pointing_game(sal, m) == 0.0
    assert not LM.pointing_hit(sal, m)


def test_negative_saliency_is_ignored_and_zero_map_is_nan() -> None:
    m = square_mask()
    sal = -np.ones(m.shape)
    sal[6, 6] = 2.0
    assert LM.energy_pointing_game(sal, m) == 1.0
    assert math.isnan(LM.energy_pointing_game(np.zeros(m.shape), m))
    assert math.isnan(LM.fraction_in(-np.ones(m.shape), m))


def test_hand_computed_epg() -> None:
    m = square_mask()
    sal = np.zeros(m.shape)
    sal[6, 6] = 3.0   # inside
    sal[0, 0] = 1.0   # outside
    assert LM.energy_pointing_game(sal, m) == pytest.approx(0.75)


def test_pointing_ties_and_tolerance() -> None:
    m = square_mask()
    sal = np.zeros(m.shape)
    sal[0, 0] = sal[7, 7] = 1.0                       # tie: one maximum inside -> hit
    assert LM.pointing_hit(sal, m)
    sal2 = np.zeros(m.shape)
    sal2[5, 3] = 1.0                                  # 2 px left of the mask
    assert not LM.pointing_hit(sal2, m)
    assert LM.pointing_hit(sal2, m, tol_px=2)


def test_sector_aware_epg() -> None:
    m = square_mask()
    sector = np.zeros_like(m)
    sector[:, :10] = 1                                # left half
    sal = np.zeros(m.shape)
    sal[6, 6] = 1.0                                   # anatomy, in sector
    sal[2, 2] = 1.0                                   # sector, not anatomy
    sal[2, 15] = 2.0                                  # outside sector: ignored by the sector-aware EPG
    r = LM.epg_relative(sal, m, sector)
    assert r["epg"] == pytest.approx(0.25)
    assert r["epg_in_sector"] == pytest.approx(0.5)
    assert r["area_frac_in_sector"] == pytest.approx(25 / 200)
    assert r["epg_in_sector_minus_area"] == pytest.approx(0.5 - 25 / 200)


def test_fraction_in_complementary_regions_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    sal = rng.random((30, 30))
    reg = rng.random((30, 30)) > 0.4
    assert LM.fraction_in(sal, reg) + LM.fraction_in(sal, ~reg) == pytest.approx(1.0)
    soft = rng.random((30, 30))
    assert LM.fraction_in(sal, soft) + LM.fraction_in(sal, 1 - soft) == pytest.approx(1.0)


def test_box_iou() -> None:
    b = (2, 3, 12, 9)
    assert LM.box_iou(b, b) == 1.0
    assert LM.box_iou(b, None) == 0.0
    assert LM.box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(50 / 150)
    assert LM.box_iou((0, 0, 2, 2), (5, 5, 6, 6)) == 0.0


def test_cam_box_iou_identical_region() -> None:
    m = np.zeros((100, 100), np.uint8)
    m[30:70, 20:80] = 1
    sal = m.astype(float) + 0.01                      # CAM box = mask box; landmark box adds a 10% margin
    iou = LM.cam_box_iou(sal, m)
    assert iou == pytest.approx((60 * 40) / (72 * 48), rel=0.05)
    assert LM.cam_box_iou(np.zeros((100, 100)), m) == 0.0
