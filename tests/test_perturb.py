"""src/audit/perturb.py: each perturbation changes only its declared region, deterministically."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from src.audit import perturb as P
from src.audit.metadata import save_masks


def toy(seed: int = 0) -> tuple[np.ndarray, dict]:
    rng = np.random.default_rng(seed)
    h, w = 200, 260
    gray = rng.integers(0, 256, (h, w), dtype=np.uint8)
    sector = np.zeros((h, w), np.uint8)
    cv2.ellipse(sector, (w // 2, -20), (190, 190), 90, -35, 35, 1, -1)
    ui = np.zeros((h, w), np.uint8)
    ui[5:15, 200:250] = 1                                   # settings text, outside the sector
    ui[100:110, 3:12] = 1                                   # ruler ticks
    marker = np.zeros((h, w), np.uint8)
    return gray, {"image_id": "toy", "masks": {"sector": sector, "sector_clean": sector.copy(), "ui": ui,
                                               "marker": marker}}


@pytest.mark.parametrize("kind", P.KINDS)
def test_only_declared_region_changes(kind: str) -> None:
    gray, meta = toy()
    out = P.Perturbation(kind)(gray, meta)
    reg = P.region(kind, meta["masks"])
    assert out.shape == gray.shape and out.dtype == np.uint8
    assert np.array_equal(out[~reg], gray[~reg])            # pixel-exact outside the region
    assert not np.array_equal(out[reg], gray[reg])          # and something changed inside it
    sector = meta["masks"]["sector_clean"].astype(bool)
    assert np.array_equal(out[sector], gray[sector])        # the anatomy is never touched


def test_region_definitions() -> None:
    _, meta = toy()
    m = meta["masks"]
    assert P.region("a_text", m).sum() < P.region("b_zero", m).sum()
    assert not (P.region("a_text", m) & m["sector_clean"].astype(bool)).any()


@pytest.mark.parametrize("kind", P.KINDS)
def test_deterministic(kind: str) -> None:
    gray, meta = toy()
    assert np.array_equal(P.Perturbation(kind)(gray, meta), P.Perturbation(kind)(gray, meta))


def test_fills() -> None:
    gray, meta = toy()
    reg = P.region("b_zero", meta["masks"])
    assert (P.Perturbation("b_zero")(gray, meta)[reg] == 0).all()
    assert (P.Perturbation("b_mean")(gray, meta)[reg] == np.uint8(np.median(gray[reg]))).all()


def test_unknown_kind_and_missing_masks() -> None:
    with pytest.raises(ValueError):
        P.Perturbation("a_marker")
    gray, _ = toy()
    with pytest.raises(ValueError, match="masks"):
        P.Perturbation("b_zero")(gray, {"image_id": "x", "masks": None})


def test_ui_transplant(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gray, meta = toy(1)
    dgray, dmeta = toy(2)
    dpath = tmp_path / "donor.png"
    cv2.imwrite(str(dpath), dgray)
    monkeypatch.setattr(P, "mask_path", lambda iid, ds: tmp_path / "donor.npz")
    save_masks(tmp_path / "donor.npz", dmeta["masks"])
    tp = P.UITransplant({"toy": (str(dpath), "donor", "fpdb")})
    out = tp(gray, meta)
    reg = P.region("e_ui_transplant", meta["masks"])
    assert np.array_equal(out[~reg], gray[~reg])
    donor_frame = np.where(dmeta["masks"]["sector_clean"].astype(bool), 0, dgray)
    assert np.array_equal(out[reg], donor_frame[reg])        # same size -> resize is the identity
    assert np.array_equal(tp(gray, {**meta, "image_id": "other"}), gray)


def test_pick_donors_crosses_machines() -> None:
    rec = pd.DataFrame({"image_id": ["a", "b", "c"], "image_path": ["x"] * 3,
                        "source_machine": ["Aloka", "Voluson E6", "Voluson S10"]})
    pool = pd.DataFrame({"image_id": ["d1", "d2"], "image_path": ["p1", "p2"], "dataset": "fpdb",
                         "source_machine": ["Voluson E6", "Aloka"]})
    d = P.pick_donors(rec, pool)
    assert d == {"a": ("p1", "d1", "fpdb"), "b": ("p2", "d2", "fpdb")}
    assert P.pick_donors(rec, pool, seed=0) == d
