"""scripts/research/r5_fm_linear_probe.py: input geometry, C selection, scoring (synthetic data; no model)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.labels import FINE_TO_COLLAPSED

spec = importlib.util.spec_from_file_location("r5fm", Path("scripts/research/r5_fm_linear_probe.py"))
fm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fm)  # type: ignore[union-attr]


# ----------------------------------------------------------------------------- input geometry
def test_fm_input_resize_squashes_to_square_rgb():
    g = np.full((300, 500), 77, np.uint8)
    out = fm.fm_input(g, "resize")
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8
    assert (out == 77).all()


def test_fm_input_letterbox_keeps_aspect_and_pads_with_zero():
    g = np.full((100, 200), 200, np.uint8)                   # 2:1 sector crop
    out = fm.fm_input(g, "sector_letterbox_inpaint")
    assert out.shape == (224, 224, 3)
    rows = np.flatnonzero(out[:, :, 0].max(axis=1) > 0)
    assert len(rows) == 112                                   # 224 / 2, centred
    assert rows[0] == 56 and rows[-1] == 167
    assert (out[:56] == 0).all() and (out[168:] == 0).all()
    assert (out[..., 0] == out[..., 1]).all() and (out[..., 1] == out[..., 2]).all()


# ----------------------------------------------------------------------------- C selection
class _Fake:
    """A 'pipeline' whose val predictions are perfect only for C == good."""

    def __init__(self, C, good):
        self.C, self.good = C, good

    def fit(self, X, y):
        return self

    def predict(self, X):
        y = X[:, 0].astype(int)
        return y if self.C == self.good else (y + 1) % 8


def test_select_and_fit_picks_best_val_macro_f1(monkeypatch):
    monkeypatch.setattr(fm, "probe_pipeline", lambda C: _Fake(C, good=1.0))
    y = np.arange(80) % 8
    X = y[:, None].astype(float)
    _, C, scores = fm.select_and_fit(X, y, X, y)
    assert C == 1.0
    assert scores[1.0] == pytest.approx(1.0) and scores[0.01] == pytest.approx(0.0)


def test_select_and_fit_ties_go_to_smaller_C(monkeypatch):
    monkeypatch.setattr(fm, "probe_pipeline", lambda C: _Fake(C, good=None))   # all equally bad
    y = np.arange(80) % 8
    X = y[:, None].astype(float)
    _, C, _ = fm.select_and_fit(X, y, X, y)
    assert C == 0.01


def test_select_and_fit_needs_every_class():
    y = np.arange(70) % 7
    X = np.random.default_rng(0).normal(size=(70, 3))
    with pytest.raises(ValueError, match="every class"):
        fm.select_and_fit(X, y, X, y)


def test_real_probe_separates_separable_classes():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(8), 30)
    X = np.eye(8)[y] * 5 + rng.normal(scale=0.3, size=(240, 8))
    model, C, scores = fm.select_and_fit(X, y, X, y)
    assert scores[C] > 0.95
    assert model.decision_function(X).shape == (240, 8)


# ----------------------------------------------------------------------------- scoring
FINE = {"Head": "Brain_Trans_thalamic", "Abdomen": "Fetal_abdomen", "Femur": "Fetal_femur", "Thorax": "Fetal_thorax"}
SETS = {"fpdb_test": list(FINE), "hc18": ["Head"], "ucl": ["Head", "Abdomen", "Femur"],
        "african": ["Head", "Abdomen", "Femur", "Thorax"]}


def _frame(ds: str, half_wrong: bool) -> pd.DataFrame:
    n = 120                                                    # divisible by 2 x (1, 3, 4) classes
    classes = SETS[ds]
    true = [classes[i % len(classes)] for i in range(n)]
    # in the 'half wrong' model every second image of each class is predicted Other
    pred = [t if (not half_wrong or (i // len(classes)) % 2 == 0) else "Other" for i, t in enumerate(true)]
    d = pd.DataFrame({"image_id": [f"{ds}{i}" for i in range(n)], "patient_id": [f"{ds}_p{i // 2}" for i in range(n)],
                      "true_collapsed": true, "pred_full": pred,
                      "true_label": [FINE[t] for t in true], "pred_fine": [FINE.get(p, "Other") for p in pred]})
    assert all(FINE_TO_COLLAPSED[f] == t for f, t in zip(d["true_label"], d["true_collapsed"]))
    return d


@pytest.fixture()
def preds_root(tmp_path: Path) -> Path:
    for mode, wrong in (("resize", False), ("sector_letterbox_inpaint", True)):
        d = tmp_path / fm.run_id(mode)
        d.mkdir()
        for ds in SETS:
            _frame(ds, wrong).to_parquet(d / f"{ds}.parquet", index=False)
        (d / "probe.meta.json").write_text(json.dumps({"C": 0.1}), encoding="utf-8")
    return tmp_path


def test_score_hand_computed(preds_root: Path):
    res = fm.score(preds_root, reference=None)
    r, lb = res["by_mode"]["resize"], res["by_mode"]["sector_letterbox_inpaint"]
    for k in ("hc18", "ucl", "african", "pooled"):
        assert r[k]["point"] == pytest.approx(1.0)
        assert lb[k]["point"] == pytest.approx(0.5)          # every class: recall 1/2
        assert res["paired_letterbox_minus_resize"][k]["delta"] == pytest.approx(-0.5)
    assert r["fpdb_test"]["point"] == pytest.approx(1.0)
    # letterbox fpdb_test: 4 classes with recall 1/2, precision 1; Other has 40 FPs, no TPs -> F1 0.
    # macro over the 5 classes present in true or pred = (4 * 2/3 + 0) / 5 = 0.5333
    assert lb["fpdb_test"]["point"] == pytest.approx(8 / 15)
    assert r["probe"]["C"] == 0.1
    md = fm.markdown(res)
    assert "Δ letterbox-inpaint − resize" in md and "reference" not in md
