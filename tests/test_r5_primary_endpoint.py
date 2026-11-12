"""scripts/research/r5_primary_endpoint.py on synthetic predictions: Holm, endpoint, non-inferiority."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.labels import FINE_TO_COLLAPSED

spec = importlib.util.spec_from_file_location("r5p", Path("scripts/research/r5_primary_endpoint.py"))
r5p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r5p)  # type: ignore[union-attr]

EXT = {"hc18": ["Head"], "ucl": ["Head", "Abdomen", "Femur"], "african": ["Head", "Abdomen", "Femur", "Thorax"]}
FINE = {"Head": "Brain_Trans_thalamic", "Abdomen": "Fetal_abdomen", "Femur": "Fetal_femur", "Thorax": "Fetal_thorax"}


def frame(ds: str, acc: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    classes = EXT.get(ds, list(FINE))
    n = 240
    true = [classes[i % len(classes)] for i in range(n)]
    pred = [t if rng.random() < acc else "Other" for t in true]
    d = pd.DataFrame({"image_id": [f"{ds}{i}" for i in range(n)], "patient_id": [f"{ds}_p{i // 3}" for i in range(n)],
                      "dataset": "fpdb" if ds == "fpdb_test" else ds, "true_collapsed": true, "pred_full": pred,
                      "centre": ["A", "B"] * (n // 2)})
    d["correct_full_i"] = (d["true_collapsed"] == d["pred_full"]).astype(np.int8)
    d["true_label"] = [FINE[t] for t in true]
    d["pred_fine"] = [FINE.get(p, "Other") for p in pred]
    assert all(FINE_TO_COLLAPSED[f] == t for f, t in zip(d["true_label"], true))
    return d


def getter(acc: dict[str, float]):
    def get(arm: str, bb: str, kind: str) -> list[pd.DataFrame]:
        sets = list(EXT) if kind == "pooled" else [kind]
        return [pd.concat([frame(ds, acc[arm], seed=s + sum(map(ord, ds))) for ds in sets], ignore_index=True)
                for s in (11, 22, 33)]
    return get


def test_holm_adjustment() -> None:
    assert r5p.holm({"a": 0.01, "b": 0.04}) == {"a": 0.02, "b": 0.04}
    assert r5p.holm({"a": 0.03, "b": 0.02}) == {"b": 0.04, "a": 0.04}   # monotone step-down
    assert r5p.holm({"a": 0.8, "b": 0.7})["a"] == 1.0


def test_clear_improvement_is_a_success_and_non_inferior() -> None:
    res = r5p.analyse(getter({"b0": 0.70, "b1": 0.72, "b2": 0.75, "afp": 0.90}))
    for bb, r in res["per_backbone"].items():
        assert r["afp_vs_b0"]["delta"] == pytest.approx(0.2, abs=0.06)
        assert r["success"] and r["holm_p"] < 0.05
        assert r["non_inferiority_macro_f1_8"]["non_inferior"]
        assert set(r["per_dataset"]) == set(EXT)


def test_no_difference_is_not_a_success() -> None:
    res = r5p.analyse(getter({"b0": 0.8, "b1": 0.8, "b2": 0.8, "afp": 0.8}))
    for r in res["per_backbone"].values():
        assert r["afp_vs_b0"]["delta"] == 0.0 and not r["success"]
