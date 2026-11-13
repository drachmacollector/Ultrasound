"""scripts/research/r5_evaluate_matrix.py: temperature scaling (hand-computed) and score() on a synthetic run."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.dataset import CANONICAL_CLASSES
from src.data.labels import COLLAPSED, FINE_TO_COLLAPSED
from src.eval import metrics as M
from src.eval.predict import build_frame

sys.path.insert(0, "scripts/research")                      # the script imports r5_run_matrix from its own folder
spec = importlib.util.spec_from_file_location("r5e", Path("scripts/research/r5_evaluate_matrix.py"))
r5e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r5e)  # type: ignore[union-attr]


def test_temperature_hand_computed():
    # logit_0 = 1, others 0. With a share q of labels = class 0 (the rest any other class), the NLL optimum has
    # p0 = q, i.e. e^(1/T) / (e^(1/T) + 7) = q  ->  T = 1 / ln(7q / (1 - q)).  q = 0.75 -> T = 1 / ln 21.
    n = 400
    logits = np.zeros((n, 8))
    logits[:, 0] = 1.0
    y = np.where(np.arange(n) < 300, 0, 1 + np.arange(n) % 7)
    assert r5e.temperature(logits, y) == pytest.approx(1 / np.log(21), rel=1e-3)


def test_temperature_identity_when_calibrated():
    # q = e / (e + 7) is exactly what softmax gives at T = 1
    n = 2000
    q = np.e / (np.e + 7)
    k = round(q * n)
    logits = np.zeros((n, 8))
    logits[:, 0] = 1.0
    y = np.where(np.arange(n) < k, 0, 1 + np.arange(n) % 7)
    assert r5e.temperature(logits, y) == pytest.approx(1.0, abs=0.01)


def _set(name: str, classes: list[str], n: int, rng, fpdb: bool):
    true = [classes[i % len(classes)] for i in range(n)]
    frame = pd.DataFrame({
        "image_path": "x", "image_id": [f"{name}{i}" for i in range(n)],
        "patient_id": [f"{name}_p{i // 2}" for i in range(n)], "dataset": name, "source_machine": "m",
        "operator": "o", "centre": "c", "true_label": true, "true_collapsed": [FINE_TO_COLLAPSED[t] for t in true]})
    logits = rng.normal(size=(n, 8))
    idx = np.array([CANONICAL_CLASSES.index(t) for t in true])
    logits[np.arange(n), idx] += 2.5                          # mostly right, over-confident when scaled up
    logits *= 2.0
    present = sorted({FINE_TO_COLLAPSED[t] for t in true}, key=COLLAPSED.index)
    return build_frame(SimpleNamespace(frame=frame, present=present, is_fpdb=fpdb), logits)


def test_score_on_synthetic_run(tmp_path: Path, monkeypatch):
    rid = "b0_convnext_s11"
    runs, preds = tmp_path / "runs", tmp_path / "preds"
    (runs / rid).mkdir(parents=True)
    (runs / rid / "DONE").write_text("", encoding="utf-8")
    (preds / rid).mkdir(parents=True)
    rng = np.random.default_rng(0)
    head = ["Brain_Trans_thalamic", "Brain_Trans_cerebellum"]
    sets = {"fpdb_val": (list(CANONICAL_CLASSES), True), "fpdb_test": (list(CANONICAL_CLASSES), True),
            "hc18": (head, False), "ucl": (head + ["Fetal_abdomen", "Fetal_femur"], False),
            "african": (head + ["Fetal_abdomen", "Fetal_femur", "Fetal_thorax"], False)}
    frames = {}
    for ds, (classes, fp) in sets.items():
        frames[ds] = _set(ds, classes, 160, rng, fp)
        frames[ds].to_parquet(preds / rid / f"{ds}.parquet", index=False)
    monkeypatch.setattr(r5e, "RUNS", runs)
    monkeypatch.setattr("src.eval.predict.PREDS_ROOT", preds)

    res = r5e.score()
    r = res["runs"][rid]
    assert 0.05 < r["temperature"] < 20
    for ds in ("fpdb_test", "hc18", "ucl", "african"):
        d = frames[ds]
        # raw ECE must equal the ECE of the stored summed-collapsed confidence
        assert r[ds]["ece_raw"] == pytest.approx(M.ece(d["conf_full_summed"].to_numpy(np.float64),
                                                        d["correct_full"].to_numpy(bool)), abs=1e-6)
        assert 0 <= r[ds]["ece_scaled"] <= 1
        assert 0 <= r[ds]["coverage_at_95"] <= 1
        assert len(r[ds]["risk_coverage"]["coverage"]) == len(r[ds]["risk_coverage"]["risk"]) > 0
    # temperature is fitted on val, so it should not make val-like test calibration worse by much
    assert r["fpdb_test"]["ece_scaled"] <= r["fpdb_test"]["ece_raw"] + 0.02
    assert "within_class_machine" not in r and "marker_flip_rate" not in r and r["epg_minus_area_true"] == {}
    assert res["by_arm"] == {}                                 # needs all 3 seeds
    assert f"`{rid}`" in r5e.markdown(res)


# ----------------------------------------------------------------------------- §6.3, §6.7, §6.9 analyses
PRESENT = {"hc18": ["Head"], "ucl": ["Head", "Abdomen", "Femur"], "african": ["Head", "Abdomen", "Femur", "Thorax"]}


def _ext(ds: str, n_other: int, n: int = 60) -> pd.DataFrame:
    """External frame: every image correct except the first ``n_other``, which are predicted Other."""
    cls = PRESENT[ds]
    true = [cls[i % len(cls)] for i in range(n)]
    pred = ["Other" if i < n_other else t for i, t in enumerate(true)]
    return pd.DataFrame({"image_id": [f"{ds}{i}" for i in range(n)], "patient_id": [f"{ds}_p{i // 2}" for i in range(n)],
                         "dataset": ds, "true_collapsed": true, "pred_full": pred, "pred_restricted": true,
                         "correct_full": [p == t for p, t in zip(pred, true)], "correct_restricted": True,
                         "true_label": "x", "pred_fine": "x"})


def test_errors_by_arm_hand_computed(tmp_path: Path, monkeypatch):
    preds = tmp_path / "preds"
    for k, s in enumerate((11, 22, 33), start=1):
        d = preds / f"b0_convnext_s{s}"
        d.mkdir(parents=True)
        for ds in PRESENT:
            _ext(ds, n_other=10 * k).to_parquet(d / f"{ds}.parquet", index=False)
    monkeypatch.setattr("src.eval.predict.PREDS_ROOT", preds)
    out = r5e.errors_by_arm()
    assert list(out) == ["b0|convnext_tiny"]                   # other arms have no files
    hc = out["b0|convnext_tiny"]["hc18"]
    assert hc["rejection_rate"]["point"] == pytest.approx((10 + 20 + 30) / 180)   # mean over seeds of k*10/60
    assert hc["err_other"]["point"] == pytest.approx(1.0)
    assert hc["err_confusion"]["point"] == pytest.approx(0.0)
    assert out["b0|convnext_tiny"]["pooled"]["rejection_rate"]["point"] == pytest.approx((30 + 60 + 90) / 540)


def _sal(ds: str, seed: int, rng, effect: float, n: int = 160) -> pd.DataFrame:
    cls = PRESENT[ds]
    epg = rng.uniform(-0.2, 0.6, n)
    conf = rng.uniform(0.5, 1.0, n)
    p = 1 / (1 + np.exp(-(effect * epg + 0.5)))
    correct = rng.random(n) < p
    return pd.DataFrame({
        "image_id": [f"{ds}{i}" for i in range(n)], "patient_id": [f"{ds}_p{i // 2}" for i in range(n)],
        "dataset": ds, "centre": ["A", "B"] * (n // 2), "true_collapsed": [cls[i % len(cls)] for i in range(n)],
        "correct_full": correct, "conf_full_summed": conf, "has_landmarks": ds != "african",
        "qc_flag": pd.array([False] * n, dtype="boolean"), "epg_minus_area_true": epg,
        "out_sector_frac_pred": np.clip(0.3 - 0.3 * correct + rng.normal(0, 0.05, n), 0, 1)})


def test_p3b_confirmatory_on_seed_pooled_synthetic(tmp_path: Path, monkeypatch):
    preds = tmp_path / "preds"
    rng = np.random.default_rng(1)
    for short in ("convnext", "effv2s"):
        for s in (11, 22, 33):
            d = preds / f"afp_{short}_s{s}"
            d.mkdir(parents=True)
            sets = ("hc18", "ucl", "african") if short == "convnext" else ("hc18", "ucl")
            for ds in sets:
                _sal(ds, s, rng, effect=12.0).to_parquet(d / f"{ds}__saliency-gradcam.parquet", index=False)
    monkeypatch.setattr("src.eval.predict.PREDS_ROOT", preds)
    import functools

    import r4_association as r4a                              # importable: scripts/research is on sys.path
    from src.eval import stats

    monkeypatch.setattr(r4a, "cluster_bootstrap", functools.partial(stats.cluster_bootstrap, B=200))
    monkeypatch.setattr(r4a, "paired_cluster_bootstrap", functools.partial(stats.paired_cluster_bootstrap, B=200))
    res = r5e.p3b_confirmatory()
    assert set(res["cells"]) == {"afp|convnext_tiny", "afp|tf_efficientnetv2_s"}
    g = res["cells"]["afp|convnext_tiny"]["landmark"]["glm"]
    assert g["n"] == 3 * 2 * 160                                # 3 seeds pooled, hc18 + ucl
    assert g["n_patients"] == 2 * 80                            # clusters are patients, not patient x seed
    assert g["or_lo"] > 1
    assert res["confirmed"] is True
    mon = res["cells"]["afp|convnext_tiny"]["african_monitor"]["discrimination"]
    assert mon["sign"] == -1.0 and mon["auroc_score"]["point"] > 0.9   # less out-of-sector energy -> correct
    assert "african_monitor" not in res["cells"]["afp|tf_efficientnetv2_s"]
    md = r5e.markdown({"runs": {}, "p3b_confirmatory": res})
    assert "Confirmatory claim" in md and "**True**" in md


def test_sensitivity_delta_sign(tmp_path: Path, monkeypatch):
    preds = tmp_path / "preds"
    for rid, wrong in (("afp_convnext_s11", 0), ("afp_cw_convnext_s11", 20)):
        d = preds / rid
        d.mkdir(parents=True)
        for ds in PRESENT:
            _ext(ds, n_other=wrong).to_parquet(d / f"{ds}.parquet", index=False)
        f = _ext("african", n_other=wrong).assign(dataset="fpdb", image_id=lambda x: "fp" + x["image_id"],
                                                    patient_id=lambda x: "fpdb_" + x["patient_id"])
        f["true_label"] = f["true_collapsed"].map({"Head": "Brain_Trans_thalamic", "Abdomen": "Fetal_abdomen",
                                                   "Femur": "Fetal_femur", "Thorax": "Fetal_thorax"})
        f["pred_fine"] = [t if p != "Other" else "Other" for t, p in zip(f["true_label"], f["pred_full"])]
        f.to_parquet(d / "fpdb_test.parquet", index=False)
    monkeypatch.setattr("src.eval.predict.PREDS_ROOT", preds)
    sv = r5e.sensitivity()
    assert sv["run"] == "afp_cw_convnext_s11" and sv["reference"] == "afp_convnext_s11"
    assert sv["pooled_bal_acc_full"]["delta"] < 0 and sv["fpdb_test_macro_f1_8"]["delta"] < 0


def test_run_matrix_has_the_sensitivity_run_last():
    spec_m = importlib.util.spec_from_file_location("r5m", Path("scripts/research/r5_run_matrix.py"))
    r5m = importlib.util.module_from_spec(spec_m)
    spec_m.loader.exec_module(r5m)  # type: ignore[union-attr]
    jobs = r5m.matrix()
    assert len(jobs) == 25 and jobs[-1] == ("afp_cw", "convnext_tiny", 11)
    assert r5m.run_id(*jobs[-1]) == "afp_cw_convnext_s11"
    assert Path(f"configs/research/{jobs[-1][0]}_{jobs[-1][1]}.yaml").exists()
