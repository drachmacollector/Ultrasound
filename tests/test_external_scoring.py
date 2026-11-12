"""src/eval/external.py + scoring.py on hand-built prediction frames."""
from __future__ import annotations

import pandas as pd
import pytest

from src.eval.external import rejection_vs_confusion_table, score_dataset, score_pooled
from src.eval.scoring import add_helper_columns


def frame(dataset: str, true: list[str], pred: list[str], prefix: str) -> pd.DataFrame:
    n = len(true)
    fine = {"Head": "Brain_Trans_thalamic", "Abdomen": "Fetal_abdomen", "Femur": "Fetal_femur",
            "Thorax": "Fetal_thorax", "Cervix": "Maternal_cervix", "Other": "Other"}
    return pd.DataFrame({
        "image_id": [f"{dataset}/{i}" for i in range(n)],
        "patient_id": [f"{prefix}_{i}" for i in range(n)],          # one image per patient
        "dataset": dataset, "true_collapsed": true, "true_label": true,
        "pred_full": pred, "pred_fine": [fine[p] for p in pred],
        "pred_restricted": [p if p in ("Head", "Abdomen", "Femur") else "Head" for p in pred],
        "correct_full": [t == p for t, p in zip(true, pred)],
        "correct_restricted": [t == (p if p in ("Head", "Abdomen", "Femur") else "Head") for t, p in zip(true, pred)],
    })


UCL = frame("ucl", ["Head", "Head", "Head", "Abdomen", "Abdomen", "Femur", "Femur", "Femur"],
            ["Head", "Other", "Thorax", "Abdomen", "Femur", "Femur", "Femur", "Cervix"], "ucl")


def test_helper_columns() -> None:
    d = add_helper_columns(UCL)
    assert d["rejected"].tolist() == [0, 1, 1, 0, 0, 0, 0, 1]
    assert d["err_other"].sum() == 1 and d["err_absent"].sum() == 2 and d["err_confusion"].sum() == 1
    assert (d["err_other"] + d["err_absent"] + d["err_confusion"] == d["err_full"]).all()


def test_score_dataset_hand() -> None:
    """acc 4/8; recalls Head 1/3, Abdomen 1/2, Femur 2/3 -> bal_acc_full = 1/2; rejection 3/8."""
    s = score_dataset(UCL)
    assert s["acc_full"].point == pytest.approx(0.5)
    assert s["bal_acc_full"].point == pytest.approx((1 / 3 + 1 / 2 + 2 / 3) / 3)
    assert s["rejection_rate"].point == pytest.approx(3 / 8)
    e = s["errors"]
    assert e["n_errors"] == 4
    shares = [e[k].point for k in ("err_other", "err_absent", "err_confusion")]
    assert shares == pytest.approx([0.25, 0.5, 0.25]) and sum(shares) == pytest.approx(1.0)
    # restricted: the 3 rejected rows map to Head -> Head row 3 and Head row 2 become correct
    assert s["recall_restricted"]["Head"].point == pytest.approx(1.0)
    assert s["present_classes"] == ["Head", "Abdomen", "Femur"]


def test_hc18_bal_acc_is_head_recall() -> None:
    hc = frame("hc18", ["Head"] * 4, ["Head", "Head", "Other", "Abdomen"], "hc18")
    s = score_dataset(hc)
    assert s["bal_acc_full"].point == s["acc_full"].point == 0.5
    assert s["rejection_rate"].point == 0.5           # Abdomen is absent from HC18 -> rejected too


def test_pooled_keeps_dataset_present_sets() -> None:
    hc = frame("hc18", ["Head"] * 2, ["Head", "Abdomen"], "hc18")
    s = score_pooled({"hc18": hc, "ucl": UCL})
    assert s["n_images"] == 10
    # Abdomen predicted on an HC18 image is "absent"; on UCL it would be "present"
    assert s["errors"]["n_err_absent"] == 3
    bad = hc.assign(patient_id="ucl_0")
    with pytest.raises(ValueError, match="spans datasets"):
        score_pooled({"hc18": bad, "ucl": UCL})


def test_rejection_table() -> None:
    t = rejection_vs_confusion_table({"m": {"ucl": score_dataset(UCL)}})
    assert t.loc[0, "rejection_rate"] == pytest.approx(3 / 8) and t.loc[0, "n_errors"] == 4
