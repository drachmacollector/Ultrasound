"""src/eval/metrics.py: hand-computed answers on toy data."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.eval import metrics as M

# 10 rows, 3 classes.
Y_TRUE = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 2])
Y_PRED = np.array([0, 0, 1, 2, 1, 1, 0, 2, 2, 2])


def test_confusion_and_accuracy() -> None:
    """CM rows: [2,1,1], [1,2,0], [0,0,3] -> accuracy 7/10."""
    cm = M.confusion(Y_TRUE, Y_PRED, 3)
    assert cm.tolist() == [[2, 1, 1], [1, 2, 0], [0, 0, 3]]
    assert M.accuracy(cm) == pytest.approx(0.7)


def test_balanced_accuracy_hand() -> None:
    """Recalls 2/4, 2/3, 3/3 -> mean = (1/2 + 2/3 + 1) / 3 = 13/18."""
    cm = M.confusion(Y_TRUE, Y_PRED, 3)
    assert M.balanced_accuracy(cm) == pytest.approx(13 / 18)
    # restricted to classes {0, 2}: (1/2 + 1) / 2
    assert M.balanced_accuracy(cm, [0, 2]) == pytest.approx(0.75)


def test_balanced_accuracy_skips_absent_classes() -> None:
    """Class 1 has no true samples: average over classes 0 and 2 only."""
    cm = M.confusion(np.array([0, 0, 2]), np.array([0, 1, 2]), 3)
    assert M.balanced_accuracy(cm) == pytest.approx((0.5 + 1.0) / 2)


def test_macro_f1_hand() -> None:
    """F1 = 2TP/(2TP+FP+FN): c0 4/7, c1 4/6, c2 6/7 -> mean 44/63."""
    cm = M.confusion(Y_TRUE, Y_PRED, 3)
    assert M.per_class_f1(cm) == pytest.approx([4 / 7, 2 / 3, 6 / 7])
    assert M.macro_f1(cm) == pytest.approx(44 / 63)


def test_macro_f1_matches_sklearn_when_all_present() -> None:
    rng = np.random.default_rng(0)
    t = rng.integers(0, 8, 500)
    p = np.where(rng.random(500) < 0.7, t, rng.integers(0, 8, 500))
    cm = M.confusion(t, p, 8)
    assert M.macro_f1(cm) == pytest.approx(f1_score(t, p, average="macro"))
    assert M.balanced_accuracy(cm) == pytest.approx(balanced_accuracy_score(t, p))


def test_macro_f1_absent_but_predicted_counts_as_zero() -> None:
    """Class 1 absent from truth but predicted once: F1 = 0 and it stays in the average."""
    cm = M.confusion(np.array([0, 0]), np.array([0, 1]), 2)
    assert M.per_class_f1(cm)[1] == 0.0
    assert M.macro_f1(cm) == pytest.approx((2 / 3 + 0) / 2)


def test_precision_recall_nan_when_undefined() -> None:
    cm = M.confusion(np.array([0, 0]), np.array([0, 0]), 2)
    assert np.isnan(M.per_class_recall(cm)[1]) and np.isnan(M.per_class_precision(cm)[1])


def test_rejection_rate_hand() -> None:
    assert M.rejection_rate(np.array(["Head", "Other", "Cervix", "Head"]), ["Head"]) == 0.5


def test_error_decomposition_hand() -> None:
    """Errors: Head->Other (rejection), Head->Abdomen (confusion), Abdomen->Thorax (absent)."""
    true = np.array(["Head", "Head", "Abdomen", "Femur", "Head"])
    pred = np.array(["Other", "Abdomen", "Thorax", "Femur", "Head"])
    d = M.error_decomposition(true, pred, ["Head", "Abdomen", "Femur"])
    assert d["n_errors"] == 3
    assert d["rejection_other"] == d["absent_class"] == d["confusion"] == pytest.approx(1 / 3)
    assert d["rejection_other"] + d["absent_class"] + d["confusion"] == pytest.approx(1.0)


def test_ece_hand() -> None:
    """Bin conf 0.9: acc 0.8 -> 0.1 x 0.5; bin conf 0.6: acc 0.4 -> 0.2 x 0.5. ECE = 0.15."""
    conf = np.array([0.9] * 5 + [0.6] * 5)
    correct = np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 0])
    assert M.ece(conf, correct) == pytest.approx(0.15)
    assert M.ece(np.ones(4), np.ones(4)) == pytest.approx(0.0)


def test_auroc() -> None:
    assert M.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert np.isnan(M.auroc(np.array([0.1, 0.2]), np.array([1, 1])))


def test_risk_coverage_hand() -> None:
    """Sorted by confidence: correct = 1, 0, 1, 1. Coverage .25/.5/1 -> risk 0, .5, .25."""
    conf = np.array([0.6, 0.9, 0.7, 0.8])
    correct = np.array([1, 1, 1, 0])
    grid, risk = M.risk_coverage(conf, correct, grid=np.array([0.25, 0.5, 1.0]))
    assert risk == pytest.approx([0.0, 0.5, 0.25])
    assert M.coverage_at_accuracy(conf, correct, 0.95) == 0.25
    assert M.coverage_at_accuracy(conf, correct, 0.70) == 1.0


def test_cramers_v() -> None:
    assert M.cramers_v(np.array([[50, 0], [0, 50]])) == pytest.approx(1.0)
    assert M.cramers_v(np.array([[25, 25], [25, 25]])) == pytest.approx(0.0)
