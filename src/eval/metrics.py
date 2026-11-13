"""
src/eval/metrics.py

Pure metric functions on arrays or confusion matrices. Every number in the study is
computed by one of these, wrapped by the bootstrap in ``src.eval.stats``.

Conventions
* A confusion matrix ``cm`` is (c, c) with rows = true class, columns = predicted class.
* ``labels`` restricts class-averaged metrics to a set of class indices (e.g. the
  classes present in an external dataset). Predictions outside ``labels`` still count
  as false negatives of the true class.
* Classes undefined in a (resampled) dataset are excluded from averages, never imputed:
  - balanced accuracy averages recall over classes in ``labels`` with >= 1 true sample;
  - macro-F1 averages F1 over classes in ``labels`` with TP+FP+FN > 0 (a class that is
    neither present nor predicted has no defined F1). On the full FPDB test set every class
    is present, so this equals ``sklearn.metrics.f1_score(average="macro")``.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

__all__ = [
    "confusion",
    "accuracy",
    "per_class_recall",
    "per_class_precision",
    "per_class_f1",
    "balanced_accuracy",
    "macro_f1",
    "rejection_rate",
    "error_decomposition",
    "ece",
    "auroc",
    "risk_coverage",
    "coverage_at_accuracy",
    "cramers_v",
    "fleiss_kappa_binary",
    "all_wrong_ratio",
]


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes) int64 confusion matrix from integer class indices."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _labels(cm: np.ndarray, labels: Sequence[int] | None) -> np.ndarray:
    return np.arange(cm.shape[0]) if labels is None else np.asarray(labels, dtype=np.int64)


def accuracy(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total) if total else float("nan")


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    """Recall per class; NaN where the class has no true samples."""
    support = cm.sum(axis=1).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(support > 0, np.diag(cm) / support, np.nan)


def per_class_precision(cm: np.ndarray) -> np.ndarray:
    """Precision per class; NaN where the class is never predicted."""
    pred = cm.sum(axis=0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(pred > 0, np.diag(cm) / pred, np.nan)


def per_class_f1(cm: np.ndarray) -> np.ndarray:
    """F1 = 2TP / (2TP + FP + FN); NaN where TP+FP+FN = 0."""
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * tp / den, np.nan)


def balanced_accuracy(cm: np.ndarray, labels: Sequence[int] | None = None) -> float:
    """Mean recall over classes in ``labels`` that have >= 1 true sample."""
    rec = per_class_recall(cm)[_labels(cm, labels)]
    rec = rec[~np.isnan(rec)]
    return float(rec.mean()) if rec.size else float("nan")


def macro_f1(cm: np.ndarray, labels: Sequence[int] | None = None) -> float:
    """Mean F1 over classes in ``labels`` with a defined F1 (see module docstring)."""
    f1 = per_class_f1(cm)[_labels(cm, labels)]
    f1 = f1[~np.isnan(f1)]
    return float(f1.mean()) if f1.size else float("nan")


def rejection_rate(pred: np.ndarray, present: Sequence) -> float:
    """Share of predictions outside the dataset's present classes (e.g. 'Other', 'Cervix')."""
    pred = np.asarray(pred)
    if pred.size == 0:
        return float("nan")
    return float((~np.isin(pred, np.asarray(list(present)))).mean())


def error_decomposition(true: np.ndarray, pred: np.ndarray, present: Sequence,
                        other_label: str = "Other") -> dict[str, float]:
    """Among errors: shares predicted Other / a non-Other class absent from the set / a present class.

    Shares sum to 1 (NaN if there are no errors). ``n_errors`` is returned as well.
    """
    true = np.asarray(true)
    pred = np.asarray(pred)
    present_arr = np.asarray(list(present))
    err = true != pred
    n = int(err.sum())
    p = pred[err]
    rej = p == other_label
    absent = (~rej) & ~np.isin(p, present_arr)
    conf = (~rej) & np.isin(p, present_arr)
    if n == 0:
        return {"n_errors": 0, "rejection_other": float("nan"), "absent_class": float("nan"),
                "confusion": float("nan")}
    return {"n_errors": n, "rejection_other": float(rej.mean()), "absent_class": float(absent.mean()),
            "confusion": float(conf.mean())}


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins on [0, 1].

    Bin b holds confidences in (b/n, (b+1)/n]; confidence 0 goes to the first bin.
    """
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = conf.size
    if n == 0:
        return float("nan")
    bins = np.clip(np.ceil(conf * n_bins).astype(np.int64) - 1, 0, n_bins - 1)
    cnt = np.bincount(bins, minlength=n_bins).astype(np.float64)
    s_conf = np.bincount(bins, weights=conf, minlength=n_bins)
    s_acc = np.bincount(bins, weights=correct, minlength=n_bins)
    nz = cnt > 0
    return float(np.sum(np.abs(s_acc[nz] - s_conf[nz])) / n)


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """ROC AUC of ``score`` for binary ``y``; NaN if only one class is present."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, dtype=np.float64)))


def risk_coverage(conf: np.ndarray, correct: np.ndarray,
                  grid: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Selective risk (1 - accuracy) when keeping the most-confident ``coverage`` share.

    Returns (coverage_grid, risk). Coverage c keeps the top ceil(c * N) images by confidence
    (stable sort, so ties keep input order).
    """
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 2) if grid is None else np.asarray(grid)
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = conf.size
    order = np.argsort(-conf, kind="stable")
    cum = np.cumsum(correct[order])
    k = np.clip(np.ceil(grid * n).astype(np.int64), 1, n)
    return grid, 1.0 - cum[k - 1] / k


def coverage_at_accuracy(conf: np.ndarray, correct: np.ndarray, target: float = 0.95) -> float:
    """Largest coverage share whose top-confidence subset has accuracy >= ``target`` (0 if none)."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    n = conf.size
    if n == 0:
        return float("nan")
    order = np.argsort(-conf, kind="stable")
    acc = np.cumsum(correct[order]) / np.arange(1, n + 1)
    ok = np.nonzero(acc >= target)[0]
    return float((ok.max() + 1) / n) if ok.size else 0.0


def cramers_v(table: np.ndarray) -> float:
    """Bias-corrected Cramér's V (Bergsma 2013) of an r x k contingency table."""
    from scipy.stats import chi2_contingency

    t = np.asarray(table, dtype=np.float64)
    t = t[t.sum(axis=1) > 0][:, t.sum(axis=0) > 0]
    n = t.sum()
    r, k = t.shape
    if n == 0 or r < 2 or k < 2:
        return float("nan")
    chi2 = chi2_contingency(t, correction=False)[0]
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    den = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / den)) if den > 0 else float("nan")


def fleiss_kappa_binary(n_pos: np.ndarray, m: int) -> float:
    """Fleiss' kappa for m raters giving a binary rating (e.g. 'wrong') to each item.

    ``n_pos[i]`` = number of the m raters that rated item i positive. 1 = perfect agreement,
    0 = chance agreement given the overall positive rate, < 0 = less than chance.
    """
    n = np.asarray(n_pos, dtype=np.float64)
    if n.size == 0 or m < 2:
        return float("nan")
    if ((n < 0) | (n > m)).any():
        raise ValueError("n_pos must lie in [0, m]")
    p_i = (n * (n - 1) + (m - n) * (m - n - 1)) / (m * (m - 1))
    p = n.sum() / (n.size * m)
    pe = p * p + (1 - p) * (1 - p)
    return float((p_i.mean() - pe) / (1 - pe)) if pe < 1 else float("nan")


def all_wrong_ratio(wrong: np.ndarray, strata: np.ndarray) -> float:
    """Observed share of items that every rater gets wrong / share expected if raters erred independently.

    ``wrong`` is (N, m) boolean. The independence expectation is computed within each stratum
    (e.g. dataset x true class) as the product of the m raters' error rates in that stratum,
    averaged over items. > 1 means errors pile up on the same items.
    """
    w = np.asarray(wrong, dtype=bool)
    s = np.asarray(strata)
    obs = w.all(axis=1).mean()
    exp = 0.0
    for v in np.unique(s):
        k = s == v
        exp += k.mean() * np.prod(w[k].mean(axis=0))
    return float(obs / exp) if exp > 0 else float("nan")
