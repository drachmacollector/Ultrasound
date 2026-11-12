"""
src/eval/scoring.py

Metric objects and the in-distribution (FPDB) scorer. Every estimate is a patient-clustered
bootstrap ``Estimate`` (``src.eval.stats``). External scoring lives in ``src.eval.external``.

Helper columns (``add_helper_columns``) turn row-level events into 0/1 columns so rates and
conditional shares use the fast ``RatioMetric`` path:
  rejected        pred_full not in the row's dataset present classes
  err_full        pred_full != true_collapsed
  err_other       error and pred_full == Other
  err_absent      error and pred_full is a non-Other class absent from the dataset
  err_confusion   error and pred_full is a present (wrong) class
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.dataset import CANONICAL_CLASSES
from src.data.labels import COLLAPSED
from src.eval import metrics as M
from src.eval.datasets import PRESENT_CLASSES
from src.eval.stats import (
    CMMetric,
    Estimate,
    PairedEstimate,
    RatioMetric,
    cluster_bootstrap,
    image_bootstrap,
)

#: present classes keyed by the parquet ``dataset`` column value.
PRESENT_BY_DATASET: dict[str, tuple[str, ...]] = {
    "fpdb": tuple(COLLAPSED),
    **{k: v for k, v in PRESENT_CLASSES.items() if not k.startswith("fpdb_")},
}


def add_helper_columns(df: pd.DataFrame,
                       present: Mapping[str, Sequence[str]] = PRESENT_BY_DATASET) -> pd.DataFrame:
    d = df.copy()
    ds = d["dataset"].astype(str)
    pred = d["pred_full"].astype(str)
    true = d["true_collapsed"].astype(str)
    sets = {name: frozenset(present[name]) for name in ds.unique()}
    in_present = pd.Series([p in sets[n] for p, n in zip(pred, ds)], index=d.index, dtype=bool)
    err = pred != true
    d["rejected"] = (~in_present).astype(np.int8)
    d["err_full"] = err.astype(np.int8)
    d["err_other"] = (err & (pred == "Other")).astype(np.int8)
    d["err_absent"] = (err & (pred != "Other") & ~in_present).astype(np.int8)
    d["err_confusion"] = (err & in_present).astype(np.int8)
    d["correct_full_i"] = d["correct_full"].astype(np.int8)
    d["correct_restricted_i"] = d["correct_restricted"].astype(np.int8)
    for c in COLLAPSED:
        is_c = true == c
        d[f"is_{c}"] = is_c.astype(np.int8)
        d[f"hit_full_{c}"] = (is_c & (pred == c)).astype(np.int8)
        d[f"hit_restricted_{c}"] = (is_c & (d["pred_restricted"].astype(str) == c)).astype(np.int8)
    return d


def _idx(classes: Sequence[str], universe: Sequence[str]) -> list[int]:
    return [list(universe).index(c) for c in classes]


# ----------------------------------------------------------------------------- metric factories
def macro_f1_8(labels: Sequence[str] = CANONICAL_CLASSES) -> CMMetric:
    li = _idx(labels, CANONICAL_CLASSES)
    return CMMetric("macro_f1_8", "true_label", "pred_fine", CANONICAL_CLASSES, lambda cm: M.macro_f1(cm, li))


def bal_acc_8(labels: Sequence[str] = CANONICAL_CLASSES) -> CMMetric:
    li = _idx(labels, CANONICAL_CLASSES)
    return CMMetric("bal_acc_8", "true_label", "pred_fine", CANONICAL_CLASSES,
                    lambda cm: M.balanced_accuracy(cm, li))


def acc_8() -> CMMetric:
    return CMMetric("acc_8", "true_label", "pred_fine", CANONICAL_CLASSES, M.accuracy)


def bal_acc_collapsed(pred_col: str, present: Sequence[str]) -> CMMetric:
    li = _idx(present, COLLAPSED)
    return CMMetric(f"bal_acc_{pred_col}", "true_collapsed", pred_col, COLLAPSED,
                    lambda cm: M.balanced_accuracy(cm, li))


def ratio(name: str, num: str, den: str | None = None) -> RatioMetric:
    return RatioMetric(name, num, den)


def other_recall_metric() -> RatioMetric:
    """Recall of the fine 'Other' class (FPDB); needs the ``with_other8`` columns."""
    return RatioMetric("other_recall", "_hit_other8", "_is_other8")


def with_other8(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 0/1 columns ``other_recall_metric`` needs (true Other; true and predicted Other)."""
    is_o = df["true_label"].astype(str) == "Other"
    return df.assign(_is_other8=is_o.astype(np.int8),
                     _hit_other8=(is_o & (df["pred_fine"].astype(str) == "Other")).astype(np.int8))


# ----------------------------------------------------------------------------- in-distribution
def score_indist(df: pd.DataFrame, image_ci: bool = False, strict_labels: bool = True,
                 max_nan_frac: float = 0.01) -> dict:
    """FPDB in-distribution metrics with cluster CIs.

    ``strict_labels``: macro-F1 / balanced accuracy over all 8 classes (whole test set). With
    False they average over classes present in ``df`` (per-machine strata, where some classes
    may be missing). ``max_nan_frac`` applies to Other recall (undefined in resamples of small
    strata without Other images).
    """
    d = add_helper_columns(with_other8(df))
    labels = (CANONICAL_CLASSES if strict_labels
              else [c for c in CANONICAL_CLASSES if c in set(d["true_label"].astype(str))])
    out: dict[str, Estimate | dict] = {
        "macro_f1_8": cluster_bootstrap(d, macro_f1_8(labels)),
        "bal_acc_8": cluster_bootstrap(d, bal_acc_8(labels)),
        "acc_8": cluster_bootstrap(d, acc_8()),
        "acc_full": cluster_bootstrap(d, ratio("acc_full", "correct_full_i")),
    }
    if d["_is_other8"].sum() > 0:
        out["other_recall"] = cluster_bootstrap(d, other_recall_metric(), max_nan_frac=max_nan_frac)
    if image_ci:
        out["macro_f1_8_image_ci"] = image_bootstrap(d, macro_f1_8(labels))
    cm = macro_f1_8(labels).cm(d)
    out["per_class"] = {
        c: {"recall": float(r), "precision": float(p), "f1": float(f), "support": int(s)}
        for c, r, p, f, s in zip(CANONICAL_CLASSES, M.per_class_recall(cm), M.per_class_precision(cm),
                                 M.per_class_f1(cm), cm.sum(axis=1))
    }
    out["n_images"] = len(d)
    out["n_patients"] = int(d["patient_id"].nunique())
    out["labels_averaged"] = list(labels)
    return out


def by_stratum(df: pd.DataFrame, col: str, scorer, min_n: int = 1, **kw) -> dict[str, dict]:
    """Apply ``scorer(sub_df, **kw)`` to each level of ``col`` (levels with < min_n rows skipped)."""
    out = {}
    for level, g in df.groupby(col, observed=True, sort=True):
        if len(g) >= min_n:
            out[str(level)] = scorer(g, **kw)
    return out


def estimate_json(obj):
    """Recursively convert (Paired)Estimates and numpy scalars for json.dump."""
    if isinstance(obj, Estimate | PairedEstimate):
        return obj.to_dict()
    if isinstance(obj, dict):
        return {str(k): estimate_json(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [estimate_json(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj
