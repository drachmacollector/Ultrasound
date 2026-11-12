"""
src/eval/external.py

External-set scoring (HC18, UCL, African, pooled). All rates carry patient-clustered CIs;
pooled clusters stay dataset-qualified (``hc18_…``, ``ucl_…``, ``afr_…``).

Endpoints
  acc_full            collapse(argmax_8) == true                (legacy scoring; G2)
  bal_acc_full        mean recall over present classes, full 8-way argmax collapsed
  bal_acc_restricted  same, argmax restricted to the dataset's present classes
  rejection_rate      prediction outside the dataset's present classes (Other, Cervix, absent class)
  error decomposition among errors: predicted Other / absent non-Other class / present wrong class

Balanced accuracy on HC18 alone is just Head recall; the pooled external endpoint is primary.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from src.data.labels import COLLAPSED
from src.eval.scoring import PRESENT_BY_DATASET, add_helper_columns, bal_acc_collapsed, ratio
from src.eval.scoring import by_stratum as _by_stratum
from src.eval.stats import Estimate, cluster_bootstrap

ERROR_SHARES = ("err_other", "err_absent", "err_confusion")


def present_of(df: pd.DataFrame, present: Mapping[str, Sequence[str]] = PRESENT_BY_DATASET) -> tuple[str, ...]:
    """Union of the present classes of the datasets in ``df`` (COLLAPSED order)."""
    s = set()
    for name in df["dataset"].astype(str).unique():
        s |= set(present[name])
    return tuple(c for c in COLLAPSED if c in s)


def score_dataset(df: pd.DataFrame, present: Mapping[str, Sequence[str]] = PRESENT_BY_DATASET,
                  max_nan_frac: float = 0.01) -> dict:
    """All external endpoints for one dataset (or a pooled / stratum frame)."""
    d = add_helper_columns(df, present)
    pres = present_of(d, present)
    observed = [c for c in pres if d[f"is_{c}"].sum() > 0]
    kw = {"max_nan_frac": max_nan_frac}
    out: dict = {
        "acc_full": cluster_bootstrap(d, ratio("acc_full", "correct_full_i"), **kw),
        "acc_restricted": cluster_bootstrap(d, ratio("acc_restricted", "correct_restricted_i"), **kw),
        "bal_acc_full": cluster_bootstrap(d, bal_acc_collapsed("pred_full", pres), **kw),
        "bal_acc_restricted": cluster_bootstrap(d, bal_acc_collapsed("pred_restricted", pres), **kw),
        "rejection_rate": cluster_bootstrap(d, ratio("rejection_rate", "rejected"), **kw),
    }
    n_err = int(d["err_full"].sum())
    out["errors"] = {"n_errors": n_err}
    if n_err:
        for s in ERROR_SHARES:
            out["errors"][s] = cluster_bootstrap(d, ratio(s, s, "err_full"), max_nan_frac=1.0)
            out["errors"][f"n_{s}"] = int(d[s].sum())
    out["recall_full"] = {c: cluster_bootstrap(d, ratio(f"recall_full_{c}", f"hit_full_{c}", f"is_{c}"),
                                               max_nan_frac=1.0) for c in observed}
    out["recall_restricted"] = {c: cluster_bootstrap(d, ratio(f"recall_restricted_{c}", f"hit_restricted_{c}",
                                                              f"is_{c}"), max_nan_frac=1.0) for c in observed}
    out["pred_full_counts"] = {str(k): int(v) for k, v in d["pred_full"].astype(str).value_counts().items()}
    out["pred_fine_counts_on_errors"] = {
        str(k): int(v) for k, v in d.loc[d["err_full"] == 1, "pred_fine"].astype(str).value_counts().items()}
    out["present_classes"] = list(pres)
    out["missing_present_classes"] = [c for c in pres if c not in observed]
    out["n_images"] = len(d)
    out["n_patients"] = int(d["patient_id"].nunique())
    return out


def score_pooled(dfs: Mapping[str, pd.DataFrame], **kw) -> dict:
    """Score the concatenation of several external sets (clusters stay dataset-qualified)."""
    d = pd.concat([df.assign(dataset=df["dataset"].astype(str)) for df in dfs.values()], ignore_index=True)
    for c in ("dataset", "pred_full", "pred_fine", "pred_restricted", "true_collapsed", "true_label",
              "source_machine", "centre", "operator", "pred_full_summed"):
        if c in d:
            d[c] = d[c].astype(str)
    if d["image_id"].duplicated().any():
        raise ValueError("pooled frames share image_id values")
    if d.groupby("patient_id")["dataset"].nunique().max() > 1:
        raise ValueError("a patient_id spans datasets; ids must be dataset-qualified")
    out = score_dataset(d, **kw)
    out["datasets"] = list(dfs)
    return out


def by_stratum(df: pd.DataFrame, col: str, **kw) -> dict[str, dict]:
    """``score_dataset`` per level of ``col`` (device / centre / machine)."""
    return _by_stratum(df, col, score_dataset, **kw)


def rejection_vs_confusion_table(scores: Mapping[str, Mapping[str, dict]]) -> pd.DataFrame:
    """Long table: model x dataset -> accuracy, rejection rate and error shares (point estimates).

    ``scores[model][dataset]`` is a ``score_dataset`` result.
    """
    rows = []
    for model, per in scores.items():
        for ds, s in per.items():
            e = s["errors"]
            row = {"model": model, "dataset": ds, "n_images": s["n_images"],
                   "acc_full": _pt(s["acc_full"]), "rejection_rate": _pt(s["rejection_rate"]),
                   "n_errors": e["n_errors"]}
            for k in ERROR_SHARES:
                row[k] = _pt(e[k]) if k in e else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def _pt(e: Estimate | float) -> float:
    return e.point if isinstance(e, Estimate) else float(e)
