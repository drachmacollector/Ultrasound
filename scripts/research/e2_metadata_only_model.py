"""
scripts/research/e2_metadata_only_model.py  (R3 E2: how much of the label is predictable without pixels? RQ1)

HistGradientBoosting on label-free acquisition metadata (src/audit/meta_features.py), fit on
train+val, evaluated on the FPDB test split. Hyperparameters are fixed in advance (plan R3 E2);
nothing is tuned. A GroupKFold(5) on train+val is reported as a sanity check only.

Tasks: (i) Other vs standard plane (binary model) -> AUROC; (ii) 8-class -> balanced accuracy;
(iii) collapsed 6-class balanced accuracy (collapse of the 8-class argmax).
Subsets: all_test; seen_machines (no Voluson S10); S10_only (where the device prior is wrong).

Outputs: results/json/E2.json, results/tables/E2.md, results/figures/E2_metadata_roc.{pdf,png},
         results/preds/e2_metadata/fpdb_test__<feature set>.parquet (per-image predictions)
Usage:   python scripts/research/e2_metadata_only_model.py [--figure]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_curve
from sklearn.model_selection import GroupKFold

sys.path.insert(0, ".")
from src.audit.meta_features import FEATURE_SETS, build_features, category_levels  # noqa: E402
from src.audit.plotting import NEUTRAL, SERIES, plt, save  # noqa: E402
from src.data.dataset import CANONICAL_CLASSES  # noqa: E402
from src.data.labels import COLLAPSED, FINE_TO_COLLAPSED  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.eval.scoring import estimate_json  # noqa: E402
from src.eval.stats import CMMetric, cluster_bootstrap, paired_cluster_bootstrap  # noqa: E402

META = Path("data/processed/metadata/acquisition_metadata.parquet")
B0_TEST = Path("results/preds/legacy_convnext_s42/fpdb_test.parquet")
OUT_JSON = Path("results/json/E2.json")
OUT_MD = Path("results/tables/E2.md")
PREDS_DIR = Path("results/preds/e2_metadata")
HGB = {"max_iter": 300, "learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0, "random_state": 0}
SUBSETS = ("all_test", "seen_machines", "S10_only")


def model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(categorical_features="from_dtype", **HGB)


def subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    if name == "all_test":
        return df
    s10 = df["source_machine"] == "Voluson S10"
    return df[~s10] if name == "seen_machines" else df[s10]


def auroc_metric(df: pd.DataFrame) -> float:
    return M.auroc(df["score_standard"].to_numpy(), (df["plane_label"] != "Other").to_numpy())


BAL8 = CMMetric("bal_acc_8", "plane_label", "pred_fine", CANONICAL_CLASSES, M.balanced_accuracy)
BAL6 = CMMetric("bal_acc_collapsed", "plane_collapsed", "pred_full", COLLAPSED, M.balanced_accuracy)


def fit_predict(tv: pd.DataFrame, te: pd.DataFrame, fs: str) -> pd.DataFrame:
    lv = category_levels(tv)
    Xtv, Xte = build_features(tv, fs, lv), build_features(te, fs, lv)
    y8 = tv["plane_label"].map(CANONICAL_CLASSES.index).to_numpy()
    yb = (tv["plane_label"] != "Other").astype(int).to_numpy()
    m8 = model().fit(Xtv, y8)
    mb = model().fit(Xtv, yb)
    out = te[["image_id", "patient_id", "source_machine", "plane_label", "plane_collapsed"]].copy()
    out["pred_fine"] = np.array(CANONICAL_CLASSES)[m8.predict(Xte)]
    out["pred_full"] = out["pred_fine"].map(FINE_TO_COLLAPSED)
    out["score_standard"] = mb.predict_proba(Xte)[:, 1]
    return out


def groupkfold_sanity(tv: pd.DataFrame, fs: str) -> dict:
    """Out-of-fold AUROC and balanced accuracy on train+val (patient-grouped; report only)."""
    oof_b = np.zeros(len(tv))
    oof_8 = np.empty(len(tv), dtype=object)
    for tr, va in GroupKFold(5).split(tv, groups=tv["patient_id"]):
        p = fit_predict(tv.iloc[tr], tv.iloc[va], fs)
        oof_b[va] = p["score_standard"].to_numpy()
        oof_8[va] = p["pred_fine"].to_numpy()
    yb = (tv["plane_label"] != "Other").to_numpy()
    cm = M.confusion(tv["plane_label"].map(CANONICAL_CLASSES.index), pd.Series(oof_8).map(CANONICAL_CLASSES.index), 8)
    return {"auroc_standard_vs_other": M.auroc(oof_b, yb), "bal_acc_8": M.balanced_accuracy(cm)}


def b0_reference() -> pd.DataFrame | None:
    if not B0_TEST.exists():
        return None
    b = pd.read_parquet(B0_TEST, columns=["image_id", "patient_id", "source_machine", "true_label", "prob_7"])
    return b.assign(plane_label=b["true_label"].astype(str), score_standard=1 - b["prob_7"],
                    source_machine=b["source_machine"].astype(str))


def main() -> None:
    if "--figure" in sys.argv:          # redraw only, from saved predictions + JSON
        figure()
        return
    meta = pd.read_parquet(META)
    tv, te = meta[meta["split"] != "test"].reset_index(drop=True), meta[meta["split"] == "test"].reset_index(drop=True)
    res: dict = {"hyperparameters": HGB, "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()},
                 "n_trainval": len(tv), "n_test": len(te), "results": {}, "preds": {}}
    preds = {}
    for fs in FEATURE_SETS:
        p = fit_predict(tv, te, fs)
        preds[fs] = p
        PREDS_DIR.mkdir(parents=True, exist_ok=True)
        p.to_parquet(PREDS_DIR / f"fpdb_test__{fs}.parquet", index=False)
        r = {}
        for sub in SUBSETS:
            d = subset(p, sub)
            r[sub] = {"n_images": len(d), "n_patients": int(d["patient_id"].nunique()),
                      "auroc_standard_vs_other": cluster_bootstrap(d, auroc_metric, max_nan_frac=0.01),
                      "bal_acc_8": cluster_bootstrap(d, BAL8, max_nan_frac=0.01),
                      "bal_acc_collapsed": cluster_bootstrap(d, BAL6, max_nan_frac=0.01)}
        r["groupkfold_trainval_sanity"] = groupkfold_sanity(tv, fs)
        res["results"][fs] = r
        print(f"  {fs}: AUROC seen {r['seen_machines']['auroc_standard_vs_other'].fmt()}", flush=True)
    res["paired_auroc_vs_prior"] = {
        sub: {fs: paired_cluster_bootstrap(subset(preds["prior"], sub), subset(preds[fs], sub), auroc_metric)
              for fs in ("geometry", "overlay", "all")} for sub in ("seen_machines", "all_test")}
    b0 = b0_reference()
    if b0 is not None:
        res["b0_full_image_auroc"] = {sub: cluster_bootstrap(subset(b0, sub), auroc_metric) for sub in SUBSETS}
    OUT_JSON.write_text(json.dumps(estimate_json({k: v for k, v in res.items() if k != "preds"}), indent=1),
                        encoding="utf-8")
    OUT_MD.write_text(markdown(res), encoding="utf-8")
    figure()
    print(f"wrote {OUT_JSON}, {OUT_MD}")


def markdown(res: dict) -> str:
    s = ["# E2: no-pixel metadata model", "",
         "HistGradientBoosting (fixed hyperparameters: " + ", ".join(f"{k}={v}" for k, v in HGB.items()) + "), fit on "
         f"train+val ({res['n_trainval']} images), evaluated on test ({res['n_test']}). 95% patient-clustered CIs.", ""]
    for sub in SUBSETS:
        n = res["results"]["prior"][sub]["n_images"]
        s += [f"## {sub} (n = {n})", "",
              "| features | AUROC standard vs Other | balanced acc 8-way (chance 0.125) | balanced acc collapsed (chance 0.167) |",
              "|---|---|---|---|"]
        for fs, r in res["results"].items():
            x = r[sub]
            s.append(f"| `{fs}` | {x['auroc_standard_vs_other'].fmt()} | {x['bal_acc_8'].fmt()} | "
                     f"{x['bal_acc_collapsed'].fmt()} |")
        if "b0_full_image_auroc" in res:
            s.append(f"| *full image, B0 convnext (reference)* | {res['b0_full_image_auroc'][sub].fmt()} | – | – |")
        s.append("")
    s += ["Train+val GroupKFold(5) sanity check (out-of-fold, report only): " + "; ".join(
        f"`{fs}` AUROC {r['groupkfold_trainval_sanity']['auroc_standard_vs_other']:.3f}, "
        f"bal acc {r['groupkfold_trainval_sanity']['bal_acc_8']:.3f}" for fs, r in res["results"].items()), "",
          "## Paired AUROC differences vs the `prior` model (same test images, paired cluster bootstrap)", "",
          "| subset | geometry − prior | overlay − prior | all − prior |", "|---|---|---|---|"]
    for sub, x in res["paired_auroc_vs_prior"].items():
        s.append(f"| {sub} | " + " | ".join(x[fs].fmt() for fs in ("geometry", "overlay", "all")) + " |")
    g = res["paired_auroc_vs_prior"]["seen_machines"]["geometry"]
    beyond = g.lo > 0
    s += ["", "*Footnote:* frame geometry is nearly collinear with the machine (E1: Cramér's V size × machine ≈ 0.8), "
          "so `geometry` was expected to match `prior`. " + (
              f"It exceeds it on seen machines (Δ AUROC {g.fmt()}): frame and sector geometry (zoom, depth, sector "
              "extent) carry label information beyond the device identity." if beyond else
              f"It does not exceed it on seen machines (Δ AUROC {g.fmt()}): the frame mainly carries the device prior.")
          + " On `S10_only` the device prior points the wrong way (Voluson in training ≈ standard plane).", ""]
    return "\n".join(s)


def _ci(e: dict) -> str:
    return f"{e['point']:.2f} [{e['lo']:.2f}, {e['hi']:.2f}]"


def figure() -> None:
    """E2_metadata_roc from the saved per-image predictions and results/json/E2.json."""
    res = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    preds = {fs: pd.read_parquet(PREDS_DIR / f"fpdb_test__{fs}.parquet") for fs in ("prior", "geometry", "all")}
    b0 = b0_reference()
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.6), sharey=True)
    for ax, sub in zip(axes, ("seen_machines", "all_test")):
        for fs, col in zip(("prior", "geometry", "all"), SERIES):
            d = subset(preds[fs], sub)
            fpr, tpr, _ = roc_curve(d["plane_label"] != "Other", d["score_standard"])
            ax.plot(fpr, tpr, color=col, lw=1.6, label=f"{fs}: {_ci(res['results'][fs][sub]['auroc_standard_vs_other'])}")
        if b0 is not None:
            d = subset(b0, sub)
            fpr, tpr, _ = roc_curve(d["plane_label"] != "Other", d["score_standard"])
            ax.plot(fpr, tpr, color=NEUTRAL, lw=1.2, ls="--",
                    label=f"full image (B0): {_ci(res['b0_full_image_auroc'][sub])}")
        ax.plot([0, 1], [0, 1], color="#cfcfcb", lw=0.8)
        ax.set_title({"seen_machines": "Seen machines (no S10)", "all_test": "All test"}[sub]
                     + f", n={res['results']['prior'][sub]['n_images']}", loc="left")
        ax.set_xlabel("false-positive rate (Other called standard)")
        ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), fontsize=6.5, title="AUROC [95% CI]",
                  title_fontsize=6.5, ncol=1, alignment="left")
        ax.set_aspect("equal")
    axes[0].set_ylabel("true-positive rate (standard planes)")
    fig.suptitle("Standard plane vs Other from metadata alone (no pixels)", x=0.02, y=1.02, ha="left", fontsize=9)
    save(fig, "E2_metadata_roc")


if __name__ == "__main__":
    main()
