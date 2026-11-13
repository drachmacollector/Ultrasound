"""
scripts/research/e8_error_concordance.py  (E8: do different architectures fail on the same images?)

POST-HOC, EXPLORATORY. Added on 2026-11-13, after Gate G4; it is not in the pre-registration and does
not touch R5. It uses only the R2 prediction parquets of the six frozen legacy checkpoints.

Question: on the external sets, are errors shared across the six architectures? If they pile up on the
same images, failures are a property of the images (how they were acquired), not of a particular model,
so changing the architecture will not fix them.

Per dataset (FPDB test as the in-distribution reference, HC18, UCL, African, pooled external):
  * Fleiss' kappa of the per-image "wrong" rating across the 6 models;
  * all-wrong ratio: observed share of images all 6 models get wrong / share expected if models erred
    independently within (dataset x true class) strata;
  * distribution of the number of models wrong (0-6), observed vs the stratified-independence expectation
    (Poisson-binomial per image);
  * images every model gets wrong: share rejected as 'Other' by most models, and per African centre.
Every estimate has a 95% patient-cluster bootstrap CI (B = 2000, seed 20261109). "Wrong" = full 8-way
argmax collapsed (``correct_full``), the R2 primary definition.

Outputs: results/json/E8.json, results/tables/E8.md, results/figures/E8_error_concordance.pdf
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.audit.plotting import NEUTRAL, SERIES, plt, save  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.eval.scoring import estimate_json  # noqa: E402
from src.eval.stats import cluster_bootstrap  # noqa: E402

PREDS = Path("results/preds")
LEGACY = {"convnext_tiny": "legacy_convnext_s42", "tf_efficientnetv2_s": "legacy_effv2s_s42",
          "repvgg_a1": "legacy_repvgga1_s42", "repvgg_a2": "legacy_repvgga2_s42",
          "mobilenetv3_large_100": "legacy_mnv3l_s42", "efficientnet_lite0": "legacy_efflite0_s42"}
SETS = ("fpdb_test", "hc18", "ucl", "african")
EXTERNAL = ("hc18", "ucl", "african")
OUT_JSON = Path("results/json/E8.json")
OUT_MD = Path("results/tables/E8.md")
W = [f"w_{k}" for k in LEGACY]
P = [f"p_{k}" for k in LEGACY]


def wide(ds: str) -> pd.DataFrame:
    """One row per image: wrong flag and prediction of each legacy model."""
    out = None
    for bb, rid in LEGACY.items():
        d = pd.read_parquet(PREDS / rid / f"{ds}.parquet",
                            columns=["image_id", "patient_id", "centre", "true_collapsed", "pred_full", "correct_full"])
        for c in ("centre", "true_collapsed", "pred_full"):
            d[c] = d[c].astype(str)
        d = d.rename(columns={"pred_full": f"p_{bb}"})
        d[f"w_{bb}"] = (~d.pop("correct_full").astype(bool)).astype(np.int8)
        out = d if out is None else out.merge(d[["image_id", f"p_{bb}", f"w_{bb}"]], on="image_id", validate="1:1")
    out["dataset"] = ds
    out["stratum"] = ds + "|" + out["true_collapsed"]
    out["n_wrong"] = out[W].sum(axis=1)
    return out.reset_index(drop=True)


def kappa(d: pd.DataFrame) -> float:
    return M.fleiss_kappa_binary(d["n_wrong"].to_numpy(), len(W))


def ratio(d: pd.DataFrame) -> float:
    return M.all_wrong_ratio(d[W].to_numpy(bool), d["stratum"].to_numpy())


def share_all_wrong(d: pd.DataFrame) -> float:
    return float((d["n_wrong"] == len(W)).mean())


def expected_hist(d: pd.DataFrame) -> np.ndarray:
    """Mean over images of the Poisson-binomial pmf of n_wrong under stratified independence."""
    rates = d.groupby("stratum")[W].mean()
    pmf_sum = np.zeros(len(W) + 1)
    for s, g in d.groupby("stratum"):
        pmf = np.array([1.0])
        for p in rates.loc[s].to_numpy():
            pmf = np.convolve(pmf, [1 - p, p])
        pmf_sum += pmf * len(g)
    return pmf_sum / len(d)


def majority_other(d: pd.DataFrame) -> pd.Series:
    return (d[P] == "Other").sum(axis=1) > len(P) / 2


def analyse() -> dict:
    frames = {ds: wide(ds) for ds in SETS}
    frames["pooled_external"] = pd.concat([frames[d] for d in EXTERNAL], ignore_index=True)
    res: dict = {"status": "post-hoc exploratory (not pre-registered)", "models": list(LEGACY),
                 "wrong": "correct_full == False (collapsed full 8-way argmax)", "by_set": {}}
    for name, d in frames.items():
        obs = np.bincount(d["n_wrong"], minlength=len(W) + 1) / len(d)
        allw = d[d["n_wrong"] == len(W)]
        r = {"n_images": len(d), "n_patients": int(d["patient_id"].nunique()),
             "fleiss_kappa": estimate_json(cluster_bootstrap(d, kappa)),
             "all_wrong_share": estimate_json(cluster_bootstrap(d, share_all_wrong)),
             "all_wrong_ratio_vs_independence": estimate_json(cluster_bootstrap(d, ratio, max_nan_frac=0.05)),
             "n_wrong_observed": obs.tolist(), "n_wrong_expected_independent": expected_hist(d).tolist(),
             "n_all_wrong": len(allw),
             "all_wrong_majority_other": float(majority_other(allw).mean()) if len(allw) else None}
        res["by_set"][name] = r
    afr = frames["african"]
    res["african_all_wrong_by_centre"] = {
        c: {"n": int(len(g)), "n_all_wrong": int((g["n_wrong"] == len(W)).sum()),
            "share_all_wrong": estimate_json(cluster_bootstrap(g, share_all_wrong))}
        for c, g in afr.groupby("centre")}
    return res


def fmt(e: dict, dp: int = 3) -> str:
    return f"{e['point']:.{dp}f} [{e['lo']:.{dp}f}, {e['hi']:.{dp}f}]"


def markdown(res: dict) -> str:
    L = ["# E8: error concordance across the six legacy architectures (post-hoc, exploratory)", "",
         "Fleiss' kappa of the per-image 'wrong' rating; all-wrong ratio = observed share of images all 6 models "
         "get wrong / share expected under independence within dataset x true-class strata. 95% patient-cluster "
         "bootstrap CIs.", "",
         "| set | images | Fleiss' kappa | all 6 wrong (share) | all-wrong ratio vs independence | all-wrong images mostly 'Other' |",
         "|---|---|---|---|---|---|"]
    for name, r in res["by_set"].items():
        mo = "–" if r["all_wrong_majority_other"] is None else f"{100 * r['all_wrong_majority_other']:.0f}% of {r['n_all_wrong']}"
        L.append(f"| {name} | {r['n_images']:,} | {fmt(r['fleiss_kappa'])} | {fmt(r['all_wrong_share'])} | "
                 f"{fmt(r['all_wrong_ratio_vs_independence'], 0)} | {mo} |")
    L += ["", "African set, images all 6 models get wrong, by centre:", "", "| centre | images | all wrong | share |",
          "|---|---|---|---|"]
    for c, r in res["african_all_wrong_by_centre"].items():
        L.append(f"| {c} | {r['n']} | {r['n_all_wrong']} | {fmt(r['share_all_wrong'])} |")
    L.append("")
    return "\n".join(L)


def figure(res: dict) -> None:
    sets = [("fpdb_test", "FPDB test"), ("hc18", "HC18"), ("ucl", "UCL"), ("african", "African")]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.3), sharey=True)
    x = np.arange(len(W) + 1)
    for ax, (k, title) in zip(axes, sets):
        r = res["by_set"][k]
        ax.bar(x - 0.2, r["n_wrong_observed"], width=0.4, color=SERIES[0], label="observed")
        ax.bar(x + 0.2, r["n_wrong_expected_independent"], width=0.4, color=NEUTRAL, hatch="///", edgecolor="white",
               label="if models erred independently")
        ax.set_yscale("log")
        ax.set_title(f"{title}  (κ = {r['fleiss_kappa']['point']:.2f})", loc="left")
        ax.set_xticks(x)
        ax.set_xlabel("models wrong (of 6)")
    axes[0].set_ylabel("share of images (log)")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, "E8_error_concordance", sources="results/preds/legacy_*_s42/{fpdb_test,hc18,ucl,african}.parquet",
         dpi=300)


def main() -> None:
    res = analyse()
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(res), encoding="utf-8")
    figure(res)
    print(markdown(res))


if __name__ == "__main__":
    main()
