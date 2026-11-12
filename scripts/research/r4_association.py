"""
scripts/research/r4_association.py  (R4.7, P3b: does on-anatomy saliency predict external correctness?)

Status: **exploratory** for the legacy checkpoints (confirmatory only for R5 models, per the
pre-registration). Primary model: convnext_tiny, Grad-CAM, true-class target, HC18 + UCL landmark
images (QC-flagged landmarks excluded).

1. GLM Binomial  correct_full ~ epg10 + conf_full_summed + C(dataset) + C(true_collapsed),
   patient-clustered covariance; epg10 = (EPG − area) / 0.1, so the OR is per +0.1 EPG − area.
2. Discrimination of correct vs incorrect: AUROC of EPG − area alone, of confidence alone, and of a
   logistic model on both (5-fold GroupKFold out-of-fold predictions, patient groups, no in-sample
   optimism); ΔAUROC (both − confidence) by paired patient-cluster bootstrap.
3. Predicted-class variant (``epg_pred``), the plan's "deployable" variant. It is **not** deployable and
   is mechanically favoured: the landmark mask marks the TRUE anatomy, so when the prediction is wrong
   the predicted-class map is scored against an anatomy it does not belong to and its EPG is low by
   construction. Reported for completeness, never as evidence.
   The only ground-truth-free monitor here is ``out_sector_frac_pred`` (predicted-class map, sector mask
   from the image), tested on the African set (no landmarks needed).
4. In-distribution analogue: FPDB-test landmark images (C(source_machine) instead of C(dataset)).
5. Every legacy backbone x passing method: the AUROC table (secondary).

Target-leakage caveat: ``epg_true`` uses the true label. Outputs: results/json/R4_association.json,
results/tables/R4_association.md, results/figures/R4_association.pdf.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")
from src.audit.plotting import INK, NEUTRAL, SERIES, plt, save  # noqa: E402
from src.eval.metrics import auroc  # noqa: E402
from src.eval.scoring import estimate_json  # noqa: E402
from src.eval.stats import RatioMetric, cluster_bootstrap, paired_cluster_bootstrap  # noqa: E402
from src.localisation import runner as R  # noqa: E402

OUT_JSON = Path("results/json/R4_association.json")
OUT_MD = Path("results/tables/R4_association.md")
G3_JSON = Path("results/json/G3.json")
PRIMARY = ("convnext_tiny", "gradcam")


def load(bb: str, sets: list[str], method: str, landmark: bool = True) -> pd.DataFrame | None:
    parts = []
    for ds in sets:
        p = R.saliency_path(R.legacy_run_id(bb), ds, method)
        if not p.exists():
            return None
        parts.append(pd.read_parquet(p))
    d = pd.concat(parts, ignore_index=True)
    for c in d.select_dtypes("category").columns:
        d[c] = d[c].astype(str)
    d["qc_flag"] = d["qc_flag"].fillna(False).astype(bool)
    d["correct"] = d["correct_full"].astype(bool).astype(int)
    if landmark:
        d = d[d["has_landmarks"].astype(bool) & ~d["qc_flag"] & d["epg_minus_area_true"].notna()]
    return d.reset_index(drop=True)


def glm(d: pd.DataFrame, score: str, strata: list[str], scale: float = 0.1) -> dict:
    x = d.assign(s10=d[score] / scale)
    terms = " + ".join(["s10", "conf_full_summed"] + [f"C({s})" for s in strata if x[s].nunique() > 1])
    groups = pd.factorize(x["patient_id"])[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.glm(f"correct ~ {terms}", data=x, family=sm.families.Binomial()).fit(
            cov_type="cluster", cov_kwds={"groups": groups})
    b, (lo, hi) = fit.params["s10"], fit.conf_int().loc["s10"]
    bc, (lc, hc) = fit.params["conf_full_summed"], fit.conf_int().loc["conf_full_summed"]
    return {"formula": f"correct ~ {terms}", "per": f"+{scale} {score}", "or": float(np.exp(b)),
            "or_lo": float(np.exp(lo)), "or_hi": float(np.exp(hi)), "p": float(fit.pvalues["s10"]),
            "conf_or_per_1": float(np.exp(bc)), "conf_or_lo": float(np.exp(lc)), "conf_or_hi": float(np.exp(hc)),
            "n": int(len(x)), "n_patients": int(x["patient_id"].nunique()), "n_incorrect": int((x["correct"] == 0).sum())}


def oof_both(d: pd.DataFrame, score: str) -> np.ndarray:
    X = d[[score, "conf_full_summed"]].to_numpy(dtype=np.float64)
    y = d["correct"].to_numpy()
    out = np.zeros(len(d))
    for tr, te in GroupKFold(5).split(X, y, d["patient_id"]):
        m = make_pipeline(StandardScaler(), LogisticRegression()).fit(X[tr], y[tr])
        out[te] = m.predict_proba(X[te])[:, 1]
    return out


class AUROC:
    def __init__(self, col: str):
        self.col, self.name = col, f"auroc_{col}"

    def __call__(self, df: pd.DataFrame) -> float:
        y = df["correct"].to_numpy()
        return auroc(df[self.col].to_numpy(dtype=np.float64), y) if 0 < y.sum() < len(y) else float("nan")


def discrimination(d: pd.DataFrame, score: str, sign: float = 1.0) -> dict:
    d = d.assign(s=sign * d[score], both=oof_both(d.assign(s=sign * d[score]), "s"), conf=d["conf_full_summed"])
    m = AUROC("v")
    e_s = cluster_bootstrap(d.assign(v=d["s"]), m)
    e_c = cluster_bootstrap(d.assign(v=d["conf"]), m)
    e_b = cluster_bootstrap(d.assign(v=d["both"]), m)
    delta = paired_cluster_bootstrap(d.assign(v=d["conf"]), d.assign(v=d["both"]), m)
    return {"score": score, "sign": sign, "auroc_score": estimate_json(e_s), "auroc_conf": estimate_json(e_c),
            "auroc_both_oof": estimate_json(e_b), "delta_both_minus_conf": estimate_json(delta)}


def deciles(d: pd.DataFrame, score: str) -> list[dict]:
    q = pd.qcut(d[score].rank(method="first"), 10, labels=False)
    out = []
    for k in range(10):
        x = d[q == k]
        e = cluster_bootstrap(x, RatioMetric("acc", "correct"))
        out.append({"decile": k + 1, "score_mean": float(x[score].mean()), **estimate_json(e)})
    return out


def passing_methods() -> list[str]:
    g3 = json.loads(G3_JSON.read_text(encoding="utf-8"))
    return [m for m, dd in g3["decision"].items() if dd["pass"]]


def analyse() -> dict:
    bb, method = PRIMARY
    ext = load(bb, ["hc18", "ucl"], method)
    res: dict = {"status": "exploratory (legacy checkpoints)", "primary_model": bb, "primary_method": method}
    res["primary"] = {"glm": glm(ext, "epg_minus_area_true", ["dataset", "true_collapsed"]),
                      "discrimination": discrimination(ext, "epg_minus_area_true"),
                      "accuracy_by_decile": deciles(ext, "epg_minus_area_true")}
    pe = ext[ext["epg_minus_area_pred"].notna()]
    res["deployable_pred_target"] = {"glm": glm(pe, "epg_minus_area_pred", ["dataset", "true_collapsed"]),
                                     "discrimination": discrimination(pe, "epg_minus_area_pred")}
    afr = load(bb, ["african"], method, landmark=False)
    afr = afr[afr["out_sector_frac_pred"].notna()]
    res["african_out_sector_pred"] = {"glm": glm(afr, "out_sector_frac_pred", ["centre", "true_collapsed"]),
                                      "discrimination": discrimination(afr, "out_sector_frac_pred", sign=-1.0),
                                      "note": "sign −1: less out-of-sector energy scored as more likely correct"}
    fp = load(bb, ["fpdb_test"], method)
    res["in_distribution_fpdb"] = {"glm": glm(fp, "epg_minus_area_true", ["source_machine", "true_collapsed"]),
                                   "discrimination": discrimination(fp, "epg_minus_area_true")}
    res["all_backbones"] = {}
    for m in passing_methods():
        for b in R.LEGACY:
            d = load(b, ["hc18", "ucl"], m)
            if d is None or (d["correct"] == 0).sum() < 10:
                continue
            res["all_backbones"][f"{b}|{m}"] = discrimination(d, "epg_minus_area_true")
            res["all_backbones"][f"{b}|{m}"]["glm"] = glm(d, "epg_minus_area_true", ["dataset", "true_collapsed"])
    return res


def figure(res: dict) -> None:
    bb, method = PRIMARY
    ext = load(bb, ["hc18", "ucl"], method)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))
    dec = res["primary"]["accuracy_by_decile"]
    ax = axes[0]
    x = [r["decile"] for r in dec]
    ax.errorbar(x, [r["point"] for r in dec], yerr=[[r["point"] - r["lo"] for r in dec], [r["hi"] - r["point"] for r in dec]],
                fmt="o-", ms=3, lw=1, capsize=2, color=SERIES[0])
    ax.set_xticks(x)
    ax.set_xlabel("decile of EPG − area (true target)")
    ax.set_ylabel("accuracy (HC18 + UCL)")
    ax.set_title("External accuracy by on-anatomy saliency", fontsize=8)
    ax = axes[1]
    both = oof_both(ext, "epg_minus_area_true")
    y = ext["correct"].to_numpy()
    for (lab, s), c in zip((("EPG − area", ext["epg_minus_area_true"]), ("confidence", ext["conf_full_summed"]),
                            ("both (out-of-fold)", both)), SERIES):
        s = np.asarray(s, dtype=np.float64)
        order = np.argsort(-s)
        tp = np.cumsum(y[order]) / y.sum()
        fpr = np.cumsum(1 - y[order]) / (1 - y).sum()
        ax.plot(np.r_[0, fpr], np.r_[0, tp], color=c, lw=1.2, label=f"{lab} {auroc(s, y):.2f}")
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=0.8, ls=":")
    ax.set_xlabel("false-positive rate (incorrect scored as correct)")
    ax.set_ylabel("true-positive rate")
    ax.set_title("Predicting correctness (AUROC)", fontsize=8)
    ax.legend(fontsize=6, loc="lower right")
    fig.text(0.5, -0.06, f"{bb} Grad-CAM; exploratory analysis of legacy checkpoints.", ha="center", fontsize=6, color=INK)
    fig.tight_layout()
    save(fig, "R4_association")


def fmt(e: dict) -> str:
    """Estimate (point) or PairedEstimate (delta, signed)."""
    if "delta" in e:
        return f"{e['delta']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"
    return f"{e['point']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"


def markdown(res: dict) -> str:
    def glm_line(g: dict) -> str:
        return (f"OR {g['or']:.2f} [{g['or_lo']:.2f}, {g['or_hi']:.2f}] per {g['per']} (p = {g['p']:.3g}); "
                f"n = {g['n']} images / {g['n_patients']} patients, {g['n_incorrect']} incorrect. `{g['formula']}`")

    def disc_line(dd: dict) -> str:
        return (f"AUROC {dd['score']}: {fmt(dd['auroc_score'])}; confidence: {fmt(dd['auroc_conf'])}; both (OOF): "
                f"{fmt(dd['auroc_both_oof'])}; Δ both − confidence {fmt(dd['delta_both_minus_conf'])} "
                f"(p = {dd['delta_both_minus_conf']['p_two_sided']:.3g})")

    L = ["# R4.7 (P3b): does on-anatomy saliency predict correctness? (exploratory, legacy checkpoints)", "",
         f"Primary: `{res['primary_model']}`, {res['primary_method']}, HC18 + UCL landmark images. 95% patient-cluster CIs.", "",
         "## Primary (true-class saliency; uses the true label, so not deployable)", "",
         "* " + glm_line(res["primary"]["glm"]), "* " + disc_line(res["primary"]["discrimination"]), "",
         "## Predicted-class saliency against the expert mask (NOT deployable; favoured by construction)", "",
         "The landmark mask marks the true anatomy: when the prediction is wrong, the predicted-class map is scored "
         "against an anatomy it does not belong to, so a low EPG for errors is built in. Reported, not interpreted.", "",
         "* " + glm_line(res["deployable_pred_target"]["glm"]), "* " + disc_line(res["deployable_pred_target"]["discrimination"]), "",
         "## African (no landmarks): out-of-sector energy of the predicted-class map (ground-truth-free monitor)", "",
         "* " + glm_line(res["african_out_sector_pred"]["glm"]), "* " + disc_line(res["african_out_sector_pred"]["discrimination"]),
         f"* {res['african_out_sector_pred']['note']}", "",
         "## In-distribution analogue (FPDB-test landmark images)", "",
         "* " + glm_line(res["in_distribution_fpdb"]["glm"]), "* " + disc_line(res["in_distribution_fpdb"]["discrimination"]), "",
         "## All backbones (HC18 + UCL, true target)", "",
         "| backbone | method | AUROC EPG − area | AUROC confidence | AUROC both (OOF) | Δ both − conf | OR per +0.1 |",
         "|---|---|---|---|---|---|---|"]
    for k, v in res["all_backbones"].items():
        b, m = k.split("|")
        g = v["glm"]
        L.append(f"| `{R.LEGACY[b]}` | {m} | {fmt(v['auroc_score'])} | {fmt(v['auroc_conf'])} | {fmt(v['auroc_both_oof'])} | "
                 f"{fmt(v['delta_both_minus_conf'])} | {g['or']:.2f} [{g['or_lo']:.2f}, {g['or_hi']:.2f}] |")
    L.append("")
    return "\n".join(L)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-only", action="store_true", help="re-render table and figure from the JSON")
    if ap.parse_args().report_only:
        res = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    else:
        res = analyse()
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(res), encoding="utf-8")
    figure(res)
    print(f"wrote {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    main()
