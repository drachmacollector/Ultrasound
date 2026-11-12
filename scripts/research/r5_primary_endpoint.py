"""
scripts/research/r5_primary_endpoint.py  (R5.9: the pre-registered primary analysis; do not modify after G4)

Primary endpoint: pooled external (HC18 + UCL + African) full-argmax balanced accuracy over the
collapsed classes present in the pool {Head, Abdomen, Femur, Thorax}. "Full argmax" = the collapse of
the 8-way argmax (``pred_full``); predictions of Cervix or Other count as errors.

* Co-primary comparisons: AFP vs B0, separately for convnext_tiny and tf_efficientnetv2_s, seed-averaged
  over seeds 11/22/33, ``stats.paired_multi_seed`` (one patient-cluster draw applied to all 6 runs of a
  backbone in each replicate), B = 2000, seed 20261109.
* Multiplicity: Holm-Bonferroni over the 2 backbones, family-wise alpha = 0.05.
* Success (per backbone): Holm-adjusted p < 0.05 and delta > 0.
* Non-inferiority in distribution: delta 8-way macro-F1 on FPDB-test (AFP − B0); the lower end of the
  one-sided 95% interval (5th percentile of the bootstrap deltas) must exceed −0.02.
* Secondary (no multiplicity correction, exploratory): B1 − B0 and B2 − B0 on the same endpoint;
  per-dataset and per-African-centre balanced accuracy; pooled accuracy.

Inputs: results/preds/<arm>_<convnext|effv2s>_s<seed>/{fpdb_test,hc18,ucl,african}.parquet.
Outputs: results/json/R5_primary.json, results/tables/R5_primary.md, results/figures/R5_forest.pdf.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.eval.scoring import bal_acc_collapsed, estimate_json, macro_f1_8, ratio  # noqa: E402
from src.eval.stats import paired_multi_seed  # noqa: E402

PREDS = Path("results/preds")
ARMS = ("b0", "b1", "b2", "afp")
BACKBONES = {"convnext_tiny": "convnext", "tf_efficientnetv2_s": "effv2s"}
SEEDS = (11, 22, 33)
EXTERNAL = ("hc18", "ucl", "african")
POOL_CLASSES = ("Head", "Abdomen", "Femur", "Thorax")
ALPHA = 0.05
NI_MARGIN = -0.02
OUT_JSON = Path("results/json/R5_primary.json")
OUT_MD = Path("results/tables/R5_primary.md")


def load(run_id: str, dataset: str) -> pd.DataFrame:
    d = pd.read_parquet(PREDS / run_id / f"{dataset}.parquet")
    for c in d.select_dtypes("category").columns:
        d[c] = d[c].astype(str)
    d["correct_full_i"] = d["correct_full"].astype(np.int8)
    return d


def pooled_external(run_id: str) -> pd.DataFrame:
    return pd.concat([load(run_id, ds) for ds in EXTERNAL], ignore_index=True)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values (step-down, monotone)."""
    order = sorted(pvals, key=pvals.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def compare(frames: dict[str, list[pd.DataFrame]], a: str, b: str, metric, alpha: float = ALPHA) -> dict:
    return estimate_json(paired_multi_seed(frames[a], frames[b], metric, alpha=alpha))


def analyse(get_frames) -> dict:
    """``get_frames(arm, backbone, kind) -> list of per-seed frames``; kind in {pooled, fpdb_test, <dataset>}."""
    primary_metric = bal_acc_collapsed("pred_full", POOL_CLASSES)
    res: dict = {"endpoint": "pooled external full-argmax balanced accuracy over {Head, Abdomen, Femur, Thorax}",
                 "per_backbone": {}}
    raw_p = {}
    for bb in BACKBONES:
        pooled = {arm: get_frames(arm, bb, "pooled") for arm in ARMS}
        r = {"afp_vs_b0": compare(pooled, "b0", "afp", primary_metric)}
        raw_p[bb] = r["afp_vs_b0"]["p_two_sided"]
        for arm in ("b1", "b2"):
            r[f"{arm}_vs_b0"] = compare(pooled, "b0", arm, primary_metric)
        r["pooled_accuracy_afp_vs_b0"] = compare(pooled, "b0", "afp", ratio("acc_full", "correct_full_i"))
        fp = {arm: get_frames(arm, bb, "fpdb_test") for arm in ("b0", "afp")}
        ni = compare(fp, "b0", "afp", macro_f1_8(), alpha=0.10)          # lo = 5th percentile of the deltas
        r["non_inferiority_macro_f1_8"] = {**ni, "one_sided_95_lower": ni["lo"], "margin": NI_MARGIN,
                                           "non_inferior": bool(ni["lo"] > NI_MARGIN)}
        per_ds = {}
        for ds in EXTERNAL:
            present = [c for c in POOL_CLASSES if c in set(get_frames("b0", bb, ds)[0]["true_collapsed"])]
            m = bal_acc_collapsed("pred_full", present)
            per_ds[ds] = {arm: compare({k: get_frames(k, bb, ds) for k in ("b0", arm)}, "b0", arm, m)
                          for arm in ("b1", "b2", "afp")}
        r["per_dataset"] = per_ds
        afr = {arm: get_frames(arm, bb, "african") for arm in ("b0", "afp")}
        centres = sorted(set(afr["b0"][0]["centre"]))
        per_c = {}
        for c in centres:
            sub = {k: [f[f["centre"] == c] for f in v] for k, v in afr.items()}
            present = [x for x in POOL_CLASSES if x in set(sub["b0"][0]["true_collapsed"])]
            per_c[c] = compare(sub, "b0", "afp", bal_acc_collapsed("pred_full", present))
        r["african_by_centre_afp_vs_b0"] = per_c
        res["per_backbone"][bb] = r
    adj = holm(raw_p)
    for bb, r in res["per_backbone"].items():
        d = r["afp_vs_b0"]
        r["holm_p"] = adj[bb]
        r["success"] = bool(adj[bb] < ALPHA and d["delta"] > 0)
    res["multiplicity"] = "Holm-Bonferroni over 2 backbones, family-wise alpha 0.05"
    return res


def default_frames(arm: str, bb: str, kind: str) -> list[pd.DataFrame]:
    ids = [f"{arm}_{BACKBONES[bb]}_s{s}" for s in SEEDS]
    if kind == "pooled":
        return [pooled_external(i) for i in ids]
    return [load(i, kind) for i in ids]


def fmt(e: dict) -> str:
    return f"{e['delta']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"


def markdown(res: dict, sha: str) -> str:
    L = ["# R5 primary endpoint (pre-registered)", "", f"Script SHA-256: `{sha}`", "",
         f"Endpoint: {res['endpoint']}. Δ = arm − B0, seed-averaged (3 seeds), paired patient-cluster bootstrap.", "",
         "| backbone | B0 | AFP | Δ AFP − B0 [95% CI] | p | Holm p | success | NI: Δ macro-F1 (5th pct) |",
         "|---|---|---|---|---|---|---|---|"]
    for bb, r in res["per_backbone"].items():
        d, ni = r["afp_vs_b0"], r["non_inferiority_macro_f1_8"]
        L.append(f"| `{bb}` | {d['point_a']:.3f} | {d['point_b']:.3f} | {fmt(d)} | {d['p_two_sided']:.4f} | {r['holm_p']:.4f} | "
                 f"{'yes' if r['success'] else 'no'} | {ni['delta']:+.3f} ({ni['one_sided_95_lower']:+.3f}; "
                 f"{'non-inferior' if ni['non_inferior'] else 'not shown'}) |")
    L += ["", "Secondary (exploratory, no multiplicity correction):", "",
          "| backbone | B1 − B0 | B2 − B0 | pooled accuracy AFP − B0 |", "|---|---|---|---|"]
    for bb, r in res["per_backbone"].items():
        L.append(f"| `{bb}` | {fmt(r['b1_vs_b0'])} | {fmt(r['b2_vs_b0'])} | {fmt(r['pooled_accuracy_afp_vs_b0'])} |")
    L.append("")
    return "\n".join(L)


def forest(res: dict) -> None:
    from src.audit.plotting import NEUTRAL, SERIES, plt, save

    rows = []
    for bb, r in res["per_backbone"].items():
        for arm in ("b1", "b2", "afp"):
            rows.append((f"{BACKBONES[bb]} {arm.upper()} pooled", r[f"{arm}_vs_b0"] if arm != "afp" else r["afp_vs_b0"]))
            for ds in EXTERNAL:
                rows.append((f"{BACKBONES[bb]} {arm.upper()} {ds}", r["per_dataset"][ds][arm]))
    fig, ax = plt.subplots(figsize=(6.0, 0.18 * len(rows) + 0.8))
    for i, (lab, e) in enumerate(rows):
        c = SERIES[("B1", "B2", "AFP").index(lab.split()[1])]
        ax.errorbar([e["delta"]], [i], xerr=[[e["delta"] - e["lo"]], [e["hi"] - e["delta"]]], fmt="o", ms=3,
                    color=c, lw=1, capsize=1.5)
    ax.axvline(0, color=NEUTRAL, lw=0.8, ls=":")
    ax.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Δ balanced accuracy vs B0 (full argmax)")
    fig.tight_layout()
    save(fig, "R5_forest")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    res = analyse(default_frames)
    res["script_sha256"] = sha
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(res, sha), encoding="utf-8")
    forest(res)
    print(f"wrote {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    main()
