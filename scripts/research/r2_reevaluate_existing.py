"""
scripts/research/r2_reevaluate_existing.py  (R2.4 + R2.5: Gate G2)

Re-evaluates the six frozen legacy checkpoints with the new evaluation core and reproduces
the legacy headline numbers.

Steps
  1. predict  6 checkpoints x {fpdb_val, fpdb_test, hc18, ucl, african}
              -> results/preds/legacy_<short>_s42/<dataset>.parquet (features on test + externals).
              Existing parquet files are reused unless --force.
  2. score    in-distribution (8-way macro-F1, balanced accuracy, Other recall; cluster CIs, plus the
              legacy image-level CI for comparison), per machine, external per dataset / pooled /
              African centre, error decomposition, pairwise backbone comparison (paired cluster
              bootstrap on pooled external bal_acc_full), errors on contradictory FPDB duplicates.
  3. write    results/json/R2_existing_models.json, results/tables/R2_existing_models.md,
              results/figures/R2_rejection_vs_confusion.{pdf,png}
  4. gate     results/json/G2.json, docs/research/GATE_G2.md

Legacy reference values (G2 targets) are quoted from archive/docs/EVAL_REPORT.md and
archive/docs/CROSS_DEVICE_BACKBONE_COMPARISON.md (tag legacy-fetscan-demo-v1).

Usage:  python scripts/research/r2_reevaluate_existing.py [--force] [--score-only] [--num-workers 4]
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")
from src.eval.external import ERROR_SHARES, rejection_vs_confusion_table, score_dataset, score_pooled  # noqa: E402
from src.eval.predict import PREDS_ROOT, git_hash, predict  # noqa: E402
from src.eval.scoring import (  # noqa: E402
    add_helper_columns,
    bal_acc_collapsed,
    by_stratum,
    estimate_json,
    ratio,
    score_indist,
)
from src.eval.stats import Estimate, paired_cluster_bootstrap  # noqa: E402

BACKBONES = {  # checkpoint dir -> run-id short name
    "convnext_tiny": "convnext",
    "tf_efficientnetv2_s": "effv2s",
    "repvgg_a1": "repvgga1",
    "repvgg_a2": "repvgga2",
    "mobilenetv3_large_100": "mnv3l",
    "efficientnet_lite0": "efflite0",
}
DATASETS = ("fpdb_val", "fpdb_test", "hc18", "ucl", "african")
FEATURES_ON = {"fpdb_test", "hc18", "ucl", "african"}
EXTERNAL = ("hc18", "ucl", "african")
MACHINES = ("Aloka", "Voluson E6", "Voluson S10", "Other")

OUT_JSON = Path("results/json/R2_existing_models.json")
OUT_MD = Path("results/tables/R2_existing_models.md")
OUT_FIG = Path("results/figures/R2_rejection_vs_confusion")
G2_JSON = Path("results/json/G2.json")
G2_MD = Path("docs/research/GATE_G2.md")
META = Path("data/processed/metadata/acquisition_metadata.parquet")

# ---- legacy reference values (archive/docs/EVAL_REPORT.md §1, §4; CROSS_DEVICE_BACKBONE_COMPARISON.md)
LEGACY_TEST_MACRO_F1 = {"convnext_tiny": 0.8927, "tf_efficientnetv2_s": 0.8853, "efficientnet_lite0": 0.8823,
                        "repvgg_a2": 0.8811, "repvgg_a1": 0.8760, "mobilenetv3_large_100": 0.8751}
LEGACY_XDEV_ACC = {"convnext_tiny": 0.836, "tf_efficientnetv2_s": 0.853, "efficientnet_lite0": 0.850,
                   "repvgg_a2": 0.843, "repvgg_a1": 0.869, "mobilenetv3_large_100": 0.823}
LEGACY_HEAD_TO_OTHER = {"convnext_tiny": 180, "tf_efficientnetv2_s": 68, "efficientnet_lite0": 95,
                        "repvgg_a2": 107, "repvgg_a1": 56, "mobilenetv3_large_100": 171}
LEGACY_CONVNEXT = {"hc18_acc": 0.814, "ucl_acc": 0.889, "head_errors": 194, "head_to_other": 180}
TOL = {"macro_f1": 0.005, "pooled_acc": 0.003, "subset_acc": 0.005, "count": 3}


def run_id(bb: str) -> str:
    return f"legacy_{BACKBONES[bb]}_s42"


# ----------------------------------------------------------------------------- 1. predict
def run_predictions(force: bool, num_workers: int, batch_size: int) -> dict:
    timings = {}
    for bb in BACKBONES:
        for ds in DATASETS:
            out = PREDS_ROOT / run_id(bb)
            pq = out / f"{ds}.parquet"
            if pq.exists() and not force:
                continue
            t = time.perf_counter()
            predict(Path("checkpoints") / bb / "best.pt", ds, out, run_id=run_id(bb), batch_size=batch_size,
                    num_workers=num_workers, save_features=ds in FEATURES_ON)
            timings[f"{bb}/{ds}"] = round(time.perf_counter() - t, 1)
            print(f"  {run_id(bb)}/{ds}: {timings[f'{bb}/{ds}']} s", flush=True)
    return timings


def load_preds() -> dict[str, dict[str, pd.DataFrame]]:
    preds = {}
    for bb in BACKBONES:
        preds[bb] = {}
        for ds in DATASETS:
            f = pd.read_parquet(PREDS_ROOT / run_id(bb) / f"{ds}.parquet")
            for c in f.select_dtypes("category").columns:
                f[c] = f[c].astype(str)
            preds[bb][ds] = f
    return preds


# ----------------------------------------------------------------------------- 2. score
def duplicate_conflicts(test_preds: dict[str, pd.DataFrame]) -> dict:
    """Errors on byte-identical FPDB test images whose copies carry different labels (R1)."""
    meta = pd.read_parquet(META, columns=["image_id", "split", "md5", "plane_label"])
    te = meta[meta["split"] == "test"]
    g = te.groupby("md5")["plane_label"]
    conflict_md5 = g.nunique()[lambda s: s > 1].index
    conf = te[te["md5"].isin(conflict_md5)]
    floor = int(sum(len(x) - x.value_counts().iloc[0] for _, x in conf.groupby("md5")["plane_label"]))
    out = {"conflict_groups": len(conflict_md5), "conflict_images": len(conf),
           "unavoidable_errors_min": floor, "by_model": {}}
    ids = set(conf["image_id"])
    for bb, f in test_preds.items():
        err = f[~f["correct_fine"].astype(bool)]
        out["by_model"][bb] = {"errors_total": len(err), "errors_on_conflict_images": int(err["image_id"].isin(ids).sum())}
    return out


def legacy_head_errors(df: pd.DataFrame) -> dict:
    """Head errors and Head->Other on HC18+UCL (legacy 'Detailed Misclassification Breakdown')."""
    h = df[(df["true_collapsed"] == "Head") & (df["pred_full"] != "Head")]
    return {"head_errors": len(h), "head_to_other": int((h["pred_fine"] == "Other").sum()),
            "head_to_by_fine": h["pred_fine"].value_counts().to_dict()}


def pairwise(preds: dict, datasets: tuple[str, ...], metric) -> list[dict]:
    """All 15 backbone pairs, paired cluster bootstrap on the pooled ``datasets`` (Δ = b − a)."""
    pooled = {bb: add_helper_columns(pd.concat([preds[bb][d] for d in datasets], ignore_index=True))
              for bb in BACKBONES}
    return [{"a": a, "b": b, **paired_cluster_bootstrap(pooled[a], pooled[b], metric).to_dict()}
            for a, b in itertools.combinations(BACKBONES, 2)]


def score_all(preds: dict) -> dict:
    res: dict = {"models": {}, "git_hash": git_hash()}
    for bb in BACKBONES:
        p = preds[bb]
        r: dict = {"run_id": run_id(bb)}
        r["fpdb_test"] = score_indist(p["fpdb_test"], image_ci=True)
        r["fpdb_val"] = score_indist(p["fpdb_val"])
        r["fpdb_test_by_machine"] = by_stratum(p["fpdb_test"], "source_machine", score_indist,
                                               strict_labels=False, max_nan_frac=1.0)
        r["external"] = {ds: score_dataset(p[ds]) for ds in EXTERNAL}
        r["external"]["pooled"] = score_pooled({ds: p[ds] for ds in EXTERNAL})
        r["external"]["hc18_ucl"] = score_pooled({ds: p[ds] for ds in ("hc18", "ucl")})
        r["african_by_centre"] = by_stratum(p["african"], "centre", score_dataset, max_nan_frac=1.0)
        hu = pd.concat([p["hc18"], p["ucl"]], ignore_index=True)
        r["legacy_head_errors_hc18_ucl"] = legacy_head_errors(hu)
        res["models"][bb] = r
        print(f"  scored {bb}", flush=True)
    res["pairwise_pooled_bal_acc_full"] = pairwise(
        preds, EXTERNAL, bal_acc_collapsed("pred_full", ("Head", "Abdomen", "Femur", "Thorax")))
    res["pairwise_hc18_ucl_acc_full"] = pairwise(preds, ("hc18", "ucl"), ratio("acc_full", "correct_full_i"))
    res["duplicate_conflicts"] = duplicate_conflicts({bb: preds[bb]["fpdb_test"] for bb in BACKBONES})
    return res


# ----------------------------------------------------------------------------- 3. tables + figure
def f3(e: Estimate) -> str:
    return e.fmt(dp=3)


def pct(e: Estimate) -> str:
    return e.fmt(pct=True, dp=3)


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def build_markdown(res: dict) -> str:
    M = res["models"]
    s = ["# R2: Re-evaluation of the six frozen legacy checkpoints",
         "",
         f"Generated by `scripts/research/r2_reevaluate_existing.py` (git `{res['git_hash']}`). "
         "All intervals: 95% patient-clustered percentile bootstrap, B = 2000, seed 20261109, "
         "unless marked *image-level*. Per-image predictions: `results/preds/legacy_<short>_s42/`.",
         "", "## 1. In-distribution (FPDB test, 5,271 images / 896 patients)", ""]
    rows = []
    for bb, r in M.items():
        t = r["fpdb_test"]
        rows.append([f"`{bb}`", f3(t["macro_f1_8"]), f3(t["macro_f1_8_image_ci"]), f3(t["bal_acc_8"]),
                     pct(t["other_recall"]), pct(t["acc_full"]), f3(r["fpdb_val"]["macro_f1_8"])])
    s += [md_table(["model", "macro-F1 (8-way)", "macro-F1, *image-level CI* (legacy method)", "balanced acc (8)",
                    "Other recall", "collapsed acc (6)", "val macro-F1"], rows), ""]
    widths = [(r["fpdb_test"]["macro_f1_8"].hi - r["fpdb_test"]["macro_f1_8"].lo) /
              (r["fpdb_test"]["macro_f1_8_image_ci"].hi - r["fpdb_test"]["macro_f1_8_image_ci"].lo) for r in M.values()]
    s += [f"CI correction: cluster CIs are {min(widths):.2f}–{max(widths):.2f}× wider than the legacy image-level CIs.", ""]

    s += ["## 2. Per machine (FPDB test)", "",
          "Macro-F1 averages over the classes present on that machine; Other recall [cluster CI].", ""]
    rows = []
    for bb, r in M.items():
        row = [f"`{bb}`"]
        for m in MACHINES:
            x = r["fpdb_test_by_machine"].get(m)
            row.append("–" if x is None else f"{f3(x['macro_f1_8'])}<br>Other {pct(x['other_recall'])}"
                       if "other_recall" in x else f3(x["macro_f1_8"]))
        rows.append(row)
    n_m = {m: M["convnext_tiny"]["fpdb_test_by_machine"][m]["n_images"] for m in MACHINES}
    s += [md_table(["model"] + [f"{m} (n={n_m[m]})" for m in MACHINES], rows), ""]

    s += ["## 3. External sets", "",
          "`acc_full` = collapse(8-way argmax) == label (legacy scoring). `bal_acc_restricted` = argmax over the "
          "dataset's present classes. `rejection` = prediction outside the dataset's classes.", ""]
    for ds, title in (("hc18", "HC18 (999 images / 806 patients; Head only)"),
                      ("ucl", "UCL (424 / 51; Head, Abdomen, Femur)"),
                      ("african", "African (450 / 127; Head, Abdomen, Femur, Thorax)"),
                      ("pooled", "Pooled HC18 + UCL + African (1,873 images)"),
                      ("hc18_ucl", "HC18 + UCL (legacy cross-device set, 1,423 images)")):
        rows = []
        for bb, r in M.items():
            x = r["external"][ds]
            e = x["errors"]
            rows.append([f"`{bb}`", pct(x["acc_full"]), pct(x["bal_acc_full"]), pct(x["bal_acc_restricted"]),
                         pct(x["rejection_rate"]), e["n_errors"],
                         " / ".join(f"{e[k].point:.0%}" if k in e else "–" for k in ERROR_SHARES)])
        s += [f"### {title}", "",
              md_table(["model", "acc_full", "bal_acc_full", "bal_acc_restricted", "rejection",
                        "errors", "error split Other / absent / confusion"], rows), ""]

    s += ["## 4. African per centre (`acc_full`, rejection rate)", "",
          "Ghana and Uganda have no Thorax images; their restricted argmax still includes Thorax "
          "(dataset-level present set).", ""]
    centres = list(M["convnext_tiny"]["african_by_centre"])
    rows = []
    for bb, r in M.items():
        rows.append([f"`{bb}`"] + [f"{pct(r['african_by_centre'][c]['acc_full'])}<br>rej "
                                   f"{pct(r['african_by_centre'][c]['rejection_rate'])}" for c in centres])
    dev = {c: M["convnext_tiny"]["african_by_centre"][c]["n_images"] for c in centres}
    s += [md_table(["model"] + [f"{c} (n={dev[c]})" for c in centres], rows), ""]

    s += ["## 5. Pairwise backbone comparisons", "",
          "Δ = b − a, paired cluster bootstrap (same images, same resamples). No multiplicity correction; "
          "15 comparisons per endpoint.", ""]
    for key, title in (("pairwise_pooled_bal_acc_full", "Pooled HC18+UCL+African `bal_acc_full` (primary)"),
                       ("pairwise_hc18_ucl_acc_full", "HC18+UCL `acc_full` (legacy endpoint)")):
        rows = [[f"`{x['a']}`", f"`{x['b']}`", f"{x['point_a']:.3f}", f"{x['point_b']:.3f}",
                 f"{100 * x['delta']:+.1f} pp [{100 * x['lo']:+.1f}, {100 * x['hi']:+.1f}]",
                 f"{x['p_two_sided']:.3g}"]
                for x in sorted(res[key], key=lambda x: x["p_two_sided"])]
        s += [f"### {title}", "", md_table(["a", "b", "a", "b", "Δ [95% CI]", "p"], rows), ""]

    d = res["duplicate_conflicts"]
    s += ["## 6. Errors on contradictory duplicates (FPDB test)", "",
          f"{d['conflict_groups']} groups of byte-identical test images carry different labels "
          f"({d['conflict_images']} images); at least {d['unavoidable_errors_min']} errors are unavoidable "
          "for any classifier.", ""]
    rows = [[f"`{bb}`", v["errors_total"], v["errors_on_conflict_images"]] for bb, v in d["by_model"].items()]
    s += [md_table(["model", "test errors (8-way)", "on conflicting duplicates"], rows), ""]
    return "\n".join(s)


def make_figure(res: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seg = [("correct", "Correct", "#d9d8d4"), ("err_other", "Predicted Other", "#2a78d6"),
           ("err_absent", "Absent class", "#eb6834"), ("err_confusion", "Wrong present class", "#1baf7a")]
    panels = [("hc18", "HC18"), ("ucl", "UCL"), ("african", "African"), ("pooled", "Pooled")]
    models = list(res["models"])
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.2), sharey=True)
    for ax, (ds, title) in zip(axes, panels):
        for i, bb in enumerate(models):
            x = res["models"][bb]["external"][ds]
            n, e = x["n_images"], x["errors"]
            vals = {"correct": 1 - e["n_errors"] / n,
                    **{k: e.get(f"n_{k}", 0) / n for k in ERROR_SHARES}}
            left = 0.0
            for key, _, col in seg:
                ax.barh(i, vals[key], left=left, color=col, edgecolor="white", linewidth=1.5, height=0.7)
                left += vals[key]
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1.0])
        ax.set_xticklabels(["0", "50%", "100%"], fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(models, fontsize=8)
    axes[0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in seg]
    fig.legend(handles, [lab for _, lab, _ in seg], loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle("External sets: share of images by outcome (legacy checkpoints, full 8-way argmax)",
                 fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG.with_suffix(".pdf"))
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- 4. gate G2
def gate(res: dict) -> dict:
    M = res["models"]
    c = M["convnext_tiny"]
    checks = []

    def add(name: str, value: float, target: float, tol: float, kind: str = "primary") -> None:
        checks.append({"check": name, "value": value, "target": target, "tolerance": tol,
                       "pass": bool(abs(value - target) <= tol + 1e-12), "kind": kind})

    add("convnext_tiny FPDB-test macro-F1 (8-way)", c["fpdb_test"]["macro_f1_8"].point,
        LEGACY_TEST_MACRO_F1["convnext_tiny"], TOL["macro_f1"])
    add("convnext_tiny HC18+UCL acc_full", c["external"]["hc18_ucl"]["acc_full"].point,
        LEGACY_XDEV_ACC["convnext_tiny"], TOL["pooled_acc"])
    add("convnext_tiny HC18 acc_full", c["external"]["hc18"]["acc_full"].point, LEGACY_CONVNEXT["hc18_acc"],
        TOL["subset_acc"])
    add("convnext_tiny UCL acc_full", c["external"]["ucl"]["acc_full"].point, LEGACY_CONVNEXT["ucl_acc"],
        TOL["subset_acc"])
    add("convnext_tiny HC18+UCL Head errors", c["legacy_head_errors_hc18_ucl"]["head_errors"],
        LEGACY_CONVNEXT["head_errors"], TOL["count"])
    add("convnext_tiny HC18+UCL Head errors predicted Other", c["legacy_head_errors_hc18_ucl"]["head_to_other"],
        LEGACY_CONVNEXT["head_to_other"], TOL["count"])
    for bb in BACKBONES:
        if bb == "convnext_tiny":
            continue
        add(f"{bb} FPDB-test macro-F1 (8-way)", M[bb]["fpdb_test"]["macro_f1_8"].point,
            LEGACY_TEST_MACRO_F1[bb], TOL["macro_f1"])
    for bb in BACKBONES:  # secondary: legacy cross-device table for every backbone
        add(f"{bb} HC18+UCL acc_full", M[bb]["external"]["hc18_ucl"]["acc_full"].point, LEGACY_XDEV_ACC[bb],
            TOL["pooled_acc"], "secondary")
        add(f"{bb} HC18+UCL Head errors predicted Other", M[bb]["legacy_head_errors_hc18_ucl"]["head_to_other"],
            LEGACY_HEAD_TO_OTHER[bb], TOL["count"], "secondary")
    primary = [x for x in checks if x["kind"] == "primary"]
    return {"passed": all(x["pass"] for x in primary), "checks": checks,
            "secondary_all_pass": all(x["pass"] for x in checks if x["kind"] == "secondary")}


def a1_counts(rows: list[dict]) -> dict:
    """How many of the other 5 backbones repvgg_a1 beats / loses to (95% CI of Δ excludes 0)."""
    better = worse = 0
    for x in rows:
        if "repvgg_a1" not in (x["a"], x["b"]):
            continue
        sign = 1 if x["b"] == "repvgg_a1" else -1        # orient Δ as (repvgg_a1 − other)
        lo, hi = sorted((sign * x["lo"], sign * x["hi"]))
        better += lo > 0
        worse += hi < 0
    return {"significantly_better_than": int(better), "significantly_worse_than": int(worse), "of": 5}


def winner_verdict(res: dict) -> dict:
    """The legacy claim is supported only if repvgg_a1 ranks first and beats all 5 on both endpoints."""
    out = {}
    for key, endpoint in (("hc18_ucl", "acc_full"), ("pooled", "bal_acc_full")):
        pts = {bb: r["external"][key][endpoint].point for bb, r in res["models"].items()}
        ranked = sorted(pts, key=lambda b: pts[b], reverse=True)
        c = a1_counts(res[f"pairwise_{key}_{endpoint}"])
        out[f"{key}_{endpoint}"] = {"ranking": ranked, "points": pts, **c,
                                    "supported": ranked[0] == "repvgg_a1" and c["significantly_better_than"] == 5}
    sup = [v["supported"] for v in out.values()]
    out["verdict"] = ("supported" if all(sup) else
                      "withdrawn (endpoint-dependent)" if any(sup) or any(
                          v["significantly_better_than"] for v in out.values()) else "withdrawn")
    return out


def g2_markdown(g: dict, verdict: dict, res: dict) -> str:
    def v(x: dict) -> str:
        val, tgt = x["value"], x["target"]
        if isinstance(tgt, int):
            return f"{int(val)} | {tgt} ± {int(x['tolerance'])}"
        if tgt < 1 and "acc" in x["check"]:
            return f"{val:.2%} | {tgt:.1%} ± {100 * x['tolerance']:.1f} pp"
        return f"{val:.4f} | {tgt:.4f} ± {x['tolerance']}"

    lines = ["# Gate G2: reproduction of the legacy numbers", "",
             f"Generated by `scripts/research/r2_reevaluate_existing.py` (git `{res['git_hash']}`). "
             "Same frozen checkpoints (SHA-256 in `checkpoints/FROZEN.md`), new evaluation core "
             "(`src/eval/predict.py`, fp32, `cudnn.benchmark=False`), current environment.",
             "", f"**G2 {'PASSED' if g['passed'] else 'FAILED'}** (primary checks); secondary legacy-table checks "
             f"{'all pass' if g['secondary_all_pass'] else 'have deviations (see below)'}.", "",
             "| check | ours | legacy target | pass | kind |", "|---|---|---|---|---|"]
    for x in g["checks"]:
        ours, tgt = v(x).split(" | ")
        lines.append(f"| {x['check']} | {ours} | {tgt} | {'✅' if x['pass'] else '❌'} | {x['kind']} |")
    lines += ["", "## Legacy 'repvgg_a1 winner' claim", "",
              "The legacy comparison ranked backbones by HC18+UCL accuracy (image-level, single seed, no paired "
              "test). Re-tested with the paired patient-clustered bootstrap on the legacy endpoint and on the "
              "study's pooled external endpoint:", "",
              "| endpoint | ranking (point estimate) | repvgg_a1 significantly better / worse than |",
              "|---|---|---|"]
    for key, label in (("hc18_ucl_acc_full", "HC18+UCL `acc_full` (legacy)"),
                       ("pooled_bal_acc_full", "HC18+UCL+African `bal_acc_full` (primary)")):
        x = verdict[key]
        rank = " > ".join(f"`{b}` {x['points'][b]:.3f}" for b in x["ranking"])
        lines.append(f"| {label} | {rank} | {x['significantly_better_than']} / "
                     f"{x['significantly_worse_than']} of {x['of']} |")
    lines += ["", f"Claim **{verdict['verdict'].upper()}**: the claim holds only if `repvgg_a1` ranks first and is "
              "significantly better than all five others on both endpoints. All six are single-seed models, so "
              "backbone differences include seed variance that R5's multi-seed runs will quantify.", ""]
    return "\n".join(lines)


# ----------------------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run predictions even if parquet files exist")
    ap.add_argument("--score-only", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    a = ap.parse_args()
    t0 = time.perf_counter()
    timings = {} if a.score_only else run_predictions(a.force, a.num_workers, a.batch_size)
    preds = load_preds()
    res = score_all(preds)
    res["prediction_timings_s"] = timings
    g = gate(res)
    verdict = winner_verdict(res)
    res["repvgg_a1_verdict"] = verdict

    rv = rejection_vs_confusion_table({bb: {ds: r["external"][ds] for ds in (*EXTERNAL, "pooled")}
                                       for bb, r in res["models"].items()})
    for p in (OUT_JSON, OUT_MD, G2_JSON, G2_MD):
        p.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(estimate_json(res), indent=1), encoding="utf-8")
    OUT_MD.write_text(build_markdown(res) + "\n## 7. Rejection vs confusion (point estimates)\n\n"
                      + rv.to_markdown(index=False, floatfmt=".3f") + "\n", encoding="utf-8")
    make_figure(res)
    G2_JSON.write_text(json.dumps(estimate_json({"gate": g, "repvgg_a1": verdict}), indent=1), encoding="utf-8")
    G2_MD.write_text(g2_markdown(g, verdict, res), encoding="utf-8")
    print(f"G2 {'PASSED' if g['passed'] else 'FAILED'}; repvgg_a1 claim {verdict['verdict']}; "
          f"{time.perf_counter() - t0:.0f} s total")
    for x in g["checks"]:
        print(f"  {'OK ' if x['pass'] else 'BAD'} {x['check']}: {x['value']} vs {x['target']}")


if __name__ == "__main__":
    main()
