"""
scripts/research/e9_ablation.py  (R7 E9 analysis, POST-HOC; fixed in docs/research/plans/R7_POSTHOC_FOLLOWUP.md
before any E9 run)

Inputs: results/preds/<arm>_<convnext|effv2s>_s<seed>/ for arms b0, b1, b2, afp (R5) and b2smp, b2sty (E9).

  1. Component contrasts on the R5 primary endpoint (pooled external full-argmax balanced accuracy over
     {Head, Abdomen, Femur, Thorax}), seed-averaged, paired patient-cluster bootstrap:
     B2SMP - B2, B2STY - B2, AFP - B2 per backbone; Holm over the 6 tests (reported, still exploratory).
  2. Decomposition of the input-cleaning gain: B1 - B0 (sector letterbox), B2 - B1 (marker inpainting +
     synthetic calipers), B2 - B0, pooled and per external dataset.
  3. Marker stratification (amendment 2026-11-14, before the E9 results): the same ladder contrasts on
     pooled-external images with vs without a detected measurement marker (R1 detector, ``n_crosses > 0``),
     balanced accuracy over the classes present in each subset.
  4. Mechanism readouts per arm (seed-averaged): pooled-external rejection rate, within-class machine
     decodability (FPDB test, R3 E3 protocol), FPDB-test 8-way macro-F1 vs B2 (in-distribution cost).

  --probes   fit the within-class machine probes for the E9 runs first (CPU, torch-free workers)

Arms whose three seeds are not all predicted are skipped, so the script can run on the R5 arms alone.
Outputs: results/json/E9.json, results/tables/E9.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
sys.path.insert(0, "scripts/research")
from src.eval.scoring import add_helper_columns, bal_acc_collapsed, estimate_json, macro_f1_8, ratio  # noqa: E402
from src.eval.stats import multi_seed_summary, paired_multi_seed  # noqa: E402

PREDS = Path("results/preds")
BACKBONES = {"convnext_tiny": "convnext", "tf_efficientnetv2_s": "effv2s"}
SEEDS = (11, 22, 33)
EXTERNAL = ("hc18", "ucl", "african")
POOL_CLASSES = ("Head", "Abdomen", "Femur", "Thorax")
ARMS = ("b0", "b1", "b2", "b2smp", "b2sty", "afp")
COMPONENTS = (("b2smp", "b2"), ("b2sty", "b2"), ("afp", "b2"))
LADDER = (("b1", "b0"), ("b2", "b1"), ("b2", "b0"))
EXT_META = Path("data/processed/metadata/external_metadata.parquet")
OUT_JSON = Path("results/json/E9.json")
OUT_MD = Path("results/tables/E9.md")
ALPHA = 0.05


def rid(arm: str, bb: str, seed: int) -> str:
    return f"{arm}_{BACKBONES[bb]}_s{seed}"


def load(run_id: str, ds: str) -> pd.DataFrame:
    d = pd.read_parquet(PREDS / run_id / f"{ds}.parquet")
    for c in d.select_dtypes("category").columns:
        d[c] = d[c].astype(str)
    d["image_id"] = d["image_id"].astype(str)
    return d


def available(arm: str, bb: str) -> bool:
    return all((PREDS / rid(arm, bb, s) / f"{ds}.parquet").exists()
               for s in SEEDS for ds in (*EXTERNAL, "fpdb_test"))


def frames(arm: str, bb: str, kind: str) -> list[pd.DataFrame]:
    if kind == "pooled":
        return [pd.concat([load(rid(arm, bb, s), ds) for ds in EXTERNAL], ignore_index=True) for s in SEEDS]
    return [load(rid(arm, bb, s), kind) for s in SEEDS]


def present(fs: list[pd.DataFrame]) -> list[str]:
    return [c for c in POOL_CLASSES if c in set(fs[0]["true_collapsed"])]


def contrast(a: list[pd.DataFrame], b: list[pd.DataFrame], metric) -> dict:
    """a - b (paired, seed-averaged)."""
    return estimate_json(paired_multi_seed(b, a, metric))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals, key=pvals.get)
    m, out, running = len(order), {}, 0.0
    for i, k in enumerate(order):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        out[k] = running
    return out


def marker_ids() -> set[str]:
    m = pd.read_parquet(EXT_META, columns=["image_id", "n_crosses"])
    return set(m.loc[m["n_crosses"] > 0, "image_id"].astype(str))


# ----------------------------------------------------------------------------- probes
def run_probes(jobs: int) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    from r5_evaluate_matrix import _probe_job

    ids = [rid(a, bb, s) for a in ("b2smp", "b2sty") for bb in BACKBONES for s in SEEDS
           if (PREDS / rid(a, bb, s) / "fpdb_test.features.npy").exists()]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for r in ex.map(_probe_job, ids):
            print(f"  probes {r}", flush=True)


# ----------------------------------------------------------------------------- analysis
def analyse() -> dict:
    from src.audit.probes import within_class_mean

    res: dict = {"status": "post-hoc exploratory (R7 E9; not pre-registered)",
                 "endpoint": "pooled external full-argmax balanced accuracy over {Head, Abdomen, Femur, Thorax}",
                 "arms_available": {}, "per_backbone": {}}
    endpoint = bal_acc_collapsed("pred_full", POOL_CLASSES)
    mk = marker_ids()
    raw_p: dict[str, float] = {}
    for bb in BACKBONES:
        av = [a for a in ARMS if available(a, bb)]
        res["arms_available"][bb] = av
        pooled = {a: frames(a, bb, "pooled") for a in av}
        fp = {a: frames(a, bb, "fpdb_test") for a in av}
        r: dict = {"components_vs_b2": {}, "ladder": {}, "ladder_by_dataset": {}, "marker_strata": {}, "per_arm": {}}
        for a, b in COMPONENTS:
            if a in av and b in av:
                e = contrast(pooled[a], pooled[b], endpoint)
                r["components_vs_b2"][f"{a}-{b}"] = e
                raw_p[f"{bb}|{a}-{b}"] = e["p_two_sided"]
        for a, b in LADDER:
            r["ladder"][f"{a}-{b}"] = contrast(pooled[a], pooled[b], endpoint)
        for ds in EXTERNAL:
            per = {a: frames(a, bb, ds) for a in ("b0", "b1", "b2")}
            m = bal_acc_collapsed("pred_full", present(per["b0"]))
            r["ladder_by_dataset"][ds] = {f"{a}-{b}": contrast(per[a], per[b], m) for a, b in LADDER}
        for name, keep in (("with_marker", True), ("without_marker", False)):
            sub = {a: [f[f["image_id"].isin(mk) == keep].reset_index(drop=True) for f in pooled[a]]
                   for a in ("b0", "b1", "b2")}
            m = bal_acc_collapsed("pred_full", present(sub["b0"]))
            r["marker_strata"][name] = {"n_images": len(sub["b0"][0]), "classes": present(sub["b0"]),
                                        **{f"{a}-{b}": contrast(sub[a], sub[b], m) for a, b in LADDER}}
        for a in av:
            hs = [add_helper_columns(f) for f in pooled[a]]
            pa = {"endpoint": estimate_json(multi_seed_summary(pooled[a], endpoint)),
                  "rejection_rate": estimate_json(multi_seed_summary(hs, ratio("rejection_rate", "rejected"))),
                  "fpdb_test_macro_f1_8": estimate_json(multi_seed_summary(fp[a], macro_f1_8()))}
            pr = [PREDS / rid(a, bb, s) / "probe_within_class_machine.parquet" for s in SEEDS]
            if all(p.exists() for p in pr):
                pa["within_class_machine"] = estimate_json(
                    multi_seed_summary([pd.read_parquet(p) for p in pr], within_class_mean, key="image_id"))
            if a != "b2" and "b2" in av:
                pa["fpdb_test_macro_f1_8_vs_b2"] = contrast(fp[a], fp["b2"], macro_f1_8())
            r["per_arm"][a] = pa
        res["per_backbone"][bb] = r
    adj = holm(raw_p) if raw_p else {}
    for k, p in adj.items():
        bb, c = k.split("|")
        res["per_backbone"][bb]["components_vs_b2"][c]["holm_p"] = p
    res["multiplicity"] = f"Holm over {len(raw_p)} component tests (reported; analysis exploratory)"
    return res


# ----------------------------------------------------------------------------- report
def ci(e: dict | None, dp: int = 3, pct: bool = False) -> str:
    if not e:
        return "–"
    k, s = (100, "") if pct else (1, "")
    sign = "+" if "delta" in e else ""
    pt = e.get("delta", e.get("point"))
    return f"{k * pt:{sign}.{dp}f} [{k * e['lo']:{sign}.{dp}f}, {k * e['hi']:{sign}.{dp}f}]{s}"


def markdown(res: dict) -> str:
    L = ["# E9: which AFP component costs external accuracy? (post-hoc, exploratory)", "",
         f"Endpoint: {res['endpoint']}. Δ = seed-averaged paired patient-cluster bootstrap (3 seeds, B = 2000). "
         f"{res['multiplicity']}.", ""]
    for bb, r in res["per_backbone"].items():
        L += [f"## {bb}", "", "**Component contrasts vs B2**", "", "| contrast | Δ [95% CI] | p | Holm p |", "|---|---|---|---|"]
        for c, e in r["components_vs_b2"].items():
            L.append(f"| {c} | {ci(e)} | {e['p_two_sided']:.4f} | {e.get('holm_p', float('nan')):.4f} |")
        L += ["", "**Input-cleaning ladder** (pooled and per dataset)", "",
              "| contrast | pooled | HC18 | UCL | African |", "|---|---|---|---|---|"]
        for c in r["ladder"]:
            L.append(f"| {c} | {ci(r['ladder'][c])} | " + " | ".join(ci(r["ladder_by_dataset"][ds][c]) for ds in EXTERNAL) + " |")
        L += ["", "**Marker stratification** (pooled external; detected measurement marker)", "",
              "| subset | images | classes | B1 − B0 | B2 − B1 | B2 − B0 |", "|---|---|---|---|---|---|"]
        for name, s in r["marker_strata"].items():
            L.append(f"| {name} | {s['n_images']} | {', '.join(s['classes'])} | {ci(s['b1-b0'])} | {ci(s['b2-b1'])} | {ci(s['b2-b0'])} |")
        L += ["", "**Per arm** (seed-averaged)", "",
              "| arm | endpoint | rejection rate (%) | within-class machine decoding | FPDB-test macro-F1 | macro-F1 vs B2 |",
              "|---|---|---|---|---|---|"]
        for a, p in r["per_arm"].items():
            L.append(f"| {a} | {ci(p['endpoint'])} | {ci(p['rejection_rate'], 1, True)} | {ci(p.get('within_class_machine'))} | "
                     f"{ci(p['fpdb_test_macro_f1_8'])} | {ci(p.get('fpdb_test_macro_f1_8_vs_b2'))} |")
        L.append("")
    return "\n".join(L)


def figure(res: dict) -> None:
    """Left: input-cleaning steps by marker stratum. Right: AFP components added to B2."""
    from src.audit.plotting import SERIES, plt, save

    bbs = [("convnext_tiny", "ConvNeXt-T"), ("tf_efficientnetv2_s", "EffNetV2-S")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=False)
    rows = [("without_marker", "b1-b0", "sector crop · no marker"), ("with_marker", "b1-b0", "sector crop · marker"),
            ("without_marker", "b2-b1", "marker removal · no marker"), ("with_marker", "b2-b1", "marker removal · marker")]
    for j, (bb, lab) in enumerate(bbs):
        st = res["per_backbone"][bb]["marker_strata"]
        for i, (s, c, _) in enumerate(rows):
            e = st[s][c]
            y = i + (j - 0.5) * 0.3
            a1.errorbar(e["delta"], y, xerr=[[e["delta"] - e["lo"]], [e["hi"] - e["delta"]]], fmt="o", ms=4,
                        color=SERIES[j], capsize=2, label=lab if i == 0 else None)
    a1.set_yticks(range(len(rows)), [r[2] for r in rows])
    a1.axvline(0, color="#999", lw=0.8, ls=":")
    a1.set_ylim(len(rows) - 0.5, -0.5)
    a1.set_xlabel("Δ external balanced accuracy")
    a1.set_title("Which cleaning step helps which images", loc="left", fontsize=9)
    comps = [("b2smp-b2", "+ balanced sampler"), ("b2sty-b2", "+ style augmentation"), ("afp-b2", "+ both (AFP)")]
    for j, (bb, lab) in enumerate(bbs):
        cv = res["per_backbone"][bb]["components_vs_b2"]
        for i, (c, _) in enumerate(comps):
            if c in cv:
                e = cv[c]
                a2.errorbar(e["delta"], i + (j - 0.5) * 0.3, xerr=[[e["delta"] - e["lo"]], [e["hi"] - e["delta"]]],
                            fmt="o", ms=4, color=SERIES[j], capsize=2)
    a2.set_yticks(range(len(comps)), [c[1] for c in comps])
    a2.axvline(0, color="#999", lw=0.8, ls=":")
    a2.set_ylim(len(comps) - 0.5, -0.5)
    a2.set_xlabel("Δ vs B2 (input cleaning only)")
    a2.set_title("Adding device-invariance training to B2", loc="left", fontsize=9)
    h, lab = a1.get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    save(fig, "E9_cleaning_and_components", sources="results/json/E9.json", dpi=300)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figure-only", action="store_true", help="redraw the figure from results/json/E9.json")
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    if a.figure_only:
        figure(json.loads(OUT_JSON.read_text(encoding="utf-8")))
        return
    if a.probes:
        run_probes(a.jobs)
    res = analyse()
    OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
    OUT_MD.write_text(markdown(res), encoding="utf-8")
    figure(res)
    print(markdown(res))


if __name__ == "__main__":
    main()
