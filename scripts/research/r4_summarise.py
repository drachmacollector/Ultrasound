"""
scripts/research/r4_summarise.py  (R4.6 analyses -> results/json/R4_localisation.json)

Reads results/preds/legacy_*_s42/<dataset>__saliency-<method>.parquet for the methods that passed
Gate G3 and reports, with 95% patient-clustered bootstrap CIs (B = 2000):

  P3a        mean EPG − area (true target) and pointing-hit rate by dataset x anatomy x backbone x method;
             Femur − Head contrast (FPDB-test).
  external   HC18 / UCL − FPDB-test, within anatomy.
  correct    correct − incorrect predictions (descriptive), FPDB-test and HC18+UCL.
  E7         share of Grad-CAM energy outside the fan sector and on detected UI, by machine and class
             (FPDB-test, all 5,271 images) and per external dataset / African centre; uniform-map baselines.
  sensitivity QC-flagged landmarks included; femur capsule width 0.05 / 0.15; Head single sub-plane
             target; predicted-class target.

Primary analysis set: landmark images with qc_flag == False and a non-zero saliency map.
Figures: R4_epg_by_class_backbone, R4_saliency_grid, R4_out_sector_by_machine.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.audit.plotting import CLASS_COLORS, INK, NEUTRAL, SERIES, plt, save  # noqa: E402
from src.data.transforms import load_grayscale  # noqa: E402
from src.eval.datasets import load_eval_set  # noqa: E402
from src.eval.scoring import estimate_json  # noqa: E402
from src.eval.stats import RatioMetric, SumsMetric, cluster_bootstrap, safe_div  # noqa: E402
from src.localisation import runner as R  # noqa: E402

OUT_JSON = Path("results/json/R4_localisation.json")
OUT_MD = Path("results/tables/R4_localisation.md")
G3_JSON = Path("results/json/G3.json")
LM_SETS = ("fpdb_test", "hc18", "ucl")
ANATOMIES = ("Head", "Abdomen", "Femur")
MACHINES = ("Aloka", "Voluson E6", "Voluson S10", "Other")
BB_ORDER = list(R.LEGACY)
CAPTION_CREDIT = ("Images: FETAL_PLANES_DB (Burgos-Artizzu et al., 2020); landmark benchmark FP/HC18/UCL "
                  "(CC BY-NC-SA 4.0; HC18 challenge terms).")


def passing_methods() -> list[str]:
    g3 = json.loads(G3_JSON.read_text(encoding="utf-8"))
    return [m for m, d in g3["decision"].items() if d["pass"]]


def load(bb: str, ds: str, method: str) -> pd.DataFrame | None:
    p = R.saliency_path(R.legacy_run_id(bb), ds, method)
    if not p.exists():
        return None
    d = pd.read_parquet(p)
    for c in d.select_dtypes("category").columns:
        d[c] = d[c].astype(str)
    d["qc_flag"] = d["qc_flag"].fillna(False).astype(bool)
    return d


def primary(d: pd.DataFrame, col: str = "epg_minus_area_true", include_qc: bool = False) -> pd.DataFrame:
    keep = d["has_landmarks"].astype(bool) & d[col].notna()
    if not include_qc:
        keep &= ~d["qc_flag"]
    return d[keep]


def mean_ci(d: pd.DataFrame, col: str) -> dict:
    d = d[d[col].notna()]
    return estimate_json(cluster_bootstrap(d, RatioMetric(col, col)))


def diff_ci(d: pd.DataFrame, col: str, group: pd.Series) -> dict:
    """mean(col | group) − mean(col | ~group) in the same cluster resample."""
    g = group.astype(bool).to_numpy()
    x = d[col].to_numpy(dtype=np.float64)
    t = pd.DataFrame({"patient_id": d["patient_id"].to_numpy(), "a": x * g, "na": g.astype(float),
                      "b": x * ~g, "nb": (~g).astype(float)})
    m = SumsMetric(f"diff_{col}", ["a", "na", "b", "nb"], lambda s: safe_div(s["a"], s["na"]) - safe_div(s["b"], s["nb"]))
    out = estimate_json(cluster_bootstrap(t, m))
    out.update({"n_a": int(g.sum()), "n_b": int((~g).sum())})
    return out


# ----------------------------------------------------------------------------- analyses
def p3a(method: str) -> dict:
    out: dict = {}
    for bb in BB_ORDER:
        out[bb] = {}
        for ds in LM_SETS:
            d = load(bb, ds, method)
            if d is None:
                continue
            d = primary(d)
            for a in ANATOMIES:
                x = d[d["anatomy"] == a]
                if len(x) == 0:
                    continue
                out[bb][f"{ds}|{a}"] = {
                    "n": len(x), "n_patients": int(x["patient_id"].nunique()),
                    "epg": float(x["epg_true"].mean()), "area_frac": float(x["area_frac"].mean()),
                    "epg_minus_area": mean_ci(x, "epg_minus_area_true"),
                    "epg_over_area": float(x["epg_over_area_true"].mean()),
                    "epg_in_sector_minus_area": mean_ci(x, "epg_in_sector_minus_area_true"),
                    "hit": mean_ci(x, "hit_true"), "iou": float(x["iou_true"].mean()),
                    "pred_target_epg_minus_area": float(x["epg_minus_area_pred"].mean()),
                }
            f = d[d["dataset"] == "fpdb"] if ds == "fpdb_test" else None
            if f is not None:
                fh = f[f["anatomy"].isin(["Femur", "Head"])]
                out[bb]["fpdb_test|femur_minus_head"] = diff_ci(fh, "epg_minus_area_true", fh["anatomy"] == "Femur")
    return out


def external_vs_fpdb(method: str) -> dict:
    out: dict = {}
    for bb in BB_ORDER:
        f = load(bb, "fpdb_test", method)
        if f is None:
            continue
        f = primary(f)
        out[bb] = {}
        for ds in ("hc18", "ucl"):
            e = load(bb, ds, method)
            if e is None:
                continue
            e = primary(e)
            for a in ANATOMIES:
                ea, fa = e[e["anatomy"] == a], f[f["anatomy"] == a]
                if len(ea) == 0:
                    continue
                both = pd.concat([ea, fa], ignore_index=True)
                out[bb][f"{ds}-fpdb|{a}"] = diff_ci(both, "epg_minus_area_true", both["dataset"] == ds)
    return out


def correct_vs_incorrect(method: str) -> dict:
    out: dict = {}
    for bb in BB_ORDER:
        out[bb] = {}
        for name, sets in (("fpdb_test", ["fpdb_test"]), ("hc18+ucl", ["hc18", "ucl"])):
            parts = [load(bb, ds, method) for ds in sets]
            if any(p is None for p in parts):
                continue
            d = primary(pd.concat(parts, ignore_index=True))
            c = d["correct_full"].astype(bool)
            r = diff_ci(d, "epg_minus_area_true", c)
            r.update({"mean_correct": float(d.loc[c, "epg_minus_area_true"].mean()),
                      "mean_incorrect": float(d.loc[~c, "epg_minus_area_true"].mean()) if (~c).any() else None})
            out[bb][name] = r
    return out


def e7(method: str = "gradcam") -> dict:
    """GT-free: out-of-sector and UI energy shares (true target), with uniform-map baselines."""
    out: dict = {}
    for bb in BB_ORDER:
        d = load(bb, "fpdb_test", method)
        if d is None:
            continue
        d = d[d["out_sector_frac_true"].notna()]
        r: dict = {"all": {"out_sector_frac": mean_ci(d, "out_sector_frac_true"),
                           "out_sector_area": float(d["out_sector_area"].mean()),
                           "overlay_frac": mean_ci(d, "overlay_frac_true"), "ui_area": float(d["ui_area"].mean())}}
        r["by_machine"] = {m: {"n": int((d["source_machine"] == m).sum()),
                               "out_sector_frac": mean_ci(d[d["source_machine"] == m], "out_sector_frac_true"),
                               "out_sector_area": float(d.loc[d["source_machine"] == m, "out_sector_area"].mean())}
                           for m in MACHINES if (d["source_machine"] == m).sum() >= 10}
        al = d[d["source_machine"].isin(["Aloka", "Voluson E6"])]
        r["aloka_minus_e6"] = diff_ci(al, "out_sector_frac_true", al["source_machine"] == "Aloka")
        r["by_class"] = {c: mean_ci(d[d["true_collapsed"] == c], "out_sector_frac_true")
                         for c in sorted(d["true_collapsed"].unique())}
        # within-class machine contrast for Other (the class most entangled with machine, R3)
        o = d[(d["true_collapsed"] == "Other") & d["source_machine"].isin(["Aloka", "Voluson E6"])]
        r["other_aloka_minus_e6"] = diff_ci(o, "out_sector_frac_true", o["source_machine"] == "Aloka")
        r["correct_minus_incorrect"] = diff_ci(d, "out_sector_frac_true", d["correct_full"].astype(bool))
        ext = {}
        for ds in ("hc18", "ucl", "african"):
            e = load(bb, ds, method)
            if e is None:
                continue
            e = e[e["out_sector_frac_true"].notna()]
            ext[ds] = {"n": len(e), "out_sector_frac": mean_ci(e, "out_sector_frac_true"),
                       "out_sector_area": float(e["out_sector_area"].mean())}
            if ds == "african":
                ext["african_by_centre"] = {c: {"n": int((e["centre"] == c).sum()),
                                                "out_sector_frac": mean_ci(e[e["centre"] == c], "out_sector_frac_true"),
                                                "out_sector_area": float(e.loc[e["centre"] == c, "out_sector_area"].mean())}
                                            for c in sorted(e["centre"].unique())}
        r["external"] = ext
        out[bb] = r
    return out


def sensitivity(method: str) -> dict:
    out: dict = {}
    for bb in BB_ORDER:
        f = load(bb, "fpdb_test", method)
        if f is None:
            continue
        r: dict = {}
        for a in ANATOMIES:
            r[f"qc_included|{a}"] = float(primary(f, include_qc=True).query("anatomy == @a")["epg_minus_area_true"].mean())
        fe = primary(f).query("anatomy == 'Femur'")
        for w in ("w05", "w15"):
            r[f"femur_{w}"] = float(fe[f"epg_minus_area_{w}_true"].mean())
        r["femur_w10"] = float(fe["epg_minus_area_true"].mean())
        he = primary(f).query("anatomy == 'Head'")
        if "epg_minus_area_truesub" in he:
            r["head_set"] = float(he["epg_minus_area_true"].mean())
            r["head_single_subplane"] = float(he["epg_minus_area_truesub"].mean())
        p = primary(f, "epg_minus_area_pred")
        r["pred_target_by_anatomy"] = {a: float(p.loc[p["anatomy"] == a, "epg_minus_area_pred"].mean()) for a in ANATOMIES}
        out[bb] = r
    return out


def coverage(methods: list[str]) -> dict:
    cov = {}
    for m in methods:
        for bb in BB_ORDER:
            for ds in ("fpdb_test", "hc18", "ucl", "african"):
                d = load(bb, ds, m)
                if d is None:
                    continue
                cov[f"{m}|{bb}|{ds}"] = {
                    "n": len(d), "n_landmark": int(d["has_landmarks"].sum()),
                    "n_landmark_qc_excluded": int((d["has_landmarks"] & d["qc_flag"]).sum()),
                    "n_zero_map_true": int(d["sal_energy_true"].eq(0).sum()),
                    "r2_agreement": float(d["pred_agrees_r2"].mean()),
                }
    return cov


# ----------------------------------------------------------------------------- figures
def fig_epg(method: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True)
    for ax, a in zip(axes, ANATOMIES):
        pos, labels = [], []
        for j, bb in enumerate(BB_ORDER):
            d = load(bb, "fpdb_test", method)
            if d is None:
                continue
            x = primary(d).query("anatomy == @a")["epg_minus_area_true"].to_numpy()
            vp = ax.violinplot([x], positions=[j], widths=0.8, showextrema=False)
            for b in vp["bodies"]:
                b.set_facecolor(CLASS_COLORS[a])
                b.set_alpha(0.35)
                b.set_edgecolor("none")
            e = mean_ci(primary(d).query("anatomy == @a"), "epg_minus_area_true")
            ax.errorbar([j], [e["point"]], yerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], fmt="o",
                        color=INK, ms=3, capsize=2, lw=1)
            pos.append(j)
            labels.append(R.LEGACY[bb])
        ax.axhline(0, color=NEUTRAL, lw=0.8, ls=":")
        ax.set_xticks(pos, labels, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"FPDB-test {a}", fontsize=8)
    axes[0].set_ylabel("EPG − area fraction (true target)")
    fig.text(0.5, -0.08, "Dot = mean with 95% patient-cluster CI; 0 = no better than a uniform map.",
             ha="center", fontsize=6, color=INK)
    fig.tight_layout()
    save(fig, f"R4_epg_by_class_backbone{'' if method == 'gradcam' else '_' + method}")


def fig_grid(method: str = "gradcam") -> None:
    """2 x 3 example grid (convnext, true target): FPDB Head/Abdomen/Femur and HC18/UCL; correct and incorrect."""
    bb = "convnext_tiny"
    rid = R.legacy_run_id(bb)
    sub = json.loads(Path("results/json/R4_subsets.json").read_text(encoding="utf-8"))["figure"]
    picks = []
    for ds, a, ok in (("fpdb_test", "Head", True), ("fpdb_test", "Abdomen", True), ("fpdb_test", "Femur", True),
                      ("hc18", "Head", False), ("ucl", "Abdomen", True), ("ucl", "Femur", False)):
        d = load(bb, ds, method)
        d = d[d["image_id"].isin(sub) & (d["anatomy"] == a) & (d["correct_full"].astype(bool) == ok)]
        if len(d) == 0:
            d = load(bb, ds, method)
            d = d[d["image_id"].isin(sub) & (d["anatomy"] == a)]
        picks.append((ds, d.sort_values("image_id").iloc[0]))
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.6))
    paths = {ds: load_eval_set(ds).frame.set_index("image_id")["image_path"] for ds in ("fpdb_test", "hc18", "ucl")}
    for ax, (ds, row) in zip(axes.ravel(), picks):
        with np.load(R.saliency_path(rid, ds, method).with_suffix(".maps.npz")) as z:
            key = row["image_id"].replace("/", "__")
            sal = z[f"{key}|true"].astype(np.float32)
            anat = z[f"{key}|anat"]
        g = cv2.resize(load_grayscale(paths[ds][row["image_id"]]), sal.shape[::-1], interpolation=cv2.INTER_LINEAR)
        ax.imshow(g, cmap="gray", vmin=0, vmax=255)
        s = np.maximum(sal, 0)
        ax.imshow(s / s.max() if s.max() > 0 else s, cmap="inferno", alpha=0.45, vmin=0, vmax=1)
        ax.contour(anat.astype(float), levels=[0.5], colors=["#1baf7a"], linewidths=1.0)
        verdict = "correct" if row["correct_full"] else f"predicted {row['pred_full']}"
        ax.set_title(f"{ds.replace('_test', '').upper()} {row['anatomy']} ({verdict})\n"
                     f"EPG {row['epg_true']:.2f}, area {row['area_frac']:.2f}", fontsize=7, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.text(0.5, -0.02, "convnext_tiny Grad-CAM (true target) over the model input; green = expert-landmark mask. "
             + CAPTION_CREDIT, ha="center", fontsize=5.5, color=INK, wrap=True)
    fig.tight_layout()
    save(fig, "R4_saliency_grid")


def fig_out_sector(res_e7: dict) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    groups = [("Aloka", "m"), ("Voluson E6", "m"), ("Voluson S10", "m"), ("hc18", "e"), ("ucl", "e"), ("african", "e")]
    w = 0.13
    for j, bb in enumerate(BB_ORDER):
        if bb not in res_e7:
            continue
        r = res_e7[bb]
        ys, lo, hi = [], [], []
        for g, kind in groups:
            e = r["by_machine"][g]["out_sector_frac"] if kind == "m" else r["external"][g]["out_sector_frac"]
            ys.append(e["point"])
            lo.append(e["point"] - e["lo"])
            hi.append(e["hi"] - e["point"])
        x = np.arange(len(groups)) + (j - 2.5) * w
        # three validated colours, then neutral grey with distinct markers (identity never by colour alone)
        ax.errorbar(x, ys, yerr=[lo, hi], fmt="osD^vP"[j], ms=3, lw=0.8, capsize=1.5,
                    color=SERIES[j] if j < 3 else NEUTRAL, label=R.LEGACY[bb])
    base = [res_e7["convnext_tiny"]["by_machine"][g]["out_sector_area"] if k == "m"
            else res_e7["convnext_tiny"]["external"][g]["out_sector_area"] for g, k in groups]
    ax.scatter(np.arange(len(groups)), base, marker="_", s=300, color=INK, label="uniform map (out-of-sector area)")
    ax.set_xticks(range(len(groups)), ["FPDB Aloka", "FPDB E6", "FPDB S10", "HC18", "UCL", "African"], fontsize=7)
    ax.set_ylabel("share of Grad-CAM energy\noutside the fan sector")
    ax.set_ylim(0, None)
    ax.legend(fontsize=6, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), handletextpad=0.3, columnspacing=0.8)
    fig.tight_layout()
    save(fig, "R4_out_sector_by_machine")


# ----------------------------------------------------------------------------- markdown
def fmt(e: dict, dp: int = 3) -> str:
    return f"{e['point']:.{dp}f} [{e['lo']:.{dp}f}, {e['hi']:.{dp}f}]"


def markdown(res: dict) -> str:
    L = ["# R4: does saliency fall on the fetal anatomy? (legacy seed-42 checkpoints)", "",
         "Primary set: expert-landmark images, QC-flagged landmarks excluded. EPG = share of positive saliency "
         "energy inside the landmark mask; **EPG − area** subtracts the mask's area fraction (0 = no better than a "
         "uniform map). Targets: true class (Head = the three brain sub-planes summed). 95% patient-cluster CIs.", ""]
    for m in res["methods"]:
        L += [f"## {m}: EPG − area and pointing-hit rate", "",
              "| backbone | set | n | EPG | area | EPG − area | hit rate | CAM-box IoU |", "|---|---|---|---|---|---|---|---|"]
        for bb, per in res["p3a"][m].items():
            for k, v in per.items():
                if "|" in k and "minus" not in k:
                    L.append(f"| `{R.LEGACY[bb]}` | {k} | {v['n']} | {v['epg']:.3f} | {v['area_frac']:.3f} | "
                             f"{fmt(v['epg_minus_area'])} | {fmt(v['hit'])} | {v['iou']:.3f} |")
        L += ["", "Femur − Head (FPDB-test), EPG − area:", ""]
        for bb, per in res["p3a"][m].items():
            if "fpdb_test|femur_minus_head" in per:
                L.append(f"* `{R.LEGACY[bb]}`: {fmt(per['fpdb_test|femur_minus_head'])}")
        L += ["", "External − FPDB-test within anatomy, EPG − area:", "", "| backbone | contrast | Δ [95% CI] | n (ext / fpdb) |",
              "|---|---|---|---|"]
        for bb, per in res["external_vs_fpdb"][m].items():
            for k, v in per.items():
                L.append(f"| `{R.LEGACY[bb]}` | {k} | {fmt(v)} | {v['n_a']} / {v['n_b']} |")
        L += ["", "Correct − incorrect predictions, EPG − area (descriptive):", "",
              "| backbone | set | mean correct | mean incorrect | Δ [95% CI] | n correct / incorrect |", "|---|---|---|---|---|---|"]
        for bb, per in res["correct_vs_incorrect"][m].items():
            for k, v in per.items():
                mi = "–" if v["mean_incorrect"] is None else f"{v['mean_incorrect']:.3f}"
                L.append(f"| `{R.LEGACY[bb]}` | {k} | {v['mean_correct']:.3f} | {mi} | {fmt(v)} | {v['n_a']} / {v['n_b']} |")
        L.append("")
    L += ["## E7: Grad-CAM energy outside the fan sector (GT-free, true target)", "",
          "Uniform-map baseline = out-of-sector area fraction of the model input.", "",
          "| backbone | FPDB all | FPDB Aloka | FPDB E6 | FPDB S10 | Aloka − E6 | Other: Aloka − E6 | HC18 | UCL | African |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for bb, r in res["e7"].items():
        bm = r["by_machine"]
        L.append(f"| `{R.LEGACY[bb]}` | {fmt(r['all']['out_sector_frac'])} | {fmt(bm['Aloka']['out_sector_frac'])} | "
                 f"{fmt(bm['Voluson E6']['out_sector_frac'])} | {fmt(bm['Voluson S10']['out_sector_frac'])} | "
                 f"{fmt(r['aloka_minus_e6'])} | {fmt(r['other_aloka_minus_e6'])} | "
                 f"{fmt(r['external']['hc18']['out_sector_frac'])} | {fmt(r['external']['ucl']['out_sector_frac'])} | "
                 f"{fmt(r['external']['african']['out_sector_frac'])} |")
    c = res["e7"]["convnext_tiny"]
    L += ["", "Uniform baselines (out-of-sector area, convnext inputs): FPDB all "
          f"{c['all']['out_sector_area']:.3f}; Aloka {c['by_machine']['Aloka']['out_sector_area']:.3f}; "
          f"E6 {c['by_machine']['Voluson E6']['out_sector_area']:.3f}; S10 {c['by_machine']['Voluson S10']['out_sector_area']:.3f}; "
          f"HC18 {c['external']['hc18']['out_sector_area']:.3f}; UCL {c['external']['ucl']['out_sector_area']:.3f}; "
          f"African {c['external']['african']['out_sector_area']:.3f}.", ""]
    L += ["## Sensitivity (FPDB-test, mean EPG − area)", "",
          "| backbone | Head set | Head single sub-plane | Femur w0.05 | w0.10 | w0.15 | QC incl. Head/Abd/Femur | predicted-class target Head/Abd/Femur |",
          "|---|---|---|---|---|---|---|---|"]
    for bb, r in res["sensitivity"][res["methods"][0]].items():
        pt = r["pred_target_by_anatomy"]
        L.append(f"| `{R.LEGACY[bb]}` | {r.get('head_set', float('nan')):.3f} | {r.get('head_single_subplane', float('nan')):.3f} | "
                 f"{r['femur_w05']:.3f} | {r['femur_w10']:.3f} | {r['femur_w15']:.3f} | "
                 f"{r['qc_included|Head']:.3f} / {r['qc_included|Abdomen']:.3f} / {r['qc_included|Femur']:.3f} | "
                 f"{pt['Head']:.3f} / {pt['Abdomen']:.3f} / {pt['Femur']:.3f} |")
    L += ["", "Not run: the plan's 'images with vs without detected overlays' split needs the caliper detector, "
          "which failed Gate G1; UI text is present on every FPDB frame, so there is no without-overlay group.", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--figures-only", action="store_true")
    a = ap.parse_args()
    methods = passing_methods()
    if a.figures_only:
        res = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    else:
        res = {"methods": methods, "coverage": coverage(methods),
               "p3a": {m: p3a(m) for m in methods},
               "external_vs_fpdb": {m: external_vs_fpdb(m) for m in methods},
               "correct_vs_incorrect": {m: correct_vs_incorrect(m) for m in methods},
               "sensitivity": {m: sensitivity(m) for m in methods},
               "e7": e7("gradcam") if "gradcam" in methods else {}}
        OUT_JSON.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        OUT_MD.write_text(markdown(res), encoding="utf-8")
    for m in methods:
        fig_epg(m)
    fig_grid(methods[0])
    if res["e7"]:
        fig_out_sector(res["e7"])
    print("redrew figures" if a.figures_only else f"wrote {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    main()
