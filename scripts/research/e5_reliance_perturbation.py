"""
scripts/research/e5_reliance_perturbation.py  (R3 E5: do unchanged models *use* non-anatomical content? P2b)

Re-predicts FPDB test with test-time perturbations (src/audit/perturb.py) and compares, image by
image, with the unperturbed predictions of the same frozen checkpoint (paired on image_id,
patient-clustered bootstrap).

    a_text, b_zero, b_mean, b_blur     all six backbones
    e_ui_transplant (exploratory)      convnext_tiny, tf_efficientnetv2_s; Aloka <-> Voluson E6 donors
                                       drawn from the train split

Not run (Gate G1): a_marker / c_control (primary marker-reliance contrast; detector recall 0.20,
0 usable test images) and d_pairs (0 caliper pairs).

Metrics per perturbation: flip rate (pred_fine, pred_full), mean Δ p(true class), mean Δ p(Other),
Δ macro-F1 and Δ Other recall (paired), mean Δ p(Other) by machine (directional device-prior test),
breakdown machine × collapsed class. Out-of-sector reliance is claimed only if the three b_* fills
agree in sign (all CIs on the same side of 0) and are of similar size.

Outputs: results/preds/legacy_*/fpdb_test__perturb-*.parquet, results/json/E5.json,
         results/tables/E5.md, results/figures/E5_reliance.{pdf,png}
Usage:   python scripts/research/e5_reliance_perturbation.py [--predict] [--score]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.audit.perturb import KINDS, Perturbation, UITransplant, pick_donors  # noqa: E402
from src.audit.plotting import NEUTRAL, SERIES, plt, save  # noqa: E402
from src.data.dataset import CANONICAL_CLASSES  # noqa: E402
from src.data.labels import FINE_TO_COLLAPSED  # noqa: E402
from src.eval.predict import PREDS_ROOT, predict  # noqa: E402
from src.eval.scoring import estimate_json, macro_f1_8, other_recall_metric, with_other8  # noqa: E402
from src.eval.stats import RatioMetric, cluster_bootstrap, paired_cluster_bootstrap  # noqa: E402

BACKBONES = {"convnext_tiny": "convnext", "tf_efficientnetv2_s": "effv2s", "repvgg_a1": "repvgga1",
             "repvgg_a2": "repvgga2", "mobilenetv3_large_100": "mnv3l", "efficientnet_lite0": "efflite0"}
TRANSPLANT_ON = ("convnext_tiny", "tf_efficientnetv2_s")
META = Path("data/processed/metadata/acquisition_metadata.parquet")
MACHINES = ["Aloka", "Voluson E6", "Voluson S10", "Other"]
OUT_JSON = Path("results/json/E5.json")
OUT_MD = Path("results/tables/E5.md")


def run_dir(bb: str) -> Path:
    return PREDS_ROOT / f"legacy_{BACKBONES[bb]}_s42"


def donors() -> dict:
    meta = pd.read_parquet(META, columns=["image_id", "image_path", "source_machine", "split", "dataset"])
    return pick_donors(meta[meta["split"] == "test"], meta[meta["split"] == "train"], seed=0)


def run_predictions(num_workers: int) -> None:
    jobs = [(bb, Perturbation(k)) for bb in BACKBONES for k in KINDS]
    tp = UITransplant(donors())
    jobs += [(bb, tp) for bb in TRANSPLANT_ON]
    for bb, fn in jobs:
        pq = run_dir(bb) / f"fpdb_test__{fn.variant}.parquet"
        if pq.exists():
            continue
        predict(Path("checkpoints") / bb / "best.pt", "fpdb_test", run_dir(bb), run_id=f"legacy_{BACKBONES[bb]}_s42",
                perturb=fn, variant=fn.variant, num_workers=num_workers)
        print(f"  {bb} {fn.variant}", flush=True)


def load(path: Path) -> pd.DataFrame:
    f = pd.read_parquet(path)
    for c in f.select_dtypes("category").columns:
        f[c] = f[c].astype(str)
    return f.sort_values("image_id").reset_index(drop=True)


def paired_frame(clean: pd.DataFrame, pert: pd.DataFrame) -> pd.DataFrame:
    """One row per image with flip indicators and probability changes (pert − clean)."""
    assert np.array_equal(clean["image_id"].to_numpy(), pert["image_id"].to_numpy())
    true_idx = clean["true_label"].map(CANONICAL_CLASSES.index).to_numpy()
    pc = clean[[f"prob_{i}" for i in range(8)]].to_numpy(np.float64)
    pp = pert[[f"prob_{i}" for i in range(8)]].to_numpy(np.float64)
    rows = np.arange(len(clean))
    return pd.DataFrame({
        "image_id": clean["image_id"], "patient_id": clean["patient_id"], "source_machine": clean["source_machine"],
        "true_collapsed": clean["true_label"].map(FINE_TO_COLLAPSED),
        "flip_fine": (clean["pred_fine"] != pert["pred_fine"]).astype(int),
        "flip_full": (clean["pred_full"] != pert["pred_full"]).astype(int),
        "d_p_true": pp[rows, true_idx] - pc[rows, true_idx],
        "d_p_other": pp[:, 7] - pc[:, 7],
    })


def score_variant(clean: pd.DataFrame, pert: pd.DataFrame) -> dict:
    d = paired_frame(clean, pert)
    out = {k: cluster_bootstrap(d, RatioMetric(k, k)) for k in ("flip_fine", "flip_full", "d_p_true", "d_p_other")}
    out["n_images"] = len(d)
    out["delta_macro_f1_8"] = paired_cluster_bootstrap(clean, pert, macro_f1_8())
    out["delta_other_recall"] = paired_cluster_bootstrap(with_other8(clean), with_other8(pert), other_recall_metric())
    out["d_p_other_by_machine"] = {}
    out["flip_fine_by_machine"] = {}
    for m in MACHINES:
        g = d[d["source_machine"] == m]
        if g["patient_id"].nunique() >= 10:
            out["d_p_other_by_machine"][m] = cluster_bootstrap(g, RatioMetric("d_p_other", "d_p_other"))
            out["flip_fine_by_machine"][m] = cluster_bootstrap(g, RatioMetric("flip_fine", "flip_fine"))
    cell = d.groupby(["source_machine", "true_collapsed"]).agg(
        n=("flip_fine", "size"), flip_fine=("flip_fine", "mean"), d_p_other=("d_p_other", "mean")).reset_index()
    out["by_machine_x_class"] = cell.round(4).to_dict(orient="records")
    return out


def consistency(per_variant: dict) -> dict:
    """b_* fills agree in sign (every CI excludes 0 on the same side) for mean Δ p(Other) and flip rate."""
    out = {}
    for key in ("d_p_other", "flip_fine"):
        es = [per_variant[f"perturb-{k}"][key] for k in ("b_zero", "b_mean", "b_blur")]
        pos = all(e.lo > 0 for e in es)
        neg = all(e.hi < 0 for e in es)
        pts = [e.point for e in es]
        out[key] = {"points": pts, "same_sign_all_ci_exclude_0": bool(pos or neg),
                    "max_over_min_abs": float(max(map(abs, pts)) / max(min(map(abs, pts)), 1e-9))}
    return out


def coverage() -> dict:
    """What the perturbations can reach: out-of-sector share of the frame and of the detected UI, per machine."""
    from src.audit.metadata import load_masks, mask_path
    from src.audit.perturb import region

    meta = pd.read_parquet(META, columns=["image_id", "split", "source_machine"])
    rows = []
    for r in meta[meta["split"] == "test"].itertuples(index=False):
        k = load_masks(mask_path(r.image_id, "fpdb"))
        ui, out = k["ui"].astype(bool), region("b_zero", k)
        rows.append({"machine": r.source_machine, "outside_frac": float(out.mean()),
                     "a_text_frac": float(region("a_text", k).mean()),
                     "ui_outside_share": float((ui & out).sum() / max(int(ui.sum()), 1))})
    d = pd.DataFrame(rows)
    q = lambda s: {"median": round(float(s.median()), 4), "p10": round(float(s.quantile(0.1)), 4),  # noqa: E731
                   "p90": round(float(s.quantile(0.9)), 4)}
    return {"all": {c: q(d[c]) for c in ("outside_frac", "a_text_frac", "ui_outside_share")},
            "by_machine": {m: {c: q(g[c]) for c in ("outside_frac", "a_text_frac", "ui_outside_share")}
                           for m, g in d.groupby("machine")}}


def score() -> dict:
    res: dict = {"coverage": coverage(), "models": {}, "not_run": {
        "a_marker/c_control": "Gate G1: caliper detector recall 0.20; 0 detector-and-audit-positive test images",
        "d_pairs": "Gate G1: 0 caliper pair candidates"}}
    for bb in BACKBONES:
        clean = load(run_dir(bb) / "fpdb_test.parquet")
        r = {}
        for pq in sorted(run_dir(bb).glob("fpdb_test__perturb-*.parquet")):
            r[pq.stem.split("__")[1]] = score_variant(clean, load(pq))
        res["models"][bb] = {"variants": r, "b_consistency": consistency(r)}
        print(f"  scored {bb}", flush=True)
    return res


def pct(e) -> str:
    return e.fmt(pct=True)


def markdown(res: dict) -> str:
    M = res["models"]
    variants = [f"perturb-{k}" for k in KINDS]
    s = ["# E5: reliance on non-anatomical content (test-time perturbations, unchanged models)", "",
         "FPDB test (5,271 images). Paired with the unperturbed predictions of the same checkpoint; 95% "
         "patient-clustered CIs (B = 2000). Perturbations change only out-of-sector pixels (`a_text`: the UI/text "
         "mask; `b_*`: the whole out-of-sector region). Not run (Gate G1): " +
         "; ".join(f"`{k}` ({v})" for k, v in res["not_run"].items()) + ".", "",
         "## 0. What the perturbations reach (FPDB test, median [p10, p90] share of the frame)", "",
         "| machine | out-of-sector region (`b_*`) | `a_text` region | detected UI lying outside the sector |",
         "|---|---|---|---|"]
    cov = res["coverage"]
    for m, x in {**cov["by_machine"], "all": cov["all"]}.items():
        s.append(f"| {m} | " + " | ".join(f"{100 * x[c]['median']:.1f}% [{100 * x[c]['p10']:.1f}, {100 * x[c]['p90']:.1f}]"
                                          for c in ("outside_frac", "a_text_frac", "ui_outside_share")) + " |")
    s += ["", "**Scope.** The perturbations remove all *detected* UI and out-of-sector content. Text and markers drawn "
          "on top of the anatomy (common on zoomed frames, where the sector fills the image) are not detected at pixel "
          "level and are left in place, so E5 measures reliance on out-of-sector content, not on every on-screen "
          "character.", "",
         "## 1. Flip rate of the 8-way prediction", "",
         "| model | " + " | ".join(f"`{v.split('-', 1)[1]}`" for v in variants) + " |", "|---|" + "---|" * len(variants)]
    for bb, r in M.items():
        s.append(f"| `{bb}` | " + " | ".join(pct(r["variants"][v]["flip_fine"]) for v in variants) + " |")
    s += ["", "## 2. Mean change in p(Other) (percentage points)", "",
          "| model | " + " | ".join(f"`{v.split('-', 1)[1]}`" for v in variants) + " |", "|---|" + "---|" * len(variants)]
    for bb, r in M.items():
        s.append(f"| `{bb}` | " + " | ".join(pct(r["variants"][v]["d_p_other"]) for v in variants) + " |")
    s += ["", "## 3. Δ macro-F1 and Δ Other recall (perturbed − clean; paired cluster bootstrap)", "",
          "| model | " + " | ".join(f"`{v.split('-', 1)[1]}` ΔF1 / ΔOther recall" for v in variants) + " |",
          "|---|" + "---|" * len(variants)]
    for bb, r in M.items():
        s.append(f"| `{bb}` | " + " | ".join(
            f"{r['variants'][v]['delta_macro_f1_8'].delta:+.3f} [{r['variants'][v]['delta_macro_f1_8'].lo:+.3f}, "
            f"{r['variants'][v]['delta_macro_f1_8'].hi:+.3f}]<br>{100 * r['variants'][v]['delta_other_recall'].delta:+.1f} pp "
            f"[{100 * r['variants'][v]['delta_other_recall'].lo:+.1f}, {100 * r['variants'][v]['delta_other_recall'].hi:+.1f}]"
            for v in variants) + " |")
    s += ["", "## 4. Directional device-prior test: mean Δ p(Other) by machine", "",
          "| model | perturbation | " + " | ".join(MACHINES[:3]) + " |", "|---|---|---|---|---|"]
    for bb, r in M.items():
        for v, x in r["variants"].items():
            s.append(f"| `{bb}` | `{v.split('-', 1)[1]}` | " + " | ".join(
                pct(x["d_p_other_by_machine"][m]) if m in x["d_p_other_by_machine"] else "–" for m in MACHINES[:3]) + " |")
    s += ["", "## 5. Consistency across the three out-of-sector fills (`b_zero`, `b_mean`, `b_blur`)", "",
          "| model | Δ p(Other): same sign, all CIs exclude 0 | max/min |Δ| | flip rate: same sign | max/min |",
          "|---|---|---|---|---|"]
    for bb, r in M.items():
        c = r["b_consistency"]
        s.append(f"| `{bb}` | {c['d_p_other']['same_sign_all_ci_exclude_0']} | {c['d_p_other']['max_over_min_abs']:.1f} | "
                 f"{c['flip_fine']['same_sign_all_ci_exclude_0']} | {c['flip_fine']['max_over_min_abs']:.1f} |")
    s += ["", "Flip rates are always > 0 by construction, so their 'same sign' is trivially true; the size ratio is "
          "the informative part. `e_ui_transplant` is exploratory (convnext, efficientnetv2 only).", ""]
    return "\n".join(s)


def figure() -> None:
    """E5_reliance from results/json/E5.json (Estimates stored as dicts)."""
    M = json.loads(OUT_JSON.read_text(encoding="utf-8"))["models"]
    variants = [f"perturb-{k}" for k in KINDS]
    models = list(M)
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    for ax, key, title in ((axes[0], "flip_fine", "Flip rate of the 8-way prediction"),
                           (axes[1], "d_p_other", "Mean change in p(Other)")):
        for j, v in enumerate(variants):
            col = SERIES[j] if j < 3 else NEUTRAL
            for i, bb in enumerate(models):
                e = M[bb]["variants"][v][key]
                y = i + (j - 1.5) * 0.18
                ax.errorbar(e["point"], y, xerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], fmt="o", ms=3.5,
                            color=col, elinewidth=1.1, label=v.split("-", 1)[1] if i == 0 else None)
        ax.axvline(0, color="#cfcfcb", lw=0.8)
        ax.set_title(title, loc="left")
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{100 * x:.1f}%"))
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=7, title="out-of-sector perturbation", title_fontsize=7,
               bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Effect of removing out-of-sector content on the frozen models' predictions (FPDB test, 95% CI)",
                 x=0.01, y=1.02, ha="left", fontsize=9)
    fig.tight_layout()
    save(fig, "E5_reliance")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--figure", action="store_true", help="redraw the figure from results/json/E5.json")
    ap.add_argument("--num-workers", type=int, default=6)
    a = ap.parse_args()
    if a.predict:
        run_predictions(a.num_workers)
    if a.figure and not a.score:
        figure()
    if a.score:
        res = score()
        OUT_JSON.write_text(json.dumps(estimate_json(res), indent=1), encoding="utf-8")
        OUT_MD.write_text(markdown(res), encoding="utf-8")
        figure()
        print(f"wrote {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    main()
