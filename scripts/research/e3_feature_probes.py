"""
scripts/research/e3_feature_probes.py  (R3 E3: is acquisition information decodable from features? P1b)

Linear probes (src/audit/probes.py: patient-grouped nested CV, balanced class weights) on the
penultimate features of the six frozen checkpoints (fine-tuned) and of the same backbones with
ImageNet weights only, FPDB test split.

Targets
  machine4              Aloka / Voluson E6 / Voluson S10 / Other machine
  machine3              without the 'Other' machine (n = 57)
  operator4             Op. 1 / Op. 2 / Op. 3 / Other
  machine|<class>       machine decoded *within* one true class (Fetal_abdomen, Fetal_femur,
                        Brain_Trans_thalamic, Other); machines with >= 30 images in that class.
                        Controls for decoding machine *through* the class.
  class|<machine>       8-way class within one machine (sanity; convnext and efficientnetv2 only)

Metric: out-of-fold balanced accuracy, 95% patient-clustered CI; chance = 1/k.
Contrast: fine-tuned − ImageNet within-class machine decoding (mean over the four classes),
paired over images (same OOF images, same resamples).

Outputs: results/preds/imagenet_<short>/fpdb_test.features.npy, results/json/E3.json,
         results/tables/E3.md, results/figures/E3_decodability.{pdf,png}
Usage:   python scripts/research/e3_feature_probes.py [--extract] [--probe] [--jobs 12]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Probe workers re-import this module (Windows spawn): keep top-level imports free of torch
# (each torch import commits ~2 GB of virtual memory) and pin BLAS to one thread per worker.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, ".")
from src.audit.probes import load_features, probe_oof  # noqa: E402
from src.eval import metrics as M  # noqa: E402
from src.eval.stats import CMMetric, cluster_bootstrap, paired_cluster_bootstrap  # noqa: E402

PREDS_ROOT = Path("results/preds")

BACKBONES = {"convnext_tiny": "convnext", "tf_efficientnetv2_s": "effv2s", "repvgg_a1": "repvgga1",
             "repvgg_a2": "repvgga2", "mobilenetv3_large_100": "mnv3l", "efficientnet_lite0": "efflite0"}
SANITY_ON = ("convnext_tiny", "tf_efficientnetv2_s")
WITHIN_CLASSES = ("Fetal_abdomen", "Fetal_femur", "Brain_Trans_thalamic", "Other")
MIN_CELL = 30
FEATURE_TYPES = ("finetuned", "imagenet")
CACHE = Path("results/preds/_e3_oof")          # per-probe OOF predictions (resumable)
OUT_JSON = Path("results/json/E3.json")
OUT_MD = Path("results/tables/E3.md")


def feat_dir(bb: str, ftype: str) -> Path:
    return PREDS_ROOT / (f"legacy_{BACKBONES[bb]}_s42" if ftype == "finetuned" else f"imagenet_{BACKBONES[bb]}")


def extract() -> None:
    from src.audit.probes import extract_imagenet_features

    for bb in BACKBONES:
        out = feat_dir(bb, "imagenet")
        if not (out / "fpdb_test.features.npy").exists():
            extract_imagenet_features(Path("checkpoints") / bb / "best.pt", out)
            print(f"  imagenet features {bb}", flush=True)


def test_meta() -> pd.DataFrame:
    f = pd.read_parquet(PREDS_ROOT / "legacy_convnext_s42" / "fpdb_test.parquet",
                        columns=["image_id", "patient_id", "source_machine", "operator", "true_label"])
    return f.astype({c: str for c in f.columns}).set_index("image_id")


def tasks(meta: pd.DataFrame) -> dict[str, pd.Series]:
    """target name -> label Series (index = image_ids in the probe)."""
    t = {"machine4": meta["source_machine"],
         "machine3": meta.loc[meta["source_machine"] != "Other", "source_machine"],
         "operator4": meta["operator"]}
    for c in WITHIN_CLASSES:
        g = meta[meta["true_label"] == c]
        keep = g["source_machine"].value_counts()[lambda s: s >= MIN_CELL].index
        t[f"machine|{c}"] = g.loc[g["source_machine"].isin(keep), "source_machine"]
    for m in ("Aloka", "Voluson E6", "Voluson S10"):
        t[f"class|{m}"] = meta.loc[meta["source_machine"] == m, "true_label"]
    return t


def _run_probe(args: tuple[str, str, str]) -> str:
    bb, ftype, target = args
    out = CACHE / f"{BACKBONES[bb]}__{ftype}__{target.replace('|', '_').replace(' ', '_')}.parquet"
    if out.exists():
        return str(out)
    meta = test_meta()
    y = tasks(meta)[target]
    X = load_features(feat_dir(bb, ftype)).loc[y.index].to_numpy(np.float64)
    t0 = time.perf_counter()
    oof, Cs = probe_oof(X, y.to_numpy(), meta.loc[y.index, "patient_id"].to_numpy())
    CACHE.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": y.index, "patient_id": meta.loc[y.index, "patient_id"].to_numpy(),
                  "y": y.to_numpy(), "oof": oof, "C_by_fold": [",".join(map(str, Cs))] * len(y),
                  "seconds": time.perf_counter() - t0}).to_parquet(out, index=False)
    return str(out)


def all_jobs() -> list[tuple[str, str, str]]:
    names = list(tasks(test_meta()))
    jobs = []
    for bb in BACKBONES:
        for ft in FEATURE_TYPES:
            for t in names:
                if t.startswith("class|") and bb not in SANITY_ON:
                    continue
                jobs.append((bb, ft, t))
    return jobs


def run_probes(n_jobs: int) -> None:
    from concurrent.futures import ProcessPoolExecutor

    jobs = all_jobs()
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for i, p in enumerate(ex.map(_run_probe, jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(f"  probes {i}/{len(jobs)}", flush=True)


def bal_acc(classes: list[str]) -> CMMetric:
    return CMMetric("oof_bal_acc", "y", "oof", classes, M.balanced_accuracy)


def within_mean(df: pd.DataFrame) -> float:
    """Mean over classes of the within-class machine-decoding balanced accuracy."""
    vals = []
    for _, g in df.groupby("cls"):
        cls = sorted(set(g["y"]) | set(g["oof"]))
        vals.append(M.balanced_accuracy(M.confusion(g["y"].map(cls.index), g["oof"].map(cls.index), len(cls))))
    return float(np.mean(vals))


def score() -> dict:
    res: dict = {"models": {}, "min_cell": MIN_CELL}
    for bb in BACKBONES:
        r: dict = {}
        within: dict = {}
        for ft in FEATURE_TYPES:
            r[ft] = {}
            frames = []
            for (b, f, t) in all_jobs():
                if (b, f) != (bb, ft):
                    continue
                d = pd.read_parquet(CACHE / f"{BACKBONES[bb]}__{ft}__{t.replace('|', '_').replace(' ', '_')}.parquet")
                classes = sorted(d["y"].unique())
                est = cluster_bootstrap(d, bal_acc(classes), max_nan_frac=0.01)
                r[ft][t] = {"bal_acc": est, "k": len(classes), "chance": 1 / len(classes), "classes": classes,
                            "n_images": len(d), "C_by_fold": d["C_by_fold"].iloc[0],
                            "seconds": round(float(d["seconds"].iloc[0]), 1)}
                if t.startswith("machine|"):
                    frames.append(d.assign(cls=t.split("|", 1)[1]))
            within[ft] = pd.concat(frames, ignore_index=True)
            r[ft]["within_class_mean"] = cluster_bootstrap(within[ft], within_mean, max_nan_frac=0.01)
        r["finetuned_minus_imagenet_within_class"] = paired_cluster_bootstrap(within["imagenet"], within["finetuned"],
                                                                              within_mean)
        res["models"][bb] = r
        print(f"  scored {bb}", flush=True)
    return res


def markdown(res: dict) -> str:
    M_ = res["models"]
    tnames = [t for t in M_["convnext_tiny"]["finetuned"] if t != "within_class_mean"]
    s = ["# E3: decodability of acquisition information from classifier features (FPDB test)", "",
         "Linear probes, patient-grouped nested CV (outer StratifiedGroupKFold 5, inner GroupKFold 3 choosing C in "
         "{0.01, 0.1, 1, 10}), balanced class weights. Out-of-fold balanced accuracy, 95% patient-clustered CI. "
         f"`machine|<class>`: machine decoded within one true class (machines with ≥ {res['min_cell']} images in "
         "that class), which removes decoding *through* the class.", ""]
    for ft in FEATURE_TYPES:
        s += [f"## {'Fine-tuned (frozen legacy checkpoints)' if ft == 'finetuned' else 'ImageNet-only weights'}", "",
              "| target (k, chance) | " + " | ".join(f"`{b}`" for b in M_) + " |", "|---|" + "---|" * len(M_)]
        for t in tnames:
            x0 = M_["convnext_tiny"][ft][t]
            row = [f"{t} (k={x0['k']}, {x0['chance']:.2f})"]
            for b in M_:
                row.append(M_[b][ft][t]["bal_acc"].fmt() if t in M_[b][ft] else "–")
            s.append("| " + " | ".join(row) + " |")
        s.append("| **within-class machine, mean of 4 classes** | " +
                 " | ".join(M_[b][ft]["within_class_mean"].fmt() for b in M_) + " |")
        s.append("")
    s += ["## Fine-tuned − ImageNet, within-class machine decoding (paired over images)", "",
          "| model | Δ balanced accuracy [95% CI] | p |", "|---|---|---|"]
    for b, r in M_.items():
        x = r["finetuned_minus_imagenet_within_class"]
        s.append(f"| `{b}` | {x.delta:+.3f} [{x.lo:+.3f}, {x.hi:+.3f}] | {x.p_two_sided:.3g} |")
    s += ["", "`class|<machine>` probes are a sanity check (should be high); run for convnext and efficientnetv2 only.",
          ""]
    return "\n".join(s)


def figure() -> None:
    """E3_decodability from results/json/E3.json (Estimates stored as dicts)."""
    from src.audit.plotting import NEUTRAL, SERIES, plt, save

    M_ = json.loads(OUT_JSON.read_text(encoding="utf-8"))["models"]
    models = list(M_)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), sharey=True)
    for ax, key, title in ((axes[0], "machine4", "Machine (4-way), all test images"),
                           (axes[1], "within_class_mean", "Machine within one true class (mean of 4 classes)")):
        for j, (ft, col) in enumerate(zip(FEATURE_TYPES, SERIES)):
            for i, bb in enumerate(models):
                x = M_[bb][ft][key]
                e = x["bal_acc"] if "bal_acc" in x else x
                y = i + (j - 0.5) * 0.38
                ax.barh(y, e["point"], height=0.36, color=col, edgecolor="white",
                        label=("fine-tuned (frozen legacy model)" if ft == "finetuned" else "ImageNet weights only")
                        if i == 0 else None)
                ax.errorbar(e["point"], y, xerr=[[e["point"] - e["lo"]], [e["hi"] - e["point"]]], fmt="none",
                            ecolor="#2b2b2a", elinewidth=0.9)
        chance = 1 / (M_["convnext_tiny"]["finetuned"]["machine4"]["k"] if key == "machine4" else 3)
        ax.axvline(chance, color=NEUTRAL, ls="--", lw=0.8)
        ax.annotate("chance", (chance, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -10),
                    textcoords="offset points", fontsize=6.5, color=NEUTRAL)
        ax.set_xlim(0, 1)
        ax.set_title(title, loc="left")
        ax.set_xlabel("out-of-fold balanced accuracy (95% CI)")
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(models)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=7, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("The acquisition machine is linearly decodable from classifier features, even within one class; "
                 "fine-tuning reduces it", x=0.01, y=1.03, ha="left", fontsize=9)
    fig.tight_layout()
    save(fig, "E3_decodability")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--figure", action="store_true", help="redraw the figure from results/json/E3.json")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    if a.extract:
        extract()
    if a.probe:
        run_probes(a.jobs)
    if a.figure and not a.score:
        figure()
    if a.score:
        from src.eval.scoring import estimate_json

        res = score()
        OUT_JSON.write_text(json.dumps(estimate_json(res), indent=1), encoding="utf-8")
        OUT_MD.write_text(markdown(res), encoding="utf-8")
        figure()
        print(f"wrote {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    main()
