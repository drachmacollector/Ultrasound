"""
scripts/research/r5_evaluate_matrix.py  (R5.7: evaluation of every finished R5 run)

For each run with DONE (runs/<arm>_<convnext|effv2s>_s<seed>/), with the run's own preprocessing:

  --predict    fpdb_val, fpdb_test, hc18, ucl, african -> results/preds/<run_id>/ (features on test + externals);
               letterbox arms also: fpdb_test with the other letterbox mode (marker sensitivity)
  --probes     within-class machine decodability on fpdb_test features (R3 E3 protocol; P4b primary index)
  --saliency   Grad-CAM (true and predicted target) on the landmark images + GT-free metrics (R4 pipeline)
  --score      calibration (temperature on val; ECE in/out of distribution, before/after), selective prediction,
               shortcut indices, EPG summaries, rejection vs confusion per arm (seed-averaged; PREREGISTRATION
               §6.3), the confirmatory P3b on R5 models with the African ground-truth-free monitor (§6.7), and the
               double-reweighting sensitivity run (§6.9)
               -> results/json/R5_evaluation.json, results/tables/R5_evaluation.md

The primary endpoint is computed separately by the pre-registered scripts/research/r5_primary_endpoint.py.
Refuses to run before Gate G4 (same check as r5_run_matrix.py). One GPU job at a time.
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
ARMS = ("b0", "b1", "b2", "afp")
BACKBONES = {"convnext_tiny": "convnext", "tf_efficientnetv2_s": "effv2s"}
SEEDS = (11, 22, 33)
DATASETS = ("fpdb_val", "fpdb_test", "hc18", "ucl", "african")
FEATURES_ON = {"fpdb_test", "hc18", "ucl", "african"}
EXTERNAL = ("hc18", "ucl", "african")
RUNS = Path("runs")
OUT_JSON = Path("results/json/R5_evaluation.json")
OUT_MD = Path("results/tables/R5_evaluation.md")
OTHER_LETTERBOX = {"sector_letterbox": "sector_letterbox_inpaint", "sector_letterbox_inpaint": "sector_letterbox"}
POOL_CLASSES = ("Head", "Abdomen", "Femur", "Thorax")      # as in r5_primary_endpoint.py
SENSITIVITY = ("afp_cw", "convnext_tiny", 11)              # PREREGISTRATION §6.9: AFP + class weights, one seed
ERROR_SHARES = ("err_other", "err_absent", "err_confusion")


def finished_runs(sensitivity: bool = False) -> list[tuple[str, str, int, str]]:
    out = []
    for s in SEEDS:
        for arm in ARMS:
            for bb, short in BACKBONES.items():
                rid = f"{arm}_{short}_s{s}"
                if (RUNS / rid / "DONE").exists():
                    out.append((arm, bb, s, rid))
    arm, bb, s = SENSITIVITY
    rid = f"{arm}_{BACKBONES[bb]}_s{s}"
    if sensitivity and (RUNS / rid / "DONE").exists():
        out.append((arm, bb, s, rid))
    return out


def _root() -> Path:
    from src.eval.predict import PREDS_ROOT       # looked up at call time (tests point it elsewhere)

    return PREDS_ROOT


def load(rid: str, stem: str) -> pd.DataFrame:
    d = pd.read_parquet(_root() / rid / f"{stem}.parquet")
    for c in d.select_dtypes("category").columns:
        d[c] = d[c].astype(str)
    return d


def arm_frames(arm: str, short: str, ds: str) -> list[pd.DataFrame] | None:
    """The three seeds' prediction frames of one arm, or None unless all three exist."""
    ids = [f"{arm}_{short}_s{s}" for s in SEEDS]
    if not all((_root() / i / f"{ds}.parquet").exists() for i in ids):
        return None
    return [load(i, ds) for i in ids]


# ----------------------------------------------------------------------------- GPU steps
def predict_all() -> None:
    from src.eval.predict import PREDS_ROOT, predict
    from src.models.loading import load_model

    for arm, bb, s, rid in finished_runs(sensitivity=True):
        ckpt = RUNS / rid / "best.pt"
        for ds in DATASETS:
            if not (PREDS_ROOT / rid / f"{ds}.parquet").exists() or (ds in FEATURES_ON and not
                                                                      (PREDS_ROOT / rid / f"{ds}.features.npy").exists()):
                predict(ckpt, ds, PREDS_ROOT / rid, run_id=rid, save_features=ds in FEATURES_ON)
        mode = load_model(ckpt).preprocess_mode
        if mode in OTHER_LETTERBOX:
            other = OTHER_LETTERBOX[mode]
            if not (PREDS_ROOT / rid / f"fpdb_test__pp-{other}.parquet").exists():
                predict(ckpt, "fpdb_test", PREDS_ROOT / rid, run_id=rid, preprocess_override=other,
                        variant=f"pp-{other}")
        print(f"  predicted {rid}", flush=True)


def saliency_all() -> None:
    from src.eval.predict import PREDS_ROOT
    from src.localisation import runner as R
    from src.models.loading import load_model

    for arm, bb, s, rid in finished_runs():
        todo = [ds for ds in ("fpdb_test", "hc18", "ucl", "african")
                if not R.saliency_path(rid, ds, "gradcam").exists()]
        if not todo:
            continue
        lm = load_model(RUNS / rid / "best.pt")
        for ds in todo:
            frame = R.load_frame(ds, PREDS_ROOT / rid)
            out, maps = R.compute(lm, R.RunSpec(rid, ds, "gradcam", frame))
            R.write(out, maps, R.saliency_path(rid, ds, "gradcam"))
        print(f"  saliency {rid}", flush=True)


# ----------------------------------------------------------------------------- CPU steps
def _probe_job(rid: str) -> str:
    from src.audit.probes import load_features, probe_oof, within_class_machine_targets
    from src.eval.predict import PREDS_ROOT

    out = PREDS_ROOT / rid / "probe_within_class_machine.parquet"
    if out.exists():
        return rid
    meta = pd.read_parquet(PREDS_ROOT / rid / "fpdb_test.parquet",
                           columns=["image_id", "patient_id", "source_machine", "true_label"])
    meta = meta.astype(str).set_index("image_id")
    feats = load_features(PREDS_ROOT / rid)
    parts = []
    for cls, y in within_class_machine_targets(meta).items():
        X = feats.loc[y.index].to_numpy(np.float64)
        oof, _ = probe_oof(X, y.to_numpy(), meta.loc[y.index, "patient_id"].to_numpy())
        parts.append(pd.DataFrame({"image_id": y.index, "patient_id": meta.loc[y.index, "patient_id"].to_numpy(),
                                   "cls": cls, "y": y.to_numpy(), "oof": oof}))
    pd.concat(parts, ignore_index=True).to_parquet(out, index=False)
    return rid


def probes_all(jobs: int) -> None:
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"           # single-threaded BLAS in torch-free workers (CLAUDE.md resource note)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        for rid in ex.map(_probe_job, [r[3] for r in finished_runs()]):
            print(f"  probes {rid}", flush=True)


def temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature T > 0 minimising the NLL of softmax(logits / T) (val only)."""
    from scipy.optimize import minimize_scalar

    def nll(logt: float) -> float:
        z = logits / np.exp(logt)
        z = z - z.max(axis=1, keepdims=True)
        lp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-lp[np.arange(len(y)), y].mean())

    return float(np.exp(minimize_scalar(nll, bounds=(-3, 3), method="bounded").x))


def score() -> dict:
    from src.audit.probes import within_class_mean
    from src.data.dataset import CANONICAL_CLASSES
    from src.data.labels import collapse_probs
    from src.eval import metrics as M
    from src.eval.predict import PREDS_ROOT
    from src.eval.scoring import estimate_json
    from src.eval.stats import RatioMetric, cluster_bootstrap, multi_seed_summary
    from src.localisation import runner as R

    res: dict = {"runs": {}, "by_arm": {}}
    for arm, bb, s, rid in finished_runs():
        r: dict = {}
        val = load(rid, "fpdb_val")
        lg = val[[f"logit_{i}" for i in range(8)]].to_numpy(np.float64)
        T = temperature(lg, val["true_label"].map(CANONICAL_CLASSES.index).to_numpy())
        r["temperature"] = T
        for ds in ("fpdb_test", *EXTERNAL):
            d = load(rid, ds)
            lg = d[[f"logit_{i}" for i in range(8)]].to_numpy(np.float64)
            out = {}
            for tag, t in (("raw", 1.0), ("scaled", T)):
                z = lg / t
                p = np.exp(z - z.max(axis=1, keepdims=True))
                p /= p.sum(axis=1, keepdims=True)
                conf = collapse_probs(p).max(axis=1)
                out[f"ece_{tag}"] = M.ece(conf, d["correct_full"].to_numpy(bool))
            out["coverage_at_95"] = M.coverage_at_accuracy(d["conf_full_summed"].to_numpy(np.float64),
                                                           d["correct_full"].to_numpy(bool))
            grid, risk = M.risk_coverage(d["conf_full_summed"].to_numpy(np.float64), d["correct_full"].to_numpy(bool))
            out["risk_coverage"] = {"coverage": grid.tolist(), "risk": risk.tolist()}
            r[ds] = out
        pr = PREDS_ROOT / rid / "probe_within_class_machine.parquet"
        if pr.exists():
            r["within_class_machine"] = estimate_json(cluster_bootstrap(pd.read_parquet(pr), within_class_mean))
        alt = [p for p in (PREDS_ROOT / rid).glob("fpdb_test__pp-*.parquet")]
        if alt:
            base, other = load(rid, "fpdb_test"), load(rid, alt[0].stem)
            meta = pd.read_parquet("data/processed/metadata/acquisition_metadata.parquet",
                                   columns=["image_id", "marker_area_frac"])
            m = base.merge(other[["image_id", "pred_fine"]], on="image_id", suffixes=("", "_alt"))
            m = m.merge(meta, on="image_id", how="left")
            m["flip"] = (m["pred_fine"] != m["pred_fine_alt"]).astype(np.int8)
            mk = m[m["marker_area_frac"] > 0]
            r["marker_flip_rate"] = {"with_marker": estimate_json(cluster_bootstrap(mk, RatioMetric("flip", "flip"))),
                                     "all": estimate_json(cluster_bootstrap(m, RatioMetric("flip", "flip"))),
                                     "n_with_marker": int(len(mk))}
        epg = {}
        for ds in ("fpdb_test", "hc18", "ucl"):
            p = R.saliency_path(rid, ds, "gradcam", PREDS_ROOT)
            if p.exists():
                d = pd.read_parquet(p)
                d = d[d["has_landmarks"].astype(bool) & ~d["qc_flag"].fillna(False).astype(bool)]
                epg[ds] = {a: float(d.loc[d["anatomy"] == a, "epg_minus_area_true"].mean())
                           for a in ("Head", "Abdomen", "Femur") if (d["anatomy"] == a).any()}
        r["epg_minus_area_true"] = epg
        res["runs"][rid] = r
    # seed-averaged within-class machine decodability per arm x backbone (P4b primary shortcut index)
    for bb, short in BACKBONES.items():
        for arm in ARMS:
            frames = [pd.read_parquet(PREDS_ROOT / f"{arm}_{short}_s{s}" / "probe_within_class_machine.parquet")
                      for s in SEEDS if (PREDS_ROOT / f"{arm}_{short}_s{s}" / "probe_within_class_machine.parquet").exists()]
            if len(frames) == len(SEEDS):
                res["by_arm"][f"{arm}|{bb}|within_class_machine"] = estimate_json(
                    multi_seed_summary(frames, within_class_mean, key="image_id"))
    res["errors_by_arm"] = errors_by_arm()
    res["p3b_confirmatory"] = p3b_confirmatory()
    res["sensitivity"] = sensitivity()
    return res


# ----------------------------------------------------------------------------- §6.3 rejection vs confusion
def errors_by_arm() -> dict:
    """Seed-averaged rejection rate and error shares per external set and pooled (R2 definitions)."""
    from src.eval.scoring import add_helper_columns, estimate_json, ratio
    from src.eval.stats import multi_seed_summary

    out = {}
    for bb, short in BACKBONES.items():
        for arm in ARMS:
            frames: dict[str, list[pd.DataFrame]] = {}
            for ds in EXTERNAL:
                fs = arm_frames(arm, short, ds)
                if fs is not None:
                    frames[ds] = fs
            if len(frames) < len(EXTERNAL):
                continue
            frames["pooled"] = [pd.concat([frames[ds][k] for ds in EXTERNAL], ignore_index=True)
                                for k in range(len(SEEDS))]
            per = {}
            for ds, fs in frames.items():
                hs = [add_helper_columns(f) for f in fs]
                per[ds] = {"rejection_rate": estimate_json(multi_seed_summary(hs, ratio("rejection_rate", "rejected"))),
                           **{e: estimate_json(multi_seed_summary(hs, ratio(e, e, "err_full"), max_nan_frac=1.0))
                              for e in ERROR_SHARES}}
            out[f"{arm}|{bb}"] = per
    return out


# ----------------------------------------------------------------------------- §6.7 confirmatory P3b
def saliency_frames(arm: str, short: str, sets: list[str], landmark: bool) -> pd.DataFrame | None:
    """Seed-pooled Grad-CAM rows of one arm (R4 columns), or None unless every file exists.

    Keys become ``image_id|s<seed>`` so pooled rows stay unique; ``patient_id`` is unchanged, so every
    bootstrap draw and every GroupKFold fold keeps all of a patient's rows (all seeds) together.
    """
    from src.localisation import runner as R

    parts = []
    for s in SEEDS:
        for ds in sets:
            p = R.saliency_path(f"{arm}_{short}_s{s}", ds, "gradcam", _root())
            if not p.exists():
                return None
            parts.append(pd.read_parquet(p).assign(seed=s))
    d = pd.concat(parts, ignore_index=True)
    for c in d.select_dtypes("category").columns:
        d[c] = d[c].astype(str)
    d["qc_flag"] = d["qc_flag"].fillna(False).astype(bool)
    d["correct"] = d["correct_full"].astype(bool).astype(int)
    if landmark:
        d = d[d["has_landmarks"].astype(bool) & ~d["qc_flag"] & d["epg_minus_area_true"].notna()]
    d["image_id"] = d["image_id"].astype(str) + "|s" + d["seed"].astype(str)
    return d.reset_index(drop=True)


def _guarded(fn) -> dict:
    """Run one model fit; a degenerate score (e.g. constant) is recorded instead of stopping the analysis."""
    try:
        return fn()
    except ValueError as e:                        # includes numpy's LinAlgError (singular design)
        return {"error": f"{type(e).__name__}: {e}"}


def p3b_confirmatory() -> dict:
    from r4_association import discrimination, glm  # same directory; the R4 implementation, unchanged

    res: dict = {"spec": "PREREGISTRATION §6.7: HC18 + UCL landmark images, Grad-CAM true target, seed-pooled, "
                         "patient clusters; confirmed if OR > 1 with 95% CI excluding 1 for AFP on both backbones",
                 "cells": {}}
    for bb, short in BACKBONES.items():
        for arm in ARMS:
            ext = saliency_frames(arm, short, ["hc18", "ucl"], landmark=True)
            if ext is None:
                continue
            cell = {"landmark": _guarded(lambda: {
                "glm": glm(ext, "epg_minus_area_true", ["dataset", "true_collapsed"]),
                "discrimination": discrimination(ext, "epg_minus_area_true")})}
            afr = saliency_frames(arm, short, ["african"], landmark=False)
            if afr is not None:
                afr = afr[afr["out_sector_frac_pred"].notna()].reset_index(drop=True)
                cell["african_monitor"] = _guarded(lambda: {
                    "glm": glm(afr, "out_sector_frac_pred", ["centre", "true_collapsed"]),
                    "discrimination": discrimination(afr, "out_sector_frac_pred", sign=-1.0),
                    "score_sd": float(afr["out_sector_frac_pred"].std())})
            res["cells"][f"{arm}|{bb}"] = cell
    afp = [res["cells"].get(f"afp|{bb}", {}).get("landmark", {}).get("glm") for bb in BACKBONES]
    if all(g is not None for g in afp):
        res["confirmed"] = all(g["or_lo"] > 1 for g in afp)
    return res


# ----------------------------------------------------------------------------- §6.9 sensitivity run
def sensitivity() -> dict:
    """AFP + class weights (double reweighting) vs AFP, same seed, paired; descriptive only."""
    from src.eval.scoring import bal_acc_collapsed, estimate_json, macro_f1_8
    from src.eval.stats import paired_cluster_bootstrap

    arm, bb, s = SENSITIVITY
    rid, ref = f"{arm}_{BACKBONES[bb]}_s{s}", f"afp_{BACKBONES[bb]}_s{s}"
    if not all((_root() / r / f"{ds}.parquet").exists() for r in (rid, ref) for ds in ("fpdb_test", *EXTERNAL)):
        return {}

    def pooled(r: str) -> pd.DataFrame:
        return pd.concat([load(r, ds) for ds in EXTERNAL], ignore_index=True)

    return {"run": rid, "reference": ref, "delta": "run − reference",
            "pooled_bal_acc_full": estimate_json(paired_cluster_bootstrap(
                pooled(ref), pooled(rid), bal_acc_collapsed("pred_full", POOL_CLASSES))),
            "fpdb_test_macro_f1_8": estimate_json(paired_cluster_bootstrap(
                load(ref, "fpdb_test"), load(rid, "fpdb_test"), macro_f1_8()))}


def markdown(res: dict) -> str:
    L = ["# R5 evaluation (secondary analyses; exploratory)", "",
         "| run | T | ECE test raw/scaled | ECE HC18 | ECE UCL | ECE African | within-class machine decoding | marker flip rate |",
         "|---|---|---|---|---|---|---|---|"]
    for rid, r in res["runs"].items():
        wc = r.get("within_class_machine")
        mf = r.get("marker_flip_rate", {}).get("with_marker")
        wc_s = "–" if wc is None else f"{wc['point']:.3f}"
        mf_s = "–" if mf is None else f"{mf['point']:.4f}"
        L.append(f"| `{rid}` | {r['temperature']:.2f} | {r['fpdb_test']['ece_raw']:.3f} / {r['fpdb_test']['ece_scaled']:.3f} | "
                 + " | ".join(f"{r[d]['ece_raw']:.3f} / {r[d]['ece_scaled']:.3f}" for d in EXTERNAL)
                 + f" | {wc_s} | {mf_s} |")
    L.append("")

    def ci(e: dict) -> str:
        if "delta" in e:
            return f"{e['delta']:+.3f} [{e['lo']:+.3f}, {e['hi']:+.3f}]"
        return f"{e['point']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"

    if res.get("errors_by_arm"):
        L += ["## Rejection vs confusion, pooled external (seed-averaged)", "",
              "| arm | rejection rate | errors: Other | errors: absent class | errors: wrong present class |",
              "|---|---|---|---|---|"]
        for k, per in res["errors_by_arm"].items():
            p = per["pooled"]
            L.append(f"| {k} | {ci(p['rejection_rate'])} | " + " | ".join(ci(p[e]) for e in ERROR_SHARES) + " |")
        L.append("")
    p3b = res.get("p3b_confirmatory", {})
    if p3b.get("cells"):
        L += ["## P3b on R5 models (confirmatory for AFP; PREREGISTRATION §6.7)", "",
              "| arm | OR per +0.1 EPG − area | AUROC EPG − area | Δ AUROC (both − confidence) | "
              "African monitor AUROC | monitor Δ AUROC |", "|---|---|---|---|---|---|"]
        for k, c in p3b["cells"].items():
            lm, am = c.get("landmark", {}), c.get("african_monitor", {})
            g, dl, da = lm.get("glm"), lm.get("discrimination"), am.get("discrimination")
            or_s = f"{g['or']:.2f} [{g['or_lo']:.2f}, {g['or_hi']:.2f}]" if g else lm.get("error", "–")
            L.append(f"| {k} | {or_s} | {ci(dl['auroc_score']) if dl else '–'} | "
                     f"{ci(dl['delta_both_minus_conf']) if dl else '–'} | "
                     f"{ci(da['auroc_score']) if da else am.get('error', '–')} | "
                     f"{ci(da['delta_both_minus_conf']) if da else '–'} |")
        if "confirmed" in p3b:
            L += ["", f"Confirmatory claim (AFP on both backbones, OR CI excludes 1): **{p3b['confirmed']}**"]
        L.append("")
    sv = res.get("sensitivity")
    if sv:
        L += ["## Sensitivity: AFP + class weights − AFP (seed 11, paired; descriptive)", "",
              f"- pooled external balanced accuracy: {ci(sv['pooled_bal_acc_full'])}",
              f"- FPDB-test 8-way macro-F1: {ci(sv['fpdb_test_macro_f1_8'])}", ""]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predict", action="store_true")
    ap.add_argument("--probes", action="store_true")
    ap.add_argument("--saliency", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    from r5_run_matrix import check_g4  # same directory; same gate

    check_g4()
    if a.predict:
        predict_all()
    if a.saliency:
        saliency_all()
    if a.probes:
        probes_all(a.jobs)
    if a.score:
        res = score()
        OUT_JSON.write_text(json.dumps(res, indent=1), encoding="utf-8")
        OUT_MD.write_text(markdown(res), encoding="utf-8")
        print(f"wrote {OUT_JSON}, {OUT_MD}")


if __name__ == "__main__":
    main()
