"""
scripts/research/e4_availability_train.py  (R3 E4: can a model learn the label without anatomy? P2a)

Trains the unchanged B0 convnext_tiny recipe on inputs with the anatomy removed
(src/preprocess/pipeline.py), then evaluates on the equally transformed FPDB test split.

    nonanat_only        seeds 11, 22, 33
    overlay_only        seeds 11, 22, 33
    sector_shape_only   seed 11

Stages
  --train      python -m src.train.train for each missing run -> runs/e4_<mode>_convnext_s<seed>/
               (a run is complete when classification_report_final.txt exists)
  --evaluate   predict fpdb_test -> results/preds/e4_<mode>_convnext_s<seed>/; score with cluster CIs
               (multi-seed summaries share the cluster draws), compare with chance, the E2 metadata
               model and the full-image B0 model -> results/json/E4.json, results/tables/E4.md

Inputs needed: the cache (scripts/research/build_cache.py --mode ...) and, for the comparison,
results/json/E2.json (scripts/research/e2_metadata_only_model.py).
Usage:  python scripts/research/e4_availability_train.py --train --evaluate
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.eval import metrics as M  # noqa: E402
from src.eval.predict import PREDS_ROOT, predict  # noqa: E402
from src.eval.scoring import bal_acc_8, estimate_json, macro_f1_8  # noqa: E402
from src.eval.stats import cluster_bootstrap, multi_seed_summary  # noqa: E402

PY = sys.executable
RUNS = {"nonanat_only": (11, 22, 33), "overlay_only": (11, 22, 33), "sector_shape_only": (11,)}
B0_TEST = PREDS_ROOT / "legacy_convnext_s42" / "fpdb_test.parquet"
E2_JSON = Path("results/json/E2.json")
OUT_JSON = Path("results/json/E4.json")
OUT_MD = Path("results/tables/E4.md")


def run_id(mode: str, seed: int) -> str:
    return f"e4_{mode}_convnext_s{seed}"


def run_dir(mode: str, seed: int) -> Path:
    return Path("runs") / run_id(mode, seed)


def train_all() -> dict:
    times = {}
    for mode, seeds in RUNS.items():
        for seed in seeds:
            out = run_dir(mode, seed)
            if (out / "classification_report_final.txt").exists():
                continue
            t = time.perf_counter()
            cmd = [PY, "-m", "src.train.train", "--config", f"configs/research/e4_{mode}_convnext_tiny.yaml",
                   "--seed", str(seed), "--out-dir", str(out)]
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            times[run_id(mode, seed)] = round(time.perf_counter() - t)
            print(f"  done {run_id(mode, seed)} in {times[run_id(mode, seed)]} s", flush=True)
    return times


# ----------------------------------------------------------------------------- evaluation
def auroc_standard(df: pd.DataFrame) -> float:
    """AUROC of 1 - p(Other) for 'standard plane' vs 'Other'."""
    return M.auroc(1.0 - df["prob_7"].to_numpy(np.float64), (df["true_label"] != "Other").to_numpy())


METRICS = {"bal_acc_8": bal_acc_8(), "macro_f1_8": macro_f1_8(), "auroc_standard_vs_other": auroc_standard}


def load_run(mode: str, seed: int) -> pd.DataFrame:
    pq = PREDS_ROOT / run_id(mode, seed) / "fpdb_test.parquet"
    if not pq.exists():
        predict(run_dir(mode, seed) / "best.pt", "fpdb_test", pq.parent, run_id=run_id(mode, seed), num_workers=4)
    f = pd.read_parquet(pq)
    for c in f.select_dtypes("category").columns:
        f[c] = f[c].astype(str)
    return f


def train_summary(mode: str, seed: int) -> dict:
    import torch

    ck = torch.load(run_dir(mode, seed) / "best.pt", map_location="cpu", weights_only=False)
    return {"best_epoch": int(ck["epoch"]), "val_macro_f1": float(ck["val_macro_f1"])}


def evaluate() -> dict:
    res: dict = {"modes": {}, "chance": {"bal_acc_8": 1 / 8, "auroc_standard_vs_other": 0.5}}
    for mode, seeds in RUNS.items():
        dfs = [load_run(mode, s) for s in seeds]
        r: dict = {"seeds": list(seeds), "per_seed": {}, "summary": {}}
        for s, d in zip(seeds, dfs):
            r["per_seed"][s] = {**train_summary(mode, s),
                                **{k: cluster_bootstrap(d, m, max_nan_frac=0.01) for k, m in METRICS.items()}}
        for k, m in METRICS.items():
            r["summary"][k] = multi_seed_summary(dfs, m)
        res["modes"][mode] = r
        print(f"  evaluated {mode}", flush=True)
    b0 = pd.read_parquet(B0_TEST)
    for c in b0.select_dtypes("category").columns:
        b0[c] = b0[c].astype(str)
    res["b0_full_image"] = {k: cluster_bootstrap(b0, m) for k, m in METRICS.items()}
    if E2_JSON.exists():
        e2 = json.loads(E2_JSON.read_text(encoding="utf-8"))
        res["e2_metadata"] = {fs: {k: e2["results"][fs]["all_test"][k] for k in ("bal_acc_8", "auroc_standard_vs_other")}
                              for fs in ("prior", "all")}
    return res


def fmt(e) -> str:
    if isinstance(e, dict):
        return f"{e['point']:.3f} [{e['lo']:.3f}, {e['hi']:.3f}]"
    return e.fmt()


def markdown(res: dict) -> str:
    s = ["# E4: availability: can the B0 recipe learn the label without anatomy?", "",
         "convnext_tiny, unchanged B0 recipe (val macro-F1 early stopping, class-weighted CE). Test = FPDB test "
         "(5,271 images / 896 patients), transformed the same way. Multi-seed rows: mean over seeds, 95% CI "
         "from a patient-cluster bootstrap that applies one draw to every seed (B = 2000).", "",
         "| input | seeds | balanced acc (8-way; chance 0.125) | macro-F1 (8-way) | AUROC standard vs Other |",
         "|---|---|---|---|---|"]
    for mode, r in res["modes"].items():
        x = r["summary"]
        s.append(f"| `{mode}` | {len(r['seeds'])} | {fmt(x['bal_acc_8'])} | {fmt(x['macro_f1_8'])} | "
                 f"{fmt(x['auroc_standard_vs_other'])} |")
    if "e2_metadata" in res:
        for fs, x in res["e2_metadata"].items():
            s.append(f"| E2 metadata model `{fs}` (no pixels) | – | {fmt(x['bal_acc_8'])} | – | "
                     f"{fmt(x['auroc_standard_vs_other'])} |")
    b = res["b0_full_image"]
    s += [f"| full image (B0 legacy convnext, seed 42) | 1 | {fmt(b['bal_acc_8'])} | {fmt(b['macro_f1_8'])} | "
          f"{fmt(b['auroc_standard_vs_other'])} |", "",
          "Per seed (best epoch, val macro-F1, test balanced accuracy):", ""]
    for mode, r in res["modes"].items():
        for sd, x in r["per_seed"].items():
            s.append(f"* `{mode}` s{sd}: epoch {x['best_epoch']}, val macro-F1 {x['val_macro_f1']:.3f}, "
                     f"test balanced acc {fmt(x['bal_acc_8'])}")
    s += ["", "Reading (pre-declared): `nonanat_only` well above chance **and** above the E2 `prior` model means "
          "non-anatomical image content carries label information beyond the device prior. Limitations: the "
          "sector shape leaks the device (hence `sector_shape_only`), and `nonanat_only` keeps one in-sector grey "
          "level per image (mean brightness of the anatomy)."]
    return "\n".join(s) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    a = ap.parse_args()
    if a.train:
        train_all()
    if a.evaluate:
        res = evaluate()
        OUT_JSON.write_text(json.dumps(estimate_json(res), indent=1), encoding="utf-8")
        OUT_MD.write_text(markdown(res), encoding="utf-8")
        print(OUT_MD.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
