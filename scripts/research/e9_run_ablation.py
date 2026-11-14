"""
scripts/research/e9_run_ablation.py  (E9, POST-HOC: which AFP component costs external accuracy?)

Planned on 2026-11-14 after the R5 results (docs/research/plans/R7_POSTHOC_FOLLOWUP.md); NOT pre-registered.
AFP = B2 + acquisition-balanced sampler (class weights off) + device-style augmentation, and AFP scored below
B2 on the pooled external endpoint. Two single-component arms isolate the cost:

    b2smp = B2 + sampler, class weights off      (configs/research/b2smp_<backbone>.yaml)
    b2sty = B2 + device-style augmentation       (configs/research/b2sty_<backbone>.yaml)

2 arms x 2 backbones x seeds 11/22/33 = 12 runs, seed-major, into runs/<arm>_<convnext|effv2s>_s<seed>/.
Every run is predicted right after training (fpdb_val/test, hc18, ucl, african; features on test + externals),
in its own process, so the GPU is free between steps. Runs with DONE + all prediction files are skipped
(resume-safe). Progress (runs done, % of estimated time, ETA) goes to stdout and runs/E9_progress.json.

Usage:  python scripts/research/e9_run_ablation.py [--dry-run]
        python scripts/research/e9_run_ablation.py --predict-one <run_id>      (internal)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts/research")
from r5_run_matrix import BACKBONES, DIVERGED, MIN_FREE_MB, free_vram_mb, train, val_macro_f1  # noqa: E402

ARMS = ("b2smp", "b2sty")
SEEDS = (11, 22, 33)
DATASETS = ("fpdb_val", "fpdb_test", "hc18", "ucl", "african")
FEATURES_ON = {"fpdb_test", "hc18", "ucl", "african"}
RUNS = Path("runs")
PREDS = Path("results/preds")
PROGRESS = RUNS / "E9_progress.json"
# Prior minutes per run (train + predict), from the R5 matrix means; replaced by observed means as runs finish.
PRIOR_MIN = {"convnext_tiny": 22.0, "tf_efficientnetv2_s": 34.0}


def jobs() -> list[tuple[str, str, int]]:
    return [(arm, bb, s) for s in SEEDS for arm in ARMS for bb in BACKBONES]


def run_id(arm: str, bb: str, seed: int) -> str:
    return f"{arm}_{BACKBONES[bb]}_s{seed}"


def predicted(rid: str) -> bool:
    d = PREDS / rid
    return all((d / f"{ds}.parquet").exists() and (ds not in FEATURES_ON or (d / f"{ds}.features.npy").exists())
               for ds in DATASETS)


def complete(rid: str) -> bool:
    return (RUNS / rid / "DONE").exists() and predicted(rid)


def predict_one(rid: str) -> None:
    from src.eval.predict import predict

    for ds in DATASETS:
        p = PREDS / rid / f"{ds}.parquet"
        if not p.exists() or (ds in FEATURES_ON and not (PREDS / rid / f"{ds}.features.npy").exists()):
            predict(RUNS / rid / "best.pt", ds, PREDS / rid, run_id=rid, save_features=ds in FEATURES_ON)


def eta(todo: list[tuple[str, str, int]], observed: dict[str, list[float]]) -> float:
    """Estimated minutes for the remaining jobs (observed per-backbone mean, else the R5 prior)."""
    per = {bb: (sum(v) / len(v) if (v := observed.get(bb)) else PRIOR_MIN[bb]) for bb in BACKBONES}
    return sum(per[bb] for _, bb, _ in todo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--predict-one")
    a = ap.parse_args()
    if a.predict_one:
        predict_one(a.predict_one)
        return
    all_jobs = jobs()
    observed: dict[str, list[float]] = {}
    t_start = time.time()
    for k, (arm, bb, seed) in enumerate(all_jobs):
        rid = run_id(arm, bb, seed)
        if complete(rid):
            continue
        if a.dry_run:
            print(f"would run {rid}")
            continue
        free = free_vram_mb()
        if 0 <= free < MIN_FREE_MB:
            raise SystemExit(f"only {free} MB GPU memory free before {rid}; close other GPU jobs")
        t0 = time.time()
        out = RUNS / rid
        if not (out / "DONE").exists():
            for attempt in (1, 2):
                rc = train(arm, bb, seed, out, smoke=False)
                if rc != 0:
                    raise SystemExit(f"{rid} failed (exit {rc}); see {out / 'stdout.log'}")
                f1 = val_macro_f1(PREDS / rid / "fpdb_val.parquet")
                if f1 >= DIVERGED or attempt == 2:
                    break
                print(f"{rid}: val macro-F1 {f1:.3f} < {DIVERGED} (diverged) -> re-run once", flush=True)
                for f in ("DONE", "best.pt"):
                    (out / f).unlink(missing_ok=True)
        rc = subprocess.run([sys.executable, "-u", __file__, "--predict-one", rid]).returncode
        if rc != 0:
            raise SystemExit(f"prediction for {rid} failed (exit {rc})")
        f1 = val_macro_f1(PREDS / rid / "fpdb_val.parquet")
        observed.setdefault(bb, []).append((time.time() - t0) / 60)
        todo = [j for j in all_jobs if not complete(run_id(*j))]
        done_n = len(all_jobs) - len(todo)
        left_min = eta(todo, observed)
        total_min = left_min + (time.time() - t_start) / 60
        pct = 100 * (1 - left_min / total_min) if total_min else 100.0
        finish = datetime.now() + timedelta(minutes=left_min)
        line = (f"[{done_n}/{len(all_jobs)} runs] {rid}: {observed[bb][-1]:.0f} min (train + predict), "
                f"val macro-F1 {f1:.4f}; this session about {pct:.0f}% done; "
                f"{len(todo)} left, about {left_min / 60:.1f} h, ETA {finish:%H:%M}")
        print(line, flush=True)
        PROGRESS.write_text(json.dumps({"done": done_n, "total": len(all_jobs), "last": rid, "val_macro_f1": f1,
                                        "minutes_left": round(left_min), "eta": f"{finish:%Y-%m-%d %H:%M}",
                                        "line": line}, indent=1), encoding="utf-8")
    if not a.dry_run:
        print("E9 training + prediction complete", flush=True)


if __name__ == "__main__":
    main()
