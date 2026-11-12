"""
scripts/research/r5_run_matrix.py  (R5.6: the 24-run training matrix)

4 arms (B0, B1, B2, AFP) x 2 backbones (convnext_tiny, tf_efficientnetv2_s) x 3 seeds (11, 22, 33),
run sequentially and ordered by seed first, so a partial matrix stays balanced:

    runs/<arm>_<convnext|effv2s>_s<seed>/   (best.pt, config.yaml, train_log.txt, env.txt, DONE, stdout.log)

Guards:
* **Gate G4** (pre-registration): refuses to start unless docs/research/GATE_G4.md has a line
  ``OSF: https://...`` and a line ``PREREGISTRATION commit: <hash>`` whose commit is an ancestor of
  HEAD. ``--smoke`` (1 epoch, runs/_smoke/) is allowed before G4.
* Runs with DONE are skipped (resume-safe). Free GPU memory is checked before each run.
* After B0 convnext seed 11: its val macro-F1 (fp32, from results/preds/<run_id>/fpdb_val.parquet) must
  lie within the legacy seed-42 val macro-F1 +- 0.02, otherwise the matrix stops for diagnosis.
* A run that ends with val macro-F1 < 0.5 (diverged) is re-run once with the same seed (pre-registered).

Usage:  python scripts/research/r5_run_matrix.py [--smoke] [--only b0_convnext_s11 ...] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

ARMS = ("b0", "b1", "b2", "afp")
BACKBONES = {"convnext_tiny": "convnext", "tf_efficientnetv2_s": "effv2s"}
SEEDS = (11, 22, 33)
GATE = Path("docs/research/GATE_G4.md")
PREDS = Path("results/preds")
LEGACY_VAL = PREDS / "legacy_convnext_s42" / "fpdb_val.parquet"
SANITY_TOL = 0.02
DIVERGED = 0.5
MIN_FREE_MB = 6000


def matrix() -> list[tuple[str, str, int]]:
    """(arm, backbone, seed), seed-major."""
    return [(arm, bb, s) for s in SEEDS for arm in ARMS for bb in BACKBONES]


def run_id(arm: str, bb: str, seed: int) -> str:
    return f"{arm}_{BACKBONES[bb]}_s{seed}"


def check_g4() -> str:
    if not GATE.exists():
        raise SystemExit(f"Gate G4 not passed: {GATE} missing (pre-register on OSF first)")
    text = GATE.read_text(encoding="utf-8")
    if not re.search(r"^OSF: https?://\S+", text, flags=re.M):
        raise SystemExit("Gate G4 not passed: no 'OSF: https://...' line in GATE_G4.md")
    m = re.search(r"^PREREGISTRATION commit: ([0-9a-f]{7,40})", text, flags=re.M)
    if not m:
        raise SystemExit("Gate G4 not passed: no 'PREREGISTRATION commit: <hash>' line in GATE_G4.md")
    ok = subprocess.run(["git", "merge-base", "--is-ancestor", m.group(1), "HEAD"]).returncode == 0
    if not ok:
        raise SystemExit(f"Gate G4: PREREGISTRATION commit {m.group(1)} is not an ancestor of HEAD")
    return m.group(1)


def free_vram_mb() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.split()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return -1


def val_macro_f1(parquet: Path) -> float:
    d = pd.read_parquet(parquet, columns=["true_label", "pred_fine"])
    return float(f1_score(d["true_label"].astype(str), d["pred_fine"].astype(str), average="macro"))


def train(arm: str, bb: str, seed: int, out: Path, smoke: bool) -> int:
    cmd = [sys.executable, "-m", "src.train.train", "--config", f"configs/research/{arm}_{bb}.yaml",
           "--out-dir", out.as_posix(), "--seed", str(seed)]
    if smoke:
        cmd += ["--epochs-override", "1"]
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "stdout.log", "a", encoding="utf-8") as log:
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {' '.join(cmd)}\n")
        log.flush()
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoke", action="store_true", help="1 epoch per run into runs/_smoke/ (allowed before G4)")
    ap.add_argument("--only", nargs="*", help="restrict to these run ids")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    prereg = None if a.smoke else check_g4()
    root = Path("runs/_smoke") if a.smoke else Path("runs")
    jobs = [j for j in matrix() if not a.only or run_id(*j) in a.only]
    done_times: list[float] = []
    for k, (arm, bb, seed) in enumerate(jobs):
        rid = run_id(arm, bb, seed)
        out = root / rid
        if (out / "DONE").exists():
            continue
        if a.dry_run:
            print(f"would run {rid}")
            continue
        free = free_vram_mb()
        if 0 <= free < MIN_FREE_MB:
            raise SystemExit(f"only {free} MB GPU memory free before {rid}; close other GPU jobs")
        t0 = time.time()
        for attempt in (1, 2):
            rc = train(arm, bb, seed, out, a.smoke)
            if rc != 0:
                raise SystemExit(f"{rid} failed (exit {rc}); see {out / 'stdout.log'}")
            f1 = val_macro_f1(PREDS / rid / "fpdb_val.parquet") if not a.smoke else float("nan")
            if a.smoke or f1 >= DIVERGED or attempt == 2:
                break
            print(f"{rid}: val macro-F1 {f1:.3f} < {DIVERGED} (diverged) -> re-run once", flush=True)
            for f in ("DONE", "best.pt"):
                (out / f).unlink(missing_ok=True)
        done_times.append(time.time() - t0)
        left = sum(1 for j in jobs[k + 1:] if not (root / run_id(*j) / "DONE").exists())
        eta_h = left * (sum(done_times) / len(done_times)) / 3600
        print(f"{rid}: done in {done_times[-1] / 60:.0f} min, val macro-F1 {f1:.4f}; {left} left, ETA {eta_h:.1f} h",
              flush=True)
        if not a.smoke and rid == "b0_convnext_s11":
            ref = val_macro_f1(LEGACY_VAL)
            if abs(f1 - ref) > SANITY_TOL:
                raise SystemExit(f"sanity check failed: B0 convnext s11 val macro-F1 {f1:.4f} vs legacy {ref:.4f} "
                                 f"(tolerance {SANITY_TOL}); diagnose before continuing")
            print(f"sanity check passed: {f1:.4f} vs legacy {ref:.4f}", flush=True)
    if prereg:
        Path("runs/MATRIX_STATUS.json").write_text(json.dumps({"prereg_commit": prereg, "jobs": [run_id(*j) for j in jobs]},
                                                              indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
