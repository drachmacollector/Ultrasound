"""
scripts/tune_tier2_mode_filter.py

Tier-2a parameter sweep: dual-grid comparison of window_frames and min_majority_frac
for the causal majority-vote filter (Tier2ModeFilter) applied on top of the locked
Tier-1 smoother (configs/smoothing_tier1.yaml).

DESIGN DECISIONS LOGGED HERE
------------------------------
1. Tier-1 parameters are FROZEN at configs/smoothing_tier1.yaml — NOT re-swept.
   The Tier-1 sweep is complete and its output is canonical (PHASE5_SMOOTHING_TUNING.md).
2. Per-clip inference is cached once (precompute pass) and reused across all Tier-2a
   combos — identical to the Tier-1 sweep strategy to avoid repeated GPU inference.
3. Two sweep grids are run and compared:
     Grid A (spec grid):    window_frames ∈ {5, 9, 13, 15, 19, 25}
     Grid B (uniform grid): window_frames ∈ {5, 7, 9, 11, 13, 15, 19, 25}
   min_majority_frac ∈ {0.5, 0.6, 0.7}  (same for both grids)
   Grid B adds {7, 11} to bracket the low-latency range more densely.
4. Tier-1's baseline on each clip is re-measured here (from cached probs) so this
   script is entirely self-contained and does not depend on a previous CSV run.

SUCCESS CRITERION (pre-declared before running):
  For Tier-2a to be judged successful, the best combo must satisfy ALL of:
    a) Residual switches/min on 202101141947512003470I1.avi < 72.0 (improvement)
    b) spurious_new == 0 for every other clip (no regression)
    c) window_frames ≤ 9 (≤ ~337ms additional lag at 23.7fps, keeping total lag
       under ~677ms = 340ms Tier-1 dwell + 337ms Tier-2a worst-case window)
  If no combo satisfies all three, document and escalate to Tier-2b.

OUTPUTS
-------
  logs/eval/tier2a_sweep_grid_A.txt    — verbose log Grid A
  logs/eval/tier2a_sweep_grid_B.txt    — verbose log Grid B
  logs/eval/tier2a_sweep_comparison.txt — side-by-side comparison summary
  data/processed/tier2a_tuning/sweep_results_grid_A.csv
  data/processed/tier2a_tuning/sweep_results_grid_B.csv
  data/processed/tier2a_tuning/sweep_results_combined.csv
  configs/smoothing_tier2a.yaml       — best combo (if success criterion met)
"""
from __future__ import annotations

import datetime
import os
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")

from scripts.tune_tier1_smoothing import load_video_paths, run_inference_on_clip
from src.realtime.model_loader import load_inference_model
from src.smoothing.tier1 import Tier1Smoother
from src.smoothing.tier2_mode_filter import Tier2ModeFilter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CKPT_PATH = "checkpoints/convnext_tiny/best.pt"
TIER1_CONFIG = "configs/smoothing_tier1.yaml"
TIER2_CONFIG_OUTPUT = "configs/smoothing_tier2a.yaml"

LOG_DIR = Path("logs/eval")
DATA_DIR = Path("data/processed/tier2a_tuning")

LOG_GRID_A = LOG_DIR / "tier2a_sweep_grid_A.txt"
LOG_GRID_B = LOG_DIR / "tier2a_sweep_grid_B.txt"
LOG_COMPARE = LOG_DIR / "tier2a_sweep_comparison.txt"
CSV_GRID_A = DATA_DIR / "sweep_results_grid_A.csv"
CSV_GRID_B = DATA_DIR / "sweep_results_grid_B.csv"
CSV_COMBINED = DATA_DIR / "sweep_results_combined.csv"

# Stubborn clip name (must match basename exactly as it appears in the video path)
STUBBORN_CLIP = "202101141947512003470I1.avi"

# Tier-1 baseline values (from Phase 5 sweep, for cross-reference)
TIER1_BASELINE_TOTAL = 788.75    # switches/min across all 46 clips (raw argmax)
TIER1_RESIDUAL_TOTAL = 72.0      # switches/min across all 46 after Tier-1 (chosen combo)
TIER1_STUBBORN_RESIDUAL = 72.0   # switches/min on the one remaining clip

# Pre-declared success criterion window
SUCCESS_MAX_WINDOW_FRAMES = 9    # must be ≤ this to keep added lag reasonable

# Grid A (spec grid from §2.2)
GRID_A_WINDOWS = [5, 9, 13, 15, 19, 25]
# Grid B (uniform, denser at low-latency range)
GRID_B_WINDOWS = [5, 7, 9, 11, 13, 15, 19, 25]
# Shared majority fractions
MAJORITY_FRACS = [0.5, 0.6, 0.7]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_tier1_config() -> dict:
    with open(TIER1_CONFIG, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def measure_tier1_then_tier2(
    all_probs: list[np.ndarray],
    fps: float,
    tier1_cfg: dict,
    window_frames: int,
    min_majority_frac: float,
) -> tuple[float, float]:
    """
    Run Tier-1 then Tier-2a over a precomputed prob sequence.

    Returns
    -------
    tier1_switches_per_min : float   — Tier-1-only output switch rate
    tier2_switches_per_min : float   — Tier-1+Tier-2a output switch rate
    """
    smoother = Tier1Smoother(
        num_classes=8,
        alpha=float(tier1_cfg["alpha"]),
        switch_threshold=float(tier1_cfg["switch_threshold"]),
        min_dwell_frames=int(tier1_cfg["min_dwell_frames"]),
        hold_floor=tier1_cfg.get("hold_floor"),
    )
    mode_filter = Tier2ModeFilter(
        window_frames=window_frames,
        min_majority_frac=min_majority_frac,
    )

    prev_t1 = None
    prev_t2 = None
    t1_switches = 0
    t2_switches = 0

    for probs in all_probs:
        t1_label, _, _, _ = smoother.step(probs)
        t2_label = mode_filter.step(t1_label)

        if prev_t1 is not None and t1_label != prev_t1:
            t1_switches += 1
        if prev_t2 is not None and t2_label != prev_t2:
            t2_switches += 1

        prev_t1 = t1_label
        prev_t2 = t2_label

    duration_sec = len(all_probs) / fps if fps > 0 else 1.0
    t1_per_min = (t1_switches / duration_sec) * 60.0
    t2_per_min = (t2_switches / duration_sec) * 60.0
    return t1_per_min, t2_per_min


def run_sweep(
    grid_name: str,
    window_frames_list: list[int],
    majority_fracs: list[float],
    clip_data: list[tuple[str, list[np.ndarray], float]],
    tier1_cfg: dict,
    log_path: Path,
) -> pd.DataFrame:
    """Run the sweep for one grid; return a DataFrame of results."""

    rows = []
    n_combos = len(window_frames_list) * len(majority_fracs)

    with open(log_path, "w", encoding="utf-8") as log:
        def lp(msg: str = "") -> None:
            print(msg)
            print(msg, file=log)

        lp("=" * 72)
        lp(f"Tier-2a Parameter Sweep — Grid {grid_name}")
        lp(f"Generated: {datetime.datetime.now().isoformat()}")
        lp("=" * 72)
        lp(f"Tier-1 config (frozen): {TIER1_CONFIG}")
        lp(f"  alpha={tier1_cfg['alpha']}  sw_thr={tier1_cfg['switch_threshold']}  "
           f"dwell={tier1_cfg['min_dwell_frames']}  hold_floor={tier1_cfg.get('hold_floor')}")
        lp(f"Grid {grid_name} window_frames: {window_frames_list}")
        lp(f"min_majority_frac values: {majority_fracs}")
        lp(f"Total combos: {n_combos}")
        lp(f"Total clips:  {len(clip_data)}")
        lp()
        lp("Tier-1 baseline (from Phase 5 sweep, for reference):")
        lp(f"  Total residual switches/min across 46 clips: {TIER1_RESIDUAL_TOTAL}")
        lp(f"  Stubborn clip ({STUBBORN_CLIP}): {TIER1_STUBBORN_RESIDUAL} switches/min")
        lp()
        lp("Pre-declared success criterion (evaluated AFTER sweep):")
        lp(f"  a) stubborn clip residual < {TIER1_STUBBORN_RESIDUAL} switches/min")
        lp(f"  b) spurious_new == 0 for all other clips")
        lp(f"  c) window_frames <= {SUCCESS_MAX_WINDOW_FRAMES}")
        lp()

        combo_idx = 0
        for window_frames, min_majority_frac in product(window_frames_list, majority_fracs):
            combo_idx += 1
            lp(f"[{combo_idx:3d}/{n_combos}] window={window_frames:2d}  "
               f"majority_frac={min_majority_frac:.1f}")

            total_t1_switches = 0.0
            total_t2_switches = 0.0
            stubborn_t1 = 0.0
            stubborn_t2 = 0.0
            spurious_clips = 0
            clip_rows = []

            for clip_name, all_probs, fps in clip_data:
                t1_per_min, t2_per_min = measure_tier1_then_tier2(
                    all_probs, fps, tier1_cfg, window_frames, min_majority_frac
                )
                # spurious: Tier-2a introduced switches on a clip that had 0 after Tier-1
                is_spurious = (t1_per_min == 0.0 and t2_per_min > 0.0)

                total_t1_switches += t1_per_min
                total_t2_switches += t2_per_min
                if is_spurious:
                    spurious_clips += 1

                if STUBBORN_CLIP in clip_name:
                    stubborn_t1 = t1_per_min
                    stubborn_t2 = t2_per_min

                clip_rows.append({
                    "clip": clip_name,
                    "t1_per_min": round(t1_per_min, 2),
                    "t2_per_min": round(t2_per_min, 2),
                    "is_spurious": is_spurious,
                })

            # Added lag at 23.7fps steady state
            fps_ss = 23.7
            added_lag_ms = (window_frames - 1) / fps_ss * 1000.0

            delta = total_t2_switches - total_t1_switches
            stubborn_delta = stubborn_t2 - stubborn_t1
            suppression_pct = (
                (TIER1_RESIDUAL_TOTAL - total_t2_switches) / TIER1_RESIDUAL_TOTAL * 100.0
                if TIER1_RESIDUAL_TOTAL > 0 else 0.0
            )

            lp(f"  Tier-1 total residual:  {total_t1_switches:.2f} switches/min")
            lp(f"  Tier-2a total residual: {total_t2_switches:.2f} switches/min")
            lp(f"  Net delta:              {delta:+.2f} switches/min")
            lp(f"  Suppression vs Tier-1:  {suppression_pct:.1f}%")
            lp(f"  Stubborn clip: Tier-1={stubborn_t1:.2f} → Tier-2a={stubborn_t2:.2f} "
               f"(delta={stubborn_delta:+.2f})")
            lp(f"  Spurious clips:         {spurious_clips}")
            lp(f"  Added lag (worst-case): {added_lag_ms:.0f} ms")
            lp()

            rows.append({
                "grid": grid_name,
                "window_frames": window_frames,
                "min_majority_frac": min_majority_frac,
                "t1_total_residual": round(total_t1_switches, 2),
                "t2_total_residual": round(total_t2_switches, 2),
                "net_delta": round(delta, 2),
                "suppression_pct": round(suppression_pct, 1),
                "stubborn_t1": round(stubborn_t1, 2),
                "stubborn_t2": round(stubborn_t2, 2),
                "stubborn_delta": round(stubborn_delta, 2),
                "spurious_clips": spurious_clips,
                "added_lag_ms": round(added_lag_ms, 0),
                "meets_criterion_a": stubborn_t2 < TIER1_STUBBORN_RESIDUAL,
                "meets_criterion_b": spurious_clips == 0,
                "meets_criterion_c": window_frames <= SUCCESS_MAX_WINDOW_FRAMES,
                "meets_all_criteria": (
                    stubborn_t2 < TIER1_STUBBORN_RESIDUAL
                    and spurious_clips == 0
                    and window_frames <= SUCCESS_MAX_WINDOW_FRAMES
                ),
            })

        df = pd.DataFrame(rows)

        lp("=" * 72)
        lp(f"SWEEP COMPLETE — Grid {grid_name}")
        lp("=" * 72)
        lp()

        passing = df[df["meets_all_criteria"]]
        lp(f"Combos meeting ALL 3 criteria: {len(passing)} / {len(df)}")
        if not passing.empty:
            best = passing.sort_values("t2_total_residual").iloc[0]
            lp(f"Best combo: window={best['window_frames']}  "
               f"frac={best['min_majority_frac']}  "
               f"stubborn={best['stubborn_t2']}  total={best['t2_total_residual']}  "
               f"added_lag={best['added_lag_ms']:.0f}ms")
        else:
            lp("No combo met all 3 criteria. See below for closest misses.")
            # Show combos meeting criteria a and b (just not c)
            partial = df[df["meets_criterion_a"] & df["meets_criterion_b"]]
            if not partial.empty:
                lp(f"\nCombos meeting (a) and (b) but not (c) — window > {SUCCESS_MAX_WINDOW_FRAMES}:")
                for _, row in partial.sort_values("window_frames").iterrows():
                    lp(f"  window={row['window_frames']}  frac={row['min_majority_frac']}  "
                       f"stubborn={row['stubborn_t2']}  added_lag={row['added_lag_ms']:.0f}ms")
            else:
                lp("No combo met criteria (a) and (b). Tier-2a cannot suppress stubborn clip "
                   "without regression.")
        lp()

        # Top-20 by total residual (ascending) among zero-spurious combos
        zero_spurious = df[df["meets_criterion_b"]].sort_values("t2_total_residual")
        lp("Top-10 combos by total residual (zero spurious only):")
        lp(f"{'window':>8}  {'frac':>6}  {'t2_total':>10}  {'stubborn_t2':>12}  "
           f"{'added_lag_ms':>13}  {'all_ok':>7}")
        for _, row in zero_spurious.head(10).iterrows():
            lp(f"{int(row['window_frames']):>8}  {row['min_majority_frac']:>6.1f}  "
               f"{row['t2_total_residual']:>10.2f}  {row['stubborn_t2']:>12.2f}  "
               f"{row['added_lag_ms']:>13.0f}  {str(row['meets_all_criteria']):>7}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Tier-2a Dual-Grid Sweep — tune_tier2_mode_filter.py")
    print(f"Started: {datetime.datetime.now().isoformat()}")
    print("=" * 72)
    print()

    # --- Load Tier-1 config (frozen) ---
    tier1_cfg = load_tier1_config()
    print(f"Tier-1 config loaded from {TIER1_CONFIG}:")
    print(f"  alpha={tier1_cfg['alpha']}  sw_thr={tier1_cfg['switch_threshold']}  "
          f"dwell={tier1_cfg['min_dwell_frames']}  hold_floor={tier1_cfg.get('hold_floor')}")
    print()

    # --- Load model (once) ---
    print("Loading model (used for precompute pass)...")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    loaded_model = load_inference_model(CKPT_PATH, device=device)
    model = loaded_model.model
    transform = loaded_model.transform
    print(f"Model loaded: backbone={loaded_model.backbone_name}  device={device}")
    print()

    # --- Precompute per-clip prob sequences (once, reused for all combos) ---
    video_paths = load_video_paths()
    print(f"Clips to process: {len(video_paths)}")
    print()

    clip_data: list[tuple[str, list[np.ndarray], float]] = []
    t_precompute_start = time.monotonic()

    for i, vpath in enumerate(video_paths):
        clip_name = Path(vpath).name
        all_probs, fps = run_inference_on_clip(vpath, model, transform, device)
        if not all_probs:
            print(f"  [{i+1:2d}/{len(video_paths)}] SKIP {clip_name} (no frames decoded)")
            continue
        clip_data.append((clip_name, all_probs, fps))
        print(f"  [{i+1:2d}/{len(video_paths)}] {clip_name}: {len(all_probs)} frames @ {fps:.1f}fps")

    precompute_s = time.monotonic() - t_precompute_start
    print(f"\nPrecompute done: {len(clip_data)} clips in {precompute_s:.1f}s")
    print()

    # Confirm stubborn clip is present
    stubborn_present = any(STUBBORN_CLIP in cd[0] for cd in clip_data)
    if not stubborn_present:
        print(f"WARNING: stubborn clip '{STUBBORN_CLIP}' not found in clip set. "
              "Success criterion (a) cannot be evaluated.")
    else:
        print(f"Stubborn clip '{STUBBORN_CLIP}' confirmed present in clip set.")
    print()

    # --- Grid A sweep ---
    print(f"Running Grid A (spec grid): windows={GRID_A_WINDOWS}")
    t0 = time.monotonic()
    df_a = run_sweep("A", GRID_A_WINDOWS, MAJORITY_FRACS, clip_data, tier1_cfg, LOG_GRID_A)
    df_a.to_csv(CSV_GRID_A, index=False, encoding="utf-8")
    print(f"Grid A complete in {time.monotonic()-t0:.1f}s → {CSV_GRID_A}")
    print()

    # --- Grid B sweep ---
    print(f"Running Grid B (uniform dense grid): windows={GRID_B_WINDOWS}")
    t0 = time.monotonic()
    df_b = run_sweep("B", GRID_B_WINDOWS, MAJORITY_FRACS, clip_data, tier1_cfg, LOG_GRID_B)
    df_b.to_csv(CSV_GRID_B, index=False, encoding="utf-8")
    print(f"Grid B complete in {time.monotonic()-t0:.1f}s → {CSV_GRID_B}")
    print()

    # --- Combined CSV ---
    df_all = pd.concat([df_a, df_b], ignore_index=True).drop_duplicates(
        subset=["window_frames", "min_majority_frac"]
    ).sort_values(["window_frames", "min_majority_frac"]).reset_index(drop=True)
    df_all.to_csv(CSV_COMBINED, index=False, encoding="utf-8")
    print(f"Combined (deduplicated) results → {CSV_COMBINED}")
    print()

    # --- Comparison summary ---
    with open(LOG_COMPARE, "w", encoding="utf-8") as cmp:
        def cp(msg: str = "") -> None:
            print(msg)
            print(msg, file=cmp)

        cp("=" * 72)
        cp("Tier-2a Dual-Grid Comparison Summary")
        cp(f"Generated: {datetime.datetime.now().isoformat()}")
        cp("=" * 72)
        cp()
        cp(f"Pre-declared success criterion:")
        cp(f"  (a) stubborn clip residual < {TIER1_STUBBORN_RESIDUAL} switches/min")
        cp(f"  (b) spurious_new == 0 for all clips")
        cp(f"  (c) window_frames <= {SUCCESS_MAX_WINDOW_FRAMES}")
        cp()

        for grid_name, df in [("A (spec grid)", df_a), ("B (uniform dense)", df_b)]:
            passing = df[df["meets_all_criteria"]]
            cp(f"Grid {grid_name}: {len(passing)} / {len(df)} combos meet ALL criteria")
            if not passing.empty:
                best = passing.sort_values("t2_total_residual").iloc[0]
                cp(f"  Best: window={int(best['window_frames'])}  "
                   f"frac={best['min_majority_frac']}  "
                   f"stubborn_t2={best['stubborn_t2']}  "
                   f"total_t2={best['t2_total_residual']}  "
                   f"added_lag={best['added_lag_ms']:.0f}ms")
            else:
                partial = df[df["meets_criterion_a"] & df["meets_criterion_b"]]
                cp(f"  No combo meets all 3. Combos meeting (a)+(b): {len(partial)}")
                if not partial.empty:
                    best_p = partial.sort_values("window_frames").iloc[0]
                    cp(f"  Closest: window={int(best_p['window_frames'])}  "
                       f"frac={best_p['min_majority_frac']}  "
                       f"stubborn_t2={best_p['stubborn_t2']}  "
                       f"added_lag={best_p['added_lag_ms']:.0f}ms")
            cp()

        # --- Global best across both grids ---
        cp("-" * 72)
        cp("GLOBAL BEST (across both grids, all criteria met, min total residual):")
        all_passing = df_all[df_all["meets_all_criteria"]]
        if not all_passing.empty:
            global_best = all_passing.sort_values("t2_total_residual").iloc[0]
            cp(f"  window_frames={int(global_best['window_frames'])}")
            cp(f"  min_majority_frac={global_best['min_majority_frac']}")
            cp(f"  stubborn_t2={global_best['stubborn_t2']} switches/min")
            cp(f"  t2_total_residual={global_best['t2_total_residual']} switches/min")
            cp(f"  added_lag_ms={global_best['added_lag_ms']:.0f} ms")
            cp(f"  meets_all_criteria=True")

            # Write config
            config_out = {
                "window_frames": int(global_best["window_frames"]),
                "min_majority_frac": float(global_best["min_majority_frac"]),
                "_comment": (
                    f"Selected by tune_tier2_mode_filter.py — global best across grids A+B. "
                    f"Stubborn clip residual: {global_best['stubborn_t2']} switches/min "
                    f"(Tier-1 baseline: {TIER1_STUBBORN_RESIDUAL}). "
                    f"Added lag: {global_best['added_lag_ms']:.0f}ms worst-case at 23.7fps."
                ),
            }
            with open(TIER2_CONFIG_OUTPUT, "w", encoding="utf-8") as cfg_fh:
                yaml.dump(config_out, cfg_fh, default_flow_style=False, allow_unicode=True)
            cp()
            cp(f"configs/smoothing_tier2a.yaml written → window_frames={int(global_best['window_frames'])}  "
               f"min_majority_frac={global_best['min_majority_frac']}")

            print()
            print("SUCCESS: Tier-2a meets all pre-declared criteria.")
            print(f"Selected: window_frames={int(global_best['window_frames'])}  "
                  f"min_majority_frac={global_best['min_majority_frac']}")
            print(f"Stubborn clip: {TIER1_STUBBORN_RESIDUAL} → {global_best['stubborn_t2']} switches/min")
        else:
            cp("  No combo across either grid meets all 3 criteria.")
            cp("  Tier-2a FAILS the pre-declared success gate.")
            cp("  Decision: escalate to Tier-2b (learned temporal head).")
            cp("  See docs/phases/phase_07/tier2_results.md for full reasoning.")
            print()
            print("FAIL: No Tier-2a combo meets all 3 success criteria.")
            print("Decision: escalate to Tier-2b. See tier2_results.md.")

        cp()
        cp("Log files:")
        cp(f"  Grid A log:      {LOG_GRID_A}")
        cp(f"  Grid B log:      {LOG_GRID_B}")
        cp(f"  CSV Grid A:      {CSV_GRID_A}")
        cp(f"  CSV Grid B:      {CSV_GRID_B}")
        cp(f"  CSV combined:    {CSV_COMBINED}")
        cp(f"  Tier-2a config:  {TIER2_CONFIG_OUTPUT}")

    print()
    print(f"Comparison summary → {LOG_COMPARE}")
    print(f"Sweep complete: {datetime.datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
