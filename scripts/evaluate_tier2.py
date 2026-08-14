"""
scripts/evaluate_tier2.py

Full-clip evaluation: Tier-1-only vs. Tier-1+Tier-2a, side-by-side.

Outputs:
  logs/eval/evaluate_tier2.txt          — verbose per-clip log + summary table
  data/processed/eval_tier2/tier2_results.csv   — machine-readable results

CRITICAL checks performed here (must all pass before Tier-2a is considered ready):
  1. Tier-2a does NOT increase switches/min on any clip that Tier-1 already drove to 0.
     ('spurious_new' column must be 0 for every clip where t1_per_min == 0.)
  2. Total switches/min across all 46 clips must not increase vs. Tier-1.
  3. Stubborn clip (202101141947512003470I1.avi) residual reported explicitly.

Run:
    conda run -n fetalplane python scripts/evaluate_tier2.py

Optionally override configs:
    conda run -n fetalplane python scripts/evaluate_tier2.py \
        --tier1-config configs/smoothing_tier1.yaml \
        --tier2-config configs/smoothing_tier2a.yaml
"""
from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, ".")

from scripts.tune_tier1_smoothing import load_video_paths, run_inference_on_clip
from src.realtime.model_loader import load_inference_model
from src.smoothing.tier1 import Tier1Smoother
from src.smoothing.tier2_mode_filter import Tier2ModeFilter

CKPT_PATH = "checkpoints/convnext_tiny/best.pt"
STUBBORN_CLIP = "202101141947512003470I1.avi"
TIER1_BASELINE_TOTAL = 788.75    # raw-argmax switches/min across all 46 clips
TIER1_RESIDUAL_TOTAL = 72.0      # Tier-1 residual (Phase 5 selected combo)

LOG_DIR = Path("logs/eval")
DATA_DIR = Path("data/processed/eval_tier2")


def run_evaluation(tier1_config_path: str, tier2_config_path: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log_path = LOG_DIR / "evaluate_tier2.txt"

    with open(log_path, "w", encoding="utf-8") as log_fh:
        def lp(msg: str = "") -> None:
            print(msg)
            print(msg, file=log_fh)

        lp("=" * 72)
        lp("Tier-2a Evaluation: Tier-1 vs. Tier-1+Tier-2a")
        lp(f"Generated: {datetime.datetime.now().isoformat()}")
        lp("=" * 72)
        lp()

        # Load configs
        with open(tier1_config_path, encoding="utf-8") as fh:
            t1_cfg = yaml.safe_load(fh)
        lp(f"Tier-1 config: {tier1_config_path}")
        lp(f"  alpha={t1_cfg['alpha']}  sw_thr={t1_cfg['switch_threshold']}  "
           f"dwell={t1_cfg['min_dwell_frames']}  hold_floor={t1_cfg.get('hold_floor')}")
        lp()

        with open(tier2_config_path, encoding="utf-8") as fh:
            t2_cfg = yaml.safe_load(fh)
        lp(f"Tier-2a config: {tier2_config_path}")
        lp(f"  window_frames={t2_cfg['window_frames']}  "
           f"min_majority_frac={t2_cfg['min_majority_frac']}")
        lp()

        # Load model
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        lp(f"Device: {device}")
        loaded_model = load_inference_model(CKPT_PATH, device=device)
        lp(f"Model: {loaded_model.backbone_name}")
        lp()

        # Precompute
        video_paths = load_video_paths()
        lp(f"Clips: {len(video_paths)}")
        lp()

        rows = []
        t_start = time.monotonic()

        for i, vpath in enumerate(video_paths):
            clip_name = Path(vpath).name
            all_probs, fps = run_inference_on_clip(
                vpath, loaded_model.model, loaded_model.transform, device
            )
            if not all_probs:
                lp(f"  SKIP {clip_name} (no frames)")
                continue

            # --- Tier-1 only ---
            t1_smoother = Tier1Smoother(
                num_classes=8,
                alpha=float(t1_cfg["alpha"]),
                switch_threshold=float(t1_cfg["switch_threshold"]),
                min_dwell_frames=int(t1_cfg["min_dwell_frames"]),
                hold_floor=t1_cfg.get("hold_floor"),
            )
            mode_filter = Tier2ModeFilter(
                window_frames=int(t2_cfg["window_frames"]),
                min_majority_frac=float(t2_cfg["min_majority_frac"]),
            )

            prev_t1, prev_t2 = None, None
            t1_switches, t2_switches = 0, 0

            for probs in all_probs:
                t1_label, _, _, _ = t1_smoother.step(probs)
                t2_label = mode_filter.step(t1_label)

                if prev_t1 is not None and t1_label != prev_t1:
                    t1_switches += 1
                if prev_t2 is not None and t2_label != prev_t2:
                    t2_switches += 1
                prev_t1 = t1_label
                prev_t2 = t2_label

            duration_sec = len(all_probs) / fps
            t1_per_min = (t1_switches / duration_sec) * 60.0
            t2_per_min = (t2_switches / duration_sec) * 60.0
            delta = t2_per_min - t1_per_min
            spurious_new = t2_per_min > 0.0 and t1_per_min == 0.0

            is_stubborn = STUBBORN_CLIP in clip_name
            lp(f"  [{i+1:2d}/{len(video_paths)}] {clip_name}")
            lp(f"    Tier-1: {t1_per_min:.2f}/min  Tier-2a: {t2_per_min:.2f}/min  "
               f"delta={delta:+.2f}  spurious={'YES ⚠' if spurious_new else 'no'}"
               + ("  ← STUBBORN CLIP" if is_stubborn else ""))
            lp()

            rows.append({
                "clip": clip_name,
                "n_frames": len(all_probs),
                "fps": round(fps, 1),
                "t1_switches_per_min": round(t1_per_min, 2),
                "t2_switches_per_min": round(t2_per_min, 2),
                "delta": round(delta, 2),
                "spurious_new": spurious_new,
                "is_stubborn": is_stubborn,
            })

        elapsed = time.monotonic() - t_start
        df = pd.DataFrame(rows)
        df.to_csv(DATA_DIR / "tier2_results.csv", index=False, encoding="utf-8")

        # --- Summary ---
        lp("=" * 72)
        lp("SUMMARY")
        lp("=" * 72)
        lp()

        t1_total = df["t1_switches_per_min"].sum()
        t2_total = df["t2_switches_per_min"].sum()
        total_delta = t2_total - t1_total
        spurious_count = df["spurious_new"].sum()

        lp(f"Clips evaluated:            {len(df)}")
        lp(f"Evaluation time:            {elapsed:.1f}s")
        lp()
        lp(f"Phase 5 raw-argmax baseline:  {TIER1_BASELINE_TOTAL:.2f} switches/min")
        lp(f"Tier-1-only total residual:   {t1_total:.2f} switches/min")
        lp(f"Tier-1+Tier-2a total residual:{t2_total:.2f} switches/min")
        lp(f"Net delta (Tier-2a effect):   {total_delta:+.2f} switches/min")
        lp()

        # Stubborn clip row
        stubborn_rows = df[df["is_stubborn"]]
        if not stubborn_rows.empty:
            for _, sr in stubborn_rows.iterrows():
                lp(f"Stubborn clip ({sr['clip']}):")
                lp(f"  Tier-1: {sr['t1_switches_per_min']:.2f}  "
                   f"Tier-2a: {sr['t2_switches_per_min']:.2f}  "
                   f"delta={sr['delta']:+.2f}")
        lp()

        # Spurious check — CRITICAL
        lp(f"Spurious switches introduced: {int(spurious_count)} clips")
        if spurious_count == 0:
            lp("  [PASS] Tier-2a introduced no spurious switches on any stable clip.")
        else:
            lp("  [FAIL] Tier-2a introduced spurious switches on the following clips:")
            for _, row in df[df["spurious_new"]].iterrows():
                lp(f"    {row['clip']}: Tier-1={row['t1_switches_per_min']}  "
                   f"Tier-2a={row['t2_switches_per_min']}")
        lp()

        # Non-regression check
        regressions = df[(df["t2_switches_per_min"] > df["t1_switches_per_min"]) & ~df["is_stubborn"]]
        lp(f"Non-stubborn clips with any regression (t2 > t1): {len(regressions)}")
        if len(regressions) == 0:
            lp("  [PASS] No regression on any non-stubborn clip.")
        else:
            for _, row in regressions.iterrows():
                lp(f"    {row['clip']}: {row['t1_switches_per_min']} → {row['t2_switches_per_min']}")
        lp()

        # Per-clip table (sorted by t2 descending)
        lp("Per-clip results (sorted by Tier-2a residual, descending):")
        lp(f"{'Clip':<50} {'T1/min':>7} {'T2/min':>7} {'delta':>7} {'Spurious':>9}")
        lp("-" * 82)
        for _, row in df.sort_values("t2_switches_per_min", ascending=False).iterrows():
            lp(f"{row['clip']:<50} {row['t1_switches_per_min']:>7.2f} "
               f"{row['t2_switches_per_min']:>7.2f} {row['delta']:>+7.2f} "
               f"{'YES' if row['spurious_new'] else '':>9}")
        lp()
        lp(f"Results CSV: {DATA_DIR / 'tier2_results.csv'}")
        lp(f"Log: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Tier-2a mode filter against Tier-1.")
    parser.add_argument("--tier1-config", default="configs/smoothing_tier1.yaml")
    parser.add_argument("--tier2-config", default="configs/smoothing_tier2a.yaml")
    args = parser.parse_args()

    if not Path(args.tier2_config).exists():
        print(f"ERROR: tier2 config not found: {args.tier2_config}")
        print("Run scripts/tune_tier2_mode_filter.py first to generate it.")
        sys.exit(1)

    run_evaluation(args.tier1_config, args.tier2_config)


if __name__ == "__main__":
    main()
