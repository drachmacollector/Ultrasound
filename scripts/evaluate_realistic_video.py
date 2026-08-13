"""
scripts/evaluate_realistic_video.py

Task 3 — Realistic-video evaluation: accuracy (synthetic only) + stability (all clips).

Per PHASE_6_KICKOFF_PROMPT.md §4. Two structurally separate halves:

  §4a  Frame-level accuracy (raw vs. smoothed) — SYNTHETIC CLIPS ONLY
       Ground truth is implicit in filename prefix (e.g. Brain_Trans_cerebellum_clip01.mp4).
       Do NOT compute or report a frame-level accuracy number against any IUGC clip;
       IUGC labels are for a different clinical task (§0.6 constraint).

  §4b  Video stability metrics — all clips (16 synthetic + up to 30 IUGC)
       - label-switches-per-minute (raw baseline vs. smoothed)
       - mean dwell time on displayed label (new metric)
       - latency-to-stabilize: NOT computed — honest limitation stated verbatim (§0.7)

Outputs:
  logs/eval/evaluate_realistic_video.txt
  data/processed/eval_realistic_video/realistic_video_results.csv
    columns: clip_name, is_synthetic, ground_truth_class_or_null,
             raw_accuracy_or_null, smoothed_accuracy_or_null,
             raw_switches_per_min, smoothed_switches_per_min, mean_dwell_ms
"""
from __future__ import annotations

import glob
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")  # ensure `src` package importable from repo root

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.dataset import CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
from src.data.transforms import prep_frame_grayscale_to_rgb
from src.realtime.model_loader import load_inference_model
from src.smoothing.tier1 import Tier1Smoother

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CKPT_PATH          = "checkpoints/convnext_tiny/best.pt"
SMOOTHING_CFG      = "configs/smoothing_tier1.yaml"
SYNTHETIC_DIR      = "data/processed/synthetic_clips"
IUGC_VIDEO_DIR     = "data/raw/iugc_video/DatasetV3"
LOG_PATH           = Path("logs/eval/evaluate_realistic_video.txt")
CSV_PATH           = Path("data/processed/eval_realistic_video/realistic_video_results.csv")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_smoothing_config(cfg_path: str) -> dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_gt_class(clip_name: str) -> str | None:
    """Extract the canonical class name from a synthetic clip filename.

    E.g. 'Brain_Trans_cerebellum_clip01.mp4' -> 'Brain_Trans_cerebellum'
    Pattern: everything before the trailing _clip<digits>.mp4
    Returns None if the filename doesn't match (e.g. for IUGC clips).
    """
    stem = Path(clip_name).stem  # drop .mp4
    m = re.match(r"^(.+)_clip\d+$", stem)
    if m:
        candidate = m.group(1)
        if candidate in CLASS_TO_IDX:
            return candidate
    return None


def load_video_paths_synthetic() -> list[str]:
    return sorted(glob.glob(str(Path(SYNTHETIC_DIR) / "*.mp4")))


def load_video_paths_iugc() -> list[str]:
    """First 10 .avi per split under DatasetV3/{train,val,test}/videos/,
    matching the exact glob established in tune_tier1_smoothing.py::load_video_paths()."""
    iugc_dir = Path(IUGC_VIDEO_DIR)
    paths: list[str] = []
    if iugc_dir.exists():
        for split in ("train", "val", "test"):
            vid_dir = iugc_dir / split / "videos"
            if vid_dir.exists():
                paths.extend(str(p) for p in sorted(vid_dir.glob("*.avi"))[:10])
    return paths


def run_inference_on_clip(
    video_path: str,
    model: torch.nn.Module,
    transform,
    device: torch.device,
) -> tuple[list[np.ndarray], float]:
    """Run model frame-by-frame and return (list_of_prob_vectors, video_fps)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [], 25.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    all_probs: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img_rgb = prep_frame_grayscale_to_rgb(frame)
        tensor = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(tensor)
            # .float() ensures float32 on both CPU and GPU (avoids BFloat16 from autocast)
            probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
        all_probs.append(probs)
    cap.release()
    return all_probs, fps


def measure_inference_fps(
    model: torch.nn.Module,
    transform,
    device: torch.device,
    n_frames: int = 100,
) -> float:
    """Warm-up + timed loop to measure sustained inference FPS for ms calibration."""
    dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    tensor = transform(image=dummy)["image"].unsqueeze(0).to(device)
    # Warm-up
    for _ in range(10):
        with torch.no_grad():
            model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_frames):
        with torch.no_grad():
            model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return n_frames / elapsed


def compute_raw_switches_per_min(all_probs: list[np.ndarray], fps: float) -> float:
    """Label-switches-per-minute from raw argmax (no smoothing)."""
    switches = 0
    prev = None
    for probs in all_probs:
        lbl = int(np.argmax(probs))
        if prev is not None and lbl != prev:
            switches += 1
        prev = lbl
    duration_sec = len(all_probs) / fps if fps > 0 else 1.0
    return (switches / duration_sec) * 60.0 if duration_sec > 0 else 0.0


def run_smoother_over_clip(
    all_probs: list[np.ndarray],
    fps: float,
    smoother_cfg: dict,
) -> tuple[list[int], float, float, float]:
    """Run a fresh Tier1Smoother over a prob sequence.

    Returns:
        smoothed_labels   : list of per-frame displayed labels
        switches_per_min  : smoothed label-switch rate
        mean_dwell_ms     : mean consecutive-frame run length × ms_per_frame
        ms_per_frame      : 1000 / fps (for logging)
    """
    smoother = Tier1Smoother(
        num_classes=NUM_CLASSES,
        alpha=smoother_cfg["alpha"],
        switch_threshold=smoother_cfg["switch_threshold"],
        min_dwell_frames=smoother_cfg["min_dwell_frames"],
        hold_floor=smoother_cfg.get("hold_floor"),  # may be None
    )
    smoothed_labels: list[int] = []
    for probs in all_probs:
        label, _, _, _ = smoother.step(probs)
        smoothed_labels.append(label)

    # Switches per minute
    switches = sum(
        1 for i in range(1, len(smoothed_labels))
        if smoothed_labels[i] != smoothed_labels[i - 1]
    )
    duration_sec = len(all_probs) / fps if fps > 0 else 1.0
    switches_per_min = (switches / duration_sec) * 60.0 if duration_sec > 0 else 0.0

    # Mean dwell time: compute run-length encoding of smoothed_labels
    ms_per_frame = 1000.0 / fps if fps > 0 else (1000.0 / 25.0)
    if smoothed_labels:
        runs: list[int] = []
        run_len = 1
        for i in range(1, len(smoothed_labels)):
            if smoothed_labels[i] == smoothed_labels[i - 1]:
                run_len += 1
            else:
                runs.append(run_len)
                run_len = 1
        runs.append(run_len)
        mean_dwell_ms = float(np.mean(runs)) * ms_per_frame
    else:
        mean_dwell_ms = 0.0

    return smoothed_labels, switches_per_min, mean_dwell_ms, ms_per_frame


def compute_frame_accuracy(
    all_probs: list[np.ndarray],
    gt_class_idx: int,
    smoothed_labels: list[int] | None = None,
) -> tuple[float, float | None]:
    """Return (raw_accuracy, smoothed_accuracy_or_None)."""
    if not all_probs:
        return 0.0, None
    raw_correct = sum(1 for p in all_probs if int(np.argmax(p)) == gt_class_idx)
    raw_acc = raw_correct / len(all_probs)
    smoothed_acc = None
    if smoothed_labels is not None:
        sm_correct = sum(1 for lbl in smoothed_labels if lbl == gt_class_idx)
        smoothed_acc = sm_correct / len(smoothed_labels)
    return raw_acc, smoothed_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Load model + smoothing config
    # ------------------------------------------------------------------ #
    log.info("Loading checkpoint: %s", CKPT_PATH)
    loaded = load_inference_model(CKPT_PATH)
    model, device, transform = loaded.model, loaded.device, loaded.transform
    log.info("Device: %s  |  Backbone: %s", device, loaded.backbone_name)

    log.info("Loading smoothing config: %s", SMOOTHING_CFG)
    smoother_cfg = load_smoothing_config(SMOOTHING_CFG)
    log.info("Smoother params: %s", smoother_cfg)

    # ------------------------------------------------------------------ #
    # Measure inference FPS for ms calibration
    # ------------------------------------------------------------------ #
    log.info("Measuring inference FPS (100-frame benchmark) …")
    infer_fps = measure_inference_fps(model, transform, device)
    log.info("Inference FPS: %.2f  →  %.2f ms/frame", infer_fps, 1000.0 / infer_fps)

    # ------------------------------------------------------------------ #
    # Gather clip paths
    # ------------------------------------------------------------------ #
    synthetic_paths = load_video_paths_synthetic()
    iugc_paths = load_video_paths_iugc()
    log.info(
        "Clips found: %d synthetic, %d IUGC  (total: %d)",
        len(synthetic_paths), len(iugc_paths), len(synthetic_paths) + len(iugc_paths),
    )

    all_paths = synthetic_paths + iugc_paths

    # ------------------------------------------------------------------ #
    # Main evaluation loop
    # ------------------------------------------------------------------ #
    rows: list[dict] = []

    # §4a tracking
    synthetic_raw_accs: list[float] = []
    synthetic_sm_accs: list[float] = []
    clip_flags: list[str] = []   # flags for per-clip anomalies

    # §4b tracking
    all_raw_spm: list[float] = []
    all_sm_spm: list[float] = []
    all_dwell_ms: list[float] = []

    lines: list[str] = []  # for text log

    def log_line(s: str) -> None:
        lines.append(s)
        log.info(s) if s.strip() else None

    lines.append("REALISTIC-VIDEO EVALUATION — Phase 6 Task 3")
    lines.append(f"Checkpoint : {CKPT_PATH}")
    lines.append(f"Smoother   : {smoother_cfg}")
    lines.append(f"Inference FPS (benchmark): {infer_fps:.2f} fps  ({1000.0/infer_fps:.2f} ms/frame)")
    lines.append(f"Clips: {len(synthetic_paths)} synthetic  +  {len(iugc_paths)} IUGC  =  {len(all_paths)} total")
    lines.append("")
    lines.append("=" * 72)
    lines.append("§4a  FRAME-LEVEL ACCURACY — SYNTHETIC CLIPS ONLY")
    lines.append("     (IUGC clips have no accuracy ground truth — §0.6 constraint)")
    lines.append("=" * 72)
    lines.append(
        f"{'Clip':<45s}  {'GT Class':<25s}  {'Raw Acc':>8s}  {'Sm Acc':>8s}  {'Delta':>8s}  Flag"
    )
    lines.append("-" * 108)

    for clip_path in all_paths:
        clip_name = Path(clip_path).name
        is_synthetic = clip_path in synthetic_paths
        gt_class = parse_gt_class(clip_name) if is_synthetic else None

        log.info("Processing: %s …", clip_name)
        all_probs, video_fps = run_inference_on_clip(clip_path, model, transform, device)

        if not all_probs:
            log.warning("  Could not read any frames from %s — skipping.", clip_name)
            continue

        # --- Smoothed labels (fresh smoother per clip) ---
        smoothed_labels, sm_spm, mean_dwell_ms, ms_per_frame = run_smoother_over_clip(
            all_probs, video_fps, smoother_cfg
        )

        # --- Raw switches per minute ---
        raw_spm = compute_raw_switches_per_min(all_probs, video_fps)

        # --- Duration ---
        duration_sec = len(all_probs) / video_fps if video_fps > 0 else 0.0


        all_raw_spm.append(raw_spm)
        all_sm_spm.append(sm_spm)
        all_dwell_ms.append(mean_dwell_ms)

        # --- §4a: accuracy only for synthetic ---
        raw_acc_val: float | None = None
        sm_acc_val: float | None = None
        flag = ""

        if is_synthetic and gt_class is not None:
            gt_idx = CLASS_TO_IDX[gt_class]
            raw_acc_val, sm_acc_val = compute_frame_accuracy(
                all_probs, gt_idx, smoothed_labels
            )
            assert sm_acc_val is not None
            synthetic_raw_accs.append(raw_acc_val)
            synthetic_sm_accs.append(sm_acc_val)
            delta = sm_acc_val - raw_acc_val

            # Flag if smoothed < raw
            if delta < 0:
                raw_was_poor = raw_acc_val < 0.5
                flag = (
                    "⚠ SMOOTHED<RAW (raw acc already poor — smoother dragging stale label)"
                    if raw_was_poor
                    else "⚠ SMOOTHED<RAW (red flag: smoother degrading good raw accuracy)"
                )
                clip_flags.append(f"{clip_name}: delta={delta:+.4f}  raw={raw_acc_val:.4f}  {flag}")

            raw_acc_str = f"{raw_acc_val:.4f}"
            sm_acc_str = f"{sm_acc_val:.4f}"
            delta_str = f"{delta:+.4f}"
            lines.append(
                f"{clip_name:<45s}  {gt_class:<25s}  {raw_acc_str:>8s}  {sm_acc_str:>8s}  {delta_str:>8s}  {flag}"
            )

        # --- Gate check: dwell time == full clip length, BUT only meaningful
        # when the raw argmax was actually switching (raw_spm > 0). When the
        # raw model never switches on a clip, mean_dwell == clip_full is the
        # expected and correct outcome — the smoother correctly holds a single
        # stable label throughout. Only flag the genuinely alarming case:
        # raw transitions existed but smoother appears permanently stuck. ---
        clip_full_ms = duration_sec * 1000.0
        if raw_spm > 0 and mean_dwell_ms > 0 and abs(mean_dwell_ms - clip_full_ms) < 1.0:
            gate_flag = (
                f"GATE ⚠: {clip_name} mean_dwell_ms ≈ full clip duration "
                f"({mean_dwell_ms:.1f} ms) despite raw_spm={raw_spm:.1f} — "
                f"smoother may be permanently stuck on cold-start label"
            )
            log.warning(gate_flag)
            lines.append(f"    {gate_flag}")

        rows.append({
            "clip_name": clip_name,
            "is_synthetic": is_synthetic,
            "ground_truth_class_or_null": gt_class if gt_class else "",
            "raw_accuracy_or_null": raw_acc_val if raw_acc_val is not None else "",
            "smoothed_accuracy_or_null": sm_acc_val if sm_acc_val is not None else "",
            "raw_switches_per_min": round(raw_spm, 4),
            "smoothed_switches_per_min": round(sm_spm, 4),
            "mean_dwell_ms": round(mean_dwell_ms, 2),
            "n_frames": len(all_probs),
            "clip_duration_sec": round(duration_sec, 3),
        })

    # ------------------------------------------------------------------
    # §4a: Aggregate accuracy summary
    # ------------------------------------------------------------------
    lines.append("")
    lines.append("-" * 72)
    lines.append("§4a AGGREGATE (mean across all synthetic clips)")
    lines.append("-" * 72)
    if synthetic_raw_accs:
        mean_raw = float(np.mean(synthetic_raw_accs))
        mean_sm = float(np.mean(synthetic_sm_accs))
        mean_delta = mean_sm - mean_raw
        lines.append(f"  Mean raw accuracy     : {mean_raw:.4f}  ({mean_raw*100:.1f}%)")
        lines.append(f"  Mean smoothed accuracy: {mean_sm:.4f}  ({mean_sm*100:.1f}%)")
        lines.append(f"  Mean delta (sm - raw) : {mean_delta:+.4f}")
        if clip_flags:
            lines.append("")
            lines.append("  Per-clip anomaly flags:")
            for f in clip_flags:
                lines.append(f"    {f}")
    else:
        lines.append("  No synthetic clips processed — cannot compute aggregate.")

    # ------------------------------------------------------------------
    # §4b: Stability metrics
    # ------------------------------------------------------------------
    lines.append("")
    lines.append("=" * 72)
    lines.append("§4b  VIDEO STABILITY METRICS — ALL CLIPS")
    lines.append("=" * 72)
    lines.append(
        f"{'Clip':<45s}  {'RawSPM':>8s}  {'SmSPM':>8s}  {'Reduction':>10s}  {'DwellMs':>10s}"
    )
    lines.append("-" * 90)

    for row in rows:
        raw_s = row["raw_switches_per_min"]
        sm_s  = row["smoothed_switches_per_min"]
        red   = (1.0 - sm_s / raw_s) * 100.0 if raw_s > 0 else 0.0
        lines.append(
            f"{row['clip_name']:<45s}  {raw_s:>8.2f}  {sm_s:>8.2f}  {red:>9.1f}%  {row['mean_dwell_ms']:>10.1f}"
        )

    lines.append("")
    lines.append("-" * 72)
    lines.append("§4b AGGREGATE (all clips)")
    lines.append("-" * 72)

    # Reproduce Phase 5's documented numbers: total switches across clip set
    # Phase 5 measured 788.75 switches/min total (Task 4) → 72.00/min (Task 6)
    agg_raw_spm  = float(np.mean(all_raw_spm)) if all_raw_spm else 0.0
    agg_sm_spm   = float(np.mean(all_sm_spm))  if all_sm_spm  else 0.0
    agg_reduction = (1.0 - agg_sm_spm / agg_raw_spm) * 100.0 if agg_raw_spm > 0 else 0.0
    mean_dwell   = float(np.mean(all_dwell_ms)) if all_dwell_ms else 0.0
    total_raw_spm_summed = float(np.sum(all_raw_spm))  # Phase 5's "total" metric
    total_sm_spm_summed  = float(np.sum(all_sm_spm))

    lines.append(f"  Clips evaluated                    : {len(rows)}")
    lines.append(f"  Mean raw switches/min  (per-clip)  : {agg_raw_spm:.2f}")
    lines.append(f"  Mean smoothed spm      (per-clip)  : {agg_sm_spm:.2f}")
    lines.append(f"  Mean reduction                     : {agg_reduction:.1f}%")
    lines.append(f"  Sum raw switches/min   (all clips) : {total_raw_spm_summed:.2f}  (Phase 5 reference: 788.75)")
    lines.append(f"  Sum smoothed spm       (all clips) : {total_sm_spm_summed:.2f}  (Phase 5 reference:  72.00)")
    lines.append(f"  Mean dwell time (smoothed)         : {mean_dwell:.1f} ms")

    # Check Phase 5 reproduction
    raw_match = abs(total_raw_spm_summed - 788.75)
    sm_match  = abs(total_sm_spm_summed  -  72.00)
    if raw_match < 50.0 and sm_match < 20.0:
        lines.append("  Phase 5 reproduction check        : PASS ✓ (within acceptable noise)")
    else:
        lines.append(
            f"  Phase 5 reproduction check        : MISMATCH ⚠ "
            f"(raw Δ={raw_match:.1f}, smoothed Δ={sm_match:.1f}) — investigate data drift or re-implementation inconsistency"
        )

    lines.append("")
    lines.append("-" * 72)
    lines.append("§4b LATENCY-TO-STABILIZE — HONEST LIMITATION STATEMENT")
    lines.append("-" * 72)
    lines.append(
        "True transition-latency validation was not possible in this project because no\n"
        "available real video source contains an annotated genuine plane-to-plane transition\n"
        "— the primary dataset (FETAL_PLANES_DB) never released source video, and manual\n"
        "annotation of the 5 IUGC candidate clips (Phase 5, Task 5) found zero raw\n"
        "model-level transitions to time. The dwell-time metric above (§4b) is the closest\n"
        "available proxy for responsiveness, but it measures steady-state holding behaviour,\n"
        "not transition-tracking speed."
    )

    # ------------------------------------------------------------------
    # Write outputs
    # ------------------------------------------------------------------
    summary_text = "\n".join(lines)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(summary_text)
    log.info("Summary log saved → %s", LOG_PATH)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    log.info("Per-clip CSV saved → %s", CSV_PATH)

    print(summary_text)
    log.info("Task 3 complete.")


if __name__ == "__main__":
    main()
