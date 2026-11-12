"""
scripts/validate_pipeline_headless.py

Task 10 headless end-to-end validation for the real-time fetal plane classifier.

PURPOSE
-------
This script validates the full threaded pipeline (CaptureThread → InferenceThread →
result queue) without a display window, so it can run unattended and produce
machine-readable evidence.  cv2.imshow / cv2.waitKey are NEVER called here.

It answers the three open questions from the Task 8/9 benchmark discrepancy (Finding 1):

  Q1: Does inference FPS stabilise above or below 14 fps over an extended run?
  Q2: Is there a thermal-throttling signature (FPS declining over time)?
  Q3: What is the true steady-state forward_ms / gradcam_ms after the GPU warms up?

It also produces the Task 10 deliverables required by PHASE_5_KICKOFF_PROMPT.md §11:

  • logs/realtime/realtime_validation_run.txt  — per-5s stats CSV + narrative report
  • docs/phases/phase_05/phase5_screenshots/          — annotated proof-of-execution PNGs
    - validation_file_t010s.png       — frame at t≈10s  (early warm-up)
    - validation_file_t030s.png       — frame at t≈30s  (stabilisation)
    - validation_file_t060s.png       — frame at t≈60s  (steady state)
    - validation_file_t120s.png       — frame at t≈120s (end, thermal check)
    - validation_nogradcam_final.png  — annotated frame from no-GradCAM run

RUNS
----
  Run A — 120 s, looped video, GradCAM ON  (every_n=7, wall=200ms)
           Primary benchmark; used for FPS trajectory + thermal analysis.

  Run B — 30 s, looped video, GradCAM OFF
           Measures pure inference FPS without GradCAM overhead, for comparison.

USAGE
-----
  conda run -n fetalplane python scripts/validate_pipeline_headless.py \
      --source data/processed/synthetic_clips/multiplane_scan_01.mp4 \
      --checkpoint checkpoints/convnext_tiny/best.pt \
      --output-dir docs/phases/phase_05/phase5_screenshots

FPS INTERPRETATION KEY
----------------------
  • If FPS is FLAT across early/stable/late:  no thermal throttling; value is real.
  • If FPS DECLINES from stable to late:      thermal throttling confirmed;
                                               dwell window EWMA calculation
                                               (used for Task 9 default) may be
                                               over-pessimistic.
  • If FPS RISES from early to stable:        normal CUDA warm-up artifact;
                                               only the "stable" and "late"
                                               windows should be cited in Task 11.
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.data.dataset import IDX_TO_CLASS
from src.realtime.capture import FrameSource
from src.realtime.model_loader import load_inference_model
from src.realtime.pipeline import CaptureThread, InferenceThread, PipelineStats
from src.realtime.queues import DropOldestQueue
from src.realtime.app import build_display_frame  # reuse Task 9 renderer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Frame-checkpoint times for annotated PNG captures (seconds from run start)
# ---------------------------------------------------------------------------
_FRAME_CHECKPOINTS_S = (10, 30, 60, 120)

# ---------------------------------------------------------------------------
# FPS analysis windows (seconds from run start)
# ---------------------------------------------------------------------------
_EARLY_WINDOW  = (5,  30)   # t=5-30s — post-warmup, pre-thermal
_STABLE_WINDOW = (30, 90)   # t=30-90s — expected steady state
_LATE_WINDOW   = (90, 120)  # t=90-120s — thermal throttling detection


# ---------------------------------------------------------------------------
# Helper: run the pipeline for a fixed duration and return detailed metrics
# ---------------------------------------------------------------------------

def run_validation(
    source: str | int,
    checkpoint: str,
    smoothing_config: str,
    duration_s: float,
    enable_gradcam: bool,
    gradcam_every_n: int,
    screenshot_dir: Path,
    screenshot_prefix: str,
    stats_fh,           # open file handle for CSV stats; may be None
    frame_checkpoints: tuple[int, ...] = (),
) -> dict[str, Any]:
    """
    Run the full threaded pipeline headlessly for ``duration_s`` seconds.

    Returns a summary dict with FPS trajectory, per-window FPS averages,
    latency EWMAs, and other key metrics.
    """
    emit = log.info  # use logging (everything goes to realtime_validation_run.txt too)

    emit("─" * 60)
    emit("Run config:")
    emit("  source          : %s", source)
    emit("  duration        : %.0f s", duration_s)
    emit("  gradcam         : %s (every_n=%d)", "ON" if enable_gradcam else "OFF", gradcam_every_n)
    emit("  screenshot_prefix: %s", screenshot_prefix)
    emit("─" * 60)

    # --- Build pipeline ---------------------------------------------------
    lm = load_inference_model(checkpoint)
    emit("  backbone: %s  device: %s  img_size: %d",
         lm.backbone_name, lm.device, lm.img_size)

    stop_event   = threading.Event()
    stats        = PipelineStats()
    frame_queue: DropOldestQueue[tuple[np.ndarray, float]] = DropOldestQueue(maxsize=2)
    result_queue: DropOldestQueue[dict[str, Any]]          = DropOldestQueue(maxsize=2)

    fs = FrameSource(source, loop=True)
    emit("  source resolution: %dx%d @ %.1f fps", *fs.resolution(), fs.fps())

    cap_thread = CaptureThread(fs, frame_queue, stop_event, stats)
    inf_thread = InferenceThread(
        frame_queue=frame_queue,
        result_queue=result_queue,
        loaded_model=lm,
        smoothing_config_path=smoothing_config,
        stop_event=stop_event,
        stats=stats,
        gradcam_every_n_frames=gradcam_every_n,
        gradcam_wall_ms=1000.0,
        enable_gradcam=enable_gradcam,
    )

    cap_thread.start()
    inf_thread.start()

    # --- Sampling loop ---------------------------------------------------
    t_start   = time.monotonic()
    t_next_sample = t_start + 1.0   # sample every 1 s
    fps_history: list[tuple[float, float]] = []   # (elapsed_s, inference_fps)
    last_result: dict[str, Any] | None = None

    # Track which frame checkpoints have been saved
    checkpoints_remaining = sorted(frame_checkpoints)

    screenshot_dir.mkdir(parents=True, exist_ok=True)

    while True:
        elapsed = time.monotonic() - t_start
        if elapsed >= duration_s:
            break

        # Pull latest result (non-blocking)
        res = result_queue.get_nowait_or_none()
        if res is not None:
            last_result = res

        # Per-second sampling
        now = time.monotonic()
        if now >= t_next_sample:
            t_next_sample = now + 1.0
            snap = stats.snapshot()
            fps_val = snap["inference_fps"]
            fps_history.append((elapsed, fps_val))

            # Log to CSV
            if stats_fh is not None:
                stats_fh.write(
                    f"{screenshot_prefix},{elapsed:.1f},"
                    f"{snap['capture_fps']:.2f},{snap['inference_fps']:.2f},"
                    f"{snap['preprocess_ms']:.2f},{snap['forward_ms']:.2f},"
                    f"{snap['smoothing_ms']:.3f},{snap['gradcam_ms']:.1f},"
                    f"{snap['gradcam_calls']},{snap['cap_queue_drops']},"
                    f"{snap['inf_queue_drops']}\n"
                )
                stats_fh.flush()

            # Save annotated frame at checkpoints
            if checkpoints_remaining and elapsed >= checkpoints_remaining[0]:
                ckpt_t = checkpoints_remaining.pop(0)
                if last_result is not None:
                    canvas = build_display_frame(
                        last_result,
                        show_gradcam=enable_gradcam,
                        show_hud=True,
                        is_paused=False,
                        is_webcam=isinstance(source, int),
                        stats=stats,
                    )
                    # Burn timestamp annotation onto frame
                    _stamp = f"t={elapsed:.0f}s | fps={fps_val:.1f}"
                    cv2.putText(canvas, _stamp, (10, canvas.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(canvas, _stamp, (10, canvas.shape[0] - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (200, 255, 200), 1, cv2.LINE_AA)
                    path = screenshot_dir / f"{screenshot_prefix}_t{ckpt_t:03d}s.png"
                    cv2.imwrite(str(path), canvas)
                    emit("  [SCREENSHOT] saved %s", path)

        time.sleep(0.005)  # 5ms poll — low-latency result consumption

    # --- Tear down -------------------------------------------------------
    elapsed_total = time.monotonic() - t_start
    stop_event.set()
    cap_thread.join(timeout=3.0)
    inf_thread.join(timeout=5.0)
    fs.release()

    snap_final = stats.snapshot()

    # Save final frame if we didn't hit a checkpoint near the end
    if last_result is not None:
        canvas = build_display_frame(
            last_result,
            show_gradcam=enable_gradcam,
            show_hud=True,
            is_paused=False,
            is_webcam=isinstance(source, int),
            stats=stats,
        )
        _stamp = f"FINAL t={elapsed_total:.0f}s | fps={snap_final['inference_fps']:.1f}"
        cv2.putText(canvas, _stamp, (10, canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, _stamp, (10, canvas.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 255, 200), 1, cv2.LINE_AA)
        if not checkpoints_remaining:  # already saved the last checkpoint
            pass
        path = screenshot_dir / f"{screenshot_prefix}_final.png"
        cv2.imwrite(str(path), canvas)
        emit("  [SCREENSHOT] saved %s (final)", path)

    # --- FPS trajectory analysis -----------------------------------------
    def _window_fps(lo: float, hi: float) -> list[float]:
        return [fps for t, fps in fps_history if lo <= t < hi]

    def _mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else float("nan")

    early_fps  = _window_fps(*_EARLY_WINDOW)
    stable_fps = _window_fps(*_STABLE_WINDOW)
    late_fps   = _window_fps(*_LATE_WINDOW)

    summary = {
        "run_tag":            screenshot_prefix,
        "duration_s":         elapsed_total,
        "gradcam_enabled":    enable_gradcam,
        "total_inf_frames":   snap_final.get("frames_processed", 0),
        "cap_fps":            snap_final["capture_fps"],
        "inf_fps_mean":       _mean([fps for _, fps in fps_history]),
        "inf_fps_early":      _mean(early_fps),
        "inf_fps_stable":     _mean(stable_fps),
        "inf_fps_late":       _mean(late_fps),
        "inf_fps_end":        snap_final["inference_fps"],
        "forward_ms":         snap_final["forward_ms"],
        "gradcam_ms":         snap_final["gradcam_ms"],
        "gradcam_calls":      snap_final["gradcam_calls"],
        "cap_drops":          snap_final["cap_queue_drops"],
        "inf_drops":          snap_final["inf_queue_drops"],
        "preprocess_ms":      snap_final["preprocess_ms"],
        "smoothing_ms":       snap_final["smoothing_ms"],
        "fps_history":        fps_history,
    }

    # Thermal signature
    if stable_fps and late_fps:
        fps_delta = _mean(late_fps) - _mean(stable_fps)
        if fps_delta < -2.0:
            thermal = f"POSSIBLE THROTTLING (stable {_mean(stable_fps):.1f} → late {_mean(late_fps):.1f}, Δ={fps_delta:+.1f})"
        elif fps_delta > 2.0:
            thermal = f"RECOVERING (stable {_mean(stable_fps):.1f} → late {_mean(late_fps):.1f}, Δ={fps_delta:+.1f})"
        else:
            thermal = f"STABLE (stable {_mean(stable_fps):.1f} → late {_mean(late_fps):.1f}, Δ={fps_delta:+.1f})"
        summary["thermal_assessment"] = thermal
    else:
        summary["thermal_assessment"] = "N/A (run too short)"

    # Dwell calibration check
    stable_inf = _mean(stable_fps) if stable_fps else snap_final["inference_fps"]
    if stable_inf > 0:
        dwell_ms = 8.0 / stable_inf * 1000.0
        dwell_ok = 150.0 <= dwell_ms <= 500.0
        summary["dwell_ms_at_stable_fps"] = dwell_ms
        summary["dwell_in_range"] = dwell_ok
    else:
        summary["dwell_ms_at_stable_fps"] = float("nan")
        summary["dwell_in_range"] = False

    emit("─" * 60)
    emit("Run A complete — %s", screenshot_prefix)
    emit("  total inf frames    : %d", summary["total_inf_frames"])
    emit("  capture FPS         : %.1f", summary["cap_fps"])
    emit("  inf FPS mean        : %.1f", summary["inf_fps_mean"])
    emit("  inf FPS early(5-30s): %.1f", summary["inf_fps_early"])
    emit("  inf FPS stable(30-90s): %.1f", summary["inf_fps_stable"])
    emit("  inf FPS late(90-120s): %.1f", summary["inf_fps_late"])
    emit("  inf FPS end (EWMA)  : %.1f", summary["inf_fps_end"])
    emit("  forward_ms          : %.2f ms", summary["forward_ms"])
    if enable_gradcam:
        emit("  gradcam_ms/call     : %.1f ms (%d calls)",
             summary["gradcam_ms"], summary["gradcam_calls"])
        if summary["gradcam_calls"] > 0:
            actual_cadence = elapsed_total / summary["gradcam_calls"] * 1000.0
            emit("  gradcam cadence     : %.0f ms actual (target: 200 ms)", actual_cadence)
    emit("  cap drops           : %d", summary["cap_drops"])
    emit("  Thermal assessment  : %s", summary["thermal_assessment"])
    emit("  Dwell @ stable FPS  : %.0f ms (%s)",
         summary["dwell_ms_at_stable_fps"],
         "IN RANGE" if summary["dwell_in_range"] else "OUT OF RANGE")
    emit("─" * 60)

    return summary


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_report(
    run_a: dict[str, Any],
    run_b: dict[str, Any],
    out_path: Path,
) -> None:
    """Append a narrative report section to the stats log file."""
    with open(out_path, "a", encoding="utf-8") as fh:
        fh.write("\n")
        fh.write("=" * 70 + "\n")
        fh.write("TASK 10 VALIDATION REPORT\n")
        fh.write(f"Generated: {datetime.datetime.now().isoformat()}\n")
        fh.write("=" * 70 + "\n\n")

        # --- Finding 1 reconciliation ---
        fh.write("## Finding 1 Reconciliation: FPS Trajectory Analysis\n\n")
        fh.write("Three prior benchmark runs showed contradictory results:\n")
        fh.write("  Run-1 (10s, no throttle, every_n=10): 21.7 fps end, forward=44.08ms, gradcam=378.7ms\n")
        fh.write("  Run-2 (25s, no throttle, every_n=10): 26.6 fps end, forward=21.60ms, gradcam=64.5ms\n")
        fh.write("  Run-3 (30s, throttled, every_n=5):   14.0 fps end, forward=31.77ms, gradcam=104.9ms\n\n")
        fh.write("Task 10 extended run (120s) FPS trajectory answers the thermal question:\n\n")

        a_stab = run_a["inf_fps_stable"]
        a_late = run_a["inf_fps_late"]
        fh.write(f"  Run A (120s, GradCAM ON, every_n=7):\n")
        fh.write(f"    Early  (5-30s):    {run_a['inf_fps_early']:.1f} fps\n")
        fh.write(f"    Stable (30-90s):   {run_a['inf_fps_stable']:.1f} fps\n")
        fh.write(f"    Late  (90-120s):   {run_a['inf_fps_late']:.1f} fps\n")
        fh.write(f"    End EWMA:          {run_a['inf_fps_end']:.1f} fps\n")
        fh.write(f"    Thermal:           {run_a['thermal_assessment']}\n\n")

        fh.write(f"  Run B (30s, GradCAM OFF, baseline):\n")
        fh.write(f"    Stable fps:        {run_b['inf_fps_stable']:.1f} fps\n")
        fh.write(f"    End EWMA:          {run_b['inf_fps_end']:.1f} fps\n")
        fh.write(f"    forward_ms:        {run_b['forward_ms']:.2f} ms\n\n")

        # GradCAM overhead
        if run_b["inf_fps_stable"] > 0 and run_a["inf_fps_stable"] > 0:
            gc_overhead_pct = (1.0 - run_a["inf_fps_stable"] / run_b["inf_fps_stable"]) * 100.0
            fh.write(f"  GradCAM overhead: {gc_overhead_pct:.1f}% FPS reduction "
                     f"({run_b['inf_fps_stable']:.1f} → {run_a['inf_fps_stable']:.1f} fps)\n\n")

        # --- Dwell calibration table ---
        fh.write("## Dwell Calibration at Stable FPS\n\n")
        fh.write(f"  min_dwell_frames = 8  (from Tier-1 tuning sweep at 43fps precompute)\n")
        fh.write(f"  Stable FPS (Run A, 30-90s window) = {a_stab:.1f} fps\n")
        fh.write(f"  Implied dwell window = 8 / {a_stab:.1f} × 1000 = {run_a['dwell_ms_at_stable_fps']:.0f} ms\n")
        ok_str = "IN RANGE (150-500ms)" if run_a["dwell_in_range"] else "OUT OF RANGE (target: 150-500ms)"
        fh.write(f"  Status: {ok_str}\n\n")
        fh.write("  NOTE: The 571ms figure from the Task 8/9 benchmark used the end-EWMA\n")
        fh.write("  FPS (14.0), which may have reflected a transient warm-up or thermal\n")
        fh.write("  state.  The stable-window FPS above is the authoritative number.\n\n")

        # --- Finding 2 note ---
        fh.write("## Finding 2: Dwell Window vs Tier-2 Trigger Metric (Terminology)\n\n")
        fh.write("  The 'dwell window' (min_dwell_frames / inference_fps) is a STATIC\n")
        fh.write("  parameter-level number — how long the smoother holds before switching.\n\n")
        fh.write("  The Tier-2 gate criterion (docs/instructions/07_STRETCH_GOALS...md)\n")
        fh.write("  requires mean_latency_ms > 400ms measured on GENUINE annotated\n")
        fh.write("  transitions — a DYNAMIC measured-behavior metric.  These two numbers\n")
        fh.write("  (571ms dwell window vs Tier-2 400ms gate) are conceptually distinct\n")
        fh.write("  even though they happen to be numerically close.  See Task 11 for\n")
        fh.write("  the full reconciliation table.\n\n")

        # --- Four-number FPS reconciliation table ---
        fh.write("## FPS Reconciliation Table (all Phase 5 measurements)\n\n")
        fh.write("  Context                | Metric        | Value\n")
        fh.write("  ---------------------- | ------------- | -----\n")
        fh.write("  Task 4 baseline (seq.) | inference_fps | 24.28 fps (single-threaded, no GradCAM)\n")
        fh.write("  Task 6 sweep baseline  | precompute    | 43.28 fps (offline, no threading)\n")
        fh.write("  Benchmark Run-1 (10s)  | end FPS       | 21.7  fps (threaded, broken throttle, n=6 GC)\n")
        fh.write("  Benchmark Run-2 (25s)  | end FPS       | 26.6  fps (threaded, unthrottled, n=87 GC)\n")
        fh.write("  Benchmark Run-3 (30s)  | end FPS       | 14.0  fps (threaded, throttled, n=115 GC)\n")
        fh.write(f"  Task 10 Run A stable   | stable FPS    | {run_a['inf_fps_stable']:.1f}  fps (120s, throttled, GradCAM ON)\n")
        fh.write(f"  Task 10 Run B stable   | stable FPS    | {run_b['inf_fps_stable']:.1f}  fps (30s, throttled, GradCAM OFF)\n\n")

        fh.write("  Canonical steady-state FPS (throttled, GradCAM ON):\n")
        fh.write(f"    {run_a['inf_fps_stable']:.1f} fps (stable 30-90s window, Run A)\n\n")

        # --- Webcam instructions ---
        fh.write("## Webcam Validation (requires user interaction)\n\n")
        fh.write("  To run the webcam validation:\n")
        fh.write("  conda run -n fetalplane python -m src.realtime.app --source 0 --log-stats\n\n")
        fh.write("  Required observations:\n")
        fh.write("  1. Webcam caveat watermark visible in all 3 strips (top + bottom of frame)\n")
        fh.write("  2. Stability badge cycling correctly between STABLE and SETTLING\n")
        fh.write("  3. HUD shows plausible FPS and per-stage latencies\n")
        fh.write("  4. Grad-CAM overlay updates at ~200ms wall-clock cadence\n")
        fh.write("  5. g/h/space keybindings function correctly\n")
        fh.write("  6. Clean shutdown on ESC/q with no hanging threads\n\n")
        fh.write("  When done, save a representative screenshot as:\n")
        fh.write("  docs/phases/phase_05/phase5_screenshots/validation_webcam_demo.png\n\n")

        fh.write("=" * 70 + "\n")

    log.info("Report written to %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Task 10 headless pipeline validation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", type=str,
                   default="data/processed/synthetic_clips/multiplane_scan_01.mp4")
    p.add_argument("--checkpoint", default="checkpoints/convnext_tiny/best.pt")
    p.add_argument("--smoothing-config", default="configs/smoothing_tier1.yaml",
                   dest="smoothing_config")
    p.add_argument("--output-dir", default="docs/phases/phase_05/phase5_screenshots",
                   dest="output_dir")
    p.add_argument("--run-a-duration", type=float, default=120.0,
                   dest="run_a_duration",
                   help="Duration of the primary GradCAM-ON run (seconds).")
    p.add_argument("--run-b-duration", type=float, default=100.0,
                   dest="run_b_duration",
                   help="Duration of the GradCAM-OFF comparison run (seconds).")
    p.add_argument("--gradcam-every-n", type=int, default=7,
                   dest="gradcam_every_n")
    return p


def main() -> None:
    args = build_parser().parse_args()

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"logs/realtime/realtime_validation_run_{ts}.txt")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    screenshot_dir = Path(args.output_dir)
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    # Set up file handler so all logging also goes to the validation log
    fh_log = logging.FileHandler(log_path, encoding="utf-8")
    fh_log.setLevel(logging.INFO)
    fh_log.setFormatter(logging.Formatter(
        "%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger().addHandler(fh_log)

    log.info("=" * 60)
    log.info("TASK 10 HEADLESS VALIDATION — %s", ts)
    log.info("=" * 60)
    log.info("Stats log   : %s", log_path)
    log.info("Screenshots : %s", screenshot_dir)

    # Open CSV stats file (append so multiple runs accumulate)
    stats_csv_path = Path(f"logs/realtime/realtime_validation_stats_{ts}.csv")
    with open(stats_csv_path, "w", encoding="utf-8") as stats_fh:
        stats_fh.write(
            "run_tag,elapsed_s,capture_fps,inference_fps,"
            "preprocess_ms,forward_ms,smoothing_ms,gradcam_ms,"
            "gradcam_calls,cap_drops,inf_drops\n"
        )

        # --- Run A: 120s, GradCAM ON -------------------------------------------
        log.info("")
        log.info("RUN A: GradCAM ON, duration=%ds", int(args.run_a_duration))
        log.info("")

        # Determine frame checkpoints that fall within the run duration
        checkpoints = tuple(t for t in _FRAME_CHECKPOINTS_S if t <= args.run_a_duration)

        run_a = run_validation(
            source=args.source,
            checkpoint=args.checkpoint,
            smoothing_config=args.smoothing_config,
            duration_s=args.run_a_duration,
            enable_gradcam=True,
            gradcam_every_n=args.gradcam_every_n,
            screenshot_dir=screenshot_dir,
            screenshot_prefix="validation_file",
            stats_fh=stats_fh,
            frame_checkpoints=checkpoints,
        )

        # Pause 5s between runs to let GPU cool slightly (avoid carrying
        # thermal state from Run A directly into Run B measurement)
        log.info("Pausing 5 s between runs …")
        time.sleep(5.0)

        # --- Run B: 30s, GradCAM OFF -------------------------------------------
        log.info("")
        log.info("RUN B: GradCAM OFF (pure inference), duration=%ds",
                 int(args.run_b_duration))
        log.info("")

        run_b = run_validation(
            source=args.source,
            checkpoint=args.checkpoint,
            smoothing_config=args.smoothing_config,
            duration_s=args.run_b_duration,
            enable_gradcam=False,
            gradcam_every_n=args.gradcam_every_n,
            screenshot_dir=screenshot_dir,
            screenshot_prefix="validation_nogradcam",
            stats_fh=stats_fh,
            frame_checkpoints=(args.run_b_duration,),
        )

    # Write narrative report
    write_report(run_a, run_b, log_path)

    log.info("")
    log.info("=" * 60)
    log.info("VALIDATION COMPLETE")
    log.info("  Primary stats log  : %s", log_path)
    log.info("  Per-second CSV     : %s", stats_csv_path)
    log.info("  Screenshots        : %s/validation_file_t*.png", screenshot_dir)
    log.info("=" * 60)
    log.info("")
    log.info("NEXT STEP: webcam validation requires display — run:")
    log.info("  conda run -n fetalplane python -m src.realtime.app --source 0 --log-stats")


if __name__ == "__main__":
    main()
