"""
scripts/benchmark_pipeline.py

Headless pipeline benchmark — measures true threaded inference FPS on the RTX 4060.

This script is the "mandatory re-measurement" required by PHASE_5_KICKOFF_PROMPT.md §9:
  "Add a simple script or flag to run the pipeline in headless mode for a fixed
   number of frames (or time) over a video file, and log the average inference FPS."

Why this is necessary:
  The single-threaded CPU baseline measured in Task 4 (~24 fps on CPU) does not reflect:
  (a) GPU inference throughput (~43 fps observed from the sweep's precompute pass), or
  (b) the threading model's effect (capture/inference overlap).
  The actual threaded GPU FPS is what downstream budgets (Grad-CAM throttle cadence,
  min_dwell_frames calibration) must be computed against.

Usage:
    conda run -n fetalplane python scripts/benchmark_pipeline.py \\
        --source data/processed/synthetic_clips/your_clip.mp4 \\
        --checkpoint checkpoints/convnext_tiny/best.pt \\
        --duration 30 \\
        [--no-gradcam] [--gradcam-every-n 10]

Output:
    Printed summary + logs/realtime_benchmark_<timestamp>.txt (UTF-8)
"""
from __future__ import annotations

import argparse
import datetime
import logging
import sys
import threading
import time
from pathlib import Path

# Ensure project root is on sys.path when run directly
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.realtime.capture import FrameSource
from src.realtime.model_loader import load_inference_model
from src.realtime.pipeline import CaptureThread, InferenceThread, PipelineStats
from src.realtime.queues import DropOldestQueue

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def run_benchmark(
    source: str | int,
    checkpoint: str,
    smoothing_config: str,
    duration_secs: float,
    gradcam_every_n: int,
    enable_gradcam: bool,
    log_path: Path,
) -> None:
    lines: list[str] = []

    def emit(msg: str) -> None:
        print(msg)
        lines.append(msg)

    emit(f"Pipeline headless benchmark")
    emit(f"  source            : {source}")
    emit(f"  checkpoint        : {checkpoint}")
    emit(f"  smoothing config  : {smoothing_config}")
    emit(f"  duration          : {duration_secs:.0f} s")
    emit(f"  gradcam_every_n   : {gradcam_every_n} ({'disabled' if not enable_gradcam else 'enabled'})")
    emit("")

    # --- Load model ----------------------------------------------------------
    emit("Loading model...")
    lm = load_inference_model(checkpoint)
    emit(f"  backbone          : {lm.backbone_name}")
    emit(f"  device            : {lm.device}")
    emit(f"  image_size        : {lm.img_size}")
    emit(f"  mean/std          : {lm.normalize_mean} / {lm.normalize_std}")
    emit("")

    # --- Build pipeline components -------------------------------------------
    stop_event = threading.Event()
    stats = PipelineStats()
    frame_queue = DropOldestQueue(maxsize=2)
    result_queue = DropOldestQueue(maxsize=2)

    fs = FrameSource(source, loop=True)  # loop so short clips don't run out
    emit(f"  source resolution : {fs.resolution()[0]}x{fs.resolution()[1]}")
    emit(f"  source fps (nominal): {fs.fps():.1f}")
    emit("")

    cap_thread = CaptureThread(fs, frame_queue, stop_event, stats)
    inf_thread = InferenceThread(
        frame_queue=frame_queue,
        result_queue=result_queue,
        loaded_model=lm,
        smoothing_config_path=smoothing_config,
        stop_event=stop_event,
        stats=stats,
        gradcam_every_n_frames=gradcam_every_n,
        gradcam_wall_ms=200.0,
        enable_gradcam=enable_gradcam,
    )

    # --- Run headless for `duration_secs` seconds ----------------------------
    emit(f"Starting benchmark for {duration_secs:.0f} s ...")
    t_start = time.monotonic()

    cap_thread.start()
    inf_thread.start()

    # Drain the result queue periodically so it never backs up
    frames_consumed = 0
    fps_history: list[float] = []
    last_sample_t = time.monotonic()

    while time.monotonic() - t_start < duration_secs:
        result = result_queue.get_nowait_or_none()
        if result is not None:
            frames_consumed += 1
        else:
            time.sleep(0.005)

        now = time.monotonic()
        if now - last_sample_t >= 1.0:
            fps_history.append(stats.inference_fps)
            last_sample_t = now

    elapsed = time.monotonic() - t_start
    stop_event.set()

    cap_thread.join(timeout=3.0)
    inf_thread.join(timeout=5.0)
    fs.release()

    # --- Collect final stats -------------------------------------------------
    snap = stats.snapshot()
    
    if fps_history:
        min_fps = min(fps_history)
        max_fps = max(fps_history)
        mean_fps = sum(fps_history) / len(fps_history)
    else:
        min_fps = max_fps = mean_fps = snap["inference_fps"]

    emit(f"\n{'='*60}")
    emit(f"Benchmark complete — {elapsed:.1f} s elapsed")
    emit(f"{'='*60}")
    emit(f"  capture FPS         : {snap['capture_fps']:.1f}")
    emit(f"  inference FPS (mean): {mean_fps:.1f}  [min: {min_fps:.1f}, max: {max_fps:.1f}]")
    emit(f"  inference FPS (end) : {snap['inference_fps']:.1f}")
    emit(f"  total inf frames    : {snap.get('frames_processed', 0)}")
    emit(f"  preprocess_ms (ewma): {snap['preprocess_ms']:.2f}")
    emit(f"  forward_ms    (ewma): {snap['forward_ms']:.2f}")
    emit(f"  smoothing_ms  (ewma): {snap['smoothing_ms']:.3f}")
    emit(f"  gradcam_ms    (ewma): {snap['gradcam_ms']:.1f} (n={snap['gradcam_calls']} runs)")
    emit(f"  cap queue drops     : {snap['cap_queue_drops']}")
    emit(f"  inf queue drops     : {snap['inf_queue_drops']}")
    emit(f"  result frames consumed: {frames_consumed}")
    emit(f"{'='*60}")
    emit("")

    # Derived: expected min_dwell_ms at tuned GPU FPS
    gpu_fps = snap["inference_fps"]
    if gpu_fps > 0:
        from src.realtime.model_loader import load_inference_model as _lm  # noqa
        import yaml
        with open(smoothing_config, encoding="utf-8") as f:
            scfg = yaml.safe_load(f)
        dwell_f = scfg["min_dwell_frames"]
        dwell_ms = dwell_f / gpu_fps * 1000.0
        emit(f"Tier-1 dwell calibration check:")
        emit(f"  min_dwell_frames = {dwell_f}")
        emit(f"  at {gpu_fps:.1f} fps → {dwell_ms:.1f} ms  (target: 150–500 ms)")
        if 150 <= dwell_ms <= 500:
            emit(f"  ✓ dwell within target range")
        else:
            emit(f"  ⚠ dwell OUT OF target range — consider re-tuning if GPU FPS changed")
        emit("")

    # --- Write log file ------------------------------------------------------
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Results written to: {log_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless pipeline FPS benchmark")
    parser.add_argument("--source", required=True,
                        help="Video file path or webcam index (int)")
    parser.add_argument("--checkpoint", default="checkpoints/convnext_tiny/best.pt")
    parser.add_argument("--smoothing-config", default="configs/smoothing_tier1.yaml")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Benchmark duration in seconds (default 30)")
    parser.add_argument("--gradcam-every-n", type=int, default=10)
    parser.add_argument("--no-gradcam", action="store_true",
                        help="Disable GradCAM entirely for maximum FPS measurement")
    args = parser.parse_args()

    # Resolve source type
    source: str | int = args.source
    try:
        source = int(source)
    except ValueError:
        pass

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(f"logs/realtime_benchmark_{ts}.txt")

    run_benchmark(
        source=source,
        checkpoint=args.checkpoint,
        smoothing_config=args.smoothing_config,
        duration_secs=args.duration,
        gradcam_every_n=args.gradcam_every_n,
        enable_gradcam=not args.no_gradcam,
        log_path=log_path,
    )


if __name__ == "__main__":
    main()
