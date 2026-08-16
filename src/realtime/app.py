"""
src/realtime/app.py

Main render loop and CLI entrypoint for the real-time fetal plane classifier.

Architecture (PHASE_5_KICKOFF_PROMPT.md §10, Task 9):

    Background threads:
        CaptureThread  ──frame_queue──►  InferenceThread ──result_queue──►
    Main thread (this file):
        pull result → build display frame → cv2.imshow → cv2.waitKey

Non-negotiable constraints honoured here:
  • cv2.imshow / cv2.waitKey are ONLY called on the main thread.
  • gradcam_every_n_frames default is computed from the Task 8 empirical
    benchmark (forward_ms ≈ 44 ms on RTX 4060) to target ~200 ms wall-clock
    Grad-CAM cadence: ceil(200 / 44) = 5.  NOT the naive "10" from the spec
    skeleton — that was explicitly called out as wrong in the Phase 5 kickoff.
  • Webcam sources burn a permanent caveat watermark into every frame per
    PHASE_5_KICKOFF_PROMPT.md §11 ("burn a persistent, visible caption …
    so this caveat is impossible to miss even if someone screenshots the
    running app later").
  • No Streamlit/Gradio, no background imshow calls.

Keybindings:
  q / ESC  — quit
  g        — toggle Grad-CAM overlay (inference keeps producing; display toggles)
  h        — toggle HUD (performance stats panel)
  space    — pause / resume rendering  (inference continues; display freezes)

Usage:
    conda run -n fetalplane python -m src.realtime.app \\
        --source data/processed/synthetic_clips/Brain_Trans_thalamic_clip01.mp4 \\
        --loop
    conda run -n fetalplane python -m src.realtime.app --source 0   # webcam
"""
from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure project root is importable when run as __main__
_root = Path(__file__).resolve().parents[2]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.data.dataset import IDX_TO_CLASS
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

# ---------------------------------------------------------------------------
# Grad-CAM default cadence
#
# Three prior benchmark runs show a wide FPS spread (21.6–44.1ms forward_ms)
# that the Task 10 extended validation run is intended to settle.  Summary:
#
#   Run-1 (10s, unthrottled, every_n=10):  forward=44.08ms, gc=378.7ms  [broken GIL]
#   Run-2 (25s, unthrottled, every_n=10):  forward=21.60ms, gc=64.5ms   [best state]
#   Run-3 (30s, throttled,   every_n=5):   forward=31.77ms, gc=104.9ms  [may be thermal]
#
# The throttle fix is confirmed effective on capture-side metrics:
#   • Queue drops:    4046→11536→416 (Run-3 far lower — correct behaviour)
#   • Capture FPS:    343→421→23.6   (Run-3 correctly throttled to 24fps ✓)
#
# Its effect on inference throughput is inconclusive from these three runs alone
# because Run-2 (unthrottled) actually shows BETTER inference FPS (26.6) than
# Run-3 (throttled, 14.0fps end).  Plausible explanations: thermal throttling in
# the longer Run-3, run-to-run variance, or gradcam_every_n differing (10 vs 5).
# Task 10's 120s validation run with per-second FPS sampling resolves this.
#
# Default frame-count guard:
# Task 10 validation showed stable inference FPS is capture-bottlenecked to 23.7fps.
# This means frames arrive roughly every 42.19ms (1000/23.7), making the pure compute
# time (forward_ms ≈ 20ms) irrelevant to the cadence calculation. To achieve a true 
# 200ms wall-clock cadence against a 23.7fps stream, we divide 200 by the interval:
#   round(200 / 42.19) = 5 frames
#
# The frame-count trigger (every_n=5) is PRIMARY. The wall-clock trigger is a
# secondary safety net to prevent stale overlays if inference stalls.
# A short wall-clock trigger (e.g. 200ms) causes a runaway feedback loop because
# Grad-CAM itself takes >100ms, immediately triggering the next wall-clock guard.
# ---------------------------------------------------------------------------
_STEADY_STATE_FPS: float = 23.7
_STEADY_STATE_FRAME_INTERVAL_MS: float = 1000.0 / _STEADY_STATE_FPS
_TARGET_GRADCAM_CADENCE_MS: float = 200.0
_GRADCAM_EVERY_N_DEFAULT: int = max(1, round(_TARGET_GRADCAM_CADENCE_MS / _STEADY_STATE_FRAME_INTERVAL_MS))
# = max(1, round(200/42.19)) = 5  — accurately hits 200ms at 23.7fps stream rate


# ---------------------------------------------------------------------------
# Colour palette (BGR)
# ---------------------------------------------------------------------------
_CLR_WHITE   = (255, 255, 255)
_CLR_BLACK   = (0,   0,   0)
_CLR_GREEN   = (50,  200,  50)   # STABLE indicator
_CLR_AMBER   = (0,   165, 255)   # SETTLING indicator (orange in BGR)
_CLR_CYAN    = (255, 200,  50)   # HUD header
_CLR_GRAY    = (140, 140, 140)
_CLR_PANEL   = (20,   20,  20)   # semi-transparent panel fill

# Webcam watermark colour — red so it reads as a caution
_CLR_WATERMARK = (50, 50, 220)   # BGR red-ish

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _put_text_outlined(
    img: np.ndarray,
    text: str,
    origin: tuple[int, int],
    font: int = cv2.FONT_HERSHEY_DUPLEX,
    scale: float = 1.0,
    color: tuple[int, int, int] = _CLR_WHITE,
    thickness: int = 1,
    outline_color: tuple[int, int, int] = _CLR_BLACK,
    outline_thickness: int = 3,
) -> None:
    """Draw text with a dark outline for legibility on any background."""
    cv2.putText(img, text, origin, font, scale,
                outline_color, outline_thickness + thickness, cv2.LINE_AA)
    cv2.putText(img, text, origin, font, scale,
                color, thickness, cv2.LINE_AA)


def _draw_watermark(canvas: np.ndarray, w: int, h: int) -> None:
    """Burn the permanent webcam-demo caveat text into the frame.

    Per PHASE_5_KICKOFF_PROMPT.md §11: must be impossible to miss even in
    a screenshot.  We draw it twice (top and bottom) with a filled background
    strip so it cannot be cropped away by aspect-ratio changes.
    """
    msg = "WEBCAM DEMO: PIPELINE MECHANICS TEST ONLY"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.50
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(msg, font, scale, thickness)
    pad = 8
    strip_h = th + baseline + pad * 2

    overlay = canvas.copy()
    for y_top in (0, h - strip_h):
        cv2.rectangle(overlay, (0, y_top), (w, y_top + strip_h), _CLR_PANEL, -1)
    
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    for y_top in (0, h - strip_h):
        x = max(0, (w - tw) // 2)
        y = y_top + pad + th
        _put_text_outlined(canvas, msg, (x, y), font, scale,
                           (230, 230, 230), thickness,
                           _CLR_BLACK, 2)


def _draw_label_panel(
    canvas: np.ndarray,
    label_name: str,
    confidence: float,
    is_stable: bool,
    w: int,
    h: int,
) -> None:
    """Overlay the predicted label, confidence, and stability badge."""
    font = cv2.FONT_HERSHEY_DUPLEX

    # ---- stability badge (top-right corner) --------------------------------
    badge_symbol = "\u25cf STABLE" if is_stable else "\u25d0 SETTLING"
    badge_color  = _CLR_GREEN if is_stable else _CLR_AMBER
    badge_scale  = 0.75
    badge_thick  = 1
    (bw, bh), _ = cv2.getTextSize(badge_symbol, font, badge_scale, badge_thick)
    pad = 8
    bx = w - bw - pad
    by = bh + pad
    # semi-transparent pill background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (bx - pad, pad), (bx + bw + pad, by + pad), _CLR_PANEL, -1)
    cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)
    _put_text_outlined(canvas, badge_symbol, (bx, by), font,
                       badge_scale, badge_color, badge_thick)

    # ---- label + confidence panel (bottom of frame) ------------------------
    label_display = label_name.replace("_", " ")
    conf_pct = f"{confidence * 100:.1f}%"

    lbl_scale = 1.05
    lbl_thick = 2
    conf_scale = 0.80
    conf_thick = 1

    (lw, lh), _ = cv2.getTextSize(label_display, font, lbl_scale, lbl_thick)
    (cw, ch), _ = cv2.getTextSize(conf_pct, font, conf_scale, conf_thick)
    panel_h = lh + ch + pad * 4
    panel_w = max(lw, cw) + pad * 4

    # panel anchored bottom-left
    px, py = pad, h - panel_h - pad
    overlay2 = canvas.copy()
    cv2.rectangle(overlay2, (px, py), (px + panel_w, py + panel_h), _CLR_PANEL, -1)
    cv2.addWeighted(overlay2, 0.70, canvas, 0.30, 0, canvas)

    # label text
    _put_text_outlined(canvas, label_display,
                       (px + pad, py + pad + lh),
                       font, lbl_scale, _CLR_WHITE, lbl_thick)
    # confidence text
    _put_text_outlined(canvas, conf_pct,
                       (px + pad, py + pad + lh + pad + ch),
                       font, conf_scale, _CLR_CYAN, conf_thick)


def _draw_bboxes(
    canvas: np.ndarray,
    bboxes: np.ndarray | None,
    labels: np.ndarray | None,
    scores: np.ndarray | None
) -> None:
    """Overlay detected anatomical structures as bounding boxes."""
    if bboxes is None or len(bboxes) == 0:
        return
        
    # Clean premium palette for structures
    bbox_colors = {
        1: (220, 130, 255), # Head (Soft Pink/Purple)
        2: (130, 255, 130), # Abdomen (Soft Green)
        3: (255, 200, 100), # Femur (Soft Cyan/Blue)
    }
    
    bbox_names = {
        1: "Head",
        2: "Abdomen",
        3: "Femur"
    }

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.5
    thick = 1
    
    for box, label, score in zip(bboxes, labels, scores):
        if score < 0.4:
            continue
            
        x1, y1, x2, y2 = map(int, box)
        label_idx = int(label)
        color = bbox_colors.get(label_idx, _CLR_CYAN)
        name = bbox_names.get(label_idx, f"Obj-{label_idx}")
        
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        
        text = f"{name} {score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        
        # Draw filled label background
        cv2.rectangle(canvas, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        # Draw black text on top of the coloured background
        cv2.putText(canvas, text, (x1 + 4, y1 - 4), font, scale, _CLR_BLACK, thick, cv2.LINE_AA)

def _draw_hud(
    canvas: np.ndarray,
    snap: dict[str, Any],
    timings: dict[str, float | None],
    result_tier2_active: bool,
    w: int,
    h: int,
) -> None:
    """Overlay the performance HUD panel (top-left corner)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.50
    thick = 1
    lh_px = 18   # line height in pixels

    gradcam_ms = timings.get("gradcam_ms")
    lines = [
        ("PERFORMANCE HUD",        _CLR_CYAN),
        (f"  capture   {snap['capture_fps']:.1f} fps",   _CLR_WHITE),
        (f"  infer     {snap['inference_fps']:.1f} fps",  _CLR_WHITE),
        (f"  preproc   {snap['preprocess_ms']:.1f} ms",  _CLR_GRAY),
        (f"  forward   {snap['forward_ms']:.1f} ms",     _CLR_GRAY),
        (f"  smooth    {snap['smoothing_ms']:.2f} ms",   _CLR_GRAY),
        (f"  gradcam   {snap['gradcam_ms']:.1f} ms ({snap['gradcam_calls']}x)" if snap['gradcam_calls'] > 0
                      else "  gradcam   --",             _CLR_GRAY),
        (f"  q-drops   {snap['cap_queue_drops']}",       _CLR_GRAY),
    ]
    # Show Tier-2a status when the result dict reports it active
    if result_tier2_active:
        lines.append(("  tier2a    ON", _CLR_AMBER))

    pad = 8
    panel_w = 230
    panel_h = len(lines) * lh_px + pad * 2
    px, py = pad, pad

    # semi-transparent background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), _CLR_PANEL, -1)
    cv2.addWeighted(overlay, 0.72, canvas, 0.28, 0, canvas)

    for i, (text, color) in enumerate(lines):
        tx = px + 6
        ty = py + pad + i * lh_px + lh_px // 2
        cv2.putText(canvas, text, (tx, ty), font, scale, _CLR_BLACK, thick + 1, cv2.LINE_AA)
        cv2.putText(canvas, text, (tx, ty), font, scale, color, thick, cv2.LINE_AA)


def _draw_paused_overlay(canvas: np.ndarray, w: int, h: int) -> None:
    """Apply a dim overlay and centred PAUSED badge."""
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)
    msg = "\u23f8 PAUSED  (space to resume)"
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.2
    thick = 2
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
    ox = (w - tw) // 2
    oy = (h + th) // 2
    _put_text_outlined(canvas, msg, (ox, oy), font, scale,
                       (200, 200, 100), thick)


def _loading_frame(w: int = 640, h: int = 480) -> np.ndarray:
    """Return a dark placeholder frame while pipeline warms up."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    msg = "Initialising pipeline..."
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 0.9
    thick = 1
    (tw, th), _ = cv2.getTextSize(msg, font, scale, thick)
    _put_text_outlined(frame, msg, ((w - tw) // 2, (h + th) // 2),
                       font, scale, _CLR_GRAY, thick)
    return frame


def build_display_frame(
    result: dict[str, Any],
    show_gradcam: bool,
    show_hud: bool,
    is_paused: bool,
    is_webcam: bool,
    stats: PipelineStats,
) -> np.ndarray:
    """Compose the fully annotated frame for imshow."""
    # Base: either the Grad-CAM overlay or the raw frame
    if show_gradcam and result.get("overlay") is not None:
        canvas: np.ndarray = result["overlay"].copy()   # BGR uint8
    else:
        canvas = result["frame"].copy()

    h, w = canvas.shape[:2]

    if is_webcam:
        _draw_watermark(canvas, w, h)

    _draw_bboxes(
        canvas,
        result.get("bboxes"),
        result.get("bbox_labels"),
        result.get("bbox_scores")
    )

    _draw_label_panel(
        canvas,
        result["label_name"],
        result["confidence"],
        result["is_stable"],
        w, h,
    )

    if show_hud:
        _draw_hud(canvas, stats.snapshot(), result["timings"],
                  result.get("tier2_active", False), w, h)

    if is_paused:
        _draw_paused_overlay(canvas, w, h)

    return canvas


# ---------------------------------------------------------------------------
# Periodic stats logger
# ---------------------------------------------------------------------------

def _stats_logger(
    stats: PipelineStats,
    log_path: Path,
    interval_s: float,
    stop_event: threading.Event,
) -> None:
    """Daemon thread: appends a stats snapshot to log_path every interval_s."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_wall = time.monotonic()
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"# Validation stats log — started at wall+0s\n")
        fh.write("elapsed_s,capture_fps,inference_fps,preprocess_ms,"
                 "forward_ms,smoothing_ms,gradcam_ms,gradcam_calls,"
                 "cap_drops,inf_drops\n")
        while not stop_event.wait(timeout=interval_s):
            snap = stats.snapshot()
            elapsed = time.monotonic() - start_wall
            fh.write(
                f"{elapsed:.1f},{snap['capture_fps']:.2f},"
                f"{snap['inference_fps']:.2f},{snap['preprocess_ms']:.2f},"
                f"{snap['forward_ms']:.2f},{snap['smoothing_ms']:.3f},"
                f"{snap['gradcam_ms']:.1f},{snap['gradcam_calls']},"
                f"{snap['cap_queue_drops']},{snap['inf_queue_drops']}\n"
            )
            fh.flush()


# ---------------------------------------------------------------------------
# Main application entry point
# ---------------------------------------------------------------------------

def run_app(args: argparse.Namespace) -> None:
    """Set up pipeline, run render loop, shut down cleanly."""

    # --- Resolve source -------------------------------------------------------
    try:
        source: int | str = int(args.source)
    except ValueError:
        source = args.source

    # --- Pre-validate --tier2-config existence --------------------------------
    # Do this before loading the model (expensive) so the user gets a clean
    # error message rather than a raw traceback after a 5-second wait.
    if args.enable_tier2:
        tier2_cfg_path = Path(args.tier2_config)
        if not tier2_cfg_path.exists():
            log.error(
                "--enable-tier2 was set but --tier2-config file not found: %s\n"
                "Run scripts/tune_tier2_mode_filter.py first to generate it, or\n"
                "pass --tier2-config <path> pointing to an existing YAML file.",
                tier2_cfg_path,
            )
            sys.exit(1)

    log.info("Starting app — source=%r  checkpoint=%s  loop=%s  gradcam=%s  every_n=%d",
             source, args.checkpoint, args.loop,
             not args.no_gradcam, args.gradcam_every_n_frames)

    # --- Load model -----------------------------------------------------------
    log.info("Loading model from %s …", args.checkpoint)
    lm = load_inference_model(args.checkpoint)
    log.info("Model loaded: backbone=%s  device=%s  img_size=%d",
             lm.backbone_name, lm.device, lm.img_size)

    # --- Pipeline components --------------------------------------------------
    stop_event = threading.Event()
    stats      = PipelineStats()
    frame_queue: DropOldestQueue[tuple[np.ndarray, float]] = DropOldestQueue(maxsize=2)
    result_queue: DropOldestQueue[dict[str, Any]]          = DropOldestQueue(maxsize=2)

    fs = FrameSource(source, loop=args.loop)
    log.info("Source opened: %dx%d @ %.1f fps  (is_webcam=%s)",
             *fs.resolution(), fs.fps(), fs.is_webcam)

    cap_thread = CaptureThread(fs, frame_queue, stop_event, stats)
    inf_thread = InferenceThread(
        frame_queue=frame_queue,
        result_queue=result_queue,
        loaded_model=lm,
        smoothing_config_path=args.smoothing_config,
        stop_event=stop_event,
        stats=stats,
        gradcam_every_n_frames=args.gradcam_every_n_frames,
        gradcam_wall_ms=1000.0,
        enable_gradcam=not args.no_gradcam,
        tier2_config_path=args.tier2_config if args.enable_tier2 else None,
    )

    # --- Optional stats logger ------------------------------------------------
    if args.log_stats:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_log_path = Path(f"logs/realtime_validation_{ts}.txt")
        stats_log_thread = threading.Thread(
            target=_stats_logger,
            args=(stats, stats_log_path, 5.0, stop_event),
            name="StatsLogger",
            daemon=True,
        )
        log.info("Stats will be logged to %s (every 5 s)", stats_log_path)
        stats_log_thread.start()
    else:
        stats_log_path = None

    # --- Start background threads ---------------------------------------------
    cap_thread.start()
    inf_thread.start()

    # --- Window setup ---------------------------------------------------------
    win = "Fetal Plane Classifier — Phase 5"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    src_w, src_h = fs.resolution()
    # Sensible initial window size: keep native resolution but cap at 1280×720
    disp_w = min(src_w or 960, 1280)
    disp_h = min(src_h or 720, 720)
    cv2.resizeWindow(win, disp_w, disp_h)

    # --- Render state ---------------------------------------------------------
    last_result: dict[str, Any] | None = None
    show_gradcam = not args.no_gradcam
    show_hud     = True
    is_paused    = False

    # Placeholder frame while pipeline warms up
    placeholder = _loading_frame(disp_w, disp_h)

    log.info("Render loop started.  Keys: q/ESC=quit  g=Grad-CAM  h=HUD  space=pause")

    try:
        while True:
            # Pull latest result (non-blocking)
            if not is_paused:
                res = result_queue.get_nowait_or_none()
                if res is not None:
                    last_result = res

            # Render
            if last_result is not None:
                display = build_display_frame(
                    last_result, show_gradcam, show_hud,
                    is_paused, fs.is_webcam, stats,
                )
            else:
                display = placeholder

            cv2.imshow(win, display)

            # Key handling (10 ms poll — gives ~100 Hz render loop)
            key = cv2.waitKey(10) & 0xFF
            if key in (ord('q'), 27):           # q / ESC
                log.info("Quit key pressed.")
                break
            elif key == ord('g'):
                show_gradcam = not show_gradcam
                log.info("Grad-CAM overlay: %s", "ON" if show_gradcam else "OFF")
            elif key == ord('h'):
                show_hud = not show_hud
                log.info("HUD: %s", "ON" if show_hud else "OFF")
            elif key == ord(' '):
                is_paused = not is_paused
                log.info("Rendering: %s", "PAUSED" if is_paused else "RESUMED")

            # Exit if source is exhausted (CaptureThread sets stop_event)
            if stop_event.is_set() and not args.loop:
                log.info("Source exhausted and loop=False — exiting.")
                break

    except KeyboardInterrupt:
        log.info("KeyboardInterrupt received.")
    finally:
        log.info("Shutting down …")
        stop_event.set()
        cap_thread.join(timeout=3.0)
        inf_thread.join(timeout=5.0)
        fs.release()
        cv2.destroyAllWindows()

        # Log final stats
        snap = stats.snapshot()
        log.info("Final stats:")
        log.info("  capture FPS   : %.1f", snap["capture_fps"])
        log.info("  inference FPS : %.1f", snap["inference_fps"])
        log.info("  forward_ms    : %.1f ms", snap["forward_ms"])
        log.info("  gradcam_ms    : %.1f ms (%d calls)", snap["gradcam_ms"], snap["gradcam_calls"])
        log.info("  cap drops     : %d", snap["cap_queue_drops"])
        if stats_log_path:
            log.info("Stats log written to: %s", stats_log_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="src.realtime.app",
        description="Real-time fetal plane classifier — Phase 5 demo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source", required=True,
        help="Webcam device index (int) or path to a video file.",
    )
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/convnext_tiny/best.pt",
        help="Path to a trained checkpoint (.pt).",
    )
    parser.add_argument(
        "--smoothing-config",
        default="configs/smoothing_tier1.yaml",
        dest="smoothing_config",
        help="Path to smoothing_tier1.yaml with alpha/switch_threshold/min_dwell_frames.",
    )
    parser.add_argument(
        "--gradcam-every-n-frames",
        type=int,
        default=_GRADCAM_EVERY_N_DEFAULT,
        dest="gradcam_every_n_frames",
        help=(
            f"Run Grad-CAM every N processed frames (secondary cadence guard). "
            f"Default {_GRADCAM_EVERY_N_DEFAULT} = round({_TARGET_GRADCAM_CADENCE_MS:.0f}ms / "
            f"{_STEADY_STATE_FRAME_INTERVAL_MS:.1f}ms interval), "
            f"targeting ~{_TARGET_GRADCAM_CADENCE_MS:.0f}ms wall-clock cadence. "
            f"The wall-clock trigger always fires first when inference is fast."
        ),
    )
    parser.add_argument(
        "--no-gradcam", action="store_true",
        help="Disable Grad-CAM entirely for maximum throughput.",
    )
    parser.add_argument(
        "--loop", action="store_true",
        help="Loop video files (ignored for webcam sources).",
    )
    parser.add_argument(
        "--log-stats", action="store_true",
        help="Periodically write performance stats to logs/realtime_validation_<ts>.txt.",
    )
    parser.add_argument(
        "--enable-tier2", action="store_true",
        dest="enable_tier2",
        help=(
            "Enable Tier-2a majority-vote filter on top of Tier-1 smoothing. "
            "Off by default — existing Phase 5/6 behaviour is unchanged when omitted. "
            "Requires --tier2-config to point to a valid smoothing_tier2a.yaml."
        ),
    )
    parser.add_argument(
        "--tier2-config",
        default="configs/smoothing_tier2a.yaml",
        dest="tier2_config",
        help=(
            "Path to smoothing_tier2a.yaml (window_frames, min_majority_frac). "
            "Only used when --enable-tier2 is set."
        ),
    )
    return parser


if __name__ == "__main__":
    run_app(build_parser().parse_args())
