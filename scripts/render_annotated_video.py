"""
scripts/render_annotated_video.py

Offline annotated-video renderer — the engine that powers the Gradio web UI.

PURPOSE
-------
Takes an input ultrasound video clip, runs the full inference pipeline
(preprocessing → forward pass → Tier-1 smoothing → optional Tier-2a →
Grad-CAM), and writes a fully-annotated H.264 MP4 using imageio-ffmpeg.

The annotated frames are produced by build_display_frame() from
src/realtime/app.py — the exact same renderer used by the live desktop app
and the headless validation script — so the visual output is byte-for-byte
identical to what a sonographer would see in the local app.

DESIGN DECISIONS
----------------
- **Single-threaded**: Unlike the live pipeline (CaptureThread + InferenceThread),
  offline processing has no hard latency budget, so the simpler sequential loop
  is preferred. Progress is reported via an optional callback for Gradio.
- **No bboxes**: The classifier-only checkpoint (convnext_tiny/best.pt) is the
  intended input. The multitask model was trained for 1 epoch only (val macro-F1
  0.8421, below the production classifier's 0.9183) and is NOT suitable for
  clinical display. build_display_frame() safely no-ops _draw_bboxes() when
  result["bboxes"] is None, which is always the case for the classifier checkpoint.
  This is documented in docs/demo1_walkthrough.md §Known Limitations.
- **H.264 via imageio-ffmpeg**: OpenCV's default mp4v fourcc produces files that
  Chrome/Safari frequently cannot play inline. imageio[ffmpeg] pipes frames through
  ffmpeg to produce browser-compatible H.264 without requiring a system ffmpeg
  install (imageio-ffmpeg ships its own binaries).
- **Persistent GradCAM**: Follows InferenceThread's pattern — create one GradCAM
  instance, keep its hooks alive for the whole run, call it directly. Does NOT
  use run_gradcam() from src/models/gradcam.py because that creates a new
  GradCAM context-manager on every call (~5-15ms hook setup/teardown overhead).
- **PipelineStats used for HUD**: A PipelineStats instance is updated on every
  frame so the rendered HUD is accurate and matches the live app's display.

USAGE
-----
  # CLI — headless    conda run -n fetalplane python scripts/render_annotated_video.py \
      --input  data/processed/synthetic_clips/multiplane_scan_01.mp4 \\
      --output /tmp/annotated.mp4 \\
      --checkpoint checkpoints/convnext_tiny/best.pt \\
      --gradcam-every-n 1

  # As a library — called by app_gradio.py
  from scripts.render_annotated_video import render_video
  result = render_video(
      input_path="clip.mp4",
      output_path="out.mp4",
      progress_cb=lambda done, total: ...,
  )
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path so this script is importable from anywhere
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.dataset import IDX_TO_CLASS, NUM_CLASSES
from src.data.transforms import prep_frame_grayscale_to_rgb
from src.models.gradcam import get_target_layer
from src.models.cam_localisation import cam_to_bbox
from src.realtime.app import build_display_frame
from src.realtime.model_loader import load_inference_model
from src.realtime.pipeline import PipelineStats
from src.smoothing.tier1 import Tier1Smoother
from src.smoothing.tier2_mode_filter import Tier2ModeFilter

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_DEFAULT_CHECKPOINT     = "checkpoints/convnext_tiny/best.pt"
_DEFAULT_SMOOTHING_CFG  = "configs/smoothing_tier1.yaml"
_DEFAULT_TIER2_CFG      = "configs/smoothing_tier2a.yaml"
_GRADCAM_WALL_MS        = 9999.0  # wall-clock fallback — effectively disabled
                                   # for offline; cadence controlled by every_n only


# ---------------------------------------------------------------------------
# Core rendering function
# ---------------------------------------------------------------------------

def render_video(
    input_path: str | Path,
    output_path: str | Path,
    checkpoint: str | Path = _DEFAULT_CHECKPOINT,
    smoothing_config: str | Path = _DEFAULT_SMOOTHING_CFG,
    tier2_config: str | Path | None = _DEFAULT_TIER2_CFG,
    enable_tier2: bool = True,
    enable_gradcam: bool = True,
    gradcam_every_n: int = 1,
    show_hud: bool = True,
    enable_cam_bbox: bool = True,
    progress_cb: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Render an annotated copy of *input_path* and save it to *output_path*.

    Args:
        input_path:      Path to the source video (any codec OpenCV can decode).
        output_path:     Path where the H.264 MP4 will be written.
        checkpoint:      Path to the .pt checkpoint. Config is read from the
                         checkpoint itself — never hardcoded here.
        smoothing_config: Path to configs/smoothing_tier1.yaml.
        tier2_config:    Path to configs/smoothing_tier2a.yaml (only used when
                         enable_tier2=True).
        enable_tier2:    If True, apply Tier-2a majority-vote filter after Tier-1.
        enable_gradcam:  If True, compute Grad-CAM every gradcam_every_n frames.
        gradcam_every_n: Frequency of Grad-CAM computation (1 = every frame).
                         Higher values trade quality for speed.
        show_hud:        If True, overlay the performance HUD on each frame.
        enable_cam_bbox: If True (default), draw the dashed approx-region box
                         derived from the Grad-CAM heatmap on frames where a
                         CAM was computed (Task 3.2, Phase 8 Stage 3).  Box is
                         labelled "approx. region (saliency-derived)" and drawn
                         with a dashed orange line to distinguish it from
                         fully-supervised detection output.  Requires
                         enable_gradcam=True; silently no-ops if gradcam is off.
        progress_cb:     Optional callback (frames_done: int, total_frames: int).
                         Called after each frame. Used by Gradio gr.Progress.

    Returns:
        A dict with keys:
          "output_path"   : str — absolute path to the written MP4.
          "total_frames"  : int — number of frames processed.
          "fps_in"        : float — source FPS.
          "label_log"     : list[dict] — per-frame label/confidence/timestamp.
          "model_info"    : dict — backbone, img_size, smoothing params.
          "elapsed_s"     : float — wall-clock render time in seconds.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t_start = time.monotonic()

    # ------------------------------------------------------------------
    # 1. Load model (config-driven — backbone/img_size/norm come from ckpt)
    # ------------------------------------------------------------------
    log.info("Loading model from %s …", checkpoint)
    lm = load_inference_model(str(checkpoint))
    log.info("  backbone=%s  device=%s  img_size=%d",
             lm.backbone_name, lm.device, lm.img_size)

    # ------------------------------------------------------------------
    # 2. Open source video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps_in      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    log.info("Source: %dx%d @ %.2f fps, %d frames", src_w, src_h, fps_in, total_frames)

    # ------------------------------------------------------------------
    # 3. Smoothers
    # ------------------------------------------------------------------
    with open(smoothing_config, encoding="utf-8") as fh:
        t1_cfg = yaml.safe_load(fh)

    smoother = Tier1Smoother(
        num_classes=NUM_CLASSES,
        alpha=float(t1_cfg["alpha"]),
        switch_threshold=float(t1_cfg["switch_threshold"]),
        min_dwell_frames=int(t1_cfg["min_dwell_frames"]),
        hold_floor=t1_cfg.get("hold_floor"),
    )
    log.info("Tier1Smoother: alpha=%.2f  sw_thr=%.2f  dwell=%d  hold_floor=%s",
             t1_cfg["alpha"], t1_cfg["switch_threshold"],
             t1_cfg["min_dwell_frames"], t1_cfg.get("hold_floor"))

    tier2: Tier2ModeFilter | None = None
    if enable_tier2 and tier2_config is not None:
        tier2 = Tier2ModeFilter.from_config(str(tier2_config))
        log.info("Tier2ModeFilter: window=%d  majority_frac=%.2f",
                 tier2.window_frames, tier2.min_majority_frac)
    else:
        log.info("Tier-2a disabled.")

    # ------------------------------------------------------------------
    # 4. Persistent GradCAM (same pattern as InferenceThread.__init__)
    # ------------------------------------------------------------------
    cam: GradCAM | None = None
    if enable_gradcam and _GRADCAM_AVAILABLE:
        target_layer = get_target_layer(lm.model, lm.backbone_name)
        cam = GradCAM(model=lm.model, target_layers=[target_layer])
        log.info("GradCAM: persistent instance, every_n=%d, target=%s",
                 gradcam_every_n, type(target_layer).__name__)
    elif enable_gradcam:
        log.warning("grad-cam package not available — GradCAM disabled.")

    # ------------------------------------------------------------------
    # 5. Output writer (H.264 via imageio-ffmpeg)
    # ------------------------------------------------------------------
    imageio = None
    try:
        import imageio
        import imageio.v3 as iio
        _writer_backend = "imageio"
    except ImportError:
        _writer_backend = None
        log.warning("imageio not installed — falling back to OpenCV mp4v writer. "
                    "Output may not play in Chrome/Safari. "
                    "Install: pip install imageio[ffmpeg]")

    # We build a list of RGB frames in-memory for imageio, OR fall back to
    # cv2.VideoWriter if imageio is unavailable. For very long clips, memory
    # could be a concern; in practice ultrasound demo clips are short (<5 min).
    # For large files, a streaming imageio writer is used instead.
    use_imageio = (_writer_backend == "imageio")
    cv2_writer: cv2.VideoWriter | None = None
    writer = None

    if use_imageio:
        assert imageio is not None, "imageio must be loaded if use_imageio is true"
        # imageio FFMPEG writer — streams frames, low memory
        writer = imageio.get_writer(
            str(output_path),
            fps=fps_in,
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            ffmpeg_params=["-crf", "23"],
            macro_block_size=1,
        )
    else:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        cv2_writer = cv2.VideoWriter(str(output_path), fourcc, fps_in, (src_w, src_h))

    # ------------------------------------------------------------------
    # 6. Per-frame inference + rendering loop
    # ------------------------------------------------------------------
    stats        = PipelineStats()
    label_log: list[dict[str, Any]] = []
    last_overlay: np.ndarray | None = None
    frame_idx   = 0
    use_amp      = (lm.device.type == "cuda")

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            t_frame_start = time.monotonic()

            # ---- Preprocess ------------------------------------------------
            t0 = time.monotonic()
            rgb_hw3: np.ndarray = prep_frame_grayscale_to_rgb(frame_bgr)
            augmented = lm.transform(image=rgb_hw3)
            tensor: torch.Tensor = augmented["image"].unsqueeze(0).to(
                lm.device, non_blocking=True
            )
            preprocess_ms = (time.monotonic() - t0) * 1000.0

            # ---- Forward pass ----------------------------------------------
            t0 = time.monotonic()
            bboxes = bbox_labels = bbox_scores = None
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp):
                    if hasattr(lm.model, "retinanet"):
                        logits, det_outputs = lm.model(tensor)
                        if len(det_outputs) > 0:
                            bboxes      = det_outputs[0]["boxes"].cpu().numpy()
                            bbox_labels = det_outputs[0]["labels"].cpu().numpy()
                            bbox_scores = det_outputs[0]["scores"].cpu().numpy()
                    else:
                        logits = lm.model(tensor)
            probs_np: np.ndarray = (
                torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
            )
            forward_ms = (time.monotonic() - t0) * 1000.0

            # ---- Tier-1 smoothing ------------------------------------------
            t0 = time.monotonic()
            label, confidence, smoothed_probs, is_stable = smoother.step(probs_np)
            smoothing_ms = (time.monotonic() - t0) * 1000.0

            # ---- Tier-2a (optional) ----------------------------------------
            if tier2 is not None:
                final_label, tier2_is_stable = tier2.step(label)
                display_is_stable = tier2_is_stable
            else:
                final_label = label
                display_is_stable = is_stable

            # ---- Grad-CAM (persistent, every N frames) ---------------------
            gradcam_ms: float | None = None
            overlay: np.ndarray | None = last_overlay

            if cam is not None and (frame_idx % gradcam_every_n == 0):
                t0 = time.monotonic()
                targets = [ClassifierOutputTarget(final_label)]
                grayscale_cam: np.ndarray = cam(
                    input_tensor=tensor, targets=targets  # type: ignore[arg-type]
                )[0]
                h_orig, w_orig = frame_bgr.shape[:2]
                cam_resized = cv2.resize(grayscale_cam, (w_orig, h_orig))
                rgb_float = (
                    cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                    / 255.0
                )
                overlay_rgb: np.ndarray = show_cam_on_image(
                    rgb_float, cam_resized, use_rgb=True
                )
                overlay = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
                last_overlay = overlay
                gradcam_ms = (time.monotonic() - t0) * 1000.0
                stats.record_gradcam(gradcam_ms)

            # ---- CAM bbox (approx. region, Task 3.2) -----------------------
            # cam_resized is float32 in [0,1] at frame resolution — exactly
            # what cam_to_bbox() expects.  Returns None on diffuse CAMs.
            cam_bbox = cam_to_bbox(cam_resized) if (cam is not None and cam_resized is not None and enable_cam_bbox) else None

            # ---- Update stats ----------------------------------------------
            stats.record_inference_frame(preprocess_ms, forward_ms, smoothing_ms)
            stats.record_capture_frame()

            # ---- Build annotated frame -------------------------------------
            result: dict[str, Any] = {
                "label":          final_label,
                "label_name":     IDX_TO_CLASS.get(final_label, f"class_{final_label}"),
                "confidence":     confidence,
                "smoothed_probs": smoothed_probs,
                "is_stable":      display_is_stable,
                "overlay":        overlay,
                "frame":          frame_bgr,
                "timings": {
                    "preprocess_ms": preprocess_ms,
                    "forward_ms":    forward_ms,
                    "smoothing_ms":  smoothing_ms,
                    "gradcam_ms":    gradcam_ms,
                },
                "tier2_active":   tier2 is not None,
                "bboxes":         bboxes,
                "bbox_labels":    bbox_labels,
                "bbox_scores":    bbox_scores,
                "cam_bbox":       cam_bbox,  # (x1,y1,x2,y2) or None (Task 3.2)
            }

            canvas = build_display_frame(
                result,
                show_gradcam=cam is not None,
                show_hud=show_hud,
                is_paused=False,
                is_webcam=False,  # never burn the webcam watermark on uploaded clips
                stats=stats,
                show_cam_bbox=enable_cam_bbox,
            )

            # ---- Write frame -----------------------------------------------
            if use_imageio:
                assert writer is not None
                # imageio expects RGB
                writer.append_data(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
            else:
                cv2_writer.write(canvas)  # type: ignore[union-attr]

            # ---- Log per-frame metadata ------------------------------------
            label_log.append({
                "frame":       frame_idx,
                "timestamp_s": round(frame_idx / fps_in, 3),
                "label":       IDX_TO_CLASS.get(final_label, f"class_{final_label}"),
                "confidence":  round(confidence, 4),
                "is_stable":   display_is_stable,
            })

            frame_idx += 1

            if progress_cb is not None:
                progress_cb(frame_idx, total_frames)

            if frame_idx % 50 == 0:
                elapsed = time.monotonic() - t_start
                log.info("  frame %d/%d (%.1f%%)  elapsed=%.1fs",
                         frame_idx, total_frames,
                         100.0 * frame_idx / max(total_frames, 1),
                         elapsed)

    finally:
        cap.release()
        if use_imageio and writer is not None:
            writer.close()
        elif cv2_writer is not None:
            cv2_writer.release()

        # Clean up GradCAM hooks
        if cam is not None:
            try:
                cam.activations_and_grads.release()
            except Exception:
                pass

    elapsed_s = time.monotonic() - t_start
    log.info(
        "Render complete: %d frames in %.1fs (%.2f fps rendered).  Output: %s",
        frame_idx, elapsed_s,
        frame_idx / elapsed_s if elapsed_s > 0 else 0.0,
        output_path,
    )

    # ------------------------------------------------------------------
    # 7. Build model_info dict for the Gradio UI / sidecar JSON
    # ------------------------------------------------------------------
    model_info: dict[str, Any] = {
        "backbone":           lm.backbone_name,
        "img_size":           lm.img_size,
        "device":             str(lm.device),
        "tier1_alpha":        t1_cfg["alpha"],
        "tier1_switch_thr":   t1_cfg["switch_threshold"],
        "tier1_dwell_frames": t1_cfg["min_dwell_frames"],
        "tier2_enabled":      tier2 is not None,
        "gradcam_enabled":    cam is not None or enable_gradcam,
        "gradcam_every_n":    gradcam_every_n,
        # Evaluation metrics from EVAL_REPORT.md (fixed at release)
        "test_macro_f1":      0.8927,
        "note_bboxes":        (
            "Structure bounding boxes are NOT shown in Demo 1. "
            "The multitask detection model was trained for 1 epoch only "
            "(val macro-F1 0.8421) and is not suitable for clinical display. "
            "Label + confidence only, using the production convnext_tiny classifier."
        ),
    }

    return {
        "output_path":   str(output_path.resolve()),
        "total_frames":  frame_idx,
        "fps_in":        fps_in,
        "label_log":     label_log,
        "model_info":    model_info,
        "elapsed_s":     round(elapsed_s, 2),
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render an annotated H.264 MP4 from an ultrasound video clip.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input",  required=True, help="Path to input video.")
    p.add_argument("--output", required=True, help="Path for output H.264 MP4.")
    p.add_argument(
        "--checkpoint", default=_DEFAULT_CHECKPOINT,
        help="Path to .pt checkpoint. Config is read from checkpoint — never hardcoded.",
    )
    p.add_argument(
        "--smoothing-config", default=_DEFAULT_SMOOTHING_CFG, dest="smoothing_config",
        help="Path to smoothing_tier1.yaml.",
    )
    p.add_argument(
        "--tier2-config", default=_DEFAULT_TIER2_CFG, dest="tier2_config",
        help="Path to smoothing_tier2a.yaml. Only used with --enable-tier2.",
    )
    p.add_argument(
        "--enable-tier2", action="store_true", dest="enable_tier2",
        help="Enable Tier-2a majority-vote filter (adds ≤338ms latency, eliminates residual flicker).",
    )
    p.add_argument(
        "--no-gradcam", action="store_true", dest="no_gradcam",
        help="Disable Grad-CAM (faster, smaller output file).",
    )
    p.add_argument(
        "--gradcam-every-n", type=int, default=1, dest="gradcam_every_n",
        help="Run Grad-CAM every N frames. 1 = every frame (max fidelity). Higher = faster.",
    )
    p.add_argument(
        "--no-hud", action="store_true", dest="no_hud",
        help="Disable the performance HUD overlay.",
    )
    p.add_argument(
        "--output-json", dest="output_json",
        help="Optional path to write per-frame label log as JSON sidecar.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    result = render_video(
        input_path=args.input,
        output_path=args.output,
        checkpoint=args.checkpoint,
        smoothing_config=args.smoothing_config,
        tier2_config=args.tier2_config if args.enable_tier2 else None,
        enable_tier2=args.enable_tier2,
        enable_gradcam=not args.no_gradcam,
        gradcam_every_n=args.gradcam_every_n,
        show_hud=not args.no_hud,
    )

    print(f"\n{'='*60}")
    print(f"  Output    : {result['output_path']}")
    print(f"  Frames    : {result['total_frames']}")
    print(f"  Source FPS: {result['fps_in']:.2f}")
    print(f"  Render time: {result['elapsed_s']:.1f}s")
    print(f"{'='*60}")

    if args.output_json:
        json_path = Path(args.output_json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"  JSON sidecar: {json_path}")


if __name__ == "__main__":
    main()
