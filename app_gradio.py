"""
app_gradio.py

Demo 1 — Fetal Plane Classifier: Upload-a-Video, Watch-it-Get-Labeled.

Gradio web application that accepts an uploaded ultrasound video clip and
returns a fully-annotated MP4 with:
  • Per-frame plane label (one of 7 anatomical planes + "Other")
  • Confidence score
  • Grad-CAM saliency overlay (highlighting regions driving the prediction)
  • Stability badge (STABLE / SETTLING) from the smoothing pipeline
  • Optional performance HUD

ARCHITECTURE
------------
  Upload → [render_video()] → Annotated MP4 → Inline playback

The render_video() function in scripts/render_annotated_video.py does the
heavy lifting: it runs the full inference pipeline (load_inference_model →
Tier-1 smoothing → optional Tier-2a → persistent GradCAM → build_display_frame)
and writes browser-compatible H.264 via imageio-ffmpeg.

This file is purely the web layer — no inference code lives here.

USAGE
-----
  python app_gradio.py                  # launches on http://127.0.0.1:7860
  python app_gradio.py --port 8080      # custom port
  python app_gradio.py --share          # public Gradio tunnel (for demos)
  python app_gradio.py --no-browser     # headless launch (CI smoke test)

KNOWN LIMITATIONS (Demo 1)
--------------------------
  • Structure bounding boxes are NOT shown. The multitask detection model was
    trained for exactly 1 epoch (val macro-F1 0.8421, below the production
    classifier's 0.9183) and is not suitable for clinical display.
    Label + confidence only, using the frozen convnext_tiny classifier.
  • Live webcam streaming is not supported. Upload a pre-recorded clip.
  • Cross-device generalization: accuracy drops from 98.0% (in-distribution)
    to 83.2% on genuinely unseen devices (HC18/UCL). On a hospital machine
    that differs from the training sources expect similar degradation.
  • Genuine plane-transition latency has not been measured on annotated clips
    (see EVAL_REPORT.md §5.1). The smoothing parameters were tuned on
    within-class stability; cross-class transition timing is an open question.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project root on sys.path — allows importing from src/ and scripts/
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import gradio as gr

# Import the rendering engine
from scripts.render_annotated_video import render_video

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CHECKPOINT    = str(_ROOT / "checkpoints" / "convnext_tiny" / "best.pt")
_DEFAULT_TIER1_CONFIG  = str(_ROOT / "configs" / "smoothing_tier1.yaml")
_DEFAULT_TIER2_CONFIG  = str(_ROOT / "configs" / "smoothing_tier2a.yaml")

_CLASSES = [
    "Brain — Trans-cerebellum",
    "Brain — Trans-thalamic",
    "Brain — Trans-ventricular",
    "Fetal abdomen",
    "Fetal femur",
    "Fetal thorax",
    "Maternal cervix",
    "Other (non-standard / transitional)",
]

_MODEL_INFO_MD = f"""
### 🧠 Model Information

| Property | Value |
|---|---|
| Backbone | `convnext_tiny.fb_in22k_ft_in1k` |
| Pre-training | ImageNet-22k → ImageNet-1k fine-tune |
| Test macro-F1 | **0.8927** (held-out patient-disjoint split) |
| Classes | 7 anatomical planes + Other (8 total) |
| Smoothing | Tier-1: EMA (α=0.2) + hysteresis + dwell |
| Tier-2a | Majority-vote window (9 frames, 70% threshold) |
| Pipeline FPS | ~23.6 fps on RTX 4060 (live mode) |

### ⚠️ Known Limitations (Demo 1)

- **No structure bounding boxes.** The multitask detection model was trained
  for 1 epoch only (val macro-F1 0.84) and is not clinically reliable.
  Boxes will be added in a future demo once the detection model is fully trained.
- **Cross-device accuracy gap.** In-distribution accuracy: 98.0%.
  On unseen devices (HC18/UCL): 83.2%. Expect similar degradation on
  hospital machines differing from the training sources.
- **Transition latency unmeasured.** Smoothing was tuned on within-class
  stability; how long the system takes to follow a genuine plane transition
  has not been formally benchmarked.

### 📚 Classes
""" + "\n".join(f"- {c}" for c in _CLASSES)

_DESCRIPTION_MD = """
Upload a fetal ultrasound video clip and receive an AI-annotated version
showing:

- 🏷️ **Plane label** — which of 7 anatomical standard planes is visible
- 📊 **Confidence score** — how certain the model is
- 🔥 **Grad-CAM overlay** — which pixels drive the prediction
- ✅ **Stability badge** — STABLE when the prediction has held for ≥8 frames

Processing is **offline** (not real-time). A 30-second clip takes roughly
60–120 seconds to annotate with Grad-CAM enabled.
"""

# ---------------------------------------------------------------------------
# Processing function — called by Gradio on button click
# ---------------------------------------------------------------------------

def process_video(
    input_video: str | None,
    enable_gradcam: bool,
    gradcam_every_n: int,
    enable_tier2: bool,
    show_hud: bool,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str | None, str, str]:
    """Run the annotated-video render pipeline.

    Args:
        input_video:    Path to uploaded video (Gradio writes it to a temp file).
        enable_gradcam: Whether to compute Grad-CAM overlays.
        gradcam_every_n: Run Grad-CAM every N frames.
        enable_tier2:   Whether to apply Tier-2a majority-vote filter.
        show_hud:       Whether to render the performance HUD.
        progress:       Gradio progress tracker.

    Returns:
        Tuple of (output_video_path, status_markdown, label_log_json).
    """
    if input_video is None:
        return None, "⚠️ Please upload a video first.", "{}"

    input_path = Path(input_video)
    if not input_path.exists():
        return None, f"⚠️ Uploaded file not found: {input_path}", "{}"

    # Write output to a temp file so Gradio can serve it
    output_dir = Path(tempfile.gettempdir()) / "ultrasound_demo1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"annotated_{input_path.stem}.mp4"

    # Progress tracking via closure
    _last_total: list[int] = [0]

    def _progress_cb(done: int, total: int) -> None:
        if total > 0:
            _last_total[0] = total
            progress(done / total, desc=f"Annotating frame {done}/{total}")

    try:
        progress(0.0, desc="Loading model…")
        result = render_video(
            input_path=str(input_path),
            output_path=str(output_path),
            checkpoint=_DEFAULT_CHECKPOINT,
            smoothing_config=_DEFAULT_TIER1_CONFIG,
            tier2_config=_DEFAULT_TIER2_CONFIG if enable_tier2 else None,
            enable_tier2=enable_tier2,
            enable_gradcam=enable_gradcam,
            gradcam_every_n=gradcam_every_n,
            show_hud=show_hud,
            progress_cb=_progress_cb,
        )
        progress(1.0, desc="Done!")

    except Exception as exc:
        log.exception("render_video failed")
        return None, f"❌ Processing failed:\n```\n{exc}\n```", "{}"

    # ------------------------------------------------------------------
    # Status summary markdown
    # ------------------------------------------------------------------
    frames  = result["total_frames"]
    elapsed = result["elapsed_s"]
    fps_in  = result["fps_in"]
    clip_s  = frames / fps_in if fps_in > 0 else 0.0

    # Compute dominant label from log
    label_counts: dict[str, int] = {}
    for entry in result["label_log"]:
        lbl = entry["label"]
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    dominant_label = max(label_counts, key=label_counts.__getitem__) if label_counts else "—"
    dominant_pct   = 100.0 * label_counts.get(dominant_label, 0) / max(frames, 1)

    status_md = f"""
### ✅ Processing Complete

| Metric | Value |
|---|---|
| Frames annotated | {frames} |
| Source duration | {clip_s:.1f}s @ {fps_in:.1f} fps |
| Render time | {elapsed:.1f}s |
| Grad-CAM | {"Every frame" if (enable_gradcam and gradcam_every_n == 1) else f"Every {gradcam_every_n} frames" if enable_gradcam else "Disabled"} |
| Tier-2a smoothing | {"✓ Enabled" if enable_tier2 else "✗ Disabled"} |
| Dominant label | **{dominant_label.replace("_", " ")}** ({dominant_pct:.1f}% of frames) |

*Note: Structure bounding boxes are not shown in Demo 1 (see Model Info tab).*
"""

    # ------------------------------------------------------------------
    # Label log JSON (truncated to first 200 entries for display)
    # ------------------------------------------------------------------
    display_log = result["label_log"][:200]
    if len(result["label_log"]) > 200:
        display_log.append({"note": f"...{len(result['label_log']) - 200} more frames truncated"})
    label_log_json = json.dumps(display_log, indent=2)

    return str(output_path), status_md, label_log_json


# ---------------------------------------------------------------------------
# Gradio UI definition
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks interface."""

    with gr.Blocks(
        title="Fetal Plane Classifier — Demo 1",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        ),
        css="""
            .gradio-container { max-width: 1100px; margin: auto; }
            .output-video video { max-height: 500px; }
            footer { display: none !important; }
            .status-box { font-family: monospace; }
        """,
    ) as demo:

        # ------------------------------------------------------------------
        # Header
        # ------------------------------------------------------------------
        gr.Markdown(
            """
# 🩺 Fetal Plane Classifier — Demo 1

**Upload an ultrasound video → receive an AI-annotated version with plane labels, confidence scores, and Grad-CAM explanations.**

Powered by `convnext_tiny.fb_in22k_ft_in1k` · Test macro-F1 **0.8927** · Tier-1 + Tier-2a temporal smoothing
"""
        )
        gr.Markdown(_DESCRIPTION_MD)

        # ------------------------------------------------------------------
        # Main two-column layout
        # ------------------------------------------------------------------
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📥 Upload Video")
                input_video = gr.Video(
                    label="Input ultrasound clip",
                    sources=["upload"],
                    elem_id="input-video",
                )

                # Options accordion
                with gr.Accordion("⚙️ Processing Options", open=False):
                    enable_gradcam = gr.Checkbox(
                        label="Grad-CAM overlay",
                        value=True,
                        info="Highlights regions driving each prediction. Increases render time.",
                    )
                    gradcam_every_n = gr.Slider(
                        label="Grad-CAM every N frames",
                        minimum=1, maximum=10, step=1, value=1,
                        info="1 = every frame (max detail). Higher = faster render.",
                    )
                    enable_tier2 = gr.Checkbox(
                        label="Tier-2a smoothing (recommended)",
                        value=True,
                        info="Majority-vote window over 9 frames — eliminates residual flicker at the cost of ≤338ms added latency.",
                    )
                    show_hud = gr.Checkbox(
                        label="Performance HUD",
                        value=True,
                        info="Overlay inference timing stats on each frame.",
                    )

                process_btn = gr.Button(
                    "▶ Process Video",
                    variant="primary",
                    size="lg",
                )

            with gr.Column(scale=1):
                gr.Markdown("### 📤 Annotated Output")
                output_video = gr.Video(
                    label="Annotated clip",
                    autoplay=True,
                    elem_classes=["output-video"],
                    elem_id="output-video",
                )

        # ------------------------------------------------------------------
        # Status + collapsible extras
        # ------------------------------------------------------------------
        with gr.Row():
            status_md = gr.Markdown(
                "Upload a clip and click **▶ Process Video** to begin.",
                elem_classes=["status-box"],
            )

        with gr.Row():
            with gr.Accordion("📊 Per-frame Label Log (first 200 frames)", open=False):
                label_log_json = gr.JSON(label="Frame-level predictions")

            with gr.Accordion("ℹ️ Model Information & Limitations", open=False):
                gr.Markdown(_MODEL_INFO_MD)

        # ------------------------------------------------------------------
        # Examples (only shown if sample clips exist)
        # ------------------------------------------------------------------
        sample_dir = _ROOT / "data" / "processed" / "synthetic_clips"
        sample_files = sorted(sample_dir.glob("*.mp4"))[:3] if sample_dir.exists() else []
        if sample_files:
            gr.Examples(
                examples=[[str(f)] for f in sample_files],
                inputs=[input_video],
                label="🎬 Example clips (synthetic — for pipeline demonstration only)",
            )

        # ------------------------------------------------------------------
        # Footer note
        # ------------------------------------------------------------------
        gr.Markdown(
            """
---
*Demo 1 · For research / demonstration only · Not validated for clinical use.*
*Trained on FETAL\\_PLANES\\_DB (Burgos-Artizzu et al., 2020) + UCL/HC18 cross-device set.*
"""
        )

        # ------------------------------------------------------------------
        # Wire up the button
        # ------------------------------------------------------------------
        process_btn.click(
            fn=process_video,
            inputs=[
                input_video,
                enable_gradcam,
                gradcam_every_n,
                enable_tier2,
                show_hud,
            ],
            outputs=[output_video, status_md, label_log_json],
            show_progress="full",
        )

    return demo


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch the Fetal Plane Classifier Demo 1 Gradio web app.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--port", type=int, default=7860, help="Port to listen on.")
    p.add_argument("--host", default="127.0.0.1", help="Host to bind to.")
    p.add_argument("--share", action="store_true",
                   help="Create a public Gradio tunnel (for live demos).")
    p.add_argument("--no-browser", action="store_true", dest="no_browser",
                   help="Do not open a browser tab on launch.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG logging.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate checkpoint exists before spending time starting the server
    ckpt = Path(_DEFAULT_CHECKPOINT)
    if not ckpt.exists():
        log.error(
            "Checkpoint not found: %s\n"
            "Train the model first:  python src/train/train.py --config configs/convnext_tiny.yaml\n"
            "Or adjust _DEFAULT_CHECKPOINT in app_gradio.py.",
            ckpt,
        )
        sys.exit(1)

    demo = build_ui()
    # theme and css passed here for Gradio 5/6 forward compatibility
    # (Gradio 6 moved them from Blocks() constructor to launch())
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
