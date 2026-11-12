"""
app_streamlit.py

FetScan — Real-Time Sonographic Plane Analysis.
Primary web interface (Streamlit).

ARCHITECTURE
------------
  Upload → [render_video()] → Annotated MP4 → st.video() inline playback

The render_video() function in scripts/render_annotated_video.py does all
the inference work (load_inference_model → Tier-1 smoothing → optional
Tier-2a → persistent GradCAM → build_display_frame) and writes
browser-compatible H.264 via imageio-ffmpeg.

This file is purely the Streamlit web layer — no inference code lives here.
The session-state pattern isolates the last result so widget interactions
(e.g. expanding the label log accordion) do not re-trigger a full render.

STYLING
-------
Custom CSS is loaded from assets/style.css at startup via _load_css().
The file is a no-op stub for Demo 1 — premium styling will be added in a
later pass without touching this file. Each UI section is factored into its
own function (_render_header, _render_sidebar, etc.) so CSS selectors have
clean, predictable targets.

USAGE
-----
  conda run -n fetalplane streamlit run app_streamlit.py
  conda run -n fetalplane streamlit run app_streamlit.py -- --port 8502

KNOWN LIMITATIONS (Demo 1)
--------------------------
  • Structure bounding boxes are NOT shown. The multitask detection model was
    trained for exactly 1 epoch (val macro-F1 0.8421, below the production
    classifier's 0.9183) and is not suitable for clinical display.
  • Live webcam streaming is not supported. Upload a pre-recorded clip.
  • Cross-device generalization: accuracy drops from 98.0% (in-distribution)
    to 83.2% on genuinely unseen devices (HC18/UCL).
  • st.file_uploader returns a BytesIO object, not a file path. The upload
    is therefore written to outputs/demo1/ before being passed to render_video().
  • Grad-CAM is OFF by default (clinical mode). Enable it in the sidebar for
    model explanation or research use.

NOTES ON STREAMLIT PROGRESS
----------------------------
Streamlit's script execution is synchronous within a single session. The
st.progress() object created before render_video() is called can be mutated
in-place by the progress_cb closure during the blocking render call. This
works because Streamlit re-renders on each progress() mutation (fragment
protocol). No threading is required.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import streamlit as st

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

# ---------------------------------------------------------------------------
# Project root on sys.path — allows importing from src/ and scripts/
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.render_annotated_video import render_video

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DEFAULT_CHECKPOINT   = str(_ROOT / "checkpoints" / "convnext_tiny" / "best.pt")
_DEFAULT_TIER1_CONFIG = str(_ROOT / "configs" / "smoothing_tier1.yaml")
_DEFAULT_TIER2_CONFIG = str(_ROOT / "configs" / "smoothing_tier2a.yaml")
_ASSETS_CSS           = _ROOT / "assets" / "style.css"
_UPLOAD_DIR           = _ROOT / "outputs" / "demo1"
_SAMPLE_CLIPS_DIR     = _ROOT / "data" / "processed" / "synthetic_clips"
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

# ---------------------------------------------------------------------------
# CSS loader
# ---------------------------------------------------------------------------

def _load_css(path: Path) -> None:
    """Inject custom CSS from assets/style.css."""
    if path.exists() and path.stat().st_size > 0:
        css_content = path.read_text(encoding="utf-8")
        if hasattr(st, "html"):
            st.html(f"<style>\n{css_content}\n</style>")
        else:
            st.markdown(f"<style>\n{css_content}\n</style>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# UI sections — separated so CSS selectors have clean targets
# ---------------------------------------------------------------------------

def _render_header() -> None:
    """Page title and subtitle in editorial style."""
    st.markdown(
        """
        <div class="fpc-masthead">
            <h1>FetScan</h1>
            <p style="font-size: 1.1rem; margin-bottom: 0.25rem;">
                An AI assistant that watches a fetal ultrasound scan in real time and tells the sonographer which standard anatomical plane is on screen — and how confident it is.
            </p>
            <div class="fpc-metadata-ticker">REAL-TIME SONOGRAPHIC PLANE ANALYSIS · 8-CLASS DEEP LEARNING SYSTEM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown(
        """
        During a fetal anatomy scan, a sonographer must correctly identify and capture ~7 standard reference planes (brain, abdomen, femur, thorax, cervix) to complete a valid exam. Missed or mislabeled planes are a leading source of scan-quality variability between operators. FetScan classifies the current frame in real time and holds a stable label even as the probe moves, so the operator gets continuous feedback on what plane they're looking at.
        """
    )
    
    # 4-Step Process Strip
    st.markdown(
        """
        <div class="fpc-step-strip">
            <div style="display: flex; flex-direction: row; justify-content: space-between; align-items: stretch; gap: 1rem;">
                <div class="fpc-step"><strong>01 / UPLOAD</strong><br/>clip in MP4/AVI</div>
                <div class="fpc-step"><strong>02 / INFERENCE</strong><br/>per-frame classification</div>
                <div class="fpc-step"><strong>03 / SMOOTHING</strong><br/>EMA + majority vote</div>
                <div class="fpc-step"><strong>04 / OUTPUT</strong><br/>labeled, stable video</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_clinical_background() -> None:
    """Renders the professional project background and clinical context."""
    st.markdown('<div class="fpc-clinical-section">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown(
            """
            ### Precision AI for Diagnostic Obstetrics
            FetScan provides sonographers and maternal-fetal medicine specialists with a highly robust, real-time "second pair of eyes" during routine fetal anatomy scans. 
            
            By applying deep learning directly to the ultrasound video stream, the system instantly identifies standard diagnostic planes, assisting in standardizing acquisitions and reducing inter-operator variability.
            """
        )
        st.image(str(_ROOT / "assets" / "clinical_sonography_setup.jpg"))
        st.markdown('<div class="fpc-figure-caption">FIG 01 — CLINICAL DIAGNOSTIC WORKSTATION INTEGRATION</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(
            """
            ### Temporal Stability & Interpretability
            Unlike standard frame-by-frame classifiers that flicker uselessly in a clinical setting, FetScan acts as a shock absorber: locking onto a diagnosis only when confident and holding it stable via dual-tier temporal smoothing.
            
            - **Tier 1:** Exponential Moving Average (EMA) prevents rapid frame-to-frame confidence swings.
            - **Tier 2a:** A sliding 9-frame majority-vote window enforces clinical dwell time before registering a plane transition.
            - **Interpretability:** A throttled Grad-CAM diagnostic overlay provides a clear visual explanation of *why* a particular anatomical structure was identified.
            """
        )
        st.image(str(_ROOT / "assets" / "pipeline_architecture_diagram.jpg"))
        st.markdown('<div class="fpc-figure-caption">FIG 02 — FETSCAN AI PIPELINE ARCHITECTURE</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)



def _render_sidebar() -> dict[str, Any]:
    """Render the options sidebar and return the user's current settings.

    Returns a dict with keys:
        enable_gradcam  : bool
        gradcam_every_n : int
        enable_tier2    : bool
        show_hud        : bool
        enable_cam_bbox : bool  (Phase 8 Stage 3 — approx. region box)
    """
    with st.sidebar:
        st.markdown("### CONTROL PANEL")
        
        st.markdown("<div style='font-family: var(--font-mono); font-size: 0.8rem; border-top: 1px solid var(--ink-black); margin-top: 1rem; padding-top: 0.5rem;'>[ 01 ] GRAD-CAM</div>", unsafe_allow_html=True)
        enable_gradcam = st.checkbox(
            "Enable Grad-CAM overlay",
            value=False,
            help=(
                "Computes a saliency heatmap showing which pixels drive each prediction. "
                "Disabled by default — adds significant render time (~2–3×). "
                "Enable for model explanation or research use."
            ),
        )
        gradcam_every_n = st.slider(
            "Grad-CAM every N frames",
            min_value=1, max_value=10, value=5, step=1,
            disabled=not enable_gradcam,
            help="Higher N = faster render. Cached overlay reused on skipped frames.",
        )

        st.markdown("<div style='font-family: var(--font-mono); font-size: 0.8rem; border-top: 1px solid var(--ink-black); margin-top: 1rem; padding-top: 0.5rem;'>[ 02 ] SMOOTHING</div>", unsafe_allow_html=True)
        enable_tier2 = st.checkbox(
            "Tier-2a smoothing",
            value=True,
            help=(
                "Majority-vote window (9 frames, 70% threshold) on top of Tier-1 EMA. "
                "Eliminates residual label flicker at the cost of ≤338ms added latency."
            ),
        )

        st.markdown("<div style='font-family: var(--font-mono); font-size: 0.8rem; border-top: 1px solid var(--ink-black); margin-top: 1rem; padding-top: 0.5rem;'>[ 03 ] DISPLAY</div>", unsafe_allow_html=True)
        show_hud = st.checkbox(
            "Performance HUD",
            value=True,
            help="Overlay per-frame inference timings on the annotated video.",
        )

        st.markdown("<div style='font-family: var(--font-mono); font-size: 0.8rem; border-top: 1px solid var(--ink-black); margin-top: 1rem; padding-top: 0.5rem;'>[ 04 ] LOCALISATION</div>", unsafe_allow_html=True)
        enable_cam_bbox = st.checkbox(
            "Approx. region box (saliency-derived)",
            value=True,
            disabled=not enable_gradcam,
            help=(
                "Draws a dashed orange bounding box around the most salient region "
                "identified by Grad-CAM, labelled 'approx. region (saliency-derived)'. "
                "This is a weakly-supervised approximate localisation — NOT a "
                "precisely measured anatomical boundary. Requires Grad-CAM enabled."
            ),
        )

        st.divider()

        # Model information panel
        with st.expander("[ MODEL INFO ]", expanded=False):
            st.markdown(
                """
| Property | Value |
|---|---|
| Backbone | `convnext_tiny.fb_in22k_ft_in1k` |
| Pre-training | ImageNet-22k → 1k fine-tune |
| Test macro-F1 | **0.8927** |
| Classes | 7 planes + Other (8 total) |
| Smoothing | Tier-1: EMA α=0.2 + hysteresis + dwell=8 |
| Tier-2a | Majority-vote window=9, threshold=70% |

**Known Limitations**
- No structure bounding boxes (detection model trained 1 epoch only).
- In-distribution accuracy: 98.0%. On unseen devices (HC18/UCL): 83.2%.
- Transition latency not formally benchmarked.
- Grad-CAM is OFF by default (clinical mode).
"""
            )

        with st.expander("[ CLASSES ]", expanded=False):
            for c in _CLASSES:
                st.markdown(f"- {c}")

    return {
        "enable_gradcam":  enable_gradcam,
        "gradcam_every_n": gradcam_every_n,
        "enable_tier2":    enable_tier2,
        "show_hud":        show_hud,
        "enable_cam_bbox": enable_cam_bbox,
    }


def _render_upload_col(col: Any) -> "UploadedFile | None":
    """Render the upload / input column.

    Returns the UploadedFile object, or None if nothing uploaded.
    """
    with col:
        # st.markdown('<div class="fpc-intake-panel">', unsafe_allow_html=True)
        st.markdown('<div class="fpc-analysis-header">INPUT / 01</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a video clip",
            type=["mp4", "avi", "mov", "mkv"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            # Preview the raw upload so the user can confirm it's the right clip
            st.video(uploaded)
            st.caption(f"📎 `{uploaded.name}` — {uploaded.size / 1024:.0f} KB")
        st.markdown('</div>', unsafe_allow_html=True)
    return uploaded


def _save_upload(uploaded: Any) -> Path:
    """Write the uploaded file bytes to disk and return the path.

    Streamlit's UploadedFile is a BytesIO-like object, not a path.
    render_video() requires a real filesystem path, so we persist it here.
    Output dir (outputs/demo1/) is the canonical project-local output location.

    """
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    # Sanitise the filename: replace spaces with underscores
    safe_name = uploaded.name.replace(" ", "_")
    dest = _UPLOAD_DIR / f"upload_{safe_name}"
    dest.write_bytes(uploaded.read())
    return dest


def _build_status_markdown(result: dict[str, Any], opts: dict[str, Any]) -> str:
    """Build the status summary markdown string from a render_video() result."""
    frames   = result["total_frames"]
    elapsed  = result["elapsed_s"]
    fps_in   = result["fps_in"]
    clip_s   = frames / fps_in if fps_in > 0 else 0.0

    label_counts: dict[str, int] = {}
    for entry in result["label_log"]:
        lbl = entry.get("label_name", str(entry.get("label", "?")))
        label_counts[lbl] = label_counts.get(lbl, 0) + 1
    dominant = max(label_counts, key=label_counts.__getitem__) if label_counts else "—"
    dominant_pct = 100.0 * label_counts.get(dominant, 0) / max(frames, 1)

    gradcam_str = (
        f"Every {opts['gradcam_every_n']} frames"
        if opts["enable_gradcam"] else "Disabled"
    )

    return f"""
| Metric | Value |
|---|---|
| Frames annotated | {frames} |
| Source duration | {clip_s:.1f}s @ {fps_in:.1f} fps |
| Render time | {elapsed:.1f}s |
| Grad-CAM | {gradcam_str} |
| Tier-2a smoothing | {"✓ Enabled" if opts["enable_tier2"] else "✗ Disabled"} |
| Dominant label | **{dominant}** ({dominant_pct:.1f}% of frames) |

*Bounding boxes not shown (classifier-only checkpoint — see Model Information).*
"""


def _render_output_col(col: Any, result: dict[str, Any] | None, opts: dict[str, Any]) -> None:
    """Render the annotated output column from a completed render result."""
    with col:
        # st.markdown('<div class="fpc-analysis-panel">', unsafe_allow_html=True)
        st.markdown('<div class="fpc-analysis-header">OUTPUT / 02</div>', unsafe_allow_html=True)
        
        if result is None:
            # We are waiting for input, display a placeholder
            st.markdown(
                """
                <div style="border: 1px dashed var(--text-secondary); padding: 3rem; text-align: center; color: var(--text-secondary); font-family: var(--font-mono); font-size: 0.9rem;">
                    AWAITING INPUT
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown('</div>', unsafe_allow_html=True)
            return

        out_path = Path(result.get("output_path", ""))
        if out_path.exists():
            st.video(str(out_path))
        else:
            st.error(f"Output file not found: {out_path}")
            st.markdown('</div>', unsafe_allow_html=True)
            return

        # Status summary
        st.markdown("### Processing Complete")
        st.markdown(_build_status_markdown(result, opts))

        # Per-frame label log
        with st.expander("[ PER-FRAME LABEL LOG ]", expanded=False):
            display_log = result["label_log"][:200]
            if len(result["label_log"]) > 200:
                display_log = display_log + [
                    {"note": f"…{len(result['label_log']) - 200} more frames truncated"}
                ]
            st.json(display_log)
        
        st.markdown('</div>', unsafe_allow_html=True)


def _render_examples() -> "str | None":
    """Render the example clips section.

    Returns the path of the clip the user clicked, or None.
    Streamlit doesn't support a direct 'click to select' on st.video, so we
    use buttons alongside each preview — one click sets the example path in
    session state, which the main loop picks up on the next rerun.
    """
    all_clips = sorted(_SAMPLE_CLIPS_DIR.glob("*.mp4"))[:3] if _SAMPLE_CLIPS_DIR.exists() else []
    
    if not all_clips:
        return None

    st.divider()
    st.markdown("### Example Clips")

    cols = st.columns(len(all_clips))
    selected: str | None = None
    
    for i, (col, clip) in enumerate(zip(cols, all_clips)):
        with col:
            st.video(str(clip))
            caption = f"FIG. {i+1:02d} — {clip.stem.replace('_', ' ').upper()}"
            st.markdown(f'<div class="fpc-figure-caption">{caption}</div>', unsafe_allow_html=True)
            
            st.caption(
                "Synthetic clips generated from single frames via depth-layered parallax compositing. "
                "For pipeline demonstration only — not real fetal ultrasound recordings."
            )

            if st.button(f"USE THIS CLIP", key=f"ex_{clip.stem}", use_container_width=True):
                selected = str(clip)
                
    return selected


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    # ---- Page config (must be first Streamlit call) --------------------------
    st.set_page_config(
        page_title="FetScan — Plane Analysis",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ---- CSS injection -------------------------------------------------------
    _load_css(_ASSETS_CSS)

    # ---- Sidebar options -----------------------------------------------------
    opts = _render_sidebar()

    # ---- Header ----------------------------------------
    _render_header()

    # ---- Session state initialisation ----------------------------------------
    # Persists across widget interactions so the output panel doesn't disappear
    # when e.g. the label-log expander is toggled.
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    # ---- Main layout: functional UI First ------------------------------------
    col_left, col_right = st.columns(2, gap="large")

    uploaded = _render_upload_col(col_left)

    # ---- Example clip selection ----------------------------------------------
    # Runs below the columns so examples span full width.
    example_path = _render_examples()
    if example_path:
        # User clicked an example: stash path in session_state and rerun
        st.session_state["example_path"] = example_path
        st.rerun()

    # ---- Process button ------------------------------------------------------
    with col_left:
        process_btn = st.button(
            "PROCESS VIDEO",
            type="primary",
            disabled=(uploaded is None and "example_path" not in st.session_state),
            use_container_width=True,
        )

    # ---- Output placeholder for loading states -------------------------------
    # This prevents the empty box bug where `with col_right` was drawing another panel.
    status_placeholder = None
    progress_bar = None

    # ---- Run render on button click ------------------------------------------
    if process_btn:
        # Determine input: real upload takes priority over example selection
        if uploaded is not None:
            input_path = _save_upload(uploaded)
        elif "example_path" in st.session_state:
            input_path = Path(st.session_state["example_path"])
        else:
            st.warning("⚠️ Please upload a video or select an example clip first.")
            st.stop()

        stem = input_path.stem
        output_path = _UPLOAD_DIR / f"annotated_{stem}.mp4"

        with col_right:
            st.markdown('<div class="fpc-analysis-panel">', unsafe_allow_html=True)
            st.markdown('<div class="fpc-analysis-header">OUTPUT / 02</div>', unsafe_allow_html=True)
            status_placeholder = st.empty()
            progress_bar = st.progress(0, text="Loading model…")

            def _progress_cb(done: int, total: int) -> None:
                if total > 0:
                    pct = done / total
                    progress_bar.progress(pct, text=f"Annotating frame {done}/{total}")

            try:
                status_placeholder.info("⏳ Processing — this may take up to 2 minutes for a 20s clip with Grad-CAM enabled…")
                result = render_video(
                    input_path=str(input_path),
                    output_path=str(output_path),
                    checkpoint=_DEFAULT_CHECKPOINT,
                    smoothing_config=_DEFAULT_TIER1_CONFIG,
                    tier2_config=_DEFAULT_TIER2_CONFIG if opts["enable_tier2"] else None,
                    enable_tier2=opts["enable_tier2"],
                    enable_gradcam=opts["enable_gradcam"],
                    gradcam_every_n=opts["gradcam_every_n"],
                    show_hud=opts["show_hud"],
                    enable_cam_bbox=opts["enable_cam_bbox"],
                    progress_cb=_progress_cb,
                )
                progress_bar.progress(1.0, text="Done!")
                status_placeholder.success("Processing complete.")
                st.session_state["last_result"] = result
                st.session_state["last_opts"] = opts.copy()
                st.markdown('</div>', unsafe_allow_html=True)
                st.rerun() # Rerun to render the actual output via _render_output_col

            except Exception as exc:
                log.exception("render_video failed")
                progress_bar.empty()
                status_placeholder.error(f"❌ Processing failed: {exc}")
                with st.expander("Error details"):
                    st.exception(exc)
                st.markdown('</div>', unsafe_allow_html=True)
                st.stop()
            
            st.markdown('</div>', unsafe_allow_html=True)

    # ---- Output panel (persistent across interactions) -----------------------
    # Only render if we aren't currently processing a click
    if not process_btn:
        last_result = st.session_state.get("last_result")
        last_opts   = st.session_state.get("last_opts", opts)
        _render_output_col(col_right, last_result, last_opts)

    # ---- Clinical Background (moved below functional UI) ---------------------
    _render_clinical_background()

    # ---- Footer --------------------------------------------------------------
    st.divider()
    st.caption(
        "**FetScan** · For research / demonstration only · Not validated for clinical use.  \n"
        "Trained on FETAL\\_PLANES\\_DB (Burgos-Artizzu et al., 2020) + UCL/HC18 cross-device set."
    )


if __name__ == "__main__":
    main()
