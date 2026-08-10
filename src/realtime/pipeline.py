"""
src/realtime/pipeline.py

Threaded two-stage pipeline: CaptureThread → InferenceThread.

Architecture (per docs/05_TEMPORAL_SMOOTHING_AND_REALTIME.md §B2 and
PHASE_5_KICKOFF_PROMPT.md §9):

    [FrameSource]
        ↓ (frame_bgr, cap_timestamp)
    [CaptureThread]  ──frame_queue (DropOldestQueue, size 2)──►
    [InferenceThread]
        1. prep_frame_grayscale_to_rgb  (matches training preprocessing exactly)
        2. albumentations eval transform  (img_size/mean/std from checkpoint config)
        3. torch.no_grad() + autocast forward pass  (RTX 4060 GPU, fp16)
        4. Tier1Smoother.step()  (tuned params from configs/smoothing_tier1.yaml)
        5. Grad-CAM (persistent instance, throttled — every N frames OR 200 ms wall)
        6. push result dict  ──result_queue (DropOldestQueue, size 2)──►
    [Main/render thread]  (cv2.imshow/waitKey strictly on this thread only)

Non-negotiable constraints enforced here:
  • backbone, image_size, normalization stats are NEVER hardcoded — always read
    from checkpoint["config"] via LoadedModel.
  • cv2.imshow and cv2.waitKey NEVER appear in this file.
  • GradCAM instance created ONCE in InferenceThread.__init__ and reused.
    run_gradcam() from src/models/gradcam.py is NOT used here because it creates
    a new GradCAM context-manager (and re-registers hooks) on every call.
  • Both threads respect a shared threading.Event for clean shutdown.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import torch
import yaml
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from src.data.dataset import IDX_TO_CLASS, NUM_CLASSES
from src.data.transforms import prep_frame_grayscale_to_rgb
from src.models.gradcam import get_target_layer
from src.realtime.capture import FrameSource
from src.realtime.model_loader import LoadedModel
from src.realtime.queues import DropOldestQueue
from src.smoothing.tier1 import Tier1Smoother

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Thread-safe performance statistics
# ---------------------------------------------------------------------------

class PipelineStats:
    """Rolling, thread-safe performance metrics for both pipeline threads.

    All properties are safe to read from the main/render thread while the
    capture and inference threads are updating.

    FPS is computed from a rolling window of frame timestamps (last _WINDOW
    frames).  Per-stage latencies use an exponential moving average (EWMA)
    so individual outlier frames don't dominate the displayed value.
    """

    _WINDOW: int = 60          # frames for rolling FPS calculation
    _LAT_ALPHA: float = 0.15   # EWMA weight for latency smoothing (lower = smoother)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Rolling timestamp deques (monotonic seconds at frame completion)
        self._cap_ts: collections.deque[float] = collections.deque(maxlen=self._WINDOW)
        self._inf_ts: collections.deque[float] = collections.deque(maxlen=self._WINDOW)
        # EWMA per-stage latencies (milliseconds)
        self._preprocess_ms: float = 0.0
        self._forward_ms: float = 0.0
        self._smoothing_ms: float = 0.0
        self._gradcam_ms: float = 0.0   # EWMA of runs that actually ran GradCAM
        self._gradcam_calls: int = 0    # total GradCAM invocations
        self.frames_processed: int = 0  # Total inference frames processed
        # Queue drop counters — updated by InferenceThread from queue.drops
        self.cap_queue_drops: int = 0
        self.inf_queue_drops: int = 0

    # ---- FPS ----------------------------------------------------------------

    @staticmethod
    def _compute_fps(ts_deque: collections.deque) -> float:
        if len(ts_deque) < 2:
            return 0.0
        span = ts_deque[-1] - ts_deque[0]
        return (len(ts_deque) - 1) / span if span > 0.0 else 0.0

    @property
    def capture_fps(self) -> float:
        with self._lock:
            return self._compute_fps(self._cap_ts)

    @property
    def inference_fps(self) -> float:
        with self._lock:
            return self._compute_fps(self._inf_ts)

    # ---- Latencies (read-only, for HUD display) ----------------------------

    @property
    def preprocess_ms(self) -> float:
        with self._lock:
            return self._preprocess_ms

    @property
    def forward_ms(self) -> float:
        with self._lock:
            return self._forward_ms

    @property
    def smoothing_ms(self) -> float:
        with self._lock:
            return self._smoothing_ms

    @property
    def gradcam_ms(self) -> float:
        with self._lock:
            return self._gradcam_ms

    @property
    def gradcam_calls(self) -> int:
        with self._lock:
            return self._gradcam_calls

    # ---- Writer methods (called from threads) ------------------------------

    def record_capture_frame(self) -> None:
        with self._lock:
            self._cap_ts.append(time.monotonic())

    def record_inference_frame(
        self,
        preprocess_ms: float,
        forward_ms: float,
        smoothing_ms: float,
    ) -> None:
        a = self._LAT_ALPHA
        with self._lock:
            self._inf_ts.append(time.monotonic())
            self._preprocess_ms = a * preprocess_ms + (1 - a) * self._preprocess_ms
            self._forward_ms    = a * forward_ms    + (1 - a) * self._forward_ms
            self._smoothing_ms  = a * smoothing_ms  + (1 - a) * self._smoothing_ms
            self.frames_processed += 1

    def record_gradcam(self, gradcam_ms: float) -> None:
        a = 0.25  # slightly higher alpha for GradCAM since it runs infrequently
        with self._lock:
            self._gradcam_calls += 1
            self._gradcam_ms = a * gradcam_ms + (1 - a) * self._gradcam_ms

    # ---- Convenience snapshot (atomic read of all fields) ------------------

    def snapshot(self) -> dict[str, Any]:
        """Return all current stats as a plain dict (no lock held outside call)."""
        with self._lock:
            return {
                "capture_fps":     self._compute_fps(self._cap_ts),
                "inference_fps":   self._compute_fps(self._inf_ts),
                "preprocess_ms":   self._preprocess_ms,
                "forward_ms":      self._forward_ms,
                "smoothing_ms":    self._smoothing_ms,
                "gradcam_ms":      self._gradcam_ms,
                "gradcam_calls":   self._gradcam_calls,
                "frames_processed": self.frames_processed,
                "cap_queue_drops": self.cap_queue_drops,
                "inf_queue_drops": self.inf_queue_drops,
            }


# ---------------------------------------------------------------------------
# Capture thread
# ---------------------------------------------------------------------------

class CaptureThread(threading.Thread):
    """Reads frames from a FrameSource as fast as possible and pushes them to
    a bounded DropOldestQueue.

    The drop-oldest policy ensures the inference thread always sees the most
    recent available frame.  Queue drops are expected and healthy when the
    camera/file provides frames faster than inference can consume them — they
    are recorded in PipelineStats for diagnostic purposes.

    Args:
        source:      An open FrameSource (file or webcam).
        frame_queue: Destination queue; shared with InferenceThread.
        stop_event:  Shared stop signal.  Set by either thread to trigger a
                     clean shutdown of the whole pipeline.
        stats:       Shared PipelineStats instance.
    """

    def __init__(
        self,
        source: FrameSource,
        frame_queue: DropOldestQueue[tuple[np.ndarray, float]],
        stop_event: threading.Event,
        stats: PipelineStats,
    ) -> None:
        super().__init__(name="CaptureThread", daemon=True)
        self._source = source
        self._queue = frame_queue
        self._stop_event = stop_event   # NOTE: NOT self._stop — that name shadows Thread._stop() in Python 3.12
        self._stats = stats

    def stop(self) -> None:
        """Signal the thread to exit at its next iteration."""
        self._stop_event.set()

    def run(self) -> None:
        log.info("CaptureThread: starting.")
        drops_before = self._queue.drops
        while not self._stop_event.is_set():
            ret, frame = self._source.read()
            if not ret or frame is None:
                log.info("CaptureThread: source exhausted — signalling stop.")
                self._stop_event.set()
                break
            self._stats.record_capture_frame()
            self._queue.put((cast(np.ndarray, frame), time.monotonic()))
            # Sync drop counter into stats (reads are approximate but safe)
            self._stats.cap_queue_drops = self._queue.drops - drops_before
        log.info("CaptureThread: exiting.")


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------

class InferenceThread(threading.Thread):
    """Pulls frames from the frame queue, runs the full inference pipeline,
    and pushes result dicts to the result queue.

    Pipeline per frame:
      1. prep_frame_grayscale_to_rgb  → HxWx3 uint8 RGB
      2. albumentations eval transform → normalised [C,H,W] tensor on device
      3. torch.no_grad() + autocast   → logits → float32 softmax probs (CPU)
      4. Tier1Smoother.step()         → (label, confidence, smoothed_probs, is_stable)
      5. GradCAM (persistent, throttled) → HxWx3 uint8 BGR overlay (or None/reused)
      6. push result dict to result_queue

    Grad-CAM notes:
      • A single GradCAM instance is created in __init__ and its hooks remain
        registered for the entire thread lifetime.  run_gradcam() from
        src/models/gradcam.py is intentionally NOT used here — that function
        creates a new GradCAM context-manager on every call, which re-registers
        hooks and incurs ~5-15ms of unnecessary overhead.
      • GradCAM runs when EITHER N frames have been processed OR 200ms of
        wall-clock time has elapsed since the last run (whichever triggers first).
      • On non-GradCAM frames the last computed overlay is reused unchanged.
      • The overlay is stored as HxWx3 uint8 in BGR colour space (matching
        frame_bgr) so the render loop can blend it directly without conversion.

    Args:
        frame_queue:           Source queue shared with CaptureThread.
        result_queue:          Destination queue for the render loop.
        loaded_model:          LoadedModel from load_inference_model().
        smoothing_config_path: Path to configs/smoothing_tier1.yaml.
        stop_event:            Shared stop signal.
        stats:                 Shared PipelineStats instance.
        gradcam_every_n_frames: Run GradCAM every N processed frames (default 10).
        gradcam_wall_ms:       Also run GradCAM if this many ms have elapsed
                               since the last run (default 200 ms).
        enable_gradcam:        Set False to disable GradCAM entirely (useful for
                               headless benchmarking).
    """

    def __init__(
        self,
        frame_queue: DropOldestQueue[tuple[np.ndarray, float]],
        result_queue: DropOldestQueue[dict[str, Any]],
        loaded_model: LoadedModel,
        smoothing_config_path: str | Path,
        stop_event: threading.Event,
        stats: PipelineStats,
        gradcam_every_n_frames: int = 10,
        gradcam_wall_ms: float = 200.0,
        enable_gradcam: bool = True,
    ) -> None:
        super().__init__(name="InferenceThread", daemon=True)

        self._frame_queue = frame_queue
        self._result_queue = result_queue
        self._model = loaded_model.model
        self._backbone_name = loaded_model.backbone_name
        self._device = loaded_model.device
        self._transform = loaded_model.transform
        self._stop_event = stop_event   # NOTE: NOT self._stop — shadows Thread._stop() in Python 3.12
        self._stats = stats
        self._gradcam_every_n = max(1, gradcam_every_n_frames)
        self._gradcam_wall_ms = gradcam_wall_ms
        self._use_amp = (self._device.type == "cuda")

        # --- Load Tier-1 smoothing config from YAML -------------------------
        cfg_path = Path(smoothing_config_path)
        with open(cfg_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self._smoother = Tier1Smoother(
            num_classes=NUM_CLASSES,
            alpha=float(cfg["alpha"]),
            switch_threshold=float(cfg["switch_threshold"]),
            min_dwell_frames=int(cfg["min_dwell_frames"]),
            hold_floor=cfg.get("hold_floor"),  # None when YAML has "null"
        )
        log.info(
            "InferenceThread: Tier1Smoother loaded — alpha=%.2f sw_thr=%.2f "
            "dwell=%d hold_floor=%s",
            cfg["alpha"], cfg["switch_threshold"],
            cfg["min_dwell_frames"], cfg.get("hold_floor"),
        )

        # --- Persistent GradCAM instance ------------------------------------
        # Created ONCE here; hooks remain registered until cleanup() is called.
        # This avoids the hook-registration overhead that run_gradcam() incurs
        # on every throttled call (~5-15 ms per call for hook setup/teardown).
        if enable_gradcam:
            target_layer = get_target_layer(self._model, self._backbone_name)
            self._cam: GradCAM | None = GradCAM(
                model=self._model,
                target_layers=[target_layer],
            )
            log.info(
                "InferenceThread: persistent GradCAM initialised for %s "
                "(target layer type: %s, throttle: every %d frames or %.0f ms).",
                self._backbone_name, type(target_layer).__name__,
                self._gradcam_every_n, self._gradcam_wall_ms,
            )
        else:
            self._cam = None
            log.info("InferenceThread: GradCAM disabled.")

        # State for throttling and overlay reuse
        self._last_overlay: np.ndarray | None = None  # BGR uint8
        self._frames_processed: int = 0
        self._last_gradcam_wall_t: float = 0.0

    # ---- Lifecycle ----------------------------------------------------------


    def stop(self) -> None:
        """Signal the thread to exit at its next iteration."""
        self._stop_event.set()



    def cleanup(self) -> None:
        """Unregister GradCAM hooks and release resources.

        In grad-cam 1.5.5 (installed version), BaseCAM.__exit__ calls
        self.activations_and_grads.release() to remove hooks.  There is no
        public remove_handlers() method in this version — we call the
        underlying release() directly, which is equivalent and stable across
        the 1.x API.

        Safe to call after join().  Idempotent.
        """
        if self._cam is not None:
            try:
                self._cam.activations_and_grads.release()
            except Exception as exc:
                log.warning("InferenceThread cleanup: GradCAM hook release failed: %s", exc)
            self._cam = None
            log.debug("InferenceThread: GradCAM hooks released.")


    # ---- Main loop ----------------------------------------------------------

    def run(self) -> None:
        log.info(
            "InferenceThread: starting (device=%s, amp=%s, gradcam=%s).",
            self._device.type, self._use_amp, self._cam is not None,
        )
        drops_before = self._result_queue.drops

        while not self._stop_event.is_set():
            # --- Pull latest frame (non-blocking; yield if nothing available) ---
            item = self._frame_queue.get_nowait_or_none()
            if item is None:
                time.sleep(0.001)
                continue

            frame_bgr: np.ndarray
            cap_ts: float
            frame_bgr, cap_ts = item
            self._frames_processed += 1

            # ----------------------------------------------------------------
            # Stage 1 — Preprocess
            # Converts BGR VideoCapture frame → normalised NCHW float32 tensor.
            # Uses prep_frame_grayscale_to_rgb + albumentations eval transform
            # to guarantee identical preprocessing to the training pipeline.
            # ----------------------------------------------------------------
            t0 = time.monotonic()
            rgb_hw3: np.ndarray = prep_frame_grayscale_to_rgb(frame_bgr)
            augmented = self._transform(image=rgb_hw3)
            # albumentations ToTensorV2 produces [C, H, W] float32; add batch dim
            tensor: torch.Tensor = augmented["image"].unsqueeze(0).to(
                self._device, non_blocking=True
            )
            preprocess_ms = (time.monotonic() - t0) * 1000.0

            # ----------------------------------------------------------------
            # Stage 2 — Forward pass (no_grad + autocast for GPU efficiency)
            # logits is left on-device until after the with-block to allow
            # autocast to handle dtype promotion internally.
            # ----------------------------------------------------------------
            t0 = time.monotonic()
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=self._use_amp):
                    logits: torch.Tensor = self._model(tensor)
            # Convert to float32 before softmax (safe even if logits is float16)
            probs_np: np.ndarray = (
                torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
            )
            forward_ms = (time.monotonic() - t0) * 1000.0

            # ----------------------------------------------------------------
            # Stage 3 — Tier-1 smoothing
            # ----------------------------------------------------------------
            t0 = time.monotonic()
            label: int
            confidence: float
            smoothed_probs: np.ndarray
            is_stable: bool
            label, confidence, smoothed_probs, is_stable = self._smoother.step(probs_np)
            smoothing_ms = (time.monotonic() - t0) * 1000.0

            # ----------------------------------------------------------------
            # Stage 4 — Grad-CAM (persistent instance, throttled)
            #
            # Decision: run GradCAM when EITHER N processed frames have elapsed
            # since the last run OR 200 ms of wall-clock time has elapsed.
            # The OR condition ensures fresh overlays even when inference is
            # slow, without waiting for the N-frame count.
            #
            # IMPORTANT: the GradCAM call is NOT wrapped in torch.no_grad() —
            # GradCAM needs gradient computation internally (it uses
            # torch.enable_grad() itself).  It IS separated from the autocast
            # block above to avoid float16 precision issues in the backward pass.
            #
            # The overlay is produced at the ORIGINAL frame resolution (not
            # 224×224) so the render loop can blend it without upscaling.
            # ----------------------------------------------------------------
            gradcam_ms: float | None = None
            overlay: np.ndarray | None = self._last_overlay  # reuse by default

            if self._cam is not None:
                now_wall = time.monotonic()
                wall_elapsed_ms = (now_wall - self._last_gradcam_wall_t) * 1000.0
                should_run_gradcam = (
                    (self._frames_processed % self._gradcam_every_n == 0)
                    or (wall_elapsed_ms >= self._gradcam_wall_ms)
                )
                if should_run_gradcam:
                    t0 = time.monotonic()
                    targets = [ClassifierOutputTarget(label)]
                    # Grad-CAM forward+backward (hooks registered persistently)
                    grayscale_cam: np.ndarray = self._cam(
                        input_tensor=tensor, targets=targets  # type: ignore[arg-type]
                    )[0]  # shape: [H_in, W_in], typically [224, 224]

                    # Resize heatmap to original frame dimensions
                    h_orig, w_orig = frame_bgr.shape[:2]
                    cam_resized = cv2.resize(grayscale_cam, (w_orig, h_orig))

                    # show_cam_on_image needs RGB float32 [0, 1] at the same
                    # spatial size as cam_resized.  We derive it from frame_bgr
                    # rather than from the preprocessed tensor (which is
                    # normalised/cropped) so the overlay aligns with what the
                    # render loop actually displays.
                    rgb_float = (
                        cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
                        / 255.0
                    )
                    overlay_rgb: np.ndarray = show_cam_on_image(
                        rgb_float, cam_resized, use_rgb=True
                    )  # HxWx3 uint8, RGB
                    # Store as BGR so the render loop can use it directly
                    overlay = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)

                    gradcam_ms = (time.monotonic() - t0) * 1000.0
                    self._last_overlay = overlay
                    self._last_gradcam_wall_t = now_wall
                    self._stats.record_gradcam(gradcam_ms)

            # ----------------------------------------------------------------
            # Stage 5 — Push result dict to render queue
            # ----------------------------------------------------------------
            timings: dict[str, float | None] = {
                "preprocess_ms": preprocess_ms,
                "forward_ms":    forward_ms,
                "smoothing_ms":  smoothing_ms,
                "gradcam_ms":    gradcam_ms,   # None on non-GradCAM frames
            }
            result: dict[str, Any] = {
                "label":          label,
                "label_name":     IDX_TO_CLASS.get(label, f"class_{label}"),
                "confidence":     confidence,
                "smoothed_probs": smoothed_probs,
                "is_stable":      is_stable,
                "overlay":        overlay,     # BGR uint8, or None before first run
                "frame":          frame_bgr,   # original BGR frame from capture
                "timings":        timings,
                "capture_ts":     cap_ts,
                "inference_ts":   time.monotonic(),
            }
            self._result_queue.put(result)
            self._stats.inf_queue_drops = self._result_queue.drops - drops_before

            # Record per-frame inference stats
            self._stats.record_inference_frame(preprocess_ms, forward_ms, smoothing_ms)

        self.cleanup()
        log.info(
            "InferenceThread: exiting after %d frames processed.",
            self._frames_processed,
        )
