# 05 — Temporal Smoothing & Real-Time Serving Pipeline

This is the phase that actually differentiates this project from the original static-image repo. `[AGENT]`-heavy, with a couple of `[MANUAL]` tuning/inspection checkpoints.

---

## Part A — Temporal Smoothing (Tier-1 first, Tier-2 only if needed)

### A1. Tier-1 design: EMA + hysteresis + minimum dwell time

Implement `src/smoothing/tier1.py` as a small stateful class the real-time loop calls once per frame:

```
State: 
  - smoothed_probs: exponential moving average of the per-frame softmax vector
  - current_displayed_label
  - frames_since_last_switch

On each new frame's raw softmax output `p`:
  1. smoothed_probs = alpha * p + (1 - alpha) * smoothed_probs   # alpha ~ 0.2-0.4, tune empirically
  2. candidate_label = argmax(smoothed_probs)
  3. candidate_confidence = smoothed_probs[candidate_label]
  4. IF candidate_label == current_displayed_label:
        hold (reset frames_since_last_switch = 0)
     ELSE IF candidate_confidence > switch_threshold  (higher than a simple "hold" threshold — this is the hysteresis)
          AND frames_since_last_switch >= min_dwell_frames:
        current_displayed_label = candidate_label
        frames_since_last_switch = 0
     ELSE:
        frames_since_last_switch += 1
        (keep displaying current_displayed_label)
  5. Return (current_displayed_label, candidate_confidence, smoothed_probs)  # for UI + logging
```

Parameters to tune empirically (not guessed): `alpha`, `switch_threshold` (should be noticeably higher than the "hold" confidence, e.g. hold at >0.4 but require >0.6 to switch — exact numbers come from tuning below), `min_dwell_frames` (translate to a target ~150-300ms of real time given your measured fps).

### A2. Tuning procedure against real video (IUGC sandbox)

1. Run the trained plane classifier frame-by-frame over a handful of IUGC video clips (yes, wrong domain for classification — that's fine, we're not scoring classification accuracy here, we're measuring **how the raw per-frame softmax output fluctuates over genuine probe motion**, which is domain-agnostic).
2. Log the raw per-frame confidence/label-switch frequency with NO smoothing applied — this is your baseline "how bad is the flicker" measurement.
3. Sweep `alpha`, `switch_threshold`, `min_dwell_frames` and measure: (a) label-switches-per-minute with smoothing on, (b) average latency-to-first-stable-label after a genuine plane change (approximate this using the video's own annotated transitions if available, or manually mark a few transition points yourself — `[MANUAL]` task: watch ~5 clips and note approximate frame indices where the sonographer clearly settles on vs. leaves a plane, to use as ground truth for "did smoothing correctly track this").
4. Pick the parameter set that minimizes spurious switches while keeping the lag-to-stabilize under ~300-400ms.
5. **Port only the tuned parameter values (alpha, thresholds, dwell frames), not any data or model weights**, into the primary system. Document the tuning result (chosen values + reasoning) in a short note.

If you skipped the IUGC download in Phase 2, fall back to tuning against the synthetic ego-motion clips only (Phase 3) — accept that this validates jitter-robustness but not genuine plane-transition dynamics, and say so explicitly in the final write-up.

### A3. Tier-2 (only if Tier-1 empirically insufficient)

Do not build this speculatively. Only implement if, after tuning A2, you still observe unacceptable flicker or lag on real test video. If needed: a small causal GRU or 1D-conv over the last 8-16 frame *embeddings* (not raw pixels — take the backbone's penultimate-layer feature vector per frame, buffer the last N, feed through a lightweight temporal head). This is a meaningfully bigger lift (needs its own training data/labels over sequences) — treat as a stretch goal, not default path.

---

## Part B — Real-Time Serving Pipeline

### B1. Video input abstraction

`src/realtime/capture.py`: a single interface that accepts either:
- A webcam device index (`cv2.VideoCapture(0)`), or
- A video file path (`cv2.VideoCapture(path)`)

so the rest of the pipeline is agnostic to source. Build and test against **file playback first** (per your stated priority), webcam as the final validation target.

### B2. Pipeline architecture

Recommend a simple multi-threaded (not necessarily multi-process — GIL contention is manageable at this scale) pipeline to avoid capture/inference/render blocking each other:

```
[Capture thread] --frame queue (size-limited, drop-oldest-on-full)--> [Inference thread] --result queue--> [Main/render thread]
```

- **Capture thread**: reads frames as fast as the source provides them, pushes to a small bounded queue (size ~2-3) with drop-oldest policy — we never want inference falling behind and processing a backlog of stale frames; always process the most recent frame available.
- **Inference thread**: preprocess → forward pass → softmax → tier-1 smoothing → push (label, confidence, smoothed_probs, optionally Grad-CAM overlay) to a result queue.
- **Grad-CAM throttling**: only run the Grad-CAM backward pass every K frames (e.g., every 5th-10th processed frame, or on a fixed wall-clock cadence like every 200ms) — never on every frame, since it roughly doubles compute per the original repo's own pipeline. Reuse the last computed CAM overlay on skipped frames (it changes slowly relative to raw classification anyway, since the anatomy is on-screen continuously).
- **Main/render thread**: pulls latest available result, overlays label + confidence + optional CAM heatmap + a simple stability indicator (e.g., a small icon/text showing "stable" vs. "settling") onto the frame, displays via `cv2.imshow` (fine for a local demo) or a lightweight web UI (see B4).

### B3. Latency/throughput instrumentation

Build in **from the start**, not as an afterthought:
- Rolling FPS counter (capture fps vs. actual inference fps — these will diverge, that's expected and fine given the drop-oldest queue policy)
- Per-stage latency logging (preprocess, forward pass, Grad-CAM when run, smoothing, render) — log to console or a lightweight overlay toggle, so you can actually see where time goes rather than guessing.

### B4. `[MANUAL] Decision point:` UI approach

Two reasonable options:
1. **`cv2.imshow` window** — fastest to build, adequate for a personal/portfolio demo, no web server needed.
2. **Simple web app (Streamlit or Gradio) streaming the annotated frames** — nicer for showing off / sharing, adds moving parts (frame streaming over HTTP has its own latency considerations).

Recommendation: build **(1) first** to validate the whole pipeline works and feels responsive, then consider **(2)** only as a polish pass once the core loop is solid — don't let UI framework choice block getting the actual real-time inference loop working.

### B5. Webcam-as-probe caveat to document

Note explicitly in the demo/README that a webcam feed is a stand-in for a probe feed purely to exercise the real-time pipeline (frame capture → inference → smoothing → overlay) — it is not simulating actual ultrasound image content, since a webcam obviously can't produce ultrasound-like images. The actual classification demo should run against **pre-recorded ultrasound video files** (or the synthetic ego-motion clips from Phase 3) for meaningful predictions; the webcam path exists to prove the pipeline's real-time mechanics generalize to a live device source, not to produce clinically meaningful output from webcam footage.

---

## Deliverables checklist

- [ ] Tier-1 smoothing implemented and parameter-tuned against real (or synthetic-fallback) video
- [ ] Tuning results documented with chosen parameter values and reasoning
- [ ] Capture abstraction supporting both file and webcam sources
- [ ] Threaded capture/inference/render pipeline implemented with bounded, drop-oldest queues
- [ ] Grad-CAM throttling implemented
- [ ] Latency/FPS instrumentation implemented and visible
- [ ] UI approach decided and built (cv2 window first)
- [ ] Webcam-as-stand-in caveat documented in README/demo notes
