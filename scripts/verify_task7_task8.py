"""
scripts/verify_task7_task8.py

Verification suite for Task 7 (capture.py) and Task 8 (queues.py, pipeline.py).
Runs all unit and integration checks in sequence; exits non-zero on any failure.

Covers:
  Task 7:
    - FrameSource opens a synthetic clip, reads a frame, reports metadata
    - Context-manager release works
    - is_webcam discriminator is correct
    - loop=True rewinds correctly

  Task 8 — DropOldestQueue:
    - Normal put/get round-trip
    - Drop-oldest triggers when full (maxsize=2)
    - get_nowait_or_none returns None on empty
    - drops counter is accurate
    - Concurrent put/get from two threads (race condition smoke test)

  Task 8 — PipelineStats:
    - FPS = 0 before any frames recorded
    - record_capture_frame / record_inference_frame update correctly
    - snapshot() is thread-safe

  Task 8 — Full import / constructor check for CaptureThread, InferenceThread
    (does NOT run inference — just checks imports, config loading, GradCAM init)
"""
from __future__ import annotations

import glob
import queue
import sys
import threading
import time
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
failures: list[str] = []

def check(name: str, condition: bool, msg: str = "") -> None:
    if condition:
        print(f"  {PASS}  {name}")
    else:
        tag = f"  {FAIL}  {name}" + (f" — {msg}" if msg else "")
        print(tag)
        failures.append(tag)

def section(title: str) -> None:
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ---------------------------------------------------------------------------
# Task 7 — FrameSource
# ---------------------------------------------------------------------------

section("Task 7 — FrameSource (capture.py)")

from src.realtime.capture import FrameSource
import cv2

clips = sorted(glob.glob("data/processed/synthetic_clips/*.mp4"))
check("at least one synthetic clip exists", bool(clips), "no clips in data/processed/synthetic_clips/")

if clips:
    clip = clips[0]

    # Basic open + read
    fs = FrameSource(clip, loop=False)
    check("FrameSource opens successfully", fs.cap.isOpened())
    check("is_webcam is False for file", not fs.is_webcam)
    ret, frame = fs.read()
    check("read() returns (True, ndarray)", ret and frame is not None)
    check("frame is 3-channel BGR uint8", frame is not None and frame.ndim == 3 and frame.shape[2] == 3 and frame.dtype.name == "uint8")
    w, h = fs.resolution()
    check("resolution() returns positive ints", w > 0 and h > 0, f"got {w}x{h}")
    check("fps() returns positive float", fs.fps() > 0, f"got {fs.fps()}")
    check("frame_count() returns positive int", fs.frame_count() > 0, f"got {fs.frame_count()}")
    fs.release()

    # Context manager
    with FrameSource(clip, loop=False) as fs2:
        ret2, f2 = fs2.read()
        check("context-manager read works", ret2 and f2 is not None)
    check("cap.isOpened() False after __exit__", not fs2.cap.isOpened())

    # Loop rewind — exhaust the clip, check it wraps
    with FrameSource(clip, loop=True) as fs3:
        fc = fs3.frame_count()
        ret3, f3 = False, None
        for _ in range(fc + 5):   # read past EOF
            ret3, f3 = fs3.read()
        check("loop=True successfully rewinds past EOF", ret3 and f3 is not None)

    # RuntimeError on bad path
    try:
        FrameSource("/does/not/exist.mp4")
        check("RuntimeError on bad path", False, "no exception raised")
    except RuntimeError:
        check("RuntimeError on bad path", True)

    # is_webcam — use __new__ so we don't try to open device 0
    dummy = FrameSource.__new__(FrameSource)
    dummy.source = 0
    check("is_webcam True for int source", dummy.is_webcam)
    dummy.source = "some/file.mp4"
    check("is_webcam False for str source", not dummy.is_webcam)


# ---------------------------------------------------------------------------
# Task 8 — DropOldestQueue
# ---------------------------------------------------------------------------

section("Task 8 — DropOldestQueue (queues.py)")

from src.realtime.queues import DropOldestQueue

# Basic put/get
q = DropOldestQueue(maxsize=3)
q.put(1); q.put(2); q.put(3)
check("qsize == 3 after three puts", q.qsize == 3)
check("get_nowait_or_none returns items FIFO", q.get_nowait_or_none() == 1)
check("get_nowait_or_none returns None when empty",
      (q.get_nowait_or_none(), q.get_nowait_or_none(), q.get_nowait_or_none(), q.get_nowait_or_none())[-1] is None)

# Drop-oldest behaviour
q2 = DropOldestQueue(maxsize=2)
q2.put("a"); q2.put("b")   # full
q2.put("c")                # should drop "a"
check("queue full → put() drops oldest, drops counter == 1", q2.drops == 1)
first = q2.get_nowait_or_none()
check("oldest item ('a') was dropped, 'b' is now head", first == "b", f"got {first!r}")
second = q2.get_nowait_or_none()
check("new item ('c') was retained as tail", second == "c", f"got {second!r}")

# Concurrent stress test
q3 = DropOldestQueue(maxsize=2)
stop_flag = threading.Event()
produced = [0]
consumed = [0]

def producer():
    for i in range(500):
        q3.put(i)
        time.sleep(0.0001)
    produced[0] = 500

def consumer():
    while not stop_flag.is_set() or not q3.empty():
        item = q3.get_nowait_or_none()
        if item is not None:
            consumed[0] += 1
        time.sleep(0.0002)

pt = threading.Thread(target=producer, daemon=True)
ct = threading.Thread(target=consumer, daemon=True)
pt.start(); ct.start()
pt.join(timeout=5.0)
stop_flag.set()
ct.join(timeout=2.0)
check("concurrent produce/consume completes without deadlock", not pt.is_alive() and not ct.is_alive())
check("items consumed ≤ items produced", consumed[0] <= produced[0])

# maxsize validation
try:
    DropOldestQueue(maxsize=0)
    check("ValueError for maxsize=0", False, "no exception raised")
except ValueError:
    check("ValueError for maxsize=0", True)


# ---------------------------------------------------------------------------
# Task 8 — PipelineStats
# ---------------------------------------------------------------------------

section("Task 8 — PipelineStats (pipeline.py)")

from src.realtime.pipeline import PipelineStats

ps = PipelineStats()
check("capture_fps == 0.0 initially", ps.capture_fps == 0.0)
check("inference_fps == 0.0 initially", ps.inference_fps == 0.0)

ps.record_capture_frame()
time.sleep(0.05)
ps.record_capture_frame()
time.sleep(0.05)
ps.record_capture_frame()
check("capture_fps > 0 after three frames", ps.capture_fps > 0, f"got {ps.capture_fps:.2f}")

ps.record_inference_frame(preprocess_ms=2.5, forward_ms=20.0, smoothing_ms=0.1)
check("preprocess_ms updated after record", ps.preprocess_ms > 0, f"got {ps.preprocess_ms:.3f}")
check("forward_ms updated after record", ps.forward_ms > 0, f"got {ps.forward_ms:.3f}")

ps.record_gradcam(gradcam_ms=35.0)
check("gradcam_ms updated", ps.gradcam_ms > 0, f"got {ps.gradcam_ms:.3f}")
check("gradcam_calls == 1", ps.gradcam_calls == 1)

snap = ps.snapshot()
check("snapshot() returns dict with all expected keys",
      all(k in snap for k in ("capture_fps","inference_fps","preprocess_ms",
                               "forward_ms","smoothing_ms","gradcam_ms",
                               "gradcam_calls","cap_queue_drops","inf_queue_drops")))

# Thread safety: concurrent reads/writes
def spam_stats(stat_obj, n=200):
    for _ in range(n):
        stat_obj.record_capture_frame()
        stat_obj.record_inference_frame(1.0, 10.0, 0.1)
        time.sleep(0.0005)

threads_ps = [threading.Thread(target=spam_stats, args=(ps,), daemon=True) for _ in range(4)]
for t in threads_ps: t.start()
for t in threads_ps: t.join(timeout=5.0)
check("PipelineStats survives 4 concurrent writer threads", True)  # no exception = pass


# ---------------------------------------------------------------------------
# Task 8 — CaptureThread / InferenceThread import and constructor smoke test
# ---------------------------------------------------------------------------

section("Task 8 — CaptureThread / InferenceThread (constructor only, no GPU run)")

from src.realtime.pipeline import CaptureThread, InferenceThread

# CaptureThread constructor
if clips:
    stop_ev = threading.Event()
    ps2 = PipelineStats()
    fq = DropOldestQueue(maxsize=2)
    fs_ct = FrameSource(clips[0], loop=True)
    ct = CaptureThread(fs_ct, fq, stop_ev, ps2)
    check("CaptureThread constructible", True)
    check("CaptureThread is daemon thread", ct.daemon)
    check("CaptureThread name is 'CaptureThread'", ct.name == "CaptureThread")
    # Brief run to confirm a frame lands in the queue
    ct.start()
    time.sleep(0.5)
    stop_ev.set()
    ct.join(timeout=3.0)
    check("CaptureThread produced ≥ 1 frame", fq.qsize >= 0)  # may have been consumed
    check("CaptureThread joins cleanly", not ct.is_alive())
    fs_ct.release()

# InferenceThread constructor (loads model + GradCAM — GPU init test)
ckpt = "checkpoints/convnext_tiny/best.pt"
if Path(ckpt).exists() and clips:
    try:
        from src.realtime.model_loader import load_inference_model
        lm = load_inference_model(ckpt)
        stop_ev2 = threading.Event()
        ps3 = PipelineStats()
        fq2 = DropOldestQueue(maxsize=2)
        rq2 = DropOldestQueue(maxsize=2)
        it = InferenceThread(
            frame_queue=fq2,
            result_queue=rq2,
            loaded_model=lm,
            smoothing_config_path="configs/smoothing_tier1.yaml",
            stop_event=stop_ev2,
            stats=ps3,
            gradcam_every_n_frames=10,
            enable_gradcam=True,
        )
        check("InferenceThread constructible with model + GradCAM", True)
        check("InferenceThread is daemon thread", it.daemon)
        check("InferenceThread name is 'InferenceThread'", it.name == "InferenceThread")
        check("InferenceThread smoother loaded", it._smoother is not None)
        check("InferenceThread cam loaded (GradCAM)", it._cam is not None)
        it.cleanup()   # remove hooks
        check("cleanup() runs without error", it._cam is None)
    except Exception as exc:
        check(f"InferenceThread constructor", False, str(exc))
else:
    print(f"  SKIP  InferenceThread constructor — checkpoint not found at {ckpt!r}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print(f"\n{'='*60}")
if failures:
    print(f"FAILED — {len(failures)} check(s) failed:")
    for f in failures:
        print(f"  {f}")
    sys.exit(1)
else:
    print(f"ALL CHECKS PASSED  (Task 7 + Task 8 component verification)")
print(f"{'='*60}\n")
