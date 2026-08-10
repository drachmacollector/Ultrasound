"""Import smoke test for app.py and pipeline components."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

errors = []

# --- queues.py ---
try:
    from src.realtime.queues import DropOldestQueue
    q = DropOldestQueue(maxsize=2)
    q.put("a"); q.put("b"); q.put("c")
    assert q.drops == 1, f"Expected 1 drop, got {q.drops}"
    print("PASS  queues.py import + basic operation")
except Exception as e:
    errors.append(f"FAIL  queues.py: {e}")

# --- capture.py ---
try:
    from src.realtime.capture import FrameSource
    print("PASS  capture.py import")
except Exception as e:
    errors.append(f"FAIL  capture.py: {e}")

# --- pipeline.py ---
try:
    from src.realtime.pipeline import CaptureThread, InferenceThread, PipelineStats
    ps = PipelineStats()
    snap = ps.snapshot()
    assert "capture_fps" in snap
    print("PASS  pipeline.py import + PipelineStats()")
except Exception as e:
    errors.append(f"FAIL  pipeline.py: {e}")

# --- app.py ---
try:
    import src.realtime.app as app_mod
    # Verify the gradcam default was correctly computed
    assert app_mod._GRADCAM_EVERY_N_DEFAULT == 7, \
        f"Expected 7, got {app_mod._GRADCAM_EVERY_N_DEFAULT}"
    # Verify build_parser() works
    p = app_mod.build_parser()
    args = p.parse_args(["--source", "data/processed/synthetic_clips/Brain_Trans_cerebellum_clip01.mp4", "--no-gradcam"])
    assert args.no_gradcam
    assert args.gradcam_every_n_frames == 7
    assert not args.loop
    print(f"PASS  app.py import  (gradcam_every_n default={app_mod._GRADCAM_EVERY_N_DEFAULT})")
except Exception as e:
    errors.append(f"FAIL  app.py: {e}")

# --- Summary ---
print()
if errors:
    for e in errors:
        print(e)
    sys.exit(1)
else:
    print("ALL PASS — Task 9 component imports verified.")
