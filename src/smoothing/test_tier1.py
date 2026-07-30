"""
src/smoothing/test_tier1.py

Unit tests for the Tier1Smoother state machine.

Three mandatory cases per PHASE_5_KICKOFF_PROMPT.md §4 (Task 3):
  1. Stable input  — same dominant class every frame, smoother stays is_stable=True.
  2. Rapid flicker — argmax alternates between two classes each frame with low
                     confidence; smoother must suppress the flicker.
  3. Genuine transition — class A confident for N frames, then clean switch to class B
                          confident for remaining frames; smoother must eventually track
                          B and reach is_stable=True, and we log/assert the frame-lag.

Additional cases:
  4. Cold-start behaviour — first frame initialises correctly.
  5. Reset clears state   — reset() returns to pristine condition.
  6. hold_floor gate      — when current confidence drops below hold_floor, switching
                            is allowed even before dwell is met.
"""
from __future__ import annotations

import sys
import numpy as np

# Allow running directly with `python -m src.smoothing.test_tier1` from project root
sys.path.insert(0, ".")
from src.smoothing.tier1 import Tier1Smoother


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NUM_CLASSES = 8


def one_hot_probs(dominant_class: int, dominant_p: float, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Return a probability vector with `dominant_p` on `dominant_class`, the rest equal."""
    probs = np.full(num_classes, (1.0 - dominant_p) / (num_classes - 1), dtype=np.float32)
    probs[dominant_class] = dominant_p
    return probs


def make_smoother(**kwargs) -> Tier1Smoother:
    """Return a fresh Tier1Smoother with test-friendly defaults (fast convergence)."""
    defaults = dict(
        num_classes=NUM_CLASSES,
        alpha=0.4,           # responsive
        switch_threshold=0.55,
        min_dwell_frames=3,
    )
    defaults.update(kwargs)
    return Tier1Smoother(**defaults)


# ---------------------------------------------------------------------------
# Case 1: Stable input
# ---------------------------------------------------------------------------

def test_stable_input():
    """Same dominant class every frame → smoother tracks immediately, is_stable=True."""
    smoother = make_smoother()
    CLASS_A = 2
    n_frames = 30

    final_label = None
    final_stable = None

    for i in range(n_frames):
        label, conf, _, stable = smoother.step(one_hot_probs(CLASS_A, 0.9))
        final_label = label
        final_stable = stable

    assert final_label == CLASS_A, (
        f"Expected displayed label {CLASS_A}, got {final_label}"
    )
    assert final_stable is True, (
        "Expected is_stable=True after many frames of the same label"
    )
    # Also verify frames_since_last_switch has been counting up
    assert smoother.frames_since_last_switch >= smoother.min_dwell_frames
    print(f"[PASS] test_stable_input: after {n_frames} frames, "
          f"label={final_label}, frames_held={smoother.frames_since_last_switch}")


# ---------------------------------------------------------------------------
# Case 2: Rapid flicker suppression
# ---------------------------------------------------------------------------

def test_rapid_flicker_suppressed():
    """
    Argmax alternates between class 0 and class 1 every frame, neither ever
    crossing switch_threshold confidently.  Smoother must suppress all switches
    and keep displaying whatever it first settled on.
    """
    smoother = make_smoother(alpha=0.3, switch_threshold=0.6, min_dwell_frames=3)
    CLASS_A = 0
    CLASS_B = 1
    # Low-confidence alternation: 0.45 is above the ~0.125 random baseline but
    # deliberately below switch_threshold=0.6.
    probs_a = one_hot_probs(CLASS_A, 0.45)
    probs_b = one_hot_probs(CLASS_B, 0.45)

    n_frames = 60
    switches = 0
    prev_label = None

    for i in range(n_frames):
        p = probs_a if i % 2 == 0 else probs_b
        label, conf, _, stable = smoother.step(p)
        if prev_label is not None and label != prev_label:
            switches += 1
        prev_label = label

    # After cold-start there may be 0 or 1 initial "switch" when the smoother first
    # picks up a dominant class.  But the alternation itself must produce 0 further
    # switches once settled.
    assert switches == 0, (
        f"Expected 0 label switches during low-confidence flicker, got {switches}"
    )
    print(f"[PASS] test_rapid_flicker_suppressed: {n_frames} frames, switches={switches}, "
          f"final_label={prev_label}")


# ---------------------------------------------------------------------------
# Case 3: Genuine class change
# ---------------------------------------------------------------------------

def test_genuine_class_change():
    """
    First N_STABLE frames: class A confidently.
    Then N_CHANGE frames:  class B confidently.
    Smoother must:
      a) Stay on A during A phase (or settle onto A within min_dwell_frames).
      b) Switch to B and reach is_stable=True within a bounded frame count.
    Log and assert the frame-lag from the first B frame to stable B.
    """
    smoother = make_smoother(alpha=0.35, switch_threshold=0.55, min_dwell_frames=3)
    CLASS_A = 3
    CLASS_B = 5
    N_STABLE = 20   # frames of A
    N_CHANGE = 40   # frames of B

    probs_a = one_hot_probs(CLASS_A, 0.90)
    probs_b = one_hot_probs(CLASS_B, 0.90)

    # Phase 1: stable A
    label = -1
    for _ in range(N_STABLE):
        label, _, _, _ = smoother.step(probs_a)
    assert label == CLASS_A, f"Expected label {CLASS_A} during stable A phase, got {label}"

    # Phase 2: switch to B — record how many frames until we're stable on B
    first_b_frame = None
    stable_b_frame = None

    for i in range(N_CHANGE):
        label, conf, smoothed, stable = smoother.step(probs_b)
        if first_b_frame is None and label == CLASS_B:
            first_b_frame = i
        if first_b_frame is not None and label == CLASS_B and stable:
            stable_b_frame = i
            break

    lag_to_switch = first_b_frame if first_b_frame is not None else N_CHANGE
    lag_to_stable = stable_b_frame if stable_b_frame is not None else N_CHANGE

    print(f"[INFO] test_genuine_class_change: "
          f"first B label at frame +{lag_to_switch}, "
          f"stable B at frame +{lag_to_stable} "
          f"(min_dwell_frames={smoother.min_dwell_frames})")

    assert label == CLASS_B, (
        f"Expected final label {CLASS_B} after {N_CHANGE} B-frames, got {label}"
    )
    assert stable_b_frame is not None, (
        "Smoother never reached is_stable=True on class B within the allotted frames"
    )
    # Frame lag should be bounded: at most ~2*min_dwell_frames + a few EMA ramp-up frames
    # (generous upper bound: 3 × min_dwell_frames is more than enough for alpha>=0.3)
    max_allowed_lag = 3 * smoother.min_dwell_frames + 5
    assert lag_to_stable <= max_allowed_lag, (
        f"Frame-lag to stable B ({lag_to_stable}) exceeded bound ({max_allowed_lag})"
    )


# ---------------------------------------------------------------------------
# Case 4: Cold-start
# ---------------------------------------------------------------------------

def test_cold_start():
    """First frame immediately assigns a displayed label (no -1 on first call)."""
    smoother = make_smoother()
    label, conf, smoothed, stable = smoother.step(one_hot_probs(4, 0.9))
    assert label == 4, f"Cold-start should immediately display dominant class 4, got {label}"
    assert conf > 0.0, "Confidence should be positive after cold start"
    assert smoothed.shape == (NUM_CLASSES,), "smoothed_probs shape mismatch"
    print(f"[PASS] test_cold_start: label={label}, conf={conf:.4f}")


# ---------------------------------------------------------------------------
# Case 5: Reset
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    """reset() returns the smoother to the cold-start condition."""
    smoother = make_smoother()
    for _ in range(10):
        smoother.step(one_hot_probs(1, 0.9))
    smoother.reset()
    assert smoother.smoothed_probs is None
    assert smoother.current_displayed_label == -1
    assert smoother.frames_since_last_switch == 0
    # Should behave like a fresh smoother after reset
    label, _, _, _ = smoother.step(one_hot_probs(6, 0.9))
    assert label == 6
    print("[PASS] test_reset_clears_state")


# ---------------------------------------------------------------------------
# Case 6: hold_floor gate
# ---------------------------------------------------------------------------

def test_hold_floor_enables_early_switch():
    """
    When current-label confidence drops below hold_floor, switching is allowed
    even if min_dwell_frames has not been met.
    """
    # Use a very high min_dwell_frames so normal switching is blocked.
    smoother = make_smoother(
        alpha=0.5,
        switch_threshold=0.5,
        min_dwell_frames=100,   # effectively never met
        hold_floor=0.35,        # low floor: if confidence of held label drops here, unlock
    )
    CLASS_A = 0
    CLASS_B = 7

    # Settle on A
    for _ in range(5):
        smoother.step(one_hot_probs(CLASS_A, 0.9))

    # Now send B frames; alpha=0.5 means B confidence rises quickly.
    # With high min_dwell_frames and no hold_floor, we'd never switch.
    # With hold_floor=0.35, once A's smoothed confidence drops below 0.35 we should switch.
    switched = False
    for i in range(30):
        label, _, smoothed, _ = smoother.step(one_hot_probs(CLASS_B, 0.9))
        if label == CLASS_B:
            switched = True
            print(f"[INFO] test_hold_floor: switched to B at frame {i}, "
                  f"A_conf={smoothed[CLASS_A]:.4f}")
            break

    assert switched, (
        "hold_floor gate should have allowed an early switch to B, but switch never happened"
    )
    print("[PASS] test_hold_floor_enables_early_switch")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cold_start()
    test_stable_input()
    test_rapid_flicker_suppressed()
    test_genuine_class_change()
    test_reset_clears_state()
    test_hold_floor_enables_early_switch()
    print("\nAll Tier1Smoother tests passed.")
