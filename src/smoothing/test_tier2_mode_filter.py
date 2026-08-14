"""
src/smoothing/test_tier2_mode_filter.py

Unit tests for Tier2ModeFilter, mirroring test_tier1.py's structure and thoroughness.

Six cases:
  1. Stable input           — all frames same label; output = input every frame (no-op).
  2. Oscillation suppressed — A/B alternation; filter holds majority (neither class wins).
  3. Genuine transition     — N frames A, then N frames B; filter adopts B after window fills.
  4. No-op on already-stable Tier-1 output — verifies the 45/46 clips composability claim.
  5. Cold-start             — first frame returns a valid label immediately (no -1).
  6. Reset                  — reset() fully clears state to cold-start condition.

Run:
    conda run -n fetalplane python -m src.smoothing.test_tier2_mode_filter
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")
from src.smoothing.tier2_mode_filter import Tier2ModeFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_filter(**kwargs) -> Tier2ModeFilter:
    """Return a fresh Tier2ModeFilter with test-friendly defaults."""
    defaults = dict(window_frames=9, min_majority_frac=0.6)
    defaults.update(kwargs)
    return Tier2ModeFilter(**defaults)


def feed(f: Tier2ModeFilter, labels: list[int]) -> list[int]:
    """Feed a list of labels through the filter and return all output labels."""
    return [f.step(lbl) for lbl in labels]


# ---------------------------------------------------------------------------
# Case 1: Stable input — no-op guarantee
# ---------------------------------------------------------------------------

def test_stable_input_noop():
    """All frames same label → output mirrors input exactly after cold-start."""
    f = make_filter(window_frames=9, min_majority_frac=0.6)
    CLASS_A = 2
    n_frames = 50

    outputs = feed(f, [CLASS_A] * n_frames)

    for i, out in enumerate(outputs):
        assert out == CLASS_A, (
            f"Frame {i}: expected {CLASS_A}, got {out} (stable-input no-op violated)"
        )
    print(
        f"[PASS] test_stable_input_noop: {n_frames} frames, all output={CLASS_A}"
    )


# ---------------------------------------------------------------------------
# Case 2: Oscillation suppression
# ---------------------------------------------------------------------------

def test_oscillation_suppressed():
    """A/B alternation — neither class should win majority; filter freezes on first label."""
    f = make_filter(window_frames=9, min_majority_frac=0.6)
    CLASS_A, CLASS_B = 0, 1
    # Strict 1:1 alternation, so top fraction is always exactly 0.5 (< 0.6 threshold).
    labels = [CLASS_A if i % 2 == 0 else CLASS_B for i in range(60)]

    outputs = feed(f, labels)

    # After cold-start the filter locks onto CLASS_A (first frame).
    # Strict 50/50 split never reaches min_majority_frac=0.6, so no switch ever fires.
    switches = sum(1 for a, b in zip(outputs, outputs[1:]) if a != b)
    assert switches == 0, (
        f"Expected 0 Tier-2 switches on strict A/B oscillation, got {switches}"
    )
    # The locked label must be CLASS_A (the cold-start value)
    assert outputs[-1] == CLASS_A, (
        f"Expected filter to hold CLASS_A, got {outputs[-1]}"
    )
    print(
        f"[PASS] test_oscillation_suppressed: 60 frames A/B alternation, "
        f"switches={switches}, held={outputs[-1]}"
    )


# ---------------------------------------------------------------------------
# Case 3: Genuine transition
# ---------------------------------------------------------------------------

def test_genuine_transition():
    """N frames A (stable), then N frames B (stable) — filter must adopt B."""
    WINDOW = 9
    MAJORITY = 0.6
    f = make_filter(window_frames=WINDOW, min_majority_frac=MAJORITY)
    CLASS_A, CLASS_B = 3, 5
    N_STABLE = 30
    N_CHANGE = 30

    labels = [CLASS_A] * N_STABLE + [CLASS_B] * N_CHANGE
    outputs = feed(f, labels)

    # Verify stable phase: output should be CLASS_A throughout
    for i, out in enumerate(outputs[:N_STABLE]):
        assert out == CLASS_A, f"Frame {i}: expected {CLASS_A} during stable A phase, got {out}"

    # Verify that B is eventually adopted in the change phase
    final_out = outputs[-1]
    assert final_out == CLASS_B, (
        f"Expected final output {CLASS_B} after {N_CHANGE} B-frames, got {final_out}"
    )

    # Compute latency: frames from first B input to first B output
    first_b_output_frame = next(
        (i for i, out in enumerate(outputs[N_STABLE:], start=N_STABLE) if out == CLASS_B),
        None,
    )
    assert first_b_output_frame is not None, "Filter never adopted B"
    lag = first_b_output_frame - N_STABLE
    # Maximum lag: need ceil(window * majority) frames of B in the window to tip the vote.
    import math
    max_lag = math.ceil(WINDOW * MAJORITY) + 1  # generous
    assert lag <= max_lag, (
        f"Transition lag {lag} frames exceeds expected max {max_lag} frames"
    )
    print(
        f"[PASS] test_genuine_transition: A→B transition detected at lag={lag} frames "
        f"(max allowed={max_lag}, window={WINDOW})"
    )


# ---------------------------------------------------------------------------
# Case 4: No-op on already-stable Tier-1 output (composability claim)
# ---------------------------------------------------------------------------

def test_noop_on_stable_tier1_output():
    """Simulate a Tier-1 output stream that is already perfectly stable (45/46 clips case).

    The filter must produce identical output to its input — no spurious switches,
    no degradation of an already-clean label stream.
    """
    f = make_filter(window_frames=13, min_majority_frac=0.6)

    # Simulate a Tier-1 stream: stable class 4 for 100 frames, single clean
    # transition to class 7 (already gated by Tier-1's dwell), then stable class 7.
    tier1_stream = [4] * 100 + [7] * 100

    outputs = feed(f, tier1_stream)

    # Phase 1: all outputs must be 4
    for i, out in enumerate(outputs[:100]):
        assert out == 4, f"Frame {i}: expected 4, got {out}"

    # Phase 2: after the transition settles in the window, all outputs must be 7
    # Allow a lag window for the filter to catch up with the Tier-1 transition
    transition_lag = f.window_frames  # at most window_frames frames to adopt new majority
    settled_start = 100 + transition_lag
    for i, out in enumerate(outputs[settled_start:], start=settled_start):
        assert out == 7, f"Frame {i}: expected 7 after settling, got {out}"

    # Critical: zero spurious switches on the stable-input segments (before and after)
    phase1_switches = sum(
        1 for a, b in zip(outputs[:100], outputs[1:100]) if a != b
    )
    assert phase1_switches == 0, (
        f"Spurious switches in stable-input phase 1: {phase1_switches}"
    )
    print(
        f"[PASS] test_noop_on_stable_tier1_output: stable phases clean, "
        f"transition adopted within {transition_lag} frames"
    )


# ---------------------------------------------------------------------------
# Case 5: Cold-start
# ---------------------------------------------------------------------------

def test_cold_start():
    """First frame returns a valid label (not -1) immediately."""
    f = make_filter()
    out = f.step(2)
    assert out == 2, f"Cold-start should immediately return the first label, got {out}"
    assert f.window_fill == 1, f"Buffer should have 1 frame after cold-start, got {f.window_fill}"
    print(f"[PASS] test_cold_start: first frame output={out}, buffer_fill={f.window_fill}")


# ---------------------------------------------------------------------------
# Case 6: Reset
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    """reset() returns filter to cold-start condition."""
    f = make_filter()
    for _ in range(20):
        f.step(3)
    f.reset()

    assert f.window_fill == 0, f"Buffer should be empty after reset, got {f.window_fill}"
    assert f._displayed_label == Tier2ModeFilter._UNINIT, (
        f"Displayed label should be UNINIT after reset, got {f._displayed_label}"
    )
    # Should behave like a fresh filter after reset
    out = f.step(6)
    assert out == 6, f"Post-reset first frame should return 6, got {out}"
    print("[PASS] test_reset_clears_state")


# ---------------------------------------------------------------------------
# Case 7: Asymmetric oscillation (one class slightly more frequent)
# ---------------------------------------------------------------------------

def test_asymmetric_oscillation():
    """A appears 2/3 of the time, B 1/3 — with window=9, A should win majority."""
    f = make_filter(window_frames=9, min_majority_frac=0.6)
    CLASS_A, CLASS_B = 1, 4
    # Pattern: A A B A A B … (2:1 ratio → A gets ~66.7% of window > 0.6 threshold)
    pattern = [CLASS_A, CLASS_A, CLASS_B]
    labels = (pattern * 30)[:90]  # 90 frames

    outputs = feed(f, labels)

    # After warm-up, A should be the stable output (it wins the vote)
    # Check the last half of the outputs (fully warmed window)
    settled_outputs = outputs[45:]
    b_count = settled_outputs.count(CLASS_B)
    assert b_count == 0, (
        f"Expected CLASS_B to be suppressed (0 occurrences), got {b_count} in settled phase"
    )
    print(
        f"[PASS] test_asymmetric_oscillation: A(2/3)+B(1/3) stream → "
        f"B suppressed in settled phase ({settled_outputs.count(CLASS_A)}/{len(settled_outputs)} frames = A)"
    )


# ---------------------------------------------------------------------------
# Case 8: Parameter validation
# ---------------------------------------------------------------------------

def test_parameter_validation():
    """Invalid parameters raise ValueError."""
    import traceback

    errors = []

    # min_majority_frac < 0.5 should raise
    try:
        Tier2ModeFilter(window_frames=9, min_majority_frac=0.49)
        errors.append("min_majority_frac=0.49 should have raised ValueError")
    except ValueError:
        pass

    # min_majority_frac exactly 0.5 should be VALID (relaxed to [0.5, 1.0])
    try:
        Tier2ModeFilter(window_frames=9, min_majority_frac=0.5)
    except ValueError:
        errors.append("min_majority_frac=0.5 should be valid (not raise)")

    # min_majority_frac > 1.0 should raise
    try:
        Tier2ModeFilter(window_frames=9, min_majority_frac=1.1)
        errors.append("min_majority_frac=1.1 should have raised ValueError")
    except ValueError:
        pass

    # window_frames < 1 should raise
    try:
        Tier2ModeFilter(window_frames=0, min_majority_frac=0.6)
        errors.append("window_frames=0 should have raised ValueError")
    except ValueError:
        pass

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        raise AssertionError(f"{len(errors)} validation tests failed")

    print("[PASS] test_parameter_validation: all invalid params correctly rejected")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_cold_start()
    test_stable_input_noop()
    test_oscillation_suppressed()
    test_genuine_transition()
    test_noop_on_stable_tier1_output()
    test_reset_clears_state()
    test_asymmetric_oscillation()
    test_parameter_validation()
    print("\nAll Tier2ModeFilter tests passed.")
