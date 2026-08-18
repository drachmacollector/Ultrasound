"""
src/smoothing/test_tier2_mode_filter.py

Unit tests for Tier2ModeFilter, mirroring test_tier1.py's structure and thoroughness.

Tests:
  1. Stable input           — all frames same label; output = input, is_stable=True.
  2. Oscillation suppressed — A/B alternation; filter freezes, 0 Tier-2 switches.
  3. Genuine transition     — N frames A, then N frames B; filter adopts B with lag.
  4. No-op on stable Tier-1 output — composability guarantee for the 45/46 clips.
  5. Cold-start             — first frame returns a valid label, no -1.
  6. Reset                  — reset() fully clears state.
  7. Asymmetric oscillation — 2:1 ratio; dominant class wins majority.
  8. Parameter validation   — invalid params raise ValueError.
  9. Tie-break fix          — even window + frac=0.5: 50/50 tie does NOT switch.
 10. is_stable tracking     — badge reflects Tier-2a's own switch cadence.

Run:
    conda run -n fetalplane python -m src.smoothing.test_tier2_mode_filter
"""
from __future__ import annotations

import math
import sys

sys.path.insert(0, ".")
from src.smoothing.tier2_mode_filter import Tier2ModeFilter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_filter(**kwargs) -> Tier2ModeFilter:
    defaults = dict(window_frames=9, min_majority_frac=0.6)
    defaults.update(kwargs)
    return Tier2ModeFilter(**defaults)


def feed(f: Tier2ModeFilter, labels: list[int]) -> tuple[list[int], list[bool]]:
    """Feed labels through filter; return (output_labels, is_stable_flags)."""
    out_labels, out_stable = [], []
    for lbl in labels:
        label, stable = f.step(lbl)
        out_labels.append(label)
        out_stable.append(stable)
    return out_labels, out_stable


# ---------------------------------------------------------------------------
# Case 1: Stable input — no-op guarantee
# ---------------------------------------------------------------------------

def test_stable_input_noop():
    f = make_filter(window_frames=9, min_majority_frac=0.6)
    CLASS_A = 2
    n_frames = 50

    outputs, stable_flags = feed(f, [CLASS_A] * n_frames)

    for i, out in enumerate(outputs):
        assert out == CLASS_A, f"Frame {i}: expected {CLASS_A}, got {out}"
    # After window fills, should be stable every frame
    for i, s in enumerate(stable_flags[9:], start=9):
        assert s, f"Frame {i}: expected is_stable=True on steady input"
    print(f"[PASS] test_stable_input_noop: {n_frames} frames, all output={CLASS_A}, stable after warm-up")


# ---------------------------------------------------------------------------
# Case 2: Oscillation suppression
# ---------------------------------------------------------------------------

def test_oscillation_suppressed():
    f = make_filter(window_frames=9, min_majority_frac=0.6)
    CLASS_A, CLASS_B = 0, 1
    labels = [CLASS_A if i % 2 == 0 else CLASS_B for i in range(60)]

    outputs, _ = feed(f, labels)

    switches = sum(1 for a, b in zip(outputs, outputs[1:]) if a != b)
    assert switches == 0, f"Expected 0 switches on A/B oscillation, got {switches}"
    assert outputs[-1] == CLASS_A
    print(f"[PASS] test_oscillation_suppressed: 60 frames A/B, switches={switches}, held={outputs[-1]}")


# ---------------------------------------------------------------------------
# Case 3: Genuine transition
# ---------------------------------------------------------------------------

def test_genuine_transition():
    WINDOW = 9
    MAJORITY = 0.6
    f = make_filter(window_frames=WINDOW, min_majority_frac=MAJORITY)
    CLASS_A, CLASS_B = 3, 5
    N_STABLE = 30
    N_CHANGE = 30

    labels = [CLASS_A] * N_STABLE + [CLASS_B] * N_CHANGE
    outputs, _ = feed(f, labels)

    for i, out in enumerate(outputs[:N_STABLE]):
        assert out == CLASS_A, f"Frame {i}: expected {CLASS_A} during stable A phase, got {out}"

    assert outputs[-1] == CLASS_B, f"Expected final output {CLASS_B}, got {outputs[-1]}"

    first_b_output_frame = next(
        (i for i, out in enumerate(outputs[N_STABLE:], start=N_STABLE) if out == CLASS_B),
        None,
    )
    assert first_b_output_frame is not None, "Filter never adopted B"
    lag = first_b_output_frame - N_STABLE
    # Need strictly more than frac*window frames of B → ceiling of next integer above
    min_frames_needed = math.floor(WINDOW * MAJORITY) + 1
    max_lag = min_frames_needed + 1
    assert lag <= max_lag, f"Transition lag {lag} frames exceeds expected max {max_lag}"
    print(f"[PASS] test_genuine_transition: A->B at lag={lag} frames (max={max_lag}, window={WINDOW})")


# ---------------------------------------------------------------------------
# Case 4: No-op on already-stable Tier-1 output (composability)
# ---------------------------------------------------------------------------

def test_noop_on_stable_tier1_output():
    f = make_filter(window_frames=13, min_majority_frac=0.6)
    tier1_stream = [4] * 100 + [7] * 100
    outputs, _ = feed(f, tier1_stream)

    for i, out in enumerate(outputs[:100]):
        assert out == 4, f"Frame {i}: expected 4, got {out}"

    transition_lag = f.window_frames
    settled_start = 100 + transition_lag
    for i, out in enumerate(outputs[settled_start:], start=settled_start):
        assert out == 7, f"Frame {i}: expected 7 after settling, got {out}"

    phase1_switches = sum(1 for a, b in zip(outputs[:100], outputs[1:100]) if a != b)
    assert phase1_switches == 0, f"Spurious switches in stable phase: {phase1_switches}"
    print(f"[PASS] test_noop_on_stable_tier1_output")


# ---------------------------------------------------------------------------
# Case 5: Cold-start
# ---------------------------------------------------------------------------

def test_cold_start():
    f = make_filter()
    label, is_stable = f.step(2)
    assert label == 2, f"Cold-start should return first label, got {label}"
    assert f.window_fill == 1
    print(f"[PASS] test_cold_start: first frame output={label}, stable={is_stable}")


# ---------------------------------------------------------------------------
# Case 6: Reset
# ---------------------------------------------------------------------------

def test_reset_clears_state():
    f = make_filter()
    for _ in range(20):
        f.step(3)
    f.reset()

    assert f.window_fill == 0
    assert f._displayed_label == Tier2ModeFilter._UNINIT
    assert f._frames_since_switch == 0

    label, _ = f.step(6)
    assert label == 6, f"Post-reset first frame should return 6, got {label}"
    print("[PASS] test_reset_clears_state")


# ---------------------------------------------------------------------------
# Case 7: Asymmetric oscillation
# ---------------------------------------------------------------------------

def test_asymmetric_oscillation():
    f = make_filter(window_frames=9, min_majority_frac=0.6)
    CLASS_A, CLASS_B = 1, 4
    pattern = [CLASS_A, CLASS_A, CLASS_B]
    labels = (pattern * 30)[:90]

    outputs, _ = feed(f, labels)

    settled_outputs = outputs[45:]
    b_count = settled_outputs.count(CLASS_B)
    assert b_count == 0, f"Expected CLASS_B suppressed, got {b_count} occurrences"
    print(f"[PASS] test_asymmetric_oscillation: A(2/3)+B(1/3) -> B suppressed in settled phase")


# ---------------------------------------------------------------------------
# Case 8: Parameter validation
# ---------------------------------------------------------------------------

def test_parameter_validation():
    errors = []

    try:
        Tier2ModeFilter(window_frames=9, min_majority_frac=0.49)
        errors.append("min_majority_frac=0.49 should have raised ValueError")
    except ValueError:
        pass

    # min_majority_frac exactly 0.5 should be VALID
    try:
        Tier2ModeFilter(window_frames=9, min_majority_frac=0.5)
    except ValueError:
        errors.append("min_majority_frac=0.5 should be valid")

    try:
        Tier2ModeFilter(window_frames=9, min_majority_frac=1.1)
        errors.append("min_majority_frac=1.1 should have raised ValueError")
    except ValueError:
        pass

    try:
        Tier2ModeFilter(window_frames=0, min_majority_frac=0.6)
        errors.append("window_frames=0 should have raised ValueError")
    except ValueError:
        pass

    if errors:
        raise AssertionError(f"{len(errors)} validation tests failed: {errors}")
    print("[PASS] test_parameter_validation")


# ---------------------------------------------------------------------------
# Case 9: Tie-break fix (even window + frac=0.5 → tie must NOT switch)
# ---------------------------------------------------------------------------

def test_tie_break_even_window():
    """
    Regression test for the strict > comparison fix.

    window_frames=8, min_majority_frac=0.5, strict alternation A/B:
    Buffer when full = [A,B,A,B,A,B,A,B] → 4:4 tie → top_count/window_size = 0.5.
    With >=, this would trigger a switch (bug).
    With > (fix), 0.5 > 0.5 is False → no switch (correct).
    """
    f = Tier2ModeFilter(window_frames=8, min_majority_frac=0.5)
    CLASS_A, CLASS_B = 10, 20
    labels = [CLASS_A if i % 2 == 0 else CLASS_B for i in range(40)]

    outputs, _ = feed(f, labels)

    switches = sum(1 for a, b in zip(outputs, outputs[1:]) if a != b)
    assert switches == 0, (
        f"Even-window tie-break bug: got {switches} Tier-2 switches on strict "
        f"A/B alternation with window=8, frac=0.5. Expected 0 (tie must not trigger switch)."
    )
    print(f"[PASS] test_tie_break_even_window: 40 frames A/B @ window=8,frac=0.5 -> switches={switches}")


# ---------------------------------------------------------------------------
# Case 10: is_stable reflects Tier-2a's own switch cadence
# ---------------------------------------------------------------------------

def test_is_stable_tracks_tier2_switches():
    """
    When Tier-1 oscillates but Tier-2a holds steady, is_stable must be True
    (driven by Tier-2a's own frames_since_switch, not Tier-1's internal state).

    When a genuine Tier-2a switch fires, is_stable must reset to False until
    min_stable_frames frames have elapsed.
    """
    f = Tier2ModeFilter(window_frames=9, min_majority_frac=0.7, min_stable_frames=5)
    CLASS_A, CLASS_B = 0, 1

    # Phase 1: Tier-1 oscillates A/B (→ Tier-2a holds steady on A)
    # After warm-up, Tier-2a should report is_stable=True despite Tier-1 bouncing.
    osc_labels = [CLASS_A if i % 2 == 0 else CLASS_B for i in range(30)]
    osc_outputs, osc_stable = feed(f, osc_labels)

    assert osc_outputs[-1] == CLASS_A, "Filter should hold CLASS_A during oscillation"
    # After 30 frames of no Tier-2a switches, is_stable must be True
    assert osc_stable[-1], "is_stable should be True when Tier-2 hasn't switched for >min_stable_frames"

    # Phase 2: Force a genuine transition (all-B for enough frames to exceed majority)
    f2 = Tier2ModeFilter(window_frames=9, min_majority_frac=0.7, min_stable_frames=5)
    # Prime with enough A's to fill window
    for _ in range(9):
        f2.step(CLASS_A)
    # Then feed enough B's to exceed majority (need >0.7*9=6.3 → 7 frames of B)
    transition_outputs = []
    transition_stable = []
    for _ in range(15):
        lbl, stb = f2.step(CLASS_B)
        transition_outputs.append(lbl)
        transition_stable.append(stb)

    # Eventually must switch to B
    assert transition_outputs[-1] == CLASS_B, "Expected transition to B"
    # Immediately after the switch, is_stable should be False
    first_b_idx = next(i for i, lbl in enumerate(transition_outputs) if lbl == CLASS_B)
    assert not transition_stable[first_b_idx], (
        f"is_stable should be False immediately after a Tier-2a switch "
        f"(min_stable_frames=5, frames_since_switch=0 at switch frame)"
    )

    print("[PASS] test_is_stable_tracks_tier2_switches")


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
    test_tie_break_even_window()
    test_is_stable_tracks_tier2_switches()
    print("\nAll Tier2ModeFilter tests passed.")
