"""
src/smoothing/tier1.py

Tier-1 temporal smoothing: EMA + hysteresis + minimum dwell time.
Per docs/05_TEMPORAL_SMOOTHING_AND_REALTIME.md §A1.

Design note — hold_floor vs. switch_threshold:
    The doc's prose (§A1) says "hold at >0.4 but require >0.6 to switch", implying two
    thresholds: a hold-floor below which we'd give up the current label, and a switch-ceiling
    that a new candidate must cross. However, the doc's own pseudocode only implements a
    single `switch_threshold` and holds unconditionally on the current label regardless of
    its smoothed confidence.

    This implementation follows the pseudocode's *actual logic* (single switch_threshold,
    unconditional hold on label match), but exposes an optional `hold_floor` parameter.
    When hold_floor is not None, the smoother additionally requires that the smoothed
    confidence of the current displayed label exceeds hold_floor to keep holding it; if
    confidence falls below this floor, the smoother becomes willing to switch (subject to
    the same switch_threshold gate) even before min_dwell_frames is met. This is disabled
    by default (None) so behaviour matches the pseudocode literally unless the Task 6/7
    tuning sweep finds a reason to enable it. Mentioned explicitly in PHASE5 walkthrough.

frames_since_last_switch counting direction — correction vs. doc:
    The doc's pseudocode resets frames_since_last_switch = 0 on hold (label match), which
    means it never accumulates while holding a stable label and is_stable would only ever
    be momentarily True/False right after each switch event. This implementation instead
    counts frames_since_last_switch UP during holding and resets to 0 only on an actual
    switch — this is necessary to make is_stable meaningful as an ongoing stability
    indicator for the render loop's "STABLE / SETTLING" overlay. Flagged explicitly in
    PHASE5 walkthrough as a deliberate deviation from the doc's literal pseudocode.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Tier1Smoother:
    """Stateful one-frame-at-a-time EMA smoother with hysteresis and dwell gating.

    Parameters
    ----------
    num_classes : int
        Number of output classes (must match the model's softmax output size).
    alpha : float
        EMA decay weight applied to the incoming raw probabilities.
        smoothed = alpha * raw + (1 - alpha) * smoothed.
        Range (0, 1]; higher = more responsive, lower = more stable.  Tune via sweep.
    switch_threshold : float
        The smoothed confidence a candidate label must exceed (strictly) before the
        displayed label is allowed to switch to it, provided min_dwell_frames has also
        been met.  Acts as hysteresis — higher = harder to switch.
    min_dwell_frames : int
        The minimum number of consecutive frames the current displayed label must have
        been held (accumulated via frames_since_last_switch counting up) before a switch
        is permitted.  Convert to milliseconds using measured inference FPS.
    hold_floor : float | None
        Optional lower confidence bound for the *current* displayed label.  When not None,
        if the smoothed confidence of the current label falls below this value, the hold
        is relaxed and switching is gated only by switch_threshold (ignoring the dwell
        requirement).  Default None = disabled (matches the pseudocode literally).
    """

    num_classes: int
    alpha: float = 0.3
    switch_threshold: float = 0.6
    min_dwell_frames: int = 5
    hold_floor: float | None = None  # see design note in module docstring; None = pseudocode

    # Internal mutable state — not constructor args
    # smoothed_probs: np.ndarray = field(init=False, default=None)
    smoothed_probs: np.ndarray | None = field(init=False, default=None)
    current_displayed_label: int = field(init=False, default=-1)
    frames_since_last_switch: int = field(init=False, default=0)

    def reset(self) -> None:
        """Reset all state to the initial (cold-start) condition.

        Call this when switching to a new video source or after a deliberate pause.
        """
        self.smoothed_probs = None
        self.current_displayed_label = -1
        self.frames_since_last_switch = 0

    def step(self, raw_probs: np.ndarray) -> tuple[int, float, np.ndarray, bool]:
        """Process one frame's raw softmax output and return the stabilised result.

        Args:
            raw_probs: 1-D float array of shape [num_classes], raw per-frame softmax
                       probabilities.  Must be non-negative and sum to approximately 1.

        Returns:
            current_displayed_label : int
                The label the UI should display (may lag behind raw argmax by design).
            candidate_confidence : float
                Smoothed confidence of the EMA argmax (NOT of the displayed label).
                Used for the on-screen confidence readout.
            smoothed_probs : np.ndarray
                Copy of the full smoothed probability vector (for logging/debugging).
            is_stable : bool
                True when frames_since_last_switch >= min_dwell_frames, i.e. the
                displayed label has held long enough to be considered settled.
                Drives the render loop's "● STABLE / ◐ SETTLING" indicator.
        """
        # --- 1. Update EMA ---
        if self.smoothed_probs is None:
            # Cold start: initialise directly from first frame so we display something
            # immediately rather than showing a -1 label for the first min_dwell_frames.
            self.smoothed_probs = raw_probs.copy()
            self.current_displayed_label = int(np.argmax(self.smoothed_probs))
            # frames_since_last_switch stays 0 — we just "switched" from nothing
        else:
            self.smoothed_probs = (
                self.alpha * raw_probs + (1.0 - self.alpha) * self.smoothed_probs
            )

        # --- 2. Compute candidate from smoothed distribution ---
        candidate_label = int(np.argmax(self.smoothed_probs))
        candidate_confidence = float(self.smoothed_probs[candidate_label])

        # --- 3. Current-label confidence (used by hold_floor check) ---
        current_confidence = float(self.smoothed_probs[self.current_displayed_label])

        # --- 4. State-machine: hold / switch decision ---
        if candidate_label == self.current_displayed_label:
            # Holding the same label — count up so is_stable can eventually be True.
            self.frames_since_last_switch += 1
        else:
            # Candidate differs from displayed label.  Two conditions to permit switch:
            #   a) candidate_confidence must exceed switch_threshold (hysteresis)
            #   b) frames_since_last_switch must have accumulated >= min_dwell_frames
            #      (unless hold_floor is active AND current confidence fell below it,
            #       in which case we relax the dwell requirement and switch on confidence
            #       alone — prevents being stuck on a low-confidence old label forever).
            floor_breached = (
                self.hold_floor is not None
                and current_confidence <= self.hold_floor
            )
            dwell_met = self.frames_since_last_switch >= self.min_dwell_frames
            confidence_met = candidate_confidence > self.switch_threshold

            if confidence_met and (dwell_met or floor_breached):
                # Genuine switch.
                self.current_displayed_label = candidate_label
                self.frames_since_last_switch = 0
            else:
                # Not ready to switch yet — keep displaying old label but keep counting.
                self.frames_since_last_switch += 1

        # --- 5. Stability flag ---
        is_stable = self.frames_since_last_switch >= self.min_dwell_frames

        return (
            self.current_displayed_label,
            candidate_confidence,
            self.smoothed_probs.copy(),
            is_stable,
        )
