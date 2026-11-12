"""
src/smoothing/tier2_mode_filter.py

Tier-2a: causal majority-vote (mode) filter over Tier-1's displayed-label stream.

PURPOSE
-------
Tier-1 (EMA + hysteresis + dwell) achieves 90.9% flicker suppression across all 46
evaluation clips, with zero spurious switches introduced on any previously-stable clip.
One clip — 202101141947512003470I1.avi (72 switches/min after Tier-1) — is a confirmed
hard structural floor for the EMA+hysteresis+dwell parameter family: no combination of
Tier-1 parameters can reduce it further without introducing spurious switches on other
clips.  This is because the oscillation on that clip satisfies Tier-1's own hysteresis
criterion on every oscillation event (briefly crosses switch_threshold with genuine
confidence), so the dwell gate fires and the switch is accepted.

Tier-2a's mechanism: keep a fixed-size trailing window of the last N Tier-1 labels.
Adopt a new displayed label only when one class holds STRICTLY MORE than
min_majority_frac of that window.  Because the oscillating clip alternates at a rate
*faster* than window_frames, neither alternating class can accumulate a strict majority,
and the filter holds the most recently settled label until one class dominates.

COMPOSITION
-----------
Tier-2a sits *after* Tier-1 in the pipeline.  It consumes Tier-1's
`current_displayed_label` output frame-by-frame and returns its own display label.
It does NOT replace Tier-1; it is a second, optional stage.  On all 45/46 clips where
Tier-1 already outputs a stable stream, Tier-2a is a no-op after the window fills
(the majority class never changes, so `_displayed_label` is never updated away from
the correct class).  This composability claim is verified explicitly in the unit tests
and in scripts/evaluate_tier2.py.

INTERFACE
---------
Matches Tier-1's per-frame step() pattern for symmetric composability:

    tier1_label, tier1_conf, tier1_probs, tier1_stable = smoother.step(raw_probs)
    tier2_label, tier2_stable = mode_filter.step(tier1_label)

Note the extended return signature: step() now returns a (label, is_stable) tuple
mirroring Tier-1's pattern, so the pipeline can surface an accurate stability badge
when Tier-2a is active rather than relying on Tier-1's internal dwell counter.

Parameters loaded from configs/smoothing_tier2a.yaml (separate from
smoothing_tier1.yaml — tiers' configs are never merged).

LATENCY COST
------------
window_frames=N adds at most (N-1) frames of *additional* lag before a genuine
majority can accumulate in the window after a real transition.  At 23.7fps steady
state (Task 10 validated):
    window=5  → max additional lag ≈  168 ms
    window=9  → max additional lag ≈  337 ms
    window=13 → max additional lag ≈  506 ms
    window=15 → max additional lag ≈  590 ms
    window=19 → max additional lag ≈  759 ms
    window=25 → max additional lag ≈ 1012 ms
Combined with Tier-1's ~340ms dwell window, the total worst-case display lag is
Tier-1_dwell + (window-1)/fps.  The parameter sweep (scripts/tune_tier2_mode_filter.py)
measures actual added lag and must stay within the 150–500ms combined budget target.

NON-NEGOTIABLE CONSTRAINTS
--------------------------
1.  This file does NOT modify tier1.py or test_tier1.py.
2.  Tier-2a is opt-in at the CLI level (--enable-tier2 flag in app.py), off by default.
3.  configs/smoothing_tier1.yaml is never touched by Tier-2a code.
4.  checkpoints/convnext_tiny/best.pt is untouched.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import yaml


@dataclass
class Tier2ModeFilter:
    """Stateful causal majority-vote filter over a Tier-1 label stream.

    Parameters
    ----------
    window_frames : int
        Length of the trailing window (in frames) used for the vote.
        Larger values suppress more oscillation but increase latency to genuine
        transitions.  Tune empirically — see scripts/tune_tier2_mode_filter.py.
    min_majority_frac : float
        The leading class must hold STRICTLY MORE than this fraction of the window
        before a switch is committed.  The comparison is strict (`>`), so at
        exactly 0.5 with an even window, a 50/50 tie never triggers a switch.
        Range [0.5, 1.0].

        Deployed value: 0.7 (window=9, odd — ties are mathematically impossible
        regardless of the strict comparison, since 5/9 = 55.6% > 50%).
    min_stable_frames : int
        Number of consecutive frames without a Tier-2a display-label switch
        required before is_stable is reported True.  Mirrors Tier-1's
        min_dwell_frames concept so the pipeline can surface an accurate
        stability badge when Tier-2a is active.  Defaults to 1 (stable as
        soon as a new label is locked, matching simple UX expectation).

    State (internal, not constructor args)
    ---------------------------------------
    _buffer               : deque of int (last window_frames Tier-1 labels)
    _displayed_label      : int  (current Tier-2 output; -1 = uninitialised)
    _frames_since_switch  : int  (frames elapsed since last Tier-2 switch)
    """

    window_frames: int = 15
    min_majority_frac: float = 0.6
    min_stable_frames: int = 1

    # Sentinel value used before the first frame is processed.
    _UNINIT: ClassVar[int] = -1

    _buffer: deque[int] = field(init=False, default_factory=deque)
    _displayed_label: int = field(init=False, default=_UNINIT)
    _frames_since_switch: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not (0.5 <= self.min_majority_frac <= 1.0):
            raise ValueError(
                f"min_majority_frac must be in [0.5, 1.0], got {self.min_majority_frac}"
            )
        if self.window_frames < 1:
            raise ValueError(f"window_frames must be >= 1, got {self.window_frames}")
        if self.min_stable_frames < 1:
            raise ValueError(f"min_stable_frames must be >= 1, got {self.min_stable_frames}")

    # ---- Lifecycle -----------------------------------------------------------

    def reset(self) -> None:
        """Return to cold-start state.  Call when switching video sources."""
        self._buffer.clear()
        self._displayed_label = self._UNINIT
        self._frames_since_switch = 0

    # ---- Per-frame step ------------------------------------------------------

    def step(self, tier1_label: int) -> tuple[int, bool]:
        """Process one frame's Tier-1 displayed label and return (tier2_label, is_stable).

        Args:
            tier1_label: The `current_displayed_label` output from Tier1Smoother.step().

        Returns:
            tier2_label : The Tier-2 displayed label (may lag behind tier1_label by up
                          to window_frames frames when a genuine transition is in progress).
            is_stable   : True once _frames_since_switch >= min_stable_frames, meaning
                          the Tier-2 display has been steady for long enough.  Drives
                          the STABLE/SETTLING HUD badge when Tier-2a is active.
        """
        # Cold-start: immediately adopt the first Tier-1 label so there is no
        # uninitialised -1 display on the very first frame.
        if self._displayed_label == self._UNINIT:
            self._displayed_label = tier1_label
            self._buffer.append(tier1_label)
            self._frames_since_switch = 1
            is_stable = self._frames_since_switch >= self.min_stable_frames
            return self._displayed_label, is_stable

        # Append to rolling window, drop oldest if over capacity.
        self._buffer.append(tier1_label)
        if len(self._buffer) > self.window_frames:
            self._buffer.popleft()

        # Count votes in the current window.
        counts = Counter(self._buffer)
        top_label, top_count = counts.most_common(1)[0]
        window_size = len(self._buffer)

        # Switch only if the leading class holds STRICTLY MORE than min_majority_frac
        # of the window AND it differs from the currently displayed label.
        #
        # Using strict > (not >=) is critical for correctness:
        #   • At min_majority_frac=0.5 with an even window (e.g., window=8), a 4:4 tie
        #     gives top_count/window_size = 0.5.  Using >= would incorrectly trigger
        #     a switch.  Using > correctly prevents it (tie → no switch).
        #   • The deployed config (window=9, odd) makes ties impossible regardless
        #     (5/9 = 55.6%), but the strict comparison ensures general correctness.
        #
        # Mechanism on oscillating clips:
        #   • On stable clips: top_label == _displayed_label every frame → no-op.
        #   • On oscillating clips (near-50/50 alternation): the top class fraction
        #     stays at or just above 50%, well below min_majority_frac=0.7.  Switch
        #     never fires; displayed label freezes.
        #   • On genuine transitions: after enough frames of the new class accumulate
        #     to exceed min_majority_frac, the switch is committed.
        switched = False
        if (
            top_label != self._displayed_label
            and (top_count / window_size) > self.min_majority_frac
        ):
            self._displayed_label = top_label
            self._frames_since_switch = 0
            switched = True

        if not switched:
            self._frames_since_switch += 1

        is_stable = self._frames_since_switch >= self.min_stable_frames
        return self._displayed_label, is_stable

    # ---- Diagnostics ---------------------------------------------------------

    @property
    def window_fill(self) -> int:
        """Number of frames currently in the buffer (<= window_frames)."""
        return len(self._buffer)

    @property
    def frames_since_switch(self) -> int:
        """Frames elapsed since the last Tier-2 display-label change."""
        return self._frames_since_switch

    @property
    def current_vote_distribution(self) -> dict[int, float]:
        """Fraction of window votes for each label currently in the buffer."""
        if not self._buffer:
            return {}
        n = len(self._buffer)
        return {label: count / n for label, count in Counter(self._buffer).items()}

    # ---- Factory: load from YAML config --------------------------------------

    @classmethod
    def from_config(cls, config_path: str | Path) -> "Tier2ModeFilter":
        """Instantiate from a YAML config file (e.g. configs/smoothing_tier2a.yaml).

        Expected YAML keys:
            window_frames:      int
            min_majority_frac:  float
            min_stable_frames:  int  (optional, default 1)

        Example configs/smoothing_tier2a.yaml:
            window_frames: 9
            min_majority_frac: 0.7
        """
        with open(config_path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        return cls(
            window_frames=int(cfg["window_frames"]),
            min_majority_frac=float(cfg["min_majority_frac"]),
            min_stable_frames=int(cfg.get("min_stable_frames", 1)),
        )
