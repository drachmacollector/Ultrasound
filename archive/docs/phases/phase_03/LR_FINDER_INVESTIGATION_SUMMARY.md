# LR Finder Investigation — Summary & Context Doc

**Purpose of this file:** a self-contained record of what happened during the
Phase 4 learning-rate range test work, why the auto-detected "suggested LR"
values were wrong, what the actual chosen LRs are, and what (if anything)
still needs fixing in `scripts/lr_finder.py`. Read this before touching LR
tuning again, so context doesn't need to be rebuilt from scratch.

---

## 1. Where this sits in the project

This is part of Phase 4 ([docs/04_MODEL_TRAINING.md](../instructions/04_MODEL_TRAINING.md),
[docs/PHASE_4_KICKOFF_PROMPT.md](PHASE_4_KICKOFF_PROMPT.md)) — specifically
§5, the Leslie Smith-style LR range test that's supposed to run *before* any
smoke test or full training run, so the `lr:` placeholder in each
`configs/<backbone>.yaml` gets replaced with a real, justified value instead
of the `3e-4` guess every config ships with.

Six backbone configs exist (five from the original plan + one extra,
`convnext_tiny`, added along the way

- `repvgg_a1`
- `repvgg_a2`
- `mobilenetv3_large_100`
- `efficientnet_lite0`
- `tf_efficientnetv2_s`
- `convnext_tiny`

---

## 2. The bug: auto-detected "steepest descent" was unreliable

### Symptom
Across many runs and several rounds of fixes to `scripts/lr_finder.py`'s
`plot_lr_finder()` function, the auto-computed "Suggested training LR"
kept landing on nonsense values:

- The very start of the sweep (`1e-7`)
- The very end / divergence zone (`3.8e-1`, `7.24e-1`)
- **Most persistently:** a value around `~1e-6` for every single backbone
  regardless of architecture, parameter count, or loss scale
  (`repvgg_a2`, `mobilenetv3_large_100`, `efficientnet_lite0`,
  `convnext_tiny` all reported `1.12e-06`; `tf_efficientnetv2_s` reported
  `1.55e-06`).

That last pattern — five architecturally different models all agreeing to
3 significant figures — was the tell that something structural was wrong,
not something about the data.

### Root cause
`train_one_epoch`/`run_lr_finder`'s loss smoothing uses a bias-corrected EMA:

```python
smoothed_loss = smoothing * raw_loss + (1 - smoothing) * smoothed_loss
bc_loss = smoothed_loss / (1 - (1 - smoothing) ** (step + 1))
```

At `step=0` the denominator is tiny, so `bc_loss` starts enormous
(e.g. `smoothed=95.0` when the true loss is `~4.75`) and decays rapidly over
the first ~15-20 steps **regardless of what the learning rate is doing**.
This is a pure artifact of the bias-correction formula settling in, not a
signal about the model or the LR.

Because this artifact-decay region is visually/numerically the steepest
part of the whole curve — steeper than the real loss-vs-LR relationship
later in the sweep — any detection method (`np.gradient` over the full
curve, `np.diff` over a supposedly-restricted window, `argmin` search) kept
re-discovering the artifact instead of the real elbow, because the "skip"
logic was applied inconsistently (sometimes only to detection math, not to
whatever region was actually being searched, and never to the plot itself).

The plot images looked like they supported the bad number because the y-axis
is **linear** (0 to ~95+), so the artifact-decay cliff dominates the visual
scale and squashes the real (much smaller-looking, but real) loss drop from
the actual LR sweep down near the bottom of the chart.

### Status of the fix
Multiple attempted patches (clipping the search window to a hard `ceil_lr`,
switching `np.gradient` → `np.diff`, searching only up to the pre-divergence
minimum) each fixed one symptom but not the underlying issue reliably across
all six configs. **As of this writing, the auto-detection logic in
`plot_lr_finder()` is not trustworthy and should not be used as-is.**

The most robust fix identified but **not yet implemented**: track the
best-loss LR live, during the sweep itself, rather than trying to recover it
from the post-hoc smoothed curve:

```python
best_loss: float = float("inf")
best_loss_lr: float = lr_start   # NEW

...
if bc_loss < best_loss:
    best_loss = bc_loss
    best_loss_lr = current_lr    # NEW
```

Then `suggested_lr = best_loss_lr / 10`. This avoids differentiating a noisy
curve entirely. It still won't perfectly handle cases where the true minimum
sits right at the edge of the divergence cliff (see `tf_efficientnetv2_s` /
`mobilenetv3_large_100` below), so pairing it with a printed loss value a
couple of log-steps past the minimum (to eyeball whether to back off further)
is recommended before trusting it blindly.

Also recommended, if the plot itself is used for visual sanity checks going
forward: skip the first `~15` points in **the plotted curve**, not just in
the detection math, so the artifact spike doesn't visually dominate.

---

## 3. What was actually used instead: reading logs directly

Since the auto-detection wasn't reliable, final LR choices were made by
reading each run's per-10-step loss log directly and applying the standard
"steepest descent ÷ 10" rule of thumb by eye — i.e., find where loss is
dropping fastest just before it turns upward/diverges, then divide that LR
by 10 for a safety margin.

## 4. Final chosen learning rates

| Backbone | Loss minimum observed | Divergence starts | Chosen `lr:` | Notes |
|---|---|---|---|---|
| `repvgg_a1` | ~step 60-65, LR ~1e-3–2e-3 | ~step 70, LR ~8e-3 | **5e-4** | Based on 4 repeated runs, all consistent |
| `repvgg_a2` | step 50, LR 3.16e-4 (loss 1.18); already rising by step 60 | step 60, LR 1.58e-3 | **3e-4** | More LR-sensitive than A1 — divergence starts earlier |
| `mobilenetv3_large_100` | step 60, LR 1.58e-3 (loss 0.61); still ok step 70 (0.94) | step 80, LR 3.98e-2 | **7e-4** | More tolerant of higher LR than RepVGG variants |
| `efficientnet_lite0` | step 60, LR 1.58e-3 (loss 1.04) is the real minimum; step 80 dip (0.85) is noise, not trend | not clearly triggered within 100 steps | **1.5e-3** | Confirmed via direct log read after the plot's visual "steepest slope" turned out to be the EMA bias-correction artifact, not real signal (see §2) |
| `tf_efficientnetv2_s` | step 70, LR 7.94e-3 (loss 0.94) | step 80, immediate blowup | **1.5e-3** | Sharp cliff right after the minimum — backed off further than ÷10 alone would suggest |
| `convnext_tiny` | unstable starting ~step 40-50, fully NaN by step 90 | early instability | **1e-4** | Most fragile of the six — picked conservatively, well below where oscillation begins |

These are the values that should be (or already have been) written into each
`configs/<backbone>.yaml`, replacing the `3e-4` placeholder comment that
shipped with every config from the initial Phase 4 build.

---

## 5. Things to double check before the smoke test

- [ ] Decide whether to implement the `best_loss_lr` fix in
      `scripts/lr_finder.py` before running it again on any future backbone,
      or whether reading logs directly (as done here) is good enough going
      forward — the current auto-detection should not be trusted as-is
      either way

## 6. Next step

Per `docs/PHASE_4_KICKOFF_PROMPT.md` §6 / `docs/04_MODEL_TRAINING.md` §5.2:
run `scripts/smoke_test.py` across all backbone configs (5-epoch mini runs)
to confirm the full pipeline — data loading, forward/backward pass,
checkpoint save/reload, TensorBoard logging — works end-to-end for every
backbone before committing to any full training run.
