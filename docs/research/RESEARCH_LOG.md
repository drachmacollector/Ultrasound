# Research Log: Anatomy or Acquisition?

One entry per task (format: `docs/research/plans/00_INDEX_AND_VERIFIED_FACTS.md §2.5`).
Numbers are quoted from `results/json/*.json`, never typed by hand. Lines starting with
`deviation:` are collected into RESULTS.md → Deviations.

---

## 2026-11-12 — R0 — Repository restructure
commit: `0de1ec0`; legacy tag `legacy-fetscan-demo-v1` → `b854a34`
run_ids: `_smoke/convnext_tiny` (1-epoch smoke run; not a result)
outputs: `archive/`, `tests/` (56 tests), `src/models/loading.py`, `data/splits/SPLITS_SHA256.txt`, `checkpoints/FROZEN.md`, new `CLAUDE.md`/`README.md`
outcome: legacy demo archived; multitask line removed; leakage and frozen-checkpoint guards added to training; test suite green; smoke training OK (frozen checkpoint hash unchanged).
deviation: split hashes use LF-normalised bytes (repo has core.autocrlf=true; raw-byte hashes would depend on checkout settings).
deviation: forbidden-token leakage check inspects the repo-relative path (or only the file name for files outside the repo) instead of the full absolute path; the absolute-path version false-positived on temp folders such as `pytest-…`.
deviation: `requirements-lock.txt` listed pandas 3.0.5 but the environment has 2.3.3; the lock now reflects the installed environment.

---

## 2026-11-12 — R1 — Data foundation (Gate G1)
commit: `8f6ba09`
run_ids: none (no model runs)
outputs: `data/processed/external/*.csv`, `data/processed/landmarks/*.parquet`, `data/processed/metadata/*.parquet`, `data/processed/masks/` (14,273 npz + 12 consensus cores), `data/processed/audit/*`, `results/json/{R1_*,G1}.json`, `results/tables/R1_*.md`, `docs/research/GATE_G1.md`
outcome: external manifests HC18 999/806 pts, UCL 424/51, African 450/127; FP landmarks 3,090/3,090 pixel-identical to FPDB, 1,486 test + 238 val kept; Head QC flags 0%; confounding shares reproduced exactly (Aloka 0.6506, E6 0.1357, S10-test 0.5163). Binary audit (300 FPDB + 42 external + 150 sector + 40 re-check + 15 landmark): calipers 3.0% [1.6, 5.6] of FPDB, text 100%; G1 → overlays SECONDARY, E5a infeasible, E5d dropped, consensus prior enabled (E6 88% → 93% after), R4 may proceed. 863 byte-identical FPDB images (384 groups, within-patient), 8 test groups with conflicting labels.
deviation: binary audit rated by a vision model (`claude-vision`) from contact sheets instead of a human, at the PI's request; rater was blind to detector output for caliper/text ratings.
deviation: FP↔FPDB registration replaced by an exact identity check (images are pixel-identical).
deviation: HC18/UCL patient IDs from filename prefixes (annotation SubjectID columns empty).
deviation: abdominal landmark QC uses a mid-point offset bound of 0.30 (head 0.15); TAD/APAD caliper pairs do not bisect each other.
deviation: sector consensus prior = union with a 0.95-coverage train-only "core" (plan: intersection with a majority mask). Observed failures were under-inclusion, and a majority mask covered 92–95% of the frame.
deviation: caliper detector recall 0.20 (train) after one tuning round; dotted-line scoring disabled (fired on speckle). Per the pre-specified rule, no manual masks were used.

---

## 2026-11-12 — R2 — Evaluation core and re-evaluation of the six frozen checkpoints (Gate G2)
commit: uncommitted working tree on `main` (base `8f6ba09`; parquet provenance records `8f6ba09a68-dirty`)
run_ids: `legacy_convnext_s42`, `legacy_effv2s_s42`, `legacy_repvgga1_s42`, `legacy_repvgga2_s42`, `legacy_mnv3l_s42`, `legacy_efflite0_s42`
outputs: `src/eval/{metrics,stats,datasets,predict,scoring,external}.py`, `results/preds/legacy_*_s42/` (30 parquet + meta + features), `results/json/{R2_existing_models,G2}.json`, `results/tables/R2_existing_models.md`, `results/figures/R2_rejection_vs_confusion.{pdf,png}`, `docs/research/GATE_G2.md`
outcome: G2 PASSED. All 6 primary and 12 secondary reproduction checks match: convnext test macro-F1 0.8929 (legacy 0.8927); HC18+UCL acc_full 83.63% (83.6%); HC18 81.38% / UCL 88.92%; Head→Other 180/194. The other 5 backbones match their macro-F1 to ≤0.0003, and their cross-device accuracy and Head→Other counts match exactly.
outcome: convnext FPDB-test macro-F1 0.893 [0.883, 0.902] (cluster CI 1.06–1.21× wider than the legacy image-level CI). Pooled external (1,873 images): acc_full 81.7% [79.6, 83.9], bal_acc_full 0.728 [0.690, 0.766], bal_acc_restricted 0.957 [0.935, 0.977]. 95.6% [93.4, 97.7] of external errors are 'Other' rejections (72–96% across backbones).
outcome: Other recall by machine, all 6 models: Aloka 0.92–0.95, Voluson E6 0.35–0.52, S10 0.81–0.92.
outcome: legacy "repvgg_a1 winner" claim WITHDRAWN (endpoint-dependent). It is best on HC18+UCL acc_full (significantly better than 4 of 5) but significantly worse than all 5 on pooled bal_acc_full (0.691 [0.655, 0.726]).
outcome: 8–9 test errors per model fall on the 8 contradictory duplicate groups (floor 8).
deviation: in-distribution scorer and metric factories are in `src/eval/scoring.py`; `external.py` only holds external scoring (plan put everything in `external.py`).
deviation: the image-level CI column is reported for all six backbones, not only convnext. Per-machine macro-F1 averages over the classes present on that machine.
deviation: macro-F1 excludes classes whose F1 is undefined in a resample (TP+FP+FN = 0). This equals sklearn on the full test set.
deviation: a `RatioMetric` fast path (sum/sum ratios) was added next to the confusion-matrix fast path.
deviation: G2 macro-F1 targets for the five other backbones use the 4-decimal values in archive/docs/EVAL_REPORT.md, not the 2-decimal classification reports. This is stricter.
deviation: extra analyses not in the plan: pairwise comparison on the legacy endpoint (HC18+UCL acc_full), HC18+UCL legacy-set table, and errors on contradictory duplicates.
deviation: `src/__init__.py` added; it sets `NO_ALBUMENTATIONS_UPDATE=1`, because albumentations' import-time version check made network calls and once timed out during the run.
