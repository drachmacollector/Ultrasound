# Research Log: Anatomy or Acquisition?

One entry per task (format: `docs/research/plans/00_INDEX_AND_VERIFIED_FACTS.md §2.5`).
Numbers are quoted from `results/json/*.json`, never typed by hand. Lines starting with
`deviation:` are collected into RESULTS.md → Deviations.

---

## 2026-11-12 — R0 — Repository restructure
commit: uncommitted working tree on branch `research/anatomy-or-acquisition` (base `79ac5a5`); legacy tag `legacy-fetscan-demo-v1` → `b854a34`
run_ids: `_smoke/convnext_tiny` (1-epoch smoke run; not a result)
outputs: `archive/`, `tests/` (56 tests), `src/models/loading.py`, `data/splits/SPLITS_SHA256.txt`, `checkpoints/FROZEN.md`, new `CLAUDE.md`/`README.md`
outcome: legacy demo archived; multitask line removed; leakage and frozen-checkpoint guards added to training; test suite green; smoke training OK (frozen checkpoint hash unchanged).
deviation: split hashes use LF-normalised bytes (repo has core.autocrlf=true; raw-byte hashes would depend on checkout settings).
deviation: forbidden-token leakage check inspects the repo-relative path (or only the file name for files outside the repo) instead of the full absolute path; the absolute-path version false-positived on temp folders such as `pytest-…`.
deviation: `requirements-lock.txt` listed pandas 3.0.5 but the environment has 2.3.3; the lock now reflects the installed environment.

---

## 2026-11-12 — R1 — Data foundation (Gate G1)
commit: uncommitted working tree on `main` (base `0de1ec0`)
run_ids: none (no model runs)
outputs: `data/processed/external/*.csv`, `data/processed/landmarks/*.parquet`, `data/processed/metadata/*.parquet`, `data/processed/masks/` (14,273 npz + 12 consensus cores), `data/processed/audit/*`, `results/json/{R1_*,G1}.json`, `results/tables/R1_*.md`, `docs/research/GATE_G1.md`
outcome: external manifests HC18 999/806 pts, UCL 424/51, African 450/127; FP landmarks 3,090/3,090 pixel-identical to FPDB, 1,486 test + 238 val kept; Head QC flags 0%; confounding shares reproduced exactly (Aloka 0.6506, E6 0.1357, S10-test 0.5163). Binary audit (300 FPDB + 42 external + 150 sector + 40 re-check + 15 landmark): calipers 3.0% [1.6, 5.6] of FPDB, text 100%; G1 → overlays SECONDARY, E5a infeasible, E5d dropped, consensus prior enabled (E6 88% → 93% after), R4 may proceed. 863 byte-identical FPDB images (384 groups, within-patient), 8 test groups with conflicting labels.
deviation: binary audit rated by a vision model (`claude-vision`) from contact sheets instead of a human, at the PI's request; rater was blind to detector output for caliper/text ratings.
deviation: FP↔FPDB registration replaced by an exact identity check (images are pixel-identical).
deviation: HC18/UCL patient IDs from filename prefixes (annotation SubjectID columns empty).
deviation: abdominal landmark QC uses a mid-point offset bound of 0.30 (head 0.15); TAD/APAD caliper pairs do not bisect each other.
deviation: sector consensus prior = union with a 0.95-coverage train-only "core" (plan: intersection with a majority mask). Observed failures were under-inclusion, and a majority mask covered 92–95% of the frame.
deviation: caliper detector recall 0.20 (train) after one tuning round; dotted-line scoring disabled (fired on speckle). Per the pre-specified rule, no manual masks were used.
