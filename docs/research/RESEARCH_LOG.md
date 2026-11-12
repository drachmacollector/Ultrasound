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
