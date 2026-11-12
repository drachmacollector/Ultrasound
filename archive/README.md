# archive/

Frozen legacy **FetScan** real-time demo: tag `legacy-fetscan-demo-v1`, commit `b854a34`.
Archived on 2026-11-12 during the research pivot (see `docs/research/plans/R0_REPO_RESTRUCTURE.md`).

* **Not maintained. May not run.** Import paths inside these files still refer to their old
  locations (`src.realtime`, `src.smoothing`, ...).
* **Must not be imported** from `src/`, `scripts/` or `tests/`. This is enforced by
  `tests/test_no_archive_imports.py`.
* To run the demo as it was, check out the tag: `git switch --detach legacy-fetscan-demo-v1`.

| Path | What it was |
|---|---|
| `src/realtime/`, `src/smoothing/`, `app_streamlit.py`, `assets/` | threaded real-time pipeline, temporal smoothing, Streamlit/OpenCV UIs |
| `src/models/acam.py`, `configs/convnext_tiny_acam.yaml`, `configs/convnext_tiny_focal.yaml` | neutral/negative ablations (ACAM contrast module, focal loss) |
| `src/eval/evaluate_test.py`, `scripts/evaluate_test_acam.py` | legacy evaluation with image-level bootstrap (superseded by `src/eval/`) |
| `scripts/*` (video, latency, smoothing, HUD, CAM spot-check), `scripts/natalia/` | demo, tuning and phantom-data scripts |
| `docs/instructions/`, `docs/phases/`, `docs/*.md` | project specs, phase walkthroughs and reports from the demo era |
| `data/processed/*`, `logs/*` | artefacts produced by the archived scripts |

Removed outright rather than archived: the multitask detection line (`src/models/multitask_model.py`,
`scripts/{build_multitask_manifest,train_multitask,verify_no_multitask_leakage}.py`,
`configs/multitask.yaml`, `checkpoints/multitask/`, `data/splits/multitask_*.csv`). It was
trained on HC18/UCL, which are external test sets in the research study. It remains available
at the tag.
