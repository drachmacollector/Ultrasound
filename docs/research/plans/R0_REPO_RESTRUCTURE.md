# R0 — Repository Restructure (legacy freeze, archive, cleanup)

**Goal:** turn the FetScan demo repo into a clean research repo without losing history. Nothing is deleted before the tag exists, and every retained import resolves.
**Effort:** ½–1 day. **No GPU work.**
**Preconditions:** working tree clean apart from `docs/01_*`, `docs/02_*` and `docs/research/plans/*`. Commit those first: `docs: add research pivot proposal, implementation plan and phase plans`.

---

## R0.1 Freeze the legacy state

```powershell
git switch main
git add docs/01_RESEARCH_PROPOSAL.md docs/02_IMPLEMENTATION_PLAN.md docs/research/plans
git commit -m "docs: add Anatomy-or-Acquisition proposal, implementation plan and phase plans"
git tag -a legacy-fetscan-demo-v1 -m "Frozen FetScan real-time demo before research pivot" b854a34
git switch -c research/anatomy-or-acquisition
```
- Tag **`b854a34`** (the audited commit), not the docs commit, so the tag is exactly the audited demo.
- `git push --tags` and pushing the branch: **ask the user first** (outward-facing).

## R0.2 Inventory → action map (authoritative list)

Use `git mv` for tracked files and a plain `Move-Item` for untracked or gitignored local artefacts. Preserve relative paths under `archive/`.

### Source code
| Path | Action | Notes |
|---|---|---|
| `src/realtime/model_loader.py` | **move** → `src/models/loading.py` | Rewrite (R0.3). Do this *before* archiving `src/realtime/`. |
| `src/realtime/{app,pipeline,capture,queues}.py` | archive → `archive/src/realtime/` | |
| `src/smoothing/` (tier1, tier2_mode_filter, `__init__`, `.gitkeep`) | archive → `archive/src/smoothing/` | |
| `src/data/synthetic_video.py` | archive → `archive/src/data/` | |
| `src/models/acam.py` | archive → `archive/src/models/` | `train.py` changes (R0.4) |
| `src/models/multitask_model.py` | **git rm** | |
| `src/eval/evaluate_test.py` | archive → `archive/src/eval/` | Replaced by `src/eval/predict.py` (R2). Keep `TEST_CSV` constant knowledge in R2. |
| `src/utils/.gitkeep` | git rm (empty dir) | |
| `app_streamlit.py` | archive → `archive/app_streamlit.py` | |
| `assets/style.css`, `assets/fetscan_uiux_fix_brief.md`, `assets/*.jpg`, `assets/Newsprint_design_prompt.md` | archive → `archive/assets/` | jpgs and the Newsprint prompt are gitignored: move locally |
| `outputs/` (gitignored) | move locally → `archive/outputs/` | |
| all `__pycache__/` | delete locally | already gitignored |

### Scripts
| Script | Action |
|---|---|
| `build_manifest.py`, `patient_split.py`, `run_sanity_checks.py`, `lr_finder.py`, `smoke_test.py` | **retain** (`run_sanity_checks.py` reads `cross_device_manifest.csv`; leave as is until R1, then point it to the new external manifests or archive it, decided in R1) |
| `build_cross_device_manifest.py` | retain until R1 supersedes it with `build_external_manifests.py`, then archive |
| `evaluate_cross_device.py` | retain until R2; update import to `src.models.loading` now; archive after R2's wrapper exists |
| `evaluate_test_acam.py` | archive (its bootstrap is re-implemented, cluster-aware, in R2 `stats.py`) |
| `derive_bboxes.py` | archive (superseded by `src/localisation/anatomy_masks.py` in R4) |
| `generate_cam_spotcheck.py` | archive |
| `build_natalia_manifest.py`, `evaluate_natalia.py`, `verify_natalia_structural.py`, `check_natalia_transitions.py` | archive → `archive/scripts/natalia/` |
| `build_natalia_showcase_clip.py`, `benchmark_pipeline.py`, `validate_pipeline_headless.py`, `render_annotated_video.py`, `save_hud_screenshot.py`, `generate_multiplane_clips.py`, `generate_sample_clips.py`, `evaluate_realistic_video.py`, `evaluate_transition_latency.py`, `evaluate_tier2.py`, `tune_tier1_smoothing.py`, `tune_tier2_mode_filter.py` | archive → `archive/scripts/` |
| `build_multitask_manifest.py`, `train_multitask.py`, `verify_no_multitask_leakage.py` | **git rm** |
| `scripts/tests/test_backbone.py`, `test_cam_localisation.py`, `test_transforms_parity.py` | **move** → `tests/` (update imports; parity test edited in R0.5) |
| `scripts/tests/test_tier1.py`, `test_tier2_mode_filter.py`, `test_acam.py` | archive → `archive/scripts/tests/` |

### Configs
| Config | Action |
|---|---|
| `convnext_tiny, tf_efficientnetv2_s, repvgg_a1, repvgg_a2, mobilenetv3_large_100, efficientnet_lite0 .yaml` | retain; add explicit new keys (R0.6) |
| `convnext_tiny_acam.yaml`, `convnext_tiny_focal.yaml`, `smoothing_tier1.yaml`, `smoothing_tier2a.yaml` | archive → `archive/configs/` |
| `multitask.yaml` | git rm |

### Data / checkpoints / logs (mostly local; partly tracked)
| Item | Action |
|---|---|
| `data/splits/multitask_{train,val}.csv` | delete locally (untracked); they encode HC18/UCL into training |
| `data/processed/{synthetic_clips,tier1_tuning,tier2a_tuning,manual_review,eval_realistic_video,eval_tier2,natalia_showcase_clips}/` | move → `archive/data/processed/` (`git mv` the tracked files inside) |
| `data/processed/bboxes.json` | move → `archive/data/processed/` |
| `data/processed/{eval_cross_device,eval_natalia,natalia_manifest.csv,natalia_excluded_*}` | retain (historical; superseded later) |
| `checkpoints/multitask/` | **delete locally + git rm tracked txt**: it was trained on HC18/UCL, and keeping it risks contaminated use |
| `checkpoints/convnext_tiny_acam`, `_focal` | keep on disk, unused |
| `checkpoints/<6 backbones>/` | **frozen**. Add `checkpoints/FROZEN.md` explaining that they are never overwritten. New training goes to `runs/`. |
| `logs/realtime/`, `logs/tuning/`, `logs/eval/*tier2*`, `*transition*`, `*realistic_video*`, `logs/multitask/`, `logs/natalia_*` | archive → `archive/logs/` |
| `logs/backbones/`, `logs/eval/evaluate_cross_device*.txt`, `logs/training/`, `logs/smoketest/`, `logs/*_training_output.txt` | retain (provenance of frozen checkpoints) |

### Docs
Follow `00_INDEX_AND_VERIFIED_FACTS.md §4` exactly. For the three superseded result docs, prepend:
```markdown
> **SUPERSEDED (2026-11, research pivot).** Numbers here use image-level bootstrap and a single seed.
> Authoritative results: `docs/research/RESULTS.md`. Kept for provenance only (tag `legacy-fetscan-demo-v1`).
```
Add `archive/README.md`:
```markdown
# archive/
Frozen legacy FetScan demo (tag `legacy-fetscan-demo-v1`, commit b854a34). Not maintained; may not run;
must not be imported from `src/`, `scripts/` or `tests/` (enforced by tests/test_no_archive_imports.py).
```

## R0.3 `src/models/loading.py` (rewrite of `model_loader.py`)

```python
@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module; backbone_name: str; device: torch.device
    img_size: int; normalize_mean: tuple[float, ...]; normalize_std: tuple[float, ...]
    preprocess_mode: str          # cfg.get("preprocess_mode", "resize")
    config: dict                  # full checkpoint config (read-only use)
    ckpt_path: Path
    def eval_transform(self) -> A.Compose: ...   # get_eval_transform(img_size, mean, std, preprocess_mode)

def load_model(ckpt_path, device=None) -> LoadedModel
```
- `torch.load(..., map_location="cpu", weights_only=False)`, then `.to(device)`. Load on CPU first (Windows CUDA teardown issue noted in `train.py`).
- **Refuse** these checkpoints with a clear error:
  - Any `model_state_dict` key that starts with `retinanet.` (multitask).
  - `cfg.get("use_acam")` is truthy (ACAM archived).
  - `cfg["train_csv"]` contains any of `multitask`, `cross_device`, `external`, `african`, `hc18`, `ucl`, `landmark` (the generalised `_assert_not_multitask_contaminated`).
- Update every import: `grep -rn "src.realtime" src scripts tests` must return nothing.

## R0.4 `src/train/train.py` minimal edits (full R5 edits come later)
- Remove `from src.models.acam import build_acam`. If `cfg.get("use_acam")`, raise `NotImplementedError("ACAM is archived (tag legacy-fetscan-demo-v1)")`.
- Add `--seed` CLI override (default: config value).
- Add the **leakage guard** (ground rule 1). It runs at start-up, before any data is read:
  ```python
  FORBIDDEN_TOKENS = ("test", "cross_device", "external", "african", "landmark", "hc18", "ucl", "multitask")
  for key in ("train_csv", "val_csv"):
      p = str(cfg[key]).lower().replace("\\", "/")
      assert not any(t in p for t in FORBIDDEN_TOKENS), f"{key}={cfg[key]} looks like a test/external set"
  ```
  Plus a patient-disjointness assertion: train ∩ val patient sets = ∅, and both ∩ `test.csv` patients = ∅.
- Smoke run: `python -m src.train.train --config configs/convnext_tiny.yaml --epochs-override 1`. To avoid touching the frozen checkpoint, the smoke run **must** pass `--out-dir runs/_smoke` (add this flag now; it overrides `checkpoint_dir` and `log_dir`).
  > The current script copies the config into `checkpoint_dir` and writes `best.pt` there. Running it unchanged against `configs/convnext_tiny.yaml` would **overwrite the frozen legacy checkpoint**. The `--out-dir` flag is mandatory for any run until R5 replaces `checkpoint_dir` with `run_root`.

## R0.5 Tests → `tests/`
- `pytest.ini`:
  ```ini
  [pytest]
  testpaths = tests
  addopts = -q
  markers =
      gpu: needs CUDA
      data: needs local datasets under data/raw (skipped if absent)
      slow: > 30 s
  ```
- `tests/conftest.py`: put the repo root on `sys.path`; provide fixtures `repo_root` and `has_data` (skip `data`-marked tests when `data/raw/fetal_planes_db/Images` is missing).
- Move and fix:
  - `test_transforms_parity.py`: drop the `prep_frame_grayscale_to_rgb` comparison (archived). Keep loader determinism, train/eval geometric parity, and add `test_resize_mode_bit_identical_to_legacy` (stub until R5 adds modes: assert `get_eval_transform()` equals the legacy pipeline output on 5 FPDB images, tolerance 0).
  - `test_backbone.py`, `test_cam_localisation.py`: update imports only.
- New in R0:
  - `tests/test_splits_frozen.py`: SHA-256 of `data/splits/{train,val,test}.csv` equals `data/splits/SPLITS_SHA256.txt`. Generate the file once in R0; it is committed and tracked.
  - `tests/test_no_archive_imports.py`: AST-scan `src/`, `scripts/` and `tests/` for `import archive`, `from archive`, `src.realtime`, `src.smoothing`, `multitask_model`, `acam`. Expect none.
  - `tests/test_no_test_leakage.py` (first version): every YAML in `configs/` and `configs/research/` passes the FORBIDDEN_TOKENS check; train/val/test patient sets are disjoint.
  - `tests/test_loading.py`: `load_model` refuses a synthetic state dict with `retinanet.` keys and a config with `use_acam: true` (build both in tmp_path, no real checkpoint needed).

## R0.6 Configs: make B0 explicit
Append to each of the 6 retained backbone YAMLs (this must not change behaviour):
```yaml
# --- research pivot (R0): explicit B0 defaults ---
preprocess_mode: resize
caliper_aug_p: 0.0
sampler: none
use_class_weights: true
style_aug: false
```
Loaders must use `cfg.get(key, default)` so the frozen checkpoints, whose embedded configs lack these keys, still load as B0.

## R0.7 Environment and requirements
- Install into `fetalplane`: `D:\miniconda\envs\fetalplane\python.exe -m pip install pytest statsmodels scikit-image`.
- `requirements.txt`: remove `streamlit`, `imageio[ffmpeg]`, `types-seaborn`; fix the corrupted line `streamlit>=1.35/implementation_plan.md`; add `pytest`, `pyarrow`, `statsmodels`, `scikit-image`, and pin `albumentations==2.0.8` and `timm==1.0.28`. Do **not** touch torch pins in `requirements-lock.txt`. Instead, append the three new packages with their installed versions (`pip show`).
- `.gitignore`: add `runs/`, `results/preds/`, `results/cache/`, `data/processed/cache/`, `archive/outputs/`. Un-ignore `results/tables/**`, `results/json/**`, and `results/figures/**/*.pdf`. Note that the blanket `*.png` ignore stays; figure PNGs are regenerated, PDFs are tracked.
  > Check: the global `*.png` rule means review figures must also be produced as PDF (R6) to be tracked.

## R0.8 Skeleton
Create with `__init__.py` where needed: `src/preprocess/`, `src/localisation/`, `src/audit/`, `scripts/research/`, `configs/research/`, `results/{tables,figures,json,preds}/`, `runs/`, `docs/research/background/`. Create `docs/research/RESEARCH_LOG.md` with a header and the R0 entry.

## R0.9 CLAUDE.md and README.md
- Replace `CLAUDE.md` with `02_IMPLEMENTATION_PLAN.md §11`, with these corrections:
  - Docs paths: `docs/research/01_RESEARCH_PROPOSAL.md`, `docs/research/02_IMPLEMENTATION_PLAN.md`, `docs/research/plans/`.
  - Environment line: `Interpreter: D:\miniconda\envs\fetalplane\python.exe (conda not on PATH in agent shells)`.
  - Commands: `python -m src.train.train ...` (module form).
  - Add hard rules: "Never write into `checkpoints/` (frozen)." "HC18/UCL patient IDs come from filename prefixes; African patient IDs are centre-qualified."
- `README.md`: rewrite as a research-study README (question, datasets with licences, how to reproduce, layout, status). Remove the demo, Streamlit, clinical language and `file:///d:/` links.
- Regenerate `project_structure_tree.txt` (or delete it and point to CLAUDE.md layout; preferred: delete, since it goes stale).

## R0.10 Acceptance (auditor runs these)
```powershell
git tag --list legacy-fetscan-demo-v1                       # exists
python -m pytest                                            # green
rg -n "streamlit|smoothing|realtime|multitask|acam" src scripts tests configs   # only the FORBIDDEN_TOKENS literal and loading.py refusal messages
python -m src.train.train --config configs/convnext_tiny.yaml --epochs-override 1 --out-dir runs/_smoke
git status --short checkpoints                              # no changes to frozen checkpoints
```
**Commits:** split R0 into reviewable commits: (1) tag and branch; (2) `git mv` to `archive/`; (3) `git rm` of multitask; (4) `loading.py` and import fixes; (5) `train.py` guard and flags; (6) tests; (7) env, gitignore, configs; (8) CLAUDE.md and README.
