# Reproducing every table and figure (R6 D.2)

**Machine.** Windows, one 8 GB GPU (RTX 4060 laptop).

**Rules for running commands:**
- Run from the repository root. In agent shells `conda` is not on PATH, so call the interpreter by absolute path: `D:\miniconda\envs\fetalplane\python.exe`, written below as `python`.
- Run one GPU job at a time. Each process that imports torch commits about 2 GB of virtual memory.
- Every step skips outputs that already exist.

**Notation.** In the tables below, GPU = needs the GPU; Human = needs human input.

## 0. Environment

```
python -m pip install -r requirements.txt        # pinned versions: requirements-lock.txt
python -m pytest                                  # all tests must pass
```

## 1. Data (R1) — CPU, about 30 min; Gate G1 needs the human audit

| step | command | output |
|---|---|---|
| External manifests | `python scripts/build_external_manifests.py` | `data/processed/external/*.csv`, `results/tables/R1_external_counts.md` |
| Landmark linkage | `python scripts/link_fp_landmarks.py` | `data/processed/landmarks/*.parquet` |
| Sector consensus | `python scripts/build_sector_consensus.py` | `data/processed/masks/consensus/` |
| Metadata and masks | `python scripts/build_acquisition_metadata.py` | `data/processed/metadata/`, `data/processed/masks/` |
| Binary audit sheets (**Human**: rate the sheets) | `python scripts/audit_sheets.py --build-sample` then `--render ...` | `data/processed/audit/` |
| Gate G1 | `python scripts/research/g1_report.py` | `docs/research/GATE_G1.md` |

## 2. Evaluation core (R2) — GPU, about 20 min

```
python scripts/research/r2_reevaluate_existing.py      # 6 frozen checkpoints x 5 datasets -> results/preds/legacy_*; GATE_G2.md
```

## 3. Confounding and reliance audit (R3)

| step | command | time |
|---|---|---|
| E1, E2, E6 (CPU) | `python scripts/research/e1_confounding_tables.py`; `e2_metadata_only_model.py`; `e6_stratified_performance.py` | minutes |
| Box 1 | `python scripts/research/render_box1.py` | seconds |
| E3 (GPU feature extraction, then CPU probes) | `python scripts/research/e3_feature_probes.py --extract --probe --score --jobs 6` | about 40 min |
| E5 (GPU) | `python scripts/research/e5_reliance_perturbation.py --predict --score` | about 1.5 h |
| E4 (CPU cache, then GPU training) | `python scripts/research/build_cache.py --mode nonanat_only overlay_only sector_shape_only`, then `python scripts/research/e4_availability_train.py --train --evaluate` | about 2 h |

## 4. Localisation (R4) — GPU

| step | command | time |
|---|---|---|
| Development, val only | `python scripts/research/r4_localisation.py --dev` | about 10 min |
| Gate G3 (needs a clean, committed `src/`) | `python scripts/research/r4_sanity_randomisation.py` | about 50 min |
| Locked run | `python scripts/research/r4_localisation.py --run` | about 75 min |
| Summaries (CPU) | `python scripts/research/r4_summarise.py` | about 1 min |
| Association (CPU) | `python scripts/research/r4_association.py` | about 13 min |

## 5. Anatomy-focused training (R5) — GPU, about 24–36 h; Gate G4 needs the PI

| step | command | notes |
|---|---|---|
| Letterbox caches (CPU) | `python scripts/research/build_cache.py --mode sector_letterbox sector_letterbox_inpaint` | about 5 min |
| Smoke runs | `python scripts/research/r5_run_matrix.py --smoke --only <run ids>` | 1 epoch each; allowed before G4 |
| **Human:** lodge `PREREGISTRATION.md` on OSF | step-by-step: `docs/research/OSF_REGISTRATION_GUIDE.md` | Paste the link into `GATE_G4.md` as `OSF: https://...` and commit |
| 24-run matrix + sensitivity run | `python scripts/research/r5_run_matrix.py` | Refuses to start without G4; 24 runs, then `afp_cw_convnext_s11` |
| Secondary analyses | `python scripts/research/r5_evaluate_matrix.py --predict --saliency --probes --score` | Includes P3b on R5 models, rejection vs confusion, sensitivity run |
| FM probe: install | `D:\miniconda\envsetalplane\python.exe -m pip install open_clip_torch` | The FetalCLIP repository pins open-clip-torch 2.26.1 with torch 2.0; record the version actually installed in `env.txt` |
| FM probe: model files | `FetalCLIP_config.json` from github.com/BioMedIA-MBZUAI/FetalCLIP; `FetalCLIP_weights.pt` (1.7 GB) from huggingface.co/numansaeed/fetalclip-model, both into `data/external_models/fetalclip/` | Licence CC BY-NC 4.0 (`DATA_CARDS.md`) |
| FM probe | `python scripts/research/r5_fm_linear_probe.py --extract --probe --score` | GPU for `--extract` (about 30 k images, ViT-L/14, fp16); refuses before G4 |
| Primary endpoint | `python scripts/research/r5_primary_endpoint.py` | Pre-registered; never edited |

## 6. Reporting (R6) — CPU

| step | command |
|---|---|
| Results | `python scripts/research/render_results.py` (fails on any missing number; `--allow-missing` gives a draft) |
| Box 1 | `python scripts/research/render_box1.py` |
| Review figures | `python scripts/research/make_review_figures.py` |
| Study figures 3, 7, 8 + index | `python scripts/research/make_figures.py` (writes `results/figures/STUDY_FIGURES.md`; run after R5 to add the R5 arms) |
| Orphan check | `python scripts/research/check_orphan_json.py` (every `results/json/*.json` is used by RESULTS.md or a figure) |
| Evidence table | `python scripts/research/check_evidence_table.py` |

Study figures 1, 2, 4, 5 and 6 are written by the experiment scripts above, and 3, 7 and 8 by `make_figures.py`. PDFs are tracked; PNGs are regenerated. `results/figures/STUDY_FIGURES.md` maps figure numbers to files.
