# 00 — Plan Index, Conventions and Verified Facts

**Study:** *Anatomy or Acquisition?* (see `docs/01_RESEARCH_PROPOSAL.md` v1.1, `docs/02_IMPLEMENTATION_PLAN.md`)
**Written:** 2026-11-12, against commit `b854a34` (branch `main`, clean apart from the two new proposal docs)
**Audience:** Claude Code (implementer) and the phase auditor.

The proposal says *what* and *why*. `02_IMPLEMENTATION_PLAN.md` gives the repository-level *what*. The files in this folder are the **executable, step-by-step plans**, one per phase. Where they disagree with `02_IMPLEMENTATION_PLAN.md`, **these files win**. Every disagreement is listed in §3 (Errata) with the evidence behind it.

---

## 1. Plan set

| File | Phase | Gate at end | Depends on | Est. effort |
|---|---|---|---|---|
| `R0_REPO_RESTRUCTURE.md` | Repo pivot: tag, archive, remove, re-home, tests, env, CLAUDE.md/README | R0 acceptance | — | ½–1 day |
| `R1_DATA_FOUNDATION.md` | Metadata, external manifests, FP landmark linkage, sector/overlay detection, binary audit tool | **G1** | R0 | 4–6 days + ~1 h human |
| `R2_EVALUATION_CORE.md` | `predict.py`, `stats.py`, `external.py`, re-evaluation of 6 frozen checkpoints | **G2** | R0, R1 §manifests | 3–4 days |
| `R3_CONFOUNDING_RELIANCE_AUDIT.md` | E1–E7 on existing checkpoints | Box 1 numbers ready | R1, R2 | 5–7 days (+ ~4 GPU-h for E4) |
| `R4_LOCALISATION.md` | Anatomy masks, saliency methods, EPG metrics, sanity check, association | **G3** | R1, R2 | 5–7 days (+ GPU) |
| `R5_AFP_TRAINING.md` | Pre-registration, preprocessing modes, caliper aug, sampler, style aug, 24-run matrix, FM probe, primary endpoint | **G4** before training | R1–R4 | 2–3 weeks (≈24 GPU-h) |
| `R6_REPORTING_AND_REVIEW.md` | RESULTS.md auto-render, figures, data cards, REPRODUCE.md, JAPI review artefacts | Definition of done | R3 (Box 1), R4, R5 | 1–2 weeks, partly parallel |

**Critical path to the JAPI review's Box 1:** R0 → R1 (metadata + manifests only) → R2 → R3 (E1, E2, E6 and the external rejection numbers). The overlay audit, E4 and E5 can follow afterwards.

```
R0 ──► R1 ──► R2 ──► R3 ──► (Box 1) ──► JAPI draft (R6 §review)
        │      │      └──► R4 ──► G3 ──► PREREG (G4) ──► R5 ──► R6
        └─ G1 (overlay audit) gates E4b/E5a/E5d/A2 emphasis
```

---

## 2. Conventions that apply to every phase

### 2.1 Environment
- Interpreter: `D:\miniconda\envs\fetalplane\python.exe` (Python 3.11.15). `conda` is **not on PATH** in the Claude Code shell. Call the interpreter by absolute path, or `& D:\miniconda\Scripts\conda.exe run -n fetalplane ...` from PowerShell.
- Installed and relevant: torch 2.11.0+cu128 (CUDA available), timm 1.0.28, albumentations **2.0.8**, opencv 5.0.0, scikit-learn 1.9.0, pandas 2.3.3, numpy 2.4.4, pyarrow 25.0.1, scipy 1.17.1, matplotlib 3.11.1, seaborn 0.13.2, tensorboard 2.21.0, grad-cam (version not exposed).
- **Missing; install in R0:** `pytest`, `statsmodels`, `scikit-image`. Optional and only if G1 needs them: `easyocr`.
- albumentations 2.0.x API: `RandomGamma(gamma_limit=...)` (renamed `gamma_range` only in 2.2.0), `Downscale(scale_range=(lo, hi))` (tuple). **Pin `albumentations==2.0.8`** in `requirements.txt`, and add a test that constructs every augmentation used, so an upgrade fails loudly.
- Run everything from the repo root. Use `python -m <module>` for `src/...` entry points; legacy scripts use `sys.path.insert(0, ".")`.

### 2.2 Directory contract (post-R0)
```
src/data        datasets, transforms, labels (CANONICAL_CLASSES, collapse maps)
src/preprocess  sector.py, overlays.py, caliper_aug.py, pipeline.py
src/models      backbone.py, loading.py, gradcam.py (→ saliency API), cam_localisation.py
src/localisation anatomy_masks.py, metrics.py
src/audit       metadata.py, probes.py, perturb.py (R3 helpers)
src/eval        predict.py, stats.py, external.py, metrics_utils.py
src/train       train.py, losses.py, sampler.py
scripts/research  one script per experiment (thin CLIs over src/)
configs/        B0 baselines (legacy, made explicit)   configs/research/  arm × backbone
data/processed/{metadata,external,landmarks,audit,cache}/
results/preds/<run_id>/<dataset>.parquet   results/tables/   results/figures/   results/json/
runs/<run_id>/  (gitignored training outputs)
docs/research/  proposal, implementation plan, plans/, RESEARCH_LOG.md, GATE_*.md, RESULTS.md
archive/        frozen legacy (tag legacy-fetscan-demo-v1)
tests/          pytest suite
```

### 2.3 Identifiers
- `image_id`: filename stem for FPDB (`Patient00168_Plane2_1_of_2`); `hc18/<stem>`, `ucl/<anatomy>/<stem>`, `african/<stem>` for externals. It is unique across datasets.
- `patient_id`: a string, always **dataset-qualified**: `fpdb_<Patient_num>`, `hc18_<prefix>`, `ucl_<prefix>`, `afr_<ISO3>_<Patient_num>`. Never an int, so numbers cannot collide across datasets.
- `run_id`: `<arm>_<backbone_short>_s<seed>` for new runs (e.g. `afp_convnext_s11`), and `legacy_<backbone_short>_s42` for the six frozen checkpoints.
- Bootstrap seed: **20261109**. Replicates: **2,000**.

### 2.4 Labels
- 8 canonical classes (order fixed in `src/data/dataset.py::CANONICAL_CLASSES`).
- Collapsed taxonomy: `{Head, Abdomen, Femur, Thorax, Cervix, Other}`. `Brain_Trans_*` → Head; `Fetal_abdomen` → Abdomen; `Fetal_femur` → Femur; `Fetal_thorax` → Thorax; `Maternal_cervix` → Cervix; `Other` → Other.
- Collapsed probability = **sum** of member-class probabilities (Head = p0+p1+p2). Full-argmax collapsed prediction = collapse(argmax over 8). Restricted prediction = argmax of collapsed probs over the classes *present in that dataset*.
  > Note: collapse(argmax₈) can differ from argmax of summed collapsed probabilities. The parquet stores both `pred_full` (collapse of 8-way argmax; matches legacy scoring) and `pred_full_summed` (argmax over the 6 summed collapsed probs). G2 reproduction uses `pred_full`.

### 2.5 Every task ends with
1. Tests green (`python -m pytest -q`).
2. An entry appended to `docs/research/RESEARCH_LOG.md`:
   ```
   ## 2026-MM-DD — <phase>.<task> — <title>
   commit: <hash>   run_ids: <...>   outputs: <paths>
   outcome: <1–3 lines, numbers with CIs where relevant>
   deviations: <none | what and why>
   ```
3. No hand-typed numbers in any doc. Numbers come from `results/json/*.json` via `scripts/research/render_results.py` (R6). Until R6 exists, phases write their JSON, and the log entry quotes it verbatim.

---

## 3. Verified facts and errata to `02_IMPLEMENTATION_PLAN.md`

All facts below were measured on 2026-11-12 from local files. Re-verify them programmatically in the phase that depends on them; each plan names the assertion.

### 3.1 FETAL_PLANES_DB splits (`data/splits/*.csv`)
- Columns: `image_path, patient_id, plane_label, brain_subplane_raw, source_machine, operator, original_split_flag`. Paths use backslashes.
- Our split = FPDB's official split: `test` ⇔ `original_split_flag == 0`. Train 6,106 / val 1,023 / test 5,271 images. Patients: 772 / 124 / 896.
- `multitask_train.csv` and `multitask_val.csv` also live in `data/splits/`. They are **REMOVE** items (R0). The SHA-256 freeze covers only `train/val/test.csv`.
- Confounding (reproduced exactly):
  | | Aloka | Voluson E6 | Voluson S10 | Other |
  |---|---|---|---|---|
  | train+val n / %Other | 3,377 / 65.1% | 3,456 / 13.6% | 11 / 0.0% | 285 / 4.2% |
  | test n / %Other | 1,737 / 52.0% | 2,406 / 8.8% | 1,071 / 51.6% | 57 / 17.5% |
- Operators: Op.1 (2,792) and Op.2 (2,435) are E6-only; Op.3 (3,560) is Aloka-only; operator "Other" spans all machines (including all S10).
- **Image size almost identifies the machine.** Modal sizes: Aloka 647×381, E6 961×663, S10 1137×787, Other 745×559. Consequences: (a) the E2 geometry-only model will be strong; (b) B0 `resize` to a square distorts aspect ratio *differently per device*, which is itself a device signature. Both must be stated in R3.
- FPDB frames carry **burned-in device names** (e.g. "Voluson E6" at top-left), acquisition-settings text (top-right), a depth ruler (left edge) and orientation marks. The fan sector usually runs **off the frame edges**, so sector segmentation must handle border-clipped sectors (R1 §4).

### 3.2 Landmark benchmark FP subset (`data/raw/ucl_hc18/{annotations,images}/FP/`)
- **All 3,090 FP images are pixel-identical to the FPDB originals** (same filename, same shape, max |Δ| = 0). *Erratum:* `02 §3.3` assumed cropping and template-matching registration. Registration is **not needed**. Keep an identity assertion (shape plus SHA-256 or exact array equality) instead. Landmark coordinates are in native FPDB pixel space.
- Every FP filename matches an FPDB filename. Distribution over **our** splits:
  | Anatomy | our test | our val | our train | FPDB label |
  |---|---|---|---|---|
  | Head | **765** | 126 | 746 | 100% `Brain_Trans_thalamic` |
  | Abdomen | **350** | 49 | 294 | `Fetal_abdomen` |
  | Femur | **371** | 63 | 326 | `Fetal_femur` |
  → **1,486 test images** with expert landmarks. Test machines: Head E6 484 / Aloka 143 / S10 120 / Other 18; Abdomen 164/91/91/4; Femur 194/96/63/18.
- The FP `Split` column does **not** align with our splits. Ignore it.
- Head landmarks exist only for the trans-thalamic plane. Head EPG therefore covers TT only; the TV and TC sub-planes get only GT-free metrics.
- Column semantics check (150-row sample): BPD is shorter than OFD in 100% of rows (median ratio 0.80), and the BPD–OFD angle lies in 70–110° in 100% of rows. *Erratum:* no "BPD under `ofd_` prefix" typo in this release. Keep the automatic check anyway (R4).
- Abdomen: TAD and APAD are not consistently ordered by length (TAD shorter in ~37%). Use the longer axis as the ellipse major axis.
- About 2.7% (4/150) of FP Head rows have a landmark outside image bounds. Clip to bounds and flag `lm_clipped=True`.
- Femur CSVs use column `Patient`; Head and Abdomen use `SubjectID`. Column order differs between files. Always select by name.
- `Algo == "AlokaFit"` marks Aloka; empty means GE.
- **Development subset:** the 238 **val-split** landmark images are used to develop and debug mask and EPG code, so test images are not inspected during development.

### 3.3 HC18 and UCL
- *Erratum:* `SubjectID` / `Patient` columns are **empty for every row** in HC18 and UCL. `02 §3.2` assumed they could be used.
  - HC18 patient = numeric prefix of the filename (`783_4HC.png` → `783`). This gives **806 unique**, matching the documented 806 subjects.
  - UCL patient = numeric prefix (`002_3HC.jpeg` → `002`). There are 47/45/47 unique prefixes per anatomy, and the documented total is 51 pregnancies. Whether the same prefix across anatomies is the same pregnancy is **undocumented**. Treat `ucl_<prefix>` as one cluster across anatomies. This is the conservative choice: merging clusters widens CIs, never narrows them.
- Counts: HC18 999 Head; UCL 159 Head, 130 Abdomen, 135 Femur (total external landmark images 1,423). These match `cross_device_manifest.csv`.
- The current `cross_device_manifest.csv` has columns `image_path, plane_label, source_subset, is_collapsed_label`, with **no patient_id**. It is superseded in R1.

### 3.4 African planes (`data/raw/african_planes/`)
- 450 PNGs in `Algeria/ Egypt/ Ghana/ Malawi/ Uganda/` + `African_planes_database.csv` (`Patient_num, Plane, Train, Center, Filename`). Filenames have no extension in the CSV.
- Plane counts: brain 125, abdomen 125, femur 125, **thorax 75**. Thorax is absent for **Ghana and Uganda** (confirmed by the source paper).
- `Patient_num` **repeats across centres** (e.g. `patient001_DZA` and `patient001_EGY` are different people). Patient ID must be centre-qualified: 127 unique `(Center, Patient_num)`. The paper states 25 patients per centre. Malawi shows 27 IDs because two IDs (14, 30) have a single image each. Report as-is.
- `Train` column = authors' 50/50 patient-disjoint split (verified disjoint). **We use all 450 images as an external test set.** The optional few-shot analysis (proposal §5.4) fine-tunes on `Train==1` only, and must then report on `Train==0` only.
- Image geometry is fixed per centre: DZA 500×430 L, **EGY 400×450 RGB**, GHA 500×500 L, MWI 500×400 L, UGA 600×500 L. The existing loader (`cv2.IMREAD_GRAYSCALE`) handles RGB correctly.
- **The authors cropped images to the field of view, excluding vendor logos and control indicators** (Sendra-Balcells 2023 §2.1.3). African images therefore have little or no out-of-sector UI. *Implication for interpretation:* part of the B0 external drop may come from "missing UI / different framing" rather than anatomy or texture. A1 (sector crop) removes that difference, and this must be discussed in R3 and R5.
- Devices (paper Table 1): Algeria Voluson S8 (GE), Egypt Voluson P8 (GE), Malawi Mindray DC-N2, Uganda Siemens ACUSON X600, Ghana EDAN DUS 60. Licence CC BY 4.0, DOI 10.5281/zenodo.7540448.

### 3.5 Caliper-pair candidates (E5d)
- In the FPDB test split, 943 `(patient, PlaneN)` groups contain ≥2 images (3,904 images). `PlaneN` is a per-patient acquisition index, **not** a class. R1 must check label homogeneity within groups before proposing pairs.

### 3.6 Open items that cannot be verified from here
- FetalCLIP weight licence (GitHub `BioMedIA-MBZUAI/FetalCLIP`, Hugging Face card). The licence was not visible in search results. Read the LICENSE file before R5 §7.
- JAPI review-article word and reference limits: no limits were found on public pages, and the instruction PDFs are undated. This is a human task (02 §13.5).

---

## 4. Legacy documentation disposition (executed in R0)

| Current path | Action | Destination |
|---|---|---|
| `docs/instructions/00–09_*.md` | archive | `archive/docs/instructions/` |
| `docs/phases/{demo1,phase_03..phase_08}/` (incl. screenshots, txt outputs) | archive | `archive/docs/phases/` |
| `docs/FetScan_Final_Audit_Report.md`, `NATALIA_INTEGRATION.md`, `CAM_LOCALISATION_SPOTCHECK.md` | archive | `archive/docs/` |
| `docs/EVAL_REPORT.md`, `EXPERIMENTS.md`, `CROSS_DEVICE_BACKBONE_COMPARISON.md` | archive + "superseded" header | `archive/docs/` |
| `docs/literature_survey_and_gap_analysis.md` | keep (background) | `docs/research/background/` |
| `docs/01_RESEARCH_PROPOSAL.md`, `docs/02_IMPLEMENTATION_PLAN.md` | move | `docs/research/` |
| `docs/research/plans/*` (this set) | new | — |
| `docs/A multicentre ... biometry.pdf` (gitignored) | keep locally | `docs/research/background/` (still gitignored) |
| `docs/realistic_ultrasound.jpg` | archive | `archive/docs/` |

**The replacements for the archived phase and instruction docs are this plan set** (`R0`–`R6`), plus `docs/research/RESEARCH_LOG.md` and the `GATE_*.md` decision records.
