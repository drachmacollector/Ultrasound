# R1 — Data Foundation (ends at Gate G1)

**Goal:** every downstream experiment reads from a small set of validated, documented tables: acquisition metadata, external manifests with patient IDs, landmark ground truth in model-independent pixel space, and per-image sector and overlay masks. The overlay detector is validated against a short binary human audit.
**Effort:** 4–6 working days plus about 1 hour of human y/n clicking.
**Inputs:** `data/splits/{train,val,test}.csv` (frozen), `data/raw/fetal_planes_db/`, `data/raw/ucl_hc18/`, `data/raw/african_planes/`.
**Reads first:** `00_INDEX_AND_VERIFIED_FACTS.md §3`. Every fact used here is asserted in code.

Order inside R1: **R1.1 → R1.2 → R1.3** (no CV needed; unblocks R2) **→ R1.4 → R1.5 → R1.6 → G1**.

---

## R1.1 Shared label and path utilities — `src/data/labels.py`, `src/data/paths.py` (NEW)

`labels.py`
```python
CANONICAL_CLASSES  # re-exported from dataset.py (single source of truth stays there)
COLLAPSED = ["Head", "Abdomen", "Femur", "Thorax", "Cervix", "Other"]
FINE_TO_COLLAPSED: dict[str, str]          # Brain_Trans_* → Head, Fetal_abdomen → Abdomen, ...
COLLAPSE_MATRIX: np.ndarray                # shape (8, 6), 0/1; probs_collapsed = probs @ COLLAPSE_MATRIX
def collapse_label(fine: str) -> str
def collapse_probs(p8: np.ndarray) -> np.ndarray   # (N,8) → (N,6)
```
`paths.py`
```python
def norm_path(p: str) -> Path        # Path(p.replace("\\", "/")), relative to repo root
def fpdb_image_id(path) -> str       # filename stem
FPDB_STEM_RE = re.compile(r"Patient(\d+)_Plane(\d+)_(\d+)_of_(\d+)")
def parse_fpdb_stem(stem) -> dict    # patient_num, plane_n, idx, of_k
```
Tests (`tests/test_labels.py`): the collapse matrix rows sum to 1; `collapse_probs` preserves row sums; every CANONICAL class maps; `norm_path` handles mixed separators.

## R1.2 External manifests — `scripts/build_external_manifests.py` (NEW; supersedes `build_cross_device_manifest.py`)

Output: `data/processed/external/{hc18,ucl,african}.csv` plus `external_all.csv` (concatenation). Columns:

| column | HC18 | UCL | African |
|---|---|---|---|
| `image_path` | `data/raw/ucl_hc18/images/HC18/Head/<f>` | `.../UCL/<Anatomy>/<f>` | `data/raw/african_planes/<Center>/<Filename>.png` |
| `image_id` | `hc18/<stem>` | `ucl/<anatomy>/<stem>` | `african/<Filename>` |
| `dataset` | `hc18` | `ucl` | `african` |
| `patient_id` | `hc18_<int(prefix)>` | `ucl_<int(prefix)>` | `afr_<ISO3>_<Patient_num>` |
| `centre` | `Radboud` | `UCLH` | `Algeria`… |
| `device` | `GE Voluson E8/730` (dataset-level; not per image) | `GE Voluson (unspecified)` | per centre (00 §3.4) |
| `plane_label_collapsed` | `Head` | `Head/Abdomen/Femur` | brain→Head, abdomen→Abdomen, femur→Femur, thorax→Thorax |
| `orig_split` | from CSV `Split` | from CSV `Split` | `Train` (0/1) |
| `width,height,channels` | read from file | | |

- Patient prefix regex: `^(\d+)_` on the image name. Assert HC18 unique patients == **806**, UCL unique patients ≤ 51 (print the number), African unique patients == **127**, images == 450, thorax == 75 and absent for Ghana and Uganda.
- **Forbidden-path assertion** (kept from `CrossDeviceDataset`): no `image_path` contains a `FP` or `MULTICENTRE` path component.
- Assert every file exists and opens. Print per-dataset × class counts and write them to `results/tables/R1_external_counts.md`.
- `data/raw/african_planes/README.md`: source, DOI 10.5281/zenodo.7540448, CC BY 4.0, citation (Sendra-Balcells et al., *Sci Rep* 2023;13:2728), device table, the note that the authors cropped to the field of view, and the note that `Train` is the authors' split.
- After this exists: update `src/data/dataset.py` so that `CrossDeviceDataset` becomes **`ExternalDataset(csv_path, transform, preprocess=None)`**, returning a dict sample `{"image", "image_id", "patient_id", "dataset", "label_collapsed"}`. Also give `FocalPlanesDataset` an optional `return_meta=True` that returns `{"image", "label", "image_id", "patient_id", "source_machine", "operator"}`. Default return types stay unchanged, so `train.py` and legacy code keep working. Remove `multitask_collate_fn` and the bbox branch.
- Archive `scripts/build_cross_device_manifest.py`, and either point `run_sanity_checks.py` at the new manifests or archive it. Prefer pointing: it gives useful sanity grids.

Tests (`tests/test_external_manifests.py`, marker `data`): column schema; patient-ID formats; no ID collision across datasets; counts as asserted above.

## R1.3 Landmark ground truth — `scripts/link_fp_landmarks.py` (NEW)

This is the only code allowed to read `data/raw/ucl_hc18/*/FP/`. It never produces anything a training loader can consume.

1. Load `annotations/FP/{Head,Abdomen,Femur}.csv`, `HC18/Head.csv`, `UCL/{Head,Abdomen,Femur}.csv`. Select columns **by name** (orders differ; Femur uses `Patient`, the others `SubjectID`).
2. FP ↔ FPDB join on filename stem. Assert 3,090/3,090 matched.
3. **Identity check (replaces registration; see 00 §3.2):** for every FP image, assert `shape` equal and `np.array_equal(FP, FPDB)`. Store `reg_ok = True`, `dx = dy = 0`, `scale = 1`. Any failure → `reg_ok = False`, excluded and listed in the log. Expected failures: 0.
4. Attach our split (`train/val/test`) and `source_machine` from the split CSVs. **Keep only `val` and `test` rows** in the output; train-split landmarks are dropped to avoid any temptation to train on them.
5. Geometry and QC columns per row:
   - Head: `ofd_len, bpd_len, axis_angle_deg, bpd_mid_offset_rel` (|mid(bpd) − mid(ofd)| / (ofd_len/2)).
   - Abdomen: `tad_len, apad_len, axis_angle_deg, major = argmax(len)`.
   - Femur: `fl_len`.
   - `lm_clipped` (any coordinate outside the image, clipped to bounds).
   - `qc_flag` if Head `bpd_len ≥ ofd_len`, angle ∉ [70°, 110°], or `bpd_mid_offset_rel > 0.15`.
   - Log the share of rows with `qc_flag`. **Assert < 5%** for Head; otherwise stop and report (column-semantics problem).
6. Outputs:
   - `data/processed/landmarks/fpdb_landmarks.parquet`: `image_id, split, anatomy, source_machine, algo, px_to_mm, <landmark cols>, <qc cols>, reg_ok`.
   - `data/processed/landmarks/external_landmarks.parquet`: same schema for HC18/UCL (`split = "external"`, `image_id` as in R1.2).
   - `results/tables/R1_landmark_linkage.md`: matched counts by anatomy × split × machine (expected test: 765 / 350 / 371).

Tests (`tests/test_landmark_linkage.py`, marker `data`): output row counts per anatomy/split; no `train` rows; identity check passes on a 20-image sample.

## R1.4 Sector segmentation — `src/preprocess/sector.py` (NEW)

```python
@dataclass
class Sector:
    mask: np.ndarray        # uint8 {0,1}, HxW
    bbox: tuple[int,int,int,int]   # x0,y0,x1,y1 inclusive-exclusive
    area_frac: float
    shape_class: str        # "fan" | "linear" | "full" (sector ≥ 97% of frame, e.g. cropped African images)
    touches_border: tuple[bool,bool,bool,bool]  # top,right,bottom,left
def segment_sector(gray: np.ndarray, *, thr: int | None = None) -> Sector
```
Algorithm. It must handle the FPDB case where the fan runs off the frame edges, and the case of burned-in text and rulers next to the fan.
1. `g = medianBlur(gray, 5)`. Threshold `g > t`, where `t = thr or max(5, 0.5 * Otsu(g))`. Fan interiors contain dark fluid, so a low threshold plus closing is needed.
2. `morphologyEx(OPEN, ellipse k=5)` removes thin text strokes and ruler ticks. Then `CLOSE` with an ellipse whose `k = max(15, round(0.03 * min(H, W)))` fills dark fluid regions.
3. Largest connected component → `convexHull` → `fillPoly`.
4. **Text detachment check:** if the hull contains > 2% of pixels that were *bright, thin components* outside the pre-hull component (top-hat response), recompute with a larger opening kernel (k=9). Text attached to the fan by closing is the known failure mode.
5. `shape_class`: `full` if area_frac ≥ 0.97; `fan` if the top-edge width of the mask is < 60% of the bottom-edge width; otherwise `linear`.
6. **Optional consensus prior (enable if the G1 sector-check fails > 10%):** per `(source_machine, W, H)` group, build the pixelwise-majority mask over **train-split** images. The final mask is `per-image ∩ dilate(consensus, 7)`. This is label-free and train-only. External sets use per-image masks only.

Tests (`tests/test_sector.py`):
- Synthetic fan (`cv2.ellipse` sector, 60° wedge) with speckle noise, dark elliptical "fluid" holes, a text string drawn outside, and a ruler of 1-px ticks at the left edge. Require mask IoU ≥ 0.95 with the true wedge, and text pixels excluded.
- The same fan shifted so it is clipped by the bottom and side edges: IoU ≥ 0.95.
- A full-frame image (African-like): `shape_class == "full"`.
- A linear rectangle: `shape_class == "linear"`.

## R1.5 Overlay detection and inpainting — `src/preprocess/overlays.py` (NEW)

```python
@dataclass
class Overlays:
    text_mask: np.ndarray       # out-of-sector UI/text components
    marker_mask: np.ndarray     # in-sector calipers / '+' / 'x' / dotted lines / orientation marks
    caliper_score: int          # number of detected caliper-like components in-sector
    text_score: int             # number of out-of-sector components
    any_overlay_mask: np.ndarray
def detect_overlays(gray, sector: Sector, params: OverlayParams = DEFAULT) -> Overlays
def inpaint_overlays(gray, mask, dilate_px=2, radius=3) -> np.ndarray   # cv2.INPAINT_TELEA
```
- **Out-of-sector (text/UI):** pixels outside `dilate(sector.mask, 3)` with `gray > 40`, as connected components with area < 2% of the image. `text_score` = the number of components. EasyOCR is **not** used unless G1 shows image-level text recall < 0.8. Thresholds are set on train audit images only.
- **In-sector markers:**
  1. White top-hat (ellipse k=7) on `gray * sector.mask`; threshold at the `params.tophat_pct` percentile (initial 99.5) **and** absolute `≥ params.min_marker_intensity` (initial 200). Calipers are near-white overlays; speckle is rarely saturated.
  2. Cross templates: synthetic '+' and '×' at sizes {7, 9, 13, 17, 21, 25} px, 1–2 px stroke. `matchTemplate(TM_CCOEFF_NORMED)` on the top-hat image, NMS radius = size, accept ≥ `params.ncc_thr` (initial 0.6).
  3. Dotted lines: small bright components (area 2–30 px). Find ≥ 4 roughly collinear components (RANSAC line, residual ≤ 2 px, spacing coefficient of variation < 0.35).
  4. `caliper_score` = number of accepted crosses + number of dotted-line runs.
- `OverlayParams` is a frozen dataclass serialised to `data/processed/audit/overlay_params.json` after tuning (R1.6). Tuning may only use `overlay_audit_train.csv`.

Tests (`tests/test_overlays.py`): a synthetic fan, three '+' marks (sizes 9/15/21), one dotted line of 6 dots, and text outside the fan. The detector must find all 3 crosses and the dotted line, with no marker detected in an equally sized speckle-only image (false-positive check). After inpainting, mean |marker pixel − local background median| ≤ 10 grey levels. Inpainting leaves pixels farther than `dilate_px + radius` from the mask unchanged (exact equality).

## R1.6 Acquisition metadata — `scripts/build_acquisition_metadata.py` (NEW) + `src/audit/metadata.py`

One row per FPDB image (all splits) → `data/processed/metadata/acquisition_metadata.parquet`:
`image_id, patient_id (fpdb_<n>), split, plane_label, plane_collapsed, source_machine, operator, plane_n, idx, of_k, width, height, aspect, mean_intensity, std_intensity, frac_nonzero, sector_area_frac, sector_bbox_x0..y1, sector_shape_class, sector_touches_{t,r,b,l}, text_score, caliper_score, overlay_area_frac, md5`.
- The same feature set (minus machine and operator) is computed for externals → `external_metadata.parquet`. It is used for the stratified descriptive analysis and for R3 E2 transfer.
- Masks themselves: `data/processed/masks/<dataset>/<image_id>.npz` (`sector`, `text`, `marker`; `np.savez_compressed`, uint8). Image IDs containing `/` map to subfolders.
- Runtime: ~17k images, all CPU. Use `concurrent.futures.ProcessPoolExecutor(max_workers=6)`. Must be resumable (skip existing npz).
- Sanity asserts: split counts 6,106 / 1,023 / 5,271; machine × split counts as in 00 §3.1; Other-share table reproduces 0.651 / 0.136 / 0.516 to 3 dp.
- E1 tables are produced in **R3** (they need `stats.py` for CIs). R1 only prints raw counts.

## R1.7 Binary audit tool — `scripts/overlay_audit_tool.py` (NEW)

Keyboard-driven OpenCV tool; binary answers only. Three modes:

| Mode | Shows | Keys | Output |
|---|---|---|---|
| `--mode overlay --split train\|test` | one raw image, letterboxed to fit 1400×900, **no detector output** | `c`/`x` calipers yes/no → `t`/`n` text yes/no → (unless `--no-sector-check`) the auto sector outline in green, `s`/`f` sector OK / failed | `data/processed/audit/overlay_audit_{train,test}.csv`: `image_id, calipers, text, sector_ok, timestamp, rater` |
| `--mode pairs --pairs <csv>` | two images side by side (sector-masked) | `y`/`n` same scan, one with and one without calipers | `caliper_pairs_confirmed.csv` |
| `--mode landmarks` | 15 images (5 per anatomy, from **val** split) with landmarks drawn as dots and the derived mask outline (R4 mask code, or ellipse preview if R4 not ready) | `y`/`n` dots sit on head/abdomen/femur | `landmark_glance.csv` |

- Common keys: `b` back one, `q` save and quit. Progress is saved after **every** answer (append + fsync), so the tool resumes at the first unanswered item. Window title shows `i/N` and the current question.
- Sample construction (`--build-sample`, run once, seed 20261109, sample CSV committed):
  - Stratify by `source_machine` (4) × `plane_collapsed` (6) on each split separately. Allocate 150 per split: first `min(4, n_cell)` per non-empty cell, then the remainder proportional to `sqrt(n_cell)`. This oversamples rare cells without exhausting them.
  - One image per patient per cell where possible, to spread over patients.
  - Train S10 has only 11 images: take all available, up to the allocation.
- Wilson CIs for prevalence come from R2 `stats.py`; the G1 report runs after R2.1 exists, or uses `statsmodels.stats.proportion.proportion_confint(method="wilson")` directly.
- The tool must run on Windows with `cv2.imshow` on the main thread. **Human time budget: ≤ 60 minutes in total.**

## R1.8 Caliper-pair discovery — `scripts/find_caliper_pairs.py` (NEW)

1. FPDB **test** split. Group by `(patient_id, plane_n)`. Keep groups with ≥ 2 images **and a single `plane_label`** (report how many groups are label-heterogeneous).
2. For each within-group image pair with equal size: SSIM (`skimage.metrics.structural_similarity`) on sector-masked, 0.5×-downscaled grey images, plus `|caliper_score_a − caliper_score_b|`.
3. Candidates: SSIM ≥ 0.90 and exactly one image with `caliper_score ≥ 2` and the other `caliper_score == 0`. Write `data/processed/audit/caliper_pairs.csv` (`image_id_a, image_id_b, ssim, cs_a, cs_b, plane_label, source_machine`).
4. If candidates > 50, sample 50 (seed 20261109, stratified by machine) for human confirmation.

## R1.9 Gate G1 — `scripts/research/g1_report.py` → `docs/research/GATE_G1.md`

The report is auto-generated, and the human appends the decision lines at the end.
1. Manual overlay prevalence (test audit): calipers and text by machine × collapsed class, with Wilson 95% CIs.
2. Detector vs human, image level, **test audit**: calipers precision/recall/F1 (`caliper_score ≥ k`, with k chosen on train audit) and text precision/recall; train-audit numbers shown separately and labelled "tuning".
3. Sector check: share `sector_ok = no` by machine.
4. Caliper pairs: candidates found, confirmed `y` count.
5. Landmarks: glance y-rate (15 images), QC-flag rate, identity-check failures.
6. African and HC18/UCL counts (from R1.2).

**Decision rules (copied from 02 §3, made precise):**
- Caliper prevalence difference (standard planes − Other, test audit, pooled over machines) ≥ 10 pp, with the CI excluding 0 → **overlays = primary axis** (E4b, E5a, E5d, A2 at full weight). Otherwise → overlays secondary; emphasise E2, E3, E6 and A3.
- Detector caliper recall < 0.8 on the train audit → one re-tune round (train audit only). Still < 0.8 → E5a is restricted to test images where detector and human both say calipers are present, and the restriction is reported. **Never** fall back to manual masks.
- Confirmed caliper pairs < 20 → drop E5d.
- Sector `ok` rate < 90% on any machine → enable the consensus prior (R1.4 step 6) and re-run the sector check on 30 new images of that machine.
- Landmark glance < 14/15 `y`, or Head QC-flag ≥ 5% → stop R4 until resolved.

**R1 acceptance:** all of the outputs above exist; `python -m pytest` is green (including `data` tests); `GATE_G1.md` is signed by the human with an emphasis decision; RESEARCH_LOG entries written.
