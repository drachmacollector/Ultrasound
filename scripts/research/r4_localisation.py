"""
scripts/research/r4_localisation.py  (R4.4 development on val, R4.6 locked run on test/external)

  --dev   convnext_tiny, all methods, the 238 FPDB-val landmark images
          -> results/preds/legacy_convnext_s42/fpdb_val__saliency-<method>.parquet,
             results/figures/R4_dev_grid_val.pdf (24 images, 3 pages)
  --run   every legacy backbone x every method that passed Gate G3 (results/json/G3.json):
            gradcam / gradcampp: all of fpdb_test (5,271), hc18, ucl, african (landmark metrics on the
                                 1,486 + 999 + 424 landmark images, GT-free metrics everywhere = E7)
            occlusion:           the fixed 600-image landmark subset (200/anatomy FPDB-test, 100 HC18, 100 UCL)
          -> results/preds/<run_id>/<dataset>__saliency-<method>.parquet (+ .maps.npz, 48 figure images)
          The test run refuses to start before G3 is recorded with a code-freeze commit.

One GPU job at a time (CLAUDE.md resource note); finished outputs are skipped on re-run.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, ".")
from src.audit.plotting import FIG_DIR, INK, plt  # noqa: E402
from src.data.transforms import load_grayscale  # noqa: E402
from src.eval.predict import PREDS_ROOT  # noqa: E402
from src.localisation import runner as R  # noqa: E402
from src.models.loading import load_model  # noqa: E402
from src.models.saliency import METHODS  # noqa: E402

G3_JSON = Path("results/json/G3.json")
SUBSETS_JSON = Path("results/json/R4_subsets.json")
TEST_SETS = ("fpdb_test", "hc18", "ucl", "african")
DEV_GRID_N = 8  # per anatomy


# ----------------------------------------------------------------------------- dev (val)
def dev(num_workers: int) -> None:
    bb = "convnext_tiny"
    run_dir = PREDS_ROOT / R.legacy_run_id(bb)
    frame = R.load_frame("fpdb_val", run_dir, landmark_only=True)
    rng = np.random.default_rng(R.SEL_SEED)
    grid_ids = sorted(i for a in ("Head", "Abdomen", "Femur")
                      for i in rng.choice(sorted(frame.loc[frame["anatomy"] == a, "image_id"]), DEV_GRID_N, replace=False))
    lm = load_model(R.legacy_ckpt(bb))
    maps = {}
    for method in METHODS:
        t0 = time.perf_counter()
        spec = R.RunSpec(R.legacy_run_id(bb), "fpdb_val", method, frame, frozenset(grid_ids), num_workers=num_workers)
        out, maps[method] = R.compute(lm, spec)
        R.write(out, maps[method], R.saliency_path(spec.run_id, "fpdb_val", method))
        dt = time.perf_counter() - t0
        ok = out[~out["qc_flag"].fillna(False).astype(bool)]
        print(f"{method:10s} n={len(out)} {dt:6.1f}s  pred agrees with R2: {out['pred_agrees_r2'].mean():.3f}")
        print(ok.groupby("anatomy")[["epg_true", "area_frac", "epg_minus_area_true", "hit_true", "iou_true",
                                     "out_sector_frac_true"]].mean().round(3).to_string())
    dev_grid(frame.set_index("image_id"), grid_ids, maps)


def _overlay(ax, gray_s: np.ndarray, sal: np.ndarray | None, anat: np.ndarray, title: str) -> None:
    ax.imshow(gray_s, cmap="gray", vmin=0, vmax=255)
    if sal is not None:
        s = np.maximum(sal.astype(np.float32), 0)
        if s.max() > 0:
            ax.imshow(s / s.max(), cmap="inferno", alpha=0.5, vmin=0, vmax=1)
    if anat.ndim == 2:
        ax.contour(anat.astype(float), levels=[0.5], colors=["#1baf7a"], linewidths=0.8)
    ax.set_title(title, fontsize=6, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])


def dev_grid(frame: pd.DataFrame, ids: list[str], maps: dict[str, dict]) -> None:
    pq = {m: pd.read_parquet(R.saliency_path(R.legacy_run_id("convnext_tiny"), "fpdb_val", m)).set_index("image_id")
          for m in METHODS}
    out = FIG_DIR / "R4_dev_grid_val.pdf"
    with PdfPages(out) as pdf:
        for page in range(0, len(ids), 8):
            fig, axes = plt.subplots(8, 4, figsize=(6.4, 13))
            for r, iid in enumerate(ids[page:page + 8]):
                row = frame.loc[iid]
                g = cv2.resize(load_grayscale(row["image_path"]), (224, 224), interpolation=cv2.INTER_LINEAR)
                anat = maps["gradcam"][f"{iid}|anat"]
                _overlay(axes[r, 0], g, None, anat, f"{row['anatomy']} | {row['source_machine']}\n"
                                                    f"{'correct' if row['correct_full'] else 'pred ' + row['pred_full']}")
                for c, m in enumerate(METHODS, start=1):
                    v = pq[m].loc[iid]
                    _overlay(axes[r, c], g, maps[m][f"{iid}|true"], anat,
                             f"{m}  EPG−area {v['epg_minus_area_true']:+.2f}  hit {int(v['hit_true'])}")
            fig.suptitle("R4 development grid (FPDB val, convnext_tiny, true target; green = landmark mask)",
                         fontsize=8)
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(FIG_DIR / f"R4_dev_grid_val_p{page // 8 + 1}.png", dpi=130, bbox_inches="tight")
            plt.close(fig)
    print(f"wrote {out}")


# ----------------------------------------------------------------------------- locked run (test)
def passing_methods() -> list[str]:
    if not G3_JSON.exists():
        raise SystemExit("Gate G3 not recorded (results/json/G3.json); run r4_sanity_randomisation.py first")
    g3 = json.loads(G3_JSON.read_text(encoding="utf-8"))
    commit = g3.get("code_freeze_commit", "")
    if not commit or commit.endswith("-dirty"):
        raise SystemExit("G3.json has no clean code_freeze_commit; freeze (commit) the R4 code and re-run G3")
    return [m for m in METHODS if g3["decision"][m]["pass"]]


def subsets() -> dict:
    if SUBSETS_JSON.exists():
        return json.loads(SUBSETS_JSON.read_text(encoding="utf-8"))
    run_dir = PREDS_ROOT / R.legacy_run_id("convnext_tiny")
    frames = {ds: R.load_frame(ds, run_dir) for ds in ("fpdb_test", "hc18", "ucl")}
    fig = R.figure_subset(frames)
    occ = R.occlusion_subset(frames, fig)
    out = {"seed": R.SEL_SEED, "figure": fig, "occlusion": occ,
           "rule": "figure: by reference-model (convnext) correctness only, non-QC landmark images; "
                   "occlusion: figure images + random fill to 200/anatomy FPDB-test, 100 HC18, 100 UCL"}
    SUBSETS_JSON.write_text(json.dumps(out, indent=1), encoding="utf-8")
    return out


def run(backbones: list[str], methods: list[str] | None, num_workers: int) -> None:
    allowed = passing_methods()
    methods = [m for m in (methods or allowed) if m in allowed]
    sub = subsets()
    keep = frozenset(sub["figure"])
    for bb in backbones:
        rid = R.legacy_run_id(bb)
        todo = [(m, ds) for m in methods for ds in TEST_SETS
                if not (m == "occlusion" and ds not in sub["occlusion"])
                and not R.saliency_path(rid, ds, m).exists()]
        if not todo:
            continue
        lm = load_model(R.legacy_ckpt(bb))
        for m, ds in todo:
            t0 = time.perf_counter()
            frame = R.load_frame(ds, PREDS_ROOT / rid)
            if m == "occlusion":
                frame = frame[frame["image_id"].isin(set(sub["occlusion"][ds]))]
            bs = 8 if m == "occlusion" else 32
            out, maps = R.compute(lm, R.RunSpec(rid, ds, m, frame, keep, batch_size=bs, num_workers=num_workers))
            R.write(out, maps, R.saliency_path(rid, ds, m))
            print(f"  {bb} {m} {ds}: n={len(out)} in {time.perf_counter() - t0:.0f}s; "
                  f"R2 agreement {out['pred_agrees_r2'].mean():.4f}", flush=True)
        del lm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--backbones", nargs="*", default=list(R.LEGACY))
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--num-workers", type=int, default=4)
    a = ap.parse_args()
    if a.dev:
        dev(a.num_workers)
    if a.run:
        run(a.backbones, a.methods, a.num_workers)


if __name__ == "__main__":
    main()
