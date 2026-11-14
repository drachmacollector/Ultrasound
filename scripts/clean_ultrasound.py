"""
scripts/clean_ultrasound.py  (R7 E11: apply the study's B2 input cleaning to your own images)

For every image in --in (png/jpg/bmp/tif, searched recursively), writes to --out, keeping the sub-folder layout:

    <name>.png             sector crop, caliper markers inpainted, zero outside the sector, letterboxed to
                           --size x --size (aspect ratio kept); --size 0 keeps the native-scale crop
    <name>_masks.png       (--save-masks) colour overlay of the detected sector (green), UI/text (blue) and
                           measurement markers (red) on the original frame, for visual QC
    clean_log.csv          one row per image: sector shape class, sector area, number of caliper crosses and
                           dotted lines, UI area, and a QC flag

What it is: the label-free preprocessing of the R5 B2 arm (src/preprocess/clean.py), which raised pooled external
balanced accuracy in both fine-tuned backbones and in a frozen FetalCLIP probe (docs/research/INPUT_CLEANING.md).
What it is not: a validated clinical tool. Check the QC overlays; the sector detector can fail on frames with no
dark background, colour Doppler, or split screens (see INPUT_CLEANING.md).

Usage:  python scripts/clean_ultrasound.py --in <folder> --out <folder> [--size 224] [--save-masks]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess.clean import clean_image, load_params  # noqa: E402

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
MIN_SECTOR_FRAC = 0.05        # QC: a sector smaller than 5% of the frame is probably a detection failure


def qc_overlay(gray: np.ndarray, masks: dict[str, np.ndarray]) -> np.ndarray:
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edge = cv2.morphologyEx(masks["sector_clean"].astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
    out[edge > 0] = (0, 200, 0)
    out[masks["ui"] > 0] = (255, 120, 0)
    out[masks["marker"] > 0] = (0, 0, 255)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=224, help="letterbox size; 0 = native-scale crop")
    ap.add_argument("--save-masks", action="store_true")
    a = ap.parse_args()
    files = sorted(p for p in a.inp.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"no images found in {a.inp}")
    params = load_params()
    rows = []
    for p in files:
        rel = p.relative_to(a.inp).with_suffix("")
        gray = cv2.imdecode(np.fromfile(str(p), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            rows.append({"file": str(rel), "status": "unreadable"})
            continue
        c = clean_image(gray, size=a.size or None, params=params)
        dst = a.out / rel.parent
        dst.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst / f"{rel.name}.png"), c.letterbox if c.letterbox is not None else c.native)
        if a.save_masks:
            cv2.imwrite(str(dst / f"{rel.name}_masks.png"), qc_overlay(gray, c.masks))
        i = c.info
        rows.append({"file": str(rel), "status": "ok", "width": i["width"], "height": i["height"],
                     "sector_shape": i["sector_shape_class"], "sector_area_frac": round(i["sector_area_frac"], 4),
                     "n_caliper_crosses": i["n_crosses"], "n_dotted_lines": i["n_dotted_lines"],
                     "ui_area_frac": round(i["ui_area_frac"], 4),
                     "qc_flag": i["sector_area_frac"] < MIN_SECTOR_FRAC or i["sector_shape_class"] == "full"})
    a.out.mkdir(parents=True, exist_ok=True)
    log = pd.DataFrame(rows)
    log.to_csv(a.out / "clean_log.csv", index=False)
    ok = log["status"].eq("ok")
    print(f"cleaned {int(ok.sum())} of {len(log)} images -> {a.out}; "
          f"{int(log.loc[ok, 'qc_flag'].sum())} flagged for visual QC (clean_log.csv)")


if __name__ == "__main__":
    main()
