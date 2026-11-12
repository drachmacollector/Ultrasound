"""
scripts/research/build_cache.py  (R3 E4 modes; R5.1 letterbox modes)

Renders preprocess modes (src/preprocess/pipeline.py) once per image. Rendering is label-free
(R1 masks only).

* E4 modes (nonanat_only, overlay_only, sector_shape_only): every FPDB image, stored pre-resized
  to the model input:   data/processed/cache/<mode>_<size>/fpdb/<image_id>.png
* Letterbox modes (sector_letterbox, sector_letterbox_inpaint): every FPDB image (all splits) and
  every external image (HC18, UCL, African), stored at native scale as the sector crop plus its
  sector mask:          data/processed/cache/<mode>/<dataset>/<image_id>.png and ..._sector.png
  A cache_manifest.json (overlay-parameter hash + pipeline version) is written after a complete
  build; training and evaluation refuse a missing or stale cache. The build report
  (build_report.json) logs the per-dataset sector shape-class distribution (all images).

Existing files are skipped unless --force.

Usage:  python scripts/research/build_cache.py --mode sector_letterbox sector_letterbox_inpaint
        python scripts/research/build_cache.py --mode nonanat_only overlay_only sector_shape_only [--size 224]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.audit.metadata import load_masks, mask_path  # noqa: E402
from src.preprocess.pipeline import (  # noqa: E402
    CACHED_MODES,
    LETTERBOX_MODES,
    cache_dir,
    cache_path,
    prepare_native,
    render,
    resize_for_cache,
    sector_cache_path,
    write_cache_manifest,
)
from src.preprocess.sector import sector_from_mask  # noqa: E402

META = Path("data/processed/metadata/acquisition_metadata.parquet")
EXTERNAL = {d: Path(f"data/processed/external/{d}.csv") for d in ("hc18", "ucl", "african")}
CHUNK = 200


def _read(p: str) -> np.ndarray:
    img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(p)
    return img


def _write(path: Path, img) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 1]):
        raise OSError(f"could not write {path}")


def _work(args: tuple[list[tuple[str, str, str]], list[str], int, bool]) -> list[tuple[str, str]]:
    rows, modes, size, force = args
    shapes = []
    for dataset, image_id, image_path in rows:
        todo = [m for m in modes if force or not cache_path(m, size, dataset, image_id).exists()]
        masks = load_masks(mask_path(image_id, dataset))
        shapes.append((dataset, sector_from_mask(masks["sector_clean"]).shape_class))
        if not todo:
            continue
        gray = _read(image_path)
        for m in todo:
            if m in LETTERBOX_MODES:
                img, sec = prepare_native(gray, masks, m)
                _write(cache_path(m, None, dataset, image_id), img)
                _write(sector_cache_path(m, dataset, image_id), sec * 255)
            else:
                _write(cache_path(m, size, dataset, image_id), resize_for_cache(render(gray, masks, m), size))
    return shapes


def rows_for(modes: list[str]) -> list[tuple[str, str, str]]:
    meta = pd.read_parquet(META, columns=["image_id", "image_path"])[["image_id", "image_path"]]
    rows = [("fpdb", i, p) for i, p in meta.itertuples(index=False, name=None)]
    if any(m in LETTERBOX_MODES for m in modes):
        for d, csv in EXTERNAL.items():
            e = pd.read_csv(csv, usecols=["image_id", "image_path"])[["image_id", "image_path"]]
            rows += [(d, i, p) for i, p in e.itertuples(index=False, name=None)]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="+", required=True, choices=CACHED_MODES + LETTERBOX_MODES)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    e4 = [m for m in a.mode if m in CACHED_MODES]
    lb = [m for m in a.mode if m in LETTERBOX_MODES]
    for modes in (e4, lb):
        if not modes:
            continue
        rows = rows_for(modes)
        if modes is e4:
            rows = [r for r in rows if r[0] == "fpdb"]
        chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            shapes = [s for part in ex.map(_work, [(c, modes, a.size, a.force) for c in chunks]) for s in part]
        for m in modes:
            missing = [r for r in rows if not cache_path(m, a.size, r[0], r[1]).exists()]
            assert not missing, f"{m}: {len(missing)} images missing from the cache"
            if m in LETTERBOX_MODES:
                write_cache_manifest(m)
                rep = pd.DataFrame(shapes, columns=["dataset", "shape_class"]).value_counts().unstack(fill_value=0)
                (cache_dir(m) / "build_report.json").write_text(
                    json.dumps({"n_images": len(rows), "sector_shape_class_by_dataset": rep.to_dict(orient="index")},
                               indent=1), encoding="utf-8")
        print(f"{modes}: {len(rows)} images complete")


if __name__ == "__main__":
    main()
