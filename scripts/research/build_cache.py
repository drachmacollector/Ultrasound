"""
scripts/research/build_cache.py  (R3 E4; reused by R5)

Renders preprocess modes (src/preprocess/pipeline.py) for every FPDB image (all splits)
and stores them pre-resized to the model input size:

    data/processed/cache/<mode>_<size>/fpdb/<image_id>.png      (greyscale)

Rendering is label-free (masks from R1). Existing files are skipped unless --force.

Usage:  python scripts/research/build_cache.py --mode nonanat_only overlay_only sector_shape_only [--size 224]
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd

sys.path.insert(0, ".")
from src.audit.metadata import load_masks, mask_path  # noqa: E402
from src.data.transforms import load_grayscale  # noqa: E402
from src.preprocess.pipeline import CACHED_MODES, cache_path, render, resize_for_cache  # noqa: E402

META = Path("data/processed/metadata/acquisition_metadata.parquet")
CHUNK = 200


def _work(args: tuple[list[tuple[str, str]], list[str], int, bool]) -> int:
    rows, modes, size, force = args
    n = 0
    for image_id, image_path in rows:
        targets = {m: cache_path(m, size, "fpdb", image_id) for m in modes}
        todo = [m for m, p in targets.items() if force or not p.exists()]
        if not todo:
            continue
        gray = load_grayscale(image_path)
        masks = load_masks(mask_path(image_id, "fpdb"))
        for m in todo:
            out = resize_for_cache(render(gray, masks, m), size)
            targets[m].parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(targets[m]), out):
                raise OSError(f"could not write {targets[m]}")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="+", required=True, choices=CACHED_MODES)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    meta = pd.read_parquet(META, columns=["image_id", "image_path"])
    rows = list(meta.itertuples(index=False, name=None))
    chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        written = sum(ex.map(_work, [(c, a.mode, a.size, a.force) for c in chunks]))
    for m in a.mode:
        n = sum(1 for _ in (cache_path(m, a.size, "fpdb", "x").parent).glob("*.png"))
        assert n == len(rows), f"{m}: {n} cached images, expected {len(rows)}"
    print(f"wrote {written} files; {len(rows)} images x {len(a.mode)} modes complete")


if __name__ == "__main__":
    main()
