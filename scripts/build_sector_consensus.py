"""
scripts/build_sector_consensus.py  (R1.4 step 6; enabled by Gate G1)

Device-geometry consensus ("core") sectors: for every (source_machine, width, height) group
with >= MIN_GROUP images in the FPDB **train** split, the pixels inside the per-image sector
in >= 95% of the group's images (src.preprocess.sector.build_consensus). Label-free and
train-only; never computed for external datasets.

Requires a first pass of build_acquisition_metadata.py (per-image masks without consensus).
Writes data/processed/masks/consensus/<machine>_<W>x<H>.npz and consensus_index.csv.

Usage:  python scripts/build_sector_consensus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.audit.metadata import CONSENSUS_ROOT as OUT  # noqa: E402
from src.audit.metadata import consensus_key, load_masks, mask_path  # noqa: E402
from src.preprocess.sector import build_consensus  # noqa: E402

META = Path("data/processed/metadata/acquisition_metadata.parquet")
MIN_GROUP = 30


def main() -> None:
    meta = pd.read_parquet(META)
    if "consensus_used" in meta and meta["consensus_used"].any():
        raise SystemExit("metadata already built with consensus; consensus must come from the first pass")
    train = meta[meta["split"] == "train"]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for (machine, w, h), g in train.groupby(["source_machine", "width", "height"]):
        if len(g) < MIN_GROUP:
            continue
        masks = [load_masks(mask_path(i, "fpdb"))["sector"] for i in g["image_id"]]
        cons = build_consensus(masks)
        key = consensus_key(machine, w, h)
        np.savez_compressed(OUT / f"{key}.npz", mask=cons)
        rows.append({"key": key, "source_machine": machine, "width": w, "height": h,
                     "n_train_images": len(g), "area_frac": round(float(cons.mean()), 4)})
    idx = pd.DataFrame(rows)
    idx.to_csv(OUT / "consensus_index.csv", index=False)
    print(idx.to_string(index=False))


if __name__ == "__main__":
    main()
