"""
scripts/find_caliper_pairs.py  (R1.8)

Within the FPDB test split, find near-duplicate image pairs from the same
(patient, PlaneN) acquisition group. Candidate *caliper pairs* (E5d) are
near-duplicates where exactly one image has detected calipers.

Also reports the overall near-duplicate structure (images that are almost the same
frame), which matters for interpreting image-level statistics.

Outputs
  data/processed/audit/near_duplicate_pairs.csv     all within-group pairs with SSIM >= 0.90
  data/processed/audit/caliper_pairs.csv            candidates (<= 50 sampled for confirmation)
  results/json/R1_caliper_pairs.json

Usage:  python scripts/find_caliper_pairs.py
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

sys.path.insert(0, ".")
from src.audit.metadata import load_masks, mask_path  # noqa: E402

META = Path("data/processed/metadata/acquisition_metadata.parquet")
AUDIT = Path("data/processed/audit")
SSIM_MIN = 0.90
MAX_CONFIRM = 50
SEED = 20261109


def _prep(row: pd.Series) -> np.ndarray:
    g = cv2.imread(row["image_path"], cv2.IMREAD_GRAYSCALE)
    m = load_masks(mask_path(row["image_id"], "fpdb"))["sector_clean"]
    g = (g * m).astype(np.uint8)
    return cv2.resize(g, (g.shape[1] // 2, g.shape[0] // 2), interpolation=cv2.INTER_AREA)


def main() -> None:
    meta = pd.read_parquet(META)
    test = meta[meta["split"] == "test"].copy()
    test["group"] = test["patient_id"] + "_P" + test["plane_n"].astype(str)
    sizes = test.groupby("group").size()
    multi = test[test["group"].isin(sizes[sizes >= 2].index)]
    hetero = multi.groupby("group")["plane_label"].nunique()
    n_hetero = int((hetero > 1).sum())

    pairs = []
    for _, g in multi.groupby("group"):
        imgs = {r["image_id"]: _prep(r) for _, r in g.iterrows()}
        for a, b in itertools.combinations(g.itertuples(), 2):
            ia, ib = imgs[a.image_id], imgs[b.image_id]
            if ia.shape != ib.shape:
                continue
            ssim = structural_similarity(ia, ib, data_range=255)
            if ssim >= SSIM_MIN:
                pairs.append({"image_id_a": a.image_id, "image_id_b": b.image_id, "ssim": round(float(ssim), 4),
                              "label_a": a.plane_label, "label_b": b.plane_label,
                              "cs_a": a.caliper_score, "cs_b": b.caliper_score,
                              "source_machine": a.source_machine, "patient_id": a.patient_id})
    nd = pd.DataFrame(pairs)
    nd.to_csv(AUDIT / "near_duplicate_pairs.csv", index=False)

    if len(nd):
        cand = nd[(nd["label_a"] == nd["label_b"]) & ((nd["cs_a"] > 0) != (nd["cs_b"] > 0))]
    else:
        cand = nd
    if len(cand) > MAX_CONFIRM:
        cand = cand.sample(MAX_CONFIRM, random_state=SEED)
    cand.to_csv(AUDIT / "caliper_pairs.csv", index=False)

    summary = {
        "test_groups_ge2": int((sizes >= 2).sum()),
        "images_in_groups_ge2": int(sizes[sizes >= 2].sum()),
        "label_heterogeneous_groups": n_hetero,
        "near_duplicate_pairs_ssim_ge_0.90": int(len(nd)),
        "near_duplicate_images": int(len(set(nd["image_id_a"]) | set(nd["image_id_b"]))) if len(nd) else 0,
        "near_duplicate_pairs_same_label": int((nd["label_a"] == nd["label_b"]).sum()) if len(nd) else 0,
        "caliper_pair_candidates": int(len(cand)),
    }
    Path("results/json/R1_caliper_pairs.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
