"""
scripts/build_cross_device_manifest.py

Builds data/processed/cross_device_manifest.csv from ONLY:
  data/raw/ucl_hc18/images/HC18/Head/
  data/raw/ucl_hc18/images/UCL/{Head,Abdomen,Femur}/

Label is implicit in the subfolder name. 'Head' is a COLLAPSED label -- at
evaluation time (Phase 6), a prediction is counted correct for a 'Head' row
if the model's argmax is ANY of {Brain_Trans_cerebellum, Brain_Trans_thalamic,
Brain_Trans_ventricular}, since none of these external datasets distinguish
fetal brain sub-planes. This manifest does NOT include an 'Other' class --
every image here is a valid standard plane by construction.

DO NOT read from images/FP/ or images/MULTICENTRE/ -- see 02_DATASETS.md §2.
"""
from pathlib import Path
import pandas as pd

BASE = Path("data/raw/ucl_hc18/images")
OUT_PATH = Path("data/processed/cross_device_manifest.csv")
VALID_EXTS = {".png", ".jpg", ".jpeg"}

# (source_subset, subfolder_name, output_label)
SOURCES = [
    ("HC18", "Head", "Head"),
    ("UCL", "Head", "Head"),
    ("UCL", "Abdomen", "Fetal_abdomen"),
    ("UCL", "Femur", "Fetal_femur"),
]

def build_cross_device_manifest():
    rows = []
    for source_subset, subfolder, label in SOURCES:
        folder = BASE / source_subset / subfolder
        assert folder.exists(), f"Expected folder not found: {folder}"
        for img_path in sorted(folder.iterdir()):
            if img_path.suffix.lower() in VALID_EXTS:
                rows.append({
                    "image_path": str(img_path),
                    "plane_label": label,
                    "source_subset": source_subset,
                    "is_collapsed_label": label == "Head",
                })

    df = pd.DataFrame(rows)
    assert len(df) > 0, "No images found -- check BASE path and folder names."

    # Guardrail: make sure nothing from FP/MULTICENTRE snuck in via a wrong BASE path.
    # Use OS-independent path separator check.
    forbidden = [
        p for p in df["image_path"]
        if (Path(p).parts and any(part in ("FP", "MULTICENTRE") for part in Path(p).parts))
    ]
    assert not forbidden, f"FP/MULTICENTRE paths leaked into cross-device manifest: {forbidden[:5]}"

    # Additional explicit check on string content for robustness
    forbidden_str = [p for p in df["image_path"] if "/FP/" in p or "\\FP\\" in p
                     or "/MULTICENTRE/" in p or "\\MULTICENTRE\\" in p]
    assert not forbidden_str, f"FP/MULTICENTRE paths leaked (string check): {forbidden_str[:5]}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df.groupby(["source_subset", "plane_label"]).size())


if __name__ == "__main__":
    build_cross_device_manifest()
