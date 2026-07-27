"""
scripts/build_manifest.py

Reads FETAL_PLANES_DB_data.csv and produces data/processed/manifest.csv with
columns: image_path, patient_id, plane_label, brain_subplane_raw, source_machine,
operator, original_split_flag.
"""
from pathlib import Path

import pandas as pd

RAW_CSV = Path("data/raw/fetal_planes_db/FETAL_PLANES_DB_data.csv")
IMAGES_DIR = Path("data/raw/fetal_planes_db/Images")
OUT_PATH = Path("data/processed/manifest.csv")

import sys

# Add project root to sys.path so we can import src.data.dataset
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CANONICAL_CLASSES

def map_to_canonical(plane: str, brain_plane) -> str:
    plane = plane.strip()
    brain_plane = str(brain_plane).strip() if pd.notna(brain_plane) else "Not A Brain"

    if plane == "Fetal brain":
        if brain_plane == "Trans-thalamic":
            return "Brain_Trans_thalamic"
        elif brain_plane == "Trans-cerebellum":
            return "Brain_Trans_cerebellum"
        elif brain_plane == "Trans-ventricular":
            return "Brain_Trans_ventricular"
        else:
            # brain_plane == 'Other' -> non-standard brain image, NOT a standard
            # brain sub-plane. This is a common source of silent mislabeling bugs
            # if not handled explicitly -- it belongs in the 'Other' class.
            return "Other"
    elif plane == "Fetal abdomen":
        return "Fetal_abdomen"
    elif plane == "Fetal femur":
        return "Fetal_femur"
    elif plane == "Fetal thorax":
        return "Fetal_thorax"
    elif plane == "Maternal cervix":
        return "Maternal_cervix"
    elif plane == "Other":
        return "Other"
    else:
        raise ValueError(f"Unrecognized Plane value: {plane!r}")


def build_manifest():
    df = pd.read_csv(RAW_CSV, sep=";")
    df.columns = [c.strip() for c in df.columns]  # Strip trailing/leading whitespace (e.g., 'Train ')

    df["plane_label"] = df.apply(
        lambda r: map_to_canonical(r["Plane"], r.get("Brain_plane")), axis=1
    )

    assert df["plane_label"].isin(CANONICAL_CLASSES).all(), \
        "Found a row that did not map to one of the 8 canonical classes."
    assert df["plane_label"].notna().all(), "Found null canonical labels."

    df["image_path"] = df["Image_name"].apply(lambda name: (IMAGES_DIR / f"{name}.png").as_posix())
    df["patient_id"] = df["Patient_num"]
    df["brain_subplane_raw"] = df.get("Brain_plane")
    df["source_machine"] = df["US_Machine"]
    df["operator"] = df["Operator"]
    df["original_split_flag"] = df["Train"].astype(int)  # 1=train pool, 0=in-distribution test

    out_cols = [
        "image_path", "patient_id", "plane_label", "brain_subplane_raw",
        "source_machine", "operator", "original_split_flag",
    ]
    out = df[out_cols]

    # Sanity: every image file referenced actually exists on disk.
    missing = [p for p in out["image_path"].sample(min(50, len(out))) if not Path(p).exists()]
    assert not missing, f"Sampled manifest rows point to missing files: {missing[:5]}"

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(out)} rows to {OUT_PATH}")
    print(out["plane_label"].value_counts())


if __name__ == "__main__":
    build_manifest()
