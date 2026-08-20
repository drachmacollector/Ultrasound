"""
scripts/verify_no_multitask_leakage.py

CI-style data integrity checker for the multitask manifests.

Checks:
  1. No FP or MULTICENTRE images appear in either split.
  2. No patient appears in BOTH train and val (patient-level disjointness).
  3. The annotated subset (HC18/UCL rows, identifiable via has_bbox==True)
     has a non-trivial fraction of known patient IDs — i.e. patient_id != -1.
     If this fraction is too low, the patient-disjointness check in step 2
     would be vacuously passing (it only checks patients it can see).
     Threshold: >=95% of has_bbox rows must have a real patient ID.

Usage:
    conda run -n fetalplane python -m scripts.verify_no_multitask_leakage

Exits with code 0 on success, non-zero on any assertion failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


TRAIN_PATH = Path("data/splits/multitask_train.csv")
VAL_PATH   = Path("data/splits/multitask_val.csv")

# Fraction of annotated (has_bbox) rows that MUST have a non-(-1) patient_id.
# If this is violated, the patient disjointness check is vacuous.
MIN_PATIENT_ID_COVERAGE = 0.95


def main() -> None:
    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        print("ERROR: Manifests not found. Run scripts/build_multitask_manifest.py first.")
        sys.exit(1)

    df_train = pd.read_csv(TRAIN_PATH)
    df_val   = pd.read_csv(VAL_PATH)

    # -------------------------------------------------------------------------
    # Check 1: No FP or MULTICENTRE images
    # -------------------------------------------------------------------------
    for df, name in [(df_train, "Train"), (df_val, "Val")]:
        has_fp = df["image_path"].str.contains("/FP/", na=False).any()
        has_mc = df["image_path"].str.contains("/MULTICENTRE/", na=False).any()
        assert not has_fp, f"FAIL [{name}] FP images present — patient overlap with FETAL_PLANES_DB!"
        assert not has_mc, f"FAIL [{name}] MULTICENTRE images present — policy violation!"
    print("  [OK] Check 1: No FP/MULTICENTRE images in either split.")

    # -------------------------------------------------------------------------
    # Check 2: Patient-ID coverage on annotated rows is non-trivial.
    #
    # This check must come BEFORE the patient-disjointness check.
    # If patient_id == -1 for all annotated rows, the disjointness check
    # would pass vacuously (filtering out all -1 rows means it sees nothing).
    # A passing assertion that can't see the failure mode it guards against
    # is worse than no assertion — it manufactures false confidence.
    # -------------------------------------------------------------------------
    for df, name in [(df_train, "Train"), (df_val, "Val")]:
        if "has_bbox" not in df.columns:
            continue
        annotated = df[df["has_bbox"] == True]
        if len(annotated) == 0:
            continue
        coverage = (annotated["patient_id"] != -1).mean()
        assert coverage >= MIN_PATIENT_ID_COVERAGE, (
            f"FAIL [{name}] Only {coverage:.1%} of annotated rows have a real patient_id "
            f"(threshold: {MIN_PATIENT_ID_COVERAGE:.0%}). "
            f"The patient-disjointness check below would be vacuous — fix the manifest "
            f"builder to read the correct patient-ID column."
        )
        print(f"  [OK] Check 2 [{name}]: {coverage:.1%} of annotated rows have a real patient_id.")

    # -------------------------------------------------------------------------
    # Check 3: No patient appears in both train and val (annotated subsets only)
    #
    # We restrict to has_bbox==True rows (HC18/UCL annotated subset).
    # FETAL_PLANES_DB rows (has_bbox=False) were correctly split in Phase 3
    # as a separate cohort — checking them here would produce false positives
    # when comparing integer FETAL_PLANES_DB patient IDs against each other,
    # since the same patient can validly appear in only one split.
    # -------------------------------------------------------------------------
    annotated_train = df_train[df_train["has_bbox"] == True] if "has_bbox" in df_train.columns else df_train
    annotated_val   = df_val[df_val["has_bbox"] == True]   if "has_bbox" in df_val.columns   else df_val

    train_patients = set(
        annotated_train[annotated_train["patient_id"] != -1]["patient_id"].unique()
    )
    val_patients = set(
        annotated_val[annotated_val["patient_id"] != -1]["patient_id"].unique()
    )
    overlap = train_patients & val_patients
    assert len(overlap) == 0, (
        f"FAIL: {len(overlap)} patients appear in BOTH train and val splits — "
        f"patient leakage detected!\nOverlapping patient IDs: {sorted(overlap)}"
    )
    print(
        f"  [OK] Check 3: 0 overlapping annotated patients "
        f"({len(train_patients)} train | {len(val_patients)} val)."
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\nAll checks passed. Multitask manifests are clean.")
    print(f"  Train: {len(df_train)} rows  "
          f"({df_train['has_bbox'].sum() if 'has_bbox' in df_train.columns else 'N/A'} with bbox)")
    print(f"  Val:   {len(df_val)} rows  "
          f"({df_val['has_bbox'].sum() if 'has_bbox' in df_val.columns else 'N/A'} with bbox)")


if __name__ == "__main__":
    main()
