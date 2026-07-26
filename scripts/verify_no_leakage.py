"""
scripts/verify_no_leakage.py
Hard-fails (raises) if any patient_id appears in more than one of train/val/test.
"""
from pathlib import Path

import pandas as pd

SPLITS_DIR = Path("data/splits")

def verify():
    train = set(pd.read_csv(SPLITS_DIR / "train.csv")["patient_id"])
    val = set(pd.read_csv(SPLITS_DIR / "val.csv")["patient_id"])
    test = set(pd.read_csv(SPLITS_DIR / "test.csv")["patient_id"])

    assert train.isdisjoint(val), f"Leakage: {len(train & val)} patients in both train and val"
    assert train.isdisjoint(test), f"Leakage: {len(train & test)} patients in both train and test"
    assert val.isdisjoint(test), f"Leakage: {len(val & test)} patients in both val and test"

    print("PASS: train/val/test patient sets are fully disjoint.")
    print(f"train patients: {len(train)}, val patients: {len(val)}, test patients: {len(test)}")

if __name__ == "__main__":
    verify()
