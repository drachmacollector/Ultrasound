"""
scripts/patient_split.py

Splits the Train==1 pool of manifest.csv into train/val at the patient level,
using StratifiedGroupKFold so per-class patient representation in val is
checked and maximized rather than left to chance. The Train==0 pool becomes
the in-distribution test set, untouched.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

MANIFEST_PATH = Path("data/processed/manifest.csv")
SPLITS_DIR = Path("data/splits")
TARGET_VAL_FRACTION = 0.15       # starting point per 03_DATA_PIPELINE.md -- adjust if step 3 below fails
MIN_PATIENTS_PER_CLASS_IN_VAL = 3  # raise/lower per actual per-class patient counts you observe
RANDOM_STATE = 42


def patient_counts_per_class(df: pd.DataFrame) -> pd.Series:
    counts = df.groupby("plane_label")["patient_id"].nunique()
    if isinstance(counts, pd.Series):
        return counts
    return pd.Series(counts)


def find_best_val_fold(train_pool: pd.DataFrame, n_splits: int):
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    X = train_pool.index.tolist()
    y = train_pool["plane_label"].tolist()
    groups = train_pool["patient_id"].tolist()

    best_fold_idx, best_min_coverage, best_train_idx, best_val_idx = None, -1, None, None
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        val_df = train_pool.iloc[val_idx]
        counts = patient_counts_per_class(val_df)
        # every one of the 8 canonical classes must be represented
        min_coverage = counts.reindex(
            [c for c in train_pool["plane_label"].unique()]
        ).fillna(0).min()
        if min_coverage > best_min_coverage:
            best_fold_idx, best_min_coverage, best_train_idx, best_val_idx = fold_idx, min_coverage, train_idx, val_idx

    return best_fold_idx, best_min_coverage, best_train_idx, best_val_idx


def build_splits():
    df = pd.read_csv(MANIFEST_PATH)

    test_df = df[df["original_split_flag"] == 0].copy()          # untouched in-distribution test
    train_pool = df[df["original_split_flag"] == 1].copy().reset_index(drop=True)

    n_splits = round(1 / TARGET_VAL_FRACTION)  # e.g. 0.15 -> 7
    best_fold_idx, best_min_coverage, best_train_idx, best_val_idx = find_best_val_fold(train_pool, n_splits)

    if best_train_idx is None or best_val_idx is None:
        raise RuntimeError("Failed to generate any valid folds.")

    if best_min_coverage < MIN_PATIENTS_PER_CLASS_IN_VAL:
        raise RuntimeError(
            f"Best achievable minimum per-class patient coverage in val is "
            f"{best_min_coverage}, below the required {MIN_PATIENTS_PER_CLASS_IN_VAL}. "
            f"Recompute per-class patient counts (see printout below) and either lower "
            f"MIN_PATIENTS_PER_CLASS_IN_VAL with justification, or adjust TARGET_VAL_FRACTION.\n"
            f"Per-class patient counts in train pool:\n{patient_counts_per_class(train_pool)}"
        )

    val_df = train_pool.iloc[best_val_idx].copy()
    train_df = train_pool.iloc[best_train_idx].copy()

    print(f"Chosen fold {best_fold_idx}/{n_splits}, min per-class patient coverage in val = {best_min_coverage}")
    print("Per-class patient counts (train pool, for reference):")
    print(patient_counts_per_class(train_pool))
    print("\nPer-class patient counts (val):")
    print(patient_counts_per_class(val_df))
    print("\nPer-class patient counts (train):")
    print(patient_counts_per_class(train_df))

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "val.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)

    print(f"\ntrain: {len(train_df)} images / {train_df['patient_id'].nunique()} patients")
    print(f"val:   {len(val_df)} images / {val_df['patient_id'].nunique()} patients")
    print(f"test:  {len(test_df)} images / {test_df['patient_id'].nunique()} patients")


if __name__ == "__main__":
    build_splits()
