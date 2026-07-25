"""
src/data/class_weights.py

Computes inverse-frequency class weights from train.csv and persists them to
data/processed/class_weights.json for use in Phase 4 training.

Weighting formula: w_i = (N_total / (N_classes * N_i)) / mean(all w_i)
This normalizes so the average weight is 1.0, which avoids needing to tune
the learning rate relative to the loss scale when switching imbalance strategies.

Design choice: committed to class-weighted cross-entropy as the primary
imbalance strategy (see 04_MODEL_TRAINING.md §3 and configs/). This is simpler
than WeightedRandomSampler and produces equivalent gradient scaling, while
being easier to reason about when combined with other regularization.
"""
import json
import pandas as pd
from pathlib import Path

CANONICAL_CLASSES = [
    "Brain_Trans_cerebellum", "Brain_Trans_thalamic", "Brain_Trans_ventricular",
    "Fetal_abdomen", "Fetal_femur", "Fetal_thorax", "Maternal_cervix", "Other",
]

def compute_class_weights(train_csv: str = "data/splits/train.csv",
                           out_path: str = "data/processed/class_weights.json"):
    df = pd.read_csv(train_csv)
    counts = df["plane_label"].value_counts().reindex(CANONICAL_CLASSES).fillna(0)
    assert (counts > 0).all(), f"A class has zero training images: {counts[counts == 0]}"

    total = counts.sum()
    n_classes = len(CANONICAL_CLASSES)
    # inverse-frequency weighting, normalized so weights average to 1.0
    raw_weights = total / (n_classes * counts)
    weights = (raw_weights / raw_weights.mean()).to_dict()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(weights, f, indent=2)

    print("Class counts:\n", counts)
    print("\nComputed weights (inverse-frequency, normalized to mean=1.0):\n", weights)
    return weights

if __name__ == "__main__":
    compute_class_weights()
