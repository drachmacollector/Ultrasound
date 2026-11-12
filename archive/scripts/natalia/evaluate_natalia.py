import argparse
import sys
import logging
from collections import defaultdict
from pathlib import Path
from PIL import Image

sys.path.insert(0, ".")  # ensure `src` package is importable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import classification_report

from src.data.dataset import (
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CANONICAL_CLASSES,
)
from src.realtime.model_loader import load_inference_model

CKPT_PATH = "checkpoints/convnext_tiny/best.pt"
MANIFEST_CSV = "data/processed/natalia_manifest.csv"
LOG_DIR = Path("logs/eval")
OUT_DIR = Path("data/processed/eval_natalia")

BRAIN_SUBCLASSES = {
    "Brain_Trans_cerebellum",
    "Brain_Trans_thalamic",
    "Brain_Trans_ventricular",
}

class NataliaDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        # handle nan in strings
        self.df = self.df.fillna("")
        self.transform = transform
        self.base_dir = Path("data/raw/natalia_pbfus1")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.base_dir / str(row['studie']) / str(row['file_name'])
        
        # determine true label
        if row['collapsed_label'] == "Head":
            true_label = "Head"
        else:
            true_label = str(row['canonical_label'])

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image_np = np.array(image)
            image = self.transform(image=image_np)["image"]

        return image, true_label

def score_row(true_label: str, pred_class_name: str) -> bool:
    if true_label == "Head":
        return pred_class_name in BRAIN_SUBCLASSES
    return pred_class_name == true_label

def run_natalia_inference(
    ckpt_path: str,
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
):
    loaded = load_inference_model(ckpt_path)
    model = loaded.model
    device = loaded.device

    dataset = NataliaDataset(csv_path=csv_path, transform=loaded.transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    results = []
    with torch.no_grad():
        for tensors, true_labels in loader:
            tensors = tensors.to(device)
            logits = model(tensors)
            pred_indices = logits.argmax(dim=1).cpu().tolist()

            for i in range(len(true_labels)):
                pred_name = IDX_TO_CLASS[pred_indices[i]]
                true_lbl = true_labels[i]
                correct = score_row(true_lbl, pred_name)
                
                # Standard vs Other mapping
                is_true_standard = true_lbl in {"Head", "Fetal_abdomen", "Fetal_femur"}
                is_pred_standard = pred_name in BRAIN_SUBCLASSES or pred_name in {"Fetal_abdomen", "Fetal_femur"}
                
                results.append({
                    "true_label": true_lbl,
                    "pred_class_name": pred_name,
                    "correct": correct,
                    "true_is_standard": is_true_standard,
                    "pred_is_standard": is_pred_standard
                })

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=CKPT_PATH)
    parser.add_argument("--manifest", default=MANIFEST_CSV)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger(__name__)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Running NatalIA inference (manifest: %s) …", args.manifest)
    results = run_natalia_inference(args.ckpt, args.manifest, args.batch_size, args.num_workers)
    log.info("Inference complete — %d images evaluated.", len(results))

    # Output CSV
    results_df = pd.DataFrame(results)
    csv_out = OUT_DIR / "natalia_results.csv"
    results_df.to_csv(csv_out, index=False, encoding="utf-8")
    log.info("Per-image CSV saved → %s", csv_out)

    # Metrics
    n_correct = sum(1 for r in results if r["correct"])
    n_total = len(results)
    overall_accuracy = n_correct / n_total if n_total > 0 else 0

    # Per-class metrics
    y_true = [r["true_label"] for r in results]
    
    # Map predictions to Head if they are brain subclasses for the classification report
    y_pred = []
    for r in results:
        p = r["pred_class_name"]
        if p in BRAIN_SUBCLASSES:
            y_pred.append("Head")
        else:
            y_pred.append(p)
            
    # filter y_pred to only valid classes for warning free report
    valid_classes = {"Head", "Fetal_abdomen", "Fetal_femur", "Other"}
    y_pred_filtered = [p if p in valid_classes else "Other" for p in y_pred]

    cls_report = classification_report(y_true, y_pred_filtered, labels=["Head", "Fetal_abdomen", "Fetal_femur", "Other"], zero_division=0)
    
    # Standard vs Other metrics
    # Positive class = Standard Plane, Negative class = Other
    true_standard = [r["true_is_standard"] for r in results]
    pred_standard = [r["pred_is_standard"] for r in results]
    
    tp = sum(1 for t, p in zip(true_standard, pred_standard) if t and p)
    tn = sum(1 for t, p in zip(true_standard, pred_standard) if not t and not p)
    fp = sum(1 for t, p in zip(true_standard, pred_standard) if not t and p)
    fn = sum(1 for t, p in zip(true_standard, pred_standard) if t and not p)

    standard_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    other_precision = tn / (tn + fn) if (tn + fn) > 0 else 0 # Precision on Other = TN / (TN + FN)
    
    # Trivial baseline (always predict Other)
    # accuracy if always predicting Other
    trivial_acc = true_standard.count(False) / len(true_standard)

    log_path = LOG_DIR / "evaluate_natalia.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        def w(msg):
            log.info(msg)
            f.write(msg + "\n")
            
        w("NATALIA PBF-US1 EVALUATION")
        w(f"Checkpoint : {args.ckpt}")
        w(f"Manifest   : {args.manifest}")
        w("")
        w("1. OVERALL ACCURACY (All Included Classes)")
        w(f"  {n_correct} / {n_total} correct  →  {overall_accuracy:.4f}  ({overall_accuracy * 100:.1f}%)")
        w("")
        w("2. PER-CLASS METRICS (Collapsed Head)")
        w(cls_report)
        w("")
        w("3. STANDARD-VS-OTHER BINARY EVALUATION")
        w(f"  True Standard Planes : {true_standard.count(True)}")
        w(f"  True Other Planes    : {true_standard.count(False)}")
        w(f"  Standard Plane Recall: {standard_recall:.4f}")
        w(f"  'Other' Precision    : {other_precision:.4f}")
        w(f"  Model Binary Accuracy: {(tp+tn)/len(results):.4f}")
        w(f"  Trivial Baseline Acc : {trivial_acc:.4f} (always predict 'Other')")
        w("")
        w("4. CONFUSION BREAKDOWN FOR MISCLASSIFIED STANDARD PLANES")
        w("  When a standard plane is wrong, what does it predict?")
        
        misses = defaultdict(int)
        for r in results:
            if r["true_is_standard"] and not r["correct"]:
                misses[r["pred_class_name"]] += 1
                
        if misses:
            for pred_cls, cnt in sorted(misses.items(), key=lambda x: -x[1]):
                w(f"    {pred_cls:<30s}  {cnt:>4d}")
        else:
            w("    No standard planes misclassified.")

    # Confusion breakdown chart
    if misses:
        labels = sorted(misses.keys(), key=lambda k: -misses[k])
        values = [misses[k] for k in labels]
        
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(labels, values, color="#4C72B0", edgecolor="white")
        ax.bar_label(bars, fmt="%d")
        ax.set_title("NatalIA Standard Plane Misclassifications")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        fig.savefig(str(OUT_DIR / "natalia_confusion.png"), dpi=150)
        plt.close(fig)

    log.info("Summary log saved → %s", log_path)

if __name__ == "__main__":
    main()
