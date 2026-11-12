"""
src/eval/evaluate_test.py

Evaluates every trained backbone checkpoint against the genuine held-out
data/splits/test.csv — the 5,271-image / 896-patient in-distribution test set
that was NEVER seen during training or checkpoint selection.

This supersedes the per-backbone classification_report_final.txt files, which
are derived from the val set (used for early stopping) and are therefore
optimistic relative to the true test set.

Usage:
    conda run -n fetalplane python -m src.eval.evaluate_test

Outputs per backbone:
    checkpoints/<backbone>/classification_report_TEST.txt
    checkpoints/<backbone>/confusion_matrix_TEST.png

Aggregate output:
    docs/test_classification_metrics_all.txt

Prints to stdout:
    Summary table — backbone | val macro-F1 | test macro-F1 | gap
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

from src.data.dataset import CANONICAL_CLASSES, NUM_CLASSES, FocalPlanesDataset
from src.data.transforms import get_eval_transform
from src.eval.metrics_utils import save_confusion_matrix
from src.models.backbone import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_CSV = "data/splits/test.csv"
DOCS_DIR = Path("docs")

# Ordered list of checkpoint dirs to evaluate.  Each must contain best.pt.
CHECKPOINT_DIRS: list[Path] = [
    Path("checkpoints/efficientnet_lite0"),
    Path("checkpoints/mobilenetv3_large_100"),
    Path("checkpoints/repvgg_a1"),
    Path("checkpoints/tf_efficientnetv2_s"),
    Path("checkpoints/repvgg_a2"),
    Path("checkpoints/convnext_tiny"),
]

# Known val macro-F1s from training (for the gap comparison table).
# Source: logs/validation_classification_metrics_all.txt
VAL_MACRO_F1: dict[str, float] = {
    "efficientnet_lite0": 0.9112,
    "mobilenetv3_large_100": 0.9045,
    "repvgg_a1": 0.8952,
    "tf_efficientnetv2_s.in21k_ft_in1k": 0.9052,
    "repvgg_a2": 0.9228,
    "convnext_tiny.fb_in22k_ft_in1k": 0.9180,
}

AGGREGATE_HEADER = """\
All models evaluated on data/splits/test.csv — the genuine in-distribution
held-out test set (5,271 images / 896 patients), NEVER used for early
stopping or checkpoint selection. This supersedes any prior file that
used validation-set numbers.
See docs/validation_classification_metrics_all.txt for the val-set numbers
(used only for checkpoint selection / early stopping, NOT reported numbers).

"""


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint(ckpt_path: Path) -> dict:
    """Load a best.pt checkpoint and evaluate it on test.csv.

    Returns:
        dict with keys: backbone_name, test_macro_f1, all_targets, all_preds,
                        report_str, img_size, normalize_mean, normalize_std
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading checkpoint: %s", ckpt_path)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    # Pull all config from the saved checkpoint — never hardcode
    cfg: dict = ckpt["config"]
    backbone_name: str = cfg["backbone"]
    img_size: int = cfg.get("image_size", 224)
    normalize_mean: tuple = tuple(cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
    normalize_std: tuple = tuple(cfg.get("normalize_std", [0.229, 0.224, 0.225]))
    batch_size: int = cfg.get("batch_size", 32)
    num_workers: int = cfg.get("num_workers", 4)

    log.info(
        "  backbone=%s  img_size=%d  mean=%s  std=%s",
        backbone_name, img_size, normalize_mean, normalize_std,
    )

    # Build model and load weights
    model = build_model(backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Dataset + DataLoader over test.csv
    transform = get_eval_transform(img_size=img_size, mean=normalize_mean, std=normalize_std)
    test_dataset = FocalPlanesDataset(TEST_CSV, transform=transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    log.info("  Running inference on %d test images...", len(test_dataset))

    all_targets: list[int] = []
    all_preds: list[int] = []

    for images, labels in test_loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)

        preds = logits.argmax(dim=1)
        all_targets.extend(labels.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    test_macro_f1: float = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    report_str: str = classification_report(
        all_targets,
        all_preds,
        target_names=CANONICAL_CLASSES,
        zero_division=0,
    )

    log.info("  Test macro-F1 = %.4f", test_macro_f1)

    return {
        "backbone_name": backbone_name,
        "test_macro_f1": test_macro_f1,
        "all_targets": all_targets,
        "all_preds": all_preds,
        "report_str": report_str,
        "img_size": img_size,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    aggregate_lines: list[str] = [AGGREGATE_HEADER]

    for ckpt_dir in CHECKPOINT_DIRS:
        ckpt_path = ckpt_dir / "best.pt"
        if not ckpt_path.exists():
            log.warning("Checkpoint not found, skipping: %s", ckpt_path)
            continue

        result = evaluate_checkpoint(ckpt_path)
        results.append(result)

        backbone_name = result["backbone_name"]

        # --- Per-backbone text report ---
        report_path = ckpt_dir / "classification_report_TEST.txt"
        header = (
            f"TEST SET EVALUATION — {backbone_name}\n"
            f"Source: data/splits/test.csv (5,271 images / 896 patients)\n"
            f"This is the genuine held-out in-distribution test set.\n"
            f"{'=' * 60}\n\n"
        )
        report_path.write_text(header + result["report_str"], encoding="utf-8")
        log.info("  Saved → %s", report_path)

        # --- Per-backbone confusion matrix ---
        cm_path = ckpt_dir / "confusion_matrix_TEST.png"
        save_confusion_matrix(result["all_targets"], result["all_preds"], cm_path, epoch=0)

        # --- Aggregate section ---
        aggregate_lines.append(f"{'=' * 60}")
        aggregate_lines.append(f"Backbone: {backbone_name}")
        aggregate_lines.append(f"Test macro-F1: {result['test_macro_f1']:.4f}")
        aggregate_lines.append("")
        aggregate_lines.append(result["report_str"])
        aggregate_lines.append("")

    # --- Write aggregate file ---
    aggregate_path = DOCS_DIR / "test_classification_metrics_all.txt"
    aggregate_path.write_text("\n".join(aggregate_lines), encoding="utf-8")
    log.info("Aggregate test report → %s", aggregate_path)

    # --- Print summary comparison table ---
    print("\n" + "=" * 75)
    print("BACKBONE COMPARISON — Val macro-F1 (selection criterion) vs. Test macro-F1 (real)")
    print("=" * 75)
    print(f"{'Backbone':<40} {'Val F1':>8} {'Test F1':>9} {'Gap':>8}")
    print("-" * 75)

    results_sorted = sorted(results, key=lambda r: r["test_macro_f1"], reverse=True)
    for r in results_sorted:
        bname = r["backbone_name"]
        test_f1 = r["test_macro_f1"]
        val_f1 = VAL_MACRO_F1.get(bname, float("nan"))
        gap = test_f1 - val_f1
        gap_str = f"{gap:+.4f}"
        print(f"{bname:<40} {val_f1:>8.4f} {test_f1:>9.4f} {gap_str:>8}")

    print("=" * 75)
    print(f"\nAggregate test report → {aggregate_path}")
    print("Per-backbone reports  → checkpoints/<backbone>/classification_report_TEST.txt")
    print("Per-backbone matrices → checkpoints/<backbone>/confusion_matrix_TEST.png")
    print("\n[MANUAL PAUSE] — Review the gap table and confusion_matrix_TEST.png for")
    print("repvgg_a2 before confirming the backbone decision and proceeding to B.4.")


if __name__ == "__main__":
    main()
