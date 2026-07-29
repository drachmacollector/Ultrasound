"""Evaluate the focal-loss convnext_tiny checkpoint on the real test set."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.eval.evaluate_test import evaluate_checkpoint
from src.eval.metrics_utils import save_confusion_matrix

ckpt = Path("checkpoints/convnext_tiny_focal/best.pt")
r = evaluate_checkpoint(ckpt)

# Save classification report
header = (
    "TEST SET EVALUATION — convnext_tiny_focal\n"
    "Source: data/splits/test.csv (5,271 images / 896 patients)\n"
    "Ablation: Focal Loss (gamma=2.0) + class-weighted, vs. CE baseline in convnext_tiny.\n"
    "=" * 60 + "\n\n"
)
p = Path("checkpoints/convnext_tiny_focal/classification_report_TEST.txt")
p.write_text(header + r["report_str"], encoding="utf-8")
print(f"Saved → {p}")

# Save confusion matrix
save_confusion_matrix(
    r["all_targets"], r["all_preds"],
    Path("checkpoints/convnext_tiny_focal/confusion_matrix_TEST.png"),
    epoch=0,
)

print(r["report_str"])
print(f"Test macro-F1: {r['test_macro_f1']:.4f}")
