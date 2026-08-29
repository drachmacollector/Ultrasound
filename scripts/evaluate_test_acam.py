"""
from __future__ import annotations

scripts/evaluate_test_acam.py

Evaluates the ACAM checkpoint (checkpoints/convnext_tiny_acam/best.pt) on
the held-out test set (data/splits/test.csv), then runs a paired bootstrap
significance test (2,000 iterations) comparing:

    convnext_tiny + ACAM  vs.  plain convnext_tiny (baseline)

§ Cross-reference: Phase 8, Stage 5, Task 5.2
  (docs/instructions/08_FINAL_EVALUATION_AND_POLISH.md §6, Task 5.2)

Baseline numbers used:
  - Baseline checkpoint: checkpoints/convnext_tiny/best.pt
  - Baseline test macro-F1: 0.8927 (from docs/phases/phase_04/bootstrap_significance_output.txt)
  - Bootstrap protocol: 2,000 iterations, n=5,271, matching the exact protocol
    already used in EXPERIMENTS.md and reproducible from
    docs/phases/phase_04/bootstrap_significance_output.txt

Verdict language (matching the three-way system already in use):
  - "robust win  (CI > 0)"         : both bounds of 95% CI of (ACAM − baseline) > 0
  - "robust loss (CI < 0)"         : both bounds < 0
  - "statistically tied (CI ∋ 0)"  : CI straddles 0

Outputs:
  checkpoints/convnext_tiny_acam/classification_report_TEST.txt
  checkpoints/convnext_tiny_acam/confusion_matrix_TEST.png
  logs/eval/bootstrap_significance_acam.txt

Usage:
    conda run -n fetalplane python -m scripts.evaluate_test_acam
    # or directly:
    conda run -n fetalplane python scripts/evaluate_test_acam.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CANONICAL_CLASSES, NUM_CLASSES, FocalPlanesDataset
from src.data.transforms import get_eval_transform
from src.eval.metrics_utils import save_confusion_matrix
from src.models.acam import build_acam
from src.models.backbone import build_model

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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
ACAM_CKPT = Path("checkpoints/convnext_tiny_acam/best.pt")
BASELINE_CKPT = Path("checkpoints/convnext_tiny/best.pt")
BOOTSTRAP_ITERS = 2_000
BOOTSTRAP_SEED = 42

# Baseline test macro-F1 from Phase 4 bootstrap evaluation
# Source: docs/phases/phase_04/bootstrap_significance_output.txt
BASELINE_KNOWN_TEST_F1 = 0.8927

LOG_DIR = Path("logs/eval")
ACAM_CKPT_DIR = Path("checkpoints/convnext_tiny_acam")


# ---------------------------------------------------------------------------
# Model loader (ACAM-aware)
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, dict]:  # type: ignore[type-arg]
    """Load a checkpoint that may or may not include ACAM wrapping.

    Returns:
        (model, cfg) — the assembled nn.Module and the config dict.
    """
    log.info("Loading checkpoint: %s", ckpt_path)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg: dict = state["config"]  # type: ignore[assignment]
    use_acam: bool = bool(state.get("use_acam", cfg.get("use_acam", False)))

    backbone_name: str = cfg["backbone"]
    backbone = build_model(backbone_name, num_classes=NUM_CLASSES, pretrained=False)

    if use_acam:
        acam = build_acam(cfg)
        model: nn.Module = nn.Sequential(acam, backbone)
        log.info("  ACAM wrapper active (K=%d, α∈[%.1f,%.1f])", acam.K, acam.alpha_min, acam.alpha_max)
    else:
        model = backbone

    model.load_state_dict(state["model_state_dict"])
    model = model.to(device)
    model.eval()
    log.info("  backbone=%s  val_macro_f1=%.4f (training)", backbone_name, state.get("val_macro_f1", float("nan")))
    return model, cfg


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(ckpt_path: Path, device: torch.device) -> tuple[list[int], list[int], dict]:  # type: ignore[type-arg]
    """Run inference over test.csv using the given checkpoint.

    Returns:
        (all_targets, all_preds, cfg)
    """
    model, cfg = load_model_from_checkpoint(ckpt_path, device)

    img_size: int = cfg.get("image_size", 224)
    normalize_mean: tuple = tuple(cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
    normalize_std: tuple = tuple(cfg.get("normalize_std", [0.229, 0.224, 0.225]))
    batch_size: int = cfg.get("batch_size", 64)
    num_workers: int = cfg.get("num_workers", 4)

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
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images)
        preds = logits.argmax(dim=1)
        all_targets.extend(labels.tolist())
        all_preds.extend(preds.cpu().tolist())

    test_f1: float = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    log.info("  Test macro-F1 = %.4f", test_f1)
    return all_targets, all_preds, cfg


# ---------------------------------------------------------------------------
# Bootstrap significance test
# ---------------------------------------------------------------------------

def bootstrap_ci(
    targets: list[int],
    preds_a: list[int],   # ACAM
    preds_b: list[int],   # baseline
    n_iters: int = BOOTSTRAP_ITERS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, float, float, float]:
    """Paired bootstrap CI on macro-F1 difference (ACAM − baseline).

    Returns:
        (f1_a, f1_b, delta_point, ci_lo, ci_hi)
    """
    rng = np.random.default_rng(seed)
    n = len(targets)
    t_arr = np.array(targets)
    a_arr = np.array(preds_a)
    b_arr = np.array(preds_b)

    f1_a: float = f1_score(t_arr, a_arr, average="macro", zero_division=0)
    f1_b: float = f1_score(t_arr, b_arr, average="macro", zero_division=0)
    delta_point: float = f1_a - f1_b

    log.info("Running %d bootstrap iterations (n=%d, seed=%d)...", n_iters, n, seed)
    deltas: list[float] = []
    for _ in range(n_iters):
        idx = rng.integers(0, n, size=n)
        d = (
            f1_score(t_arr[idx], a_arr[idx], average="macro", zero_division=0)
            - f1_score(t_arr[idx], b_arr[idx], average="macro", zero_division=0)
        )
        deltas.append(d)

    deltas_arr = np.array(deltas)
    ci_lo: float = float(np.percentile(deltas_arr, 2.5))
    ci_hi: float = float(np.percentile(deltas_arr, 97.5))
    return f1_a, f1_b, delta_point, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

def verdict(ci_lo: float, ci_hi: float) -> str:
    """Return the three-way verdict string matching the project's existing language."""
    if ci_lo > 0:
        return "robust win  (CI > 0)"
    elif ci_hi < 0:
        return "robust loss (CI < 0)"
    else:
        return "statistically tied (CI ∋ 0)"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "bootstrap_significance_acam.txt"

    # Add file handler
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logging.getLogger().addHandler(fh)

    log.info("=" * 72)
    log.info("ACAM Ablation — Test-Set Evaluation + Bootstrap Significance Test")
    log.info("Phase 8, Stage 5, Task 5.2  (arXiv:2509.00808)")
    log.info("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # ------------------------------------------------------------------ #
    # 1. ACAM checkpoint — test-set evaluation
    # ------------------------------------------------------------------ #
    log.info("-" * 60)
    log.info("Step 1: ACAM checkpoint inference")
    acam_targets, acam_preds, acam_cfg = run_inference(ACAM_CKPT, device)

    acam_report = classification_report(acam_targets, acam_preds, target_names=CANONICAL_CLASSES, zero_division=0)
    acam_test_f1: float = f1_score(acam_targets, acam_preds, average="macro", zero_division=0)

    # Save per-class report
    report_path = ACAM_CKPT_DIR / "classification_report_TEST.txt"
    header = (
        "TEST SET EVALUATION — convnext_tiny + ACAM (arXiv:2509.00808)\n"
        "Source: data/splits/test.csv (5,271 images / 896 patients)\n"
        "Checkpoint: checkpoints/convnext_tiny_acam/best.pt (epoch 16, val macro-F1=0.9129)\n"
        "This is the genuine held-out in-distribution test set.\n"
        f"{'=' * 60}\n\n"
    )
    report_path.write_text(header + acam_report, encoding="utf-8")
    log.info("Per-class report → %s", report_path)

    # Save confusion matrix
    cm_path = ACAM_CKPT_DIR / "confusion_matrix_TEST.png"
    save_confusion_matrix(acam_targets, acam_preds, cm_path, epoch=0)
    log.info("Confusion matrix → %s", cm_path)

    # ------------------------------------------------------------------ #
    # 2. Baseline checkpoint — test-set inference (for paired bootstrap)
    # ------------------------------------------------------------------ #
    log.info("-" * 60)
    log.info("Step 2: Baseline (convnext_tiny) checkpoint inference")
    baseline_targets, baseline_preds, _ = run_inference(BASELINE_CKPT, device)
    baseline_test_f1: float = f1_score(baseline_targets, baseline_preds, average="macro", zero_division=0)
    log.info("Baseline test macro-F1 = %.4f  (documented: %.4f)", baseline_test_f1, BASELINE_KNOWN_TEST_F1)

    # Sanity-check targets are consistent
    assert acam_targets == baseline_targets, (
        "Target lists don't match between ACAM and baseline — same test.csv must be used"
    )

    # ------------------------------------------------------------------ #
    # 3. Paired bootstrap significance test
    # ------------------------------------------------------------------ #
    log.info("-" * 60)
    log.info("Step 3: Paired bootstrap (2,000 iterations)")
    f1_a, f1_b, delta_point, ci_lo, ci_hi = bootstrap_ci(
        acam_targets, acam_preds, baseline_preds
    )
    result_verdict = verdict(ci_lo, ci_hi)

    # ------------------------------------------------------------------ #
    # 4. Write structured output
    # ------------------------------------------------------------------ #
    sep = "=" * 72
    output_lines = [
        sep,
        "BOOTSTRAP SIGNIFICANCE — 2,000 iterations, n=5,271 images",
        "Comparison: convnext_tiny + ACAM  vs.  convnext_tiny (baseline)",
        "Phase 8, Stage 5  (08_FINAL_EVALUATION_AND_POLISH.md §6, Task 5.2)",
        sep,
        "",
        f"  ACAM     checkpoint : checkpoints/convnext_tiny_acam/best.pt",
        f"  Baseline checkpoint : checkpoints/convnext_tiny/best.pt",
        "",
        f"  ACAM     test macro-F1 : {f1_a:.4f}",
        f"  Baseline test macro-F1 : {f1_b:.4f}",
        "",
        f"  Point Δ (ACAM − baseline) : {delta_point:+.4f}",
        f"  Bootstrap 95% CI of Δ     : [{ci_lo:+.4f}, {ci_hi:+.4f}]",
        "",
        f"  VERDICT: {result_verdict}",
        "",
        sep,
        "",
        "Verdict language (matches project standard from EXPERIMENTS.md):",
        "  robust win  (CI > 0)       — both CI bounds > 0  → ACAM robustly outperforms baseline",
        "  statistically tied (CI ∋ 0)— CI straddles 0      → no reliable difference",
        "  robust loss (CI < 0)       — both CI bounds < 0  → ACAM robustly underperforms",
        "",
        sep,
        "",
        "ACAM per-class test-set performance:",
        "",
        acam_report,
        "",
        "Full log: logs/eval/bootstrap_significance_acam.txt",
    ]

    output_text = "\n".join(output_lines)

    # Write to log file
    log_path.write_text(output_text, encoding="utf-8")

    # Also print to stdout
    print("\n" + output_text)

    log.info("Bootstrap output → %s", log_path)
    log.info("DONE.")


if __name__ == "__main__":
    main()
