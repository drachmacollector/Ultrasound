"""
scripts/evaluate_cross_device.py

Evaluates any backbone checkpoint on data/processed/cross_device_manifest.csv
(HC18 + UCL only — FP/MULTICENTRE are excluded upstream by build_cross_device_manifest.py
and CrossDeviceDataset's own guardrail assert).

Phase 8, Stage 4 (Task 4.1): extended to support multi-backbone runs via --ckpt.
Output filenames are derived from the checkpoint directory stem so successive runs
never overwrite each other (e.g. repvgg_a2 → logs/eval/evaluate_cross_device_repvgg_a2.txt).

Scoring rule for the collapsed "Head" label:
  A "Head" row is scored CORRECT if the model's argmax is ANY of:
    {Brain_Trans_cerebellum, Brain_Trans_thalamic, Brain_Trans_ventricular}
  "Fetal_abdomen" / "Fetal_femur" rows use exact-match scoring (single canonical class each).

No "Other" class exists in this manifest by construction (every image is a valid standard
plane) -- this evaluation validates plane-IDENTITY classification cross-device, NOT the
standard-vs-Other decision. State this explicitly in the report.

No "Fetal_thorax" / "Maternal_cervix" coverage exists in ANY available external dataset --
state this explicitly too, do not imply those two classes were cross-device validated.

Outputs (backbone-suffixed — never overwrites another backbone's artifacts):
  logs/eval/evaluate_cross_device_{backbone}.txt
  data/processed/eval_cross_device/{backbone}_cross_device_results.csv
  data/processed/eval_cross_device/{backbone}_cross_device_confusion.png
"""
from __future__ import annotations

import argparse
import sys
import logging
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")  # ensure `src` package is importable when run from repo root

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# --- project imports (reuse existing modules per §0.4 constraint) ---------
from src.data.dataset import (
    CrossDeviceDataset,
    FocalPlanesDataset,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CANONICAL_CLASSES,
)
from src.realtime.model_loader import load_inference_model

# -------------------------------------------------------------------------

CKPT_PATH = "checkpoints/convnext_tiny/best.pt"
CROSS_DEVICE_CSV = "data/processed/cross_device_manifest.csv"
TEST_CSV = "data/splits/test.csv"
LOG_DIR = Path("logs/eval")
OUT_DIR = Path("data/processed/eval_cross_device")

BRAIN_SUBCLASSES: set[str] = {
    "Brain_Trans_cerebellum",
    "Brain_Trans_thalamic",
    "Brain_Trans_ventricular",
}


def score_row(true_label: str, pred_class_name: str) -> bool:
    """Collapsed-label scoring per Phase 6 spec:
    - 'Head' rows: correct iff prediction is ANY brain sub-plane.
    - Non-head rows: exact match only.
    """
    if true_label == "Head":
        return pred_class_name in BRAIN_SUBCLASSES
    return pred_class_name == true_label


def _assert_not_multitask_contaminated(ckpt_path: str) -> None:
    """Hard-fail if this checkpoint's training data included HC18/UCL,
    since that invalidates cross-device generalization evaluation."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_csv = ckpt.get("config", {}).get("train_csv", "")
    is_multitask_trained = "multitask" in train_csv
    if is_multitask_trained:
        raise RuntimeError(
            f"REFUSING to evaluate {ckpt_path} against cross_device_manifest.csv: "
            f"its train_csv ({train_csv}) is the multitask manifest, which folds "
            f"HC18/UCL into training data (see build_multitask_manifest.py). "
            f"HC18/UCL are NOT held-out for this checkpoint. See EVAL_REPORT.md §2 warning."
        )

def run_cross_device_inference(
    ckpt_path: str,
    csv_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> list[dict]:
    """Run inference over cross_device_manifest.csv and return per-row result dicts."""
    _assert_not_multitask_contaminated(ckpt_path)
    loaded = load_inference_model(ckpt_path)
    model = loaded.model
    device = loaded.device

    # CrossDeviceDataset uses loaded.transform internally via get_eval_transform()
    # but we must pass the model's own transform so img_size/mean/std are consistent.
    dataset = CrossDeviceDataset(
        csv_path=csv_path,
        transform=loaded.transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    results = []
    with torch.no_grad():
        for batch in loader:
            tensors, true_labels, is_collapsed, source_subsets = batch
            tensors = tensors.to(device)
            logits = model(tensors)
            pred_indices = logits.argmax(dim=1).cpu().tolist()

            for i in range(len(true_labels)):
                pred_name = IDX_TO_CLASS[pred_indices[i]]
                true_lbl = true_labels[i]
                correct = score_row(true_lbl, pred_name)
                results.append({
                    "true_label": true_lbl,
                    "pred_class_name": pred_name,
                    "source_subset": source_subsets[i],
                    "correct": correct,
                    "is_collapsed": bool(is_collapsed[i]),
                })

    return results


def run_indist_collapsed_baselines(
    ckpt_path: str,
    test_csv: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict[str, float]:
    """Compute in-distribution collapsed-Head, Fetal_abdomen, and Fetal_femur accuracies
    from test.csv as the apples-to-apples baselines for the generalization gap table.

    Collapsed-Head: a brain-subclass image is scored correct if the argmax is also a brain
    subclass (any-to-any), matching the cross-device collapsed-Head scoring rule exactly.
    Abdomen/Femur use exact-match.
    """
    loaded = load_inference_model(ckpt_path)
    model = loaded.model
    device = loaded.device

    dataset = FocalPlanesDataset(csv_path=test_csv, transform=loaded.transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    per_class_correct: dict[str, int] = defaultdict(int)
    per_class_total: dict[str, int] = defaultdict(int)
    brain_correct = 0
    brain_total = 0

    with torch.no_grad():
        for tensors, label_indices in loader:
            tensors = tensors.to(device)
            logits = model(tensors)
            pred_indices = logits.argmax(dim=1).cpu().tolist()
            label_indices = label_indices.tolist()

            for true_idx, pred_idx in zip(label_indices, pred_indices):
                true_name = IDX_TO_CLASS[true_idx]
                pred_name = IDX_TO_CLASS[pred_idx]

                if true_name in BRAIN_SUBCLASSES:
                    brain_total += 1
                    if pred_name in BRAIN_SUBCLASSES:
                        brain_correct += 1
                elif true_name in ("Fetal_abdomen", "Fetal_femur"):
                    per_class_total[true_name] += 1
                    if pred_name == true_name:
                        per_class_correct[true_name] += 1

    baselines: dict[str, float] = {}
    baselines["Head"] = brain_correct / brain_total if brain_total > 0 else float("nan")
    for cls in ("Fetal_abdomen", "Fetal_femur"):
        total = per_class_total[cls]
        baselines[cls] = per_class_correct[cls] / total if total > 0 else float("nan")

    return baselines


def save_head_misclass_bar(
    results: list[dict],
    out_path: Path,
) -> None:
    """Bar chart of non-brain predicted class for misclassified Head rows.

    Uses matplotlib directly (consistent with src.eval.metrics_utils plotting style).
    Saved to out_path.
    """
    head_misses = [
        r["pred_class_name"]
        for r in results
        if r["true_label"] == "Head" and not r["correct"]
    ]
    if not head_misses:
        logging.getLogger(__name__).info(
            "No Head misclassifications — skipping bar chart."
        )
        return

    counts: dict[str, int] = defaultdict(int)
    for pred in head_misses:
        counts[pred] += 1

    labels = sorted(counts.keys(), key=lambda k: -counts[k])
    values = [counts[k] for k in labels]
    short_labels = [
        lbl.replace("Brain_Trans_", "BT_")
           .replace("Fetal_", "F_")
           .replace("Maternal_", "M_")
        for lbl in labels
    ]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    bars = ax.bar(short_labels, values, color="#4C72B0", edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=10)
    ax.set_title("Head row misclassifications — predicted class distribution", fontsize=13)
    ax.set_xlabel("Predicted class (non-brain)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_ylim(0, max(values) * 1.25)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logging.getLogger(__name__).info("Head-misclass bar chart saved → %s", out_path)


def build_summary_tables(results: list[dict]) -> dict:
    """Compute all required metrics from raw per-row results.

    Returns a dict with keys:
        overall_accuracy         -- combined HC18+UCL, all labels
        per_subset               -- {subset: {label: {correct, total, accuracy}}}
        head_misclass_breakdown  -- {pred_class: count} for misclassified Head rows
    """
    per_subset: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))

    for r in results:
        subset = r["source_subset"]
        lbl = r["true_label"]
        per_subset[subset][lbl]["total"] += 1
        if r["correct"]:
            per_subset[subset][lbl]["correct"] += 1

    # Compute accuracies
    for subset in per_subset:
        for lbl in per_subset[subset]:
            entry = per_subset[subset][lbl]
            entry["accuracy"] = entry["correct"] / entry["total"] if entry["total"] > 0 else float("nan")

    # Overall accuracy (combined)
    n_correct = sum(1 for r in results if r["correct"])
    n_total = len(results)
    overall_accuracy = n_correct / n_total if n_total > 0 else float("nan")

    # Head misclass breakdown
    head_misclass: dict[str, int] = defaultdict(int)
    for r in results:
        if r["true_label"] == "Head" and not r["correct"]:
            head_misclass[r["pred_class_name"]] += 1

    return {
        "overall_accuracy": overall_accuracy,
        "n_correct": n_correct,
        "n_total": n_total,
        "per_subset": per_subset,
        "head_misclass_breakdown": dict(head_misclass),
    }


def format_summary(
    metrics: dict,
    indist_baselines: dict[str, float],
) -> str:
    """Format the full summary text for logs/eval/evaluate_cross_device.txt."""
    lines: list[str] = []

    def section(title: str) -> None:
        lines.append("")
        lines.append("=" * 70)
        lines.append(title)
        lines.append("=" * 70)

    lines.append("CROSS-DEVICE GENERALIZATION EVALUATION")
    lines.append("Checkpoint : checkpoints/convnext_tiny/best.pt")
    lines.append("Manifest   : data/processed/cross_device_manifest.csv (HC18 + UCL only)")
    lines.append("")
    lines.append("SCOPE LIMITATIONS (explicit, per Phase 6 spec):")
    lines.append("  1. Every image in HC18/UCL is a valid standard plane by construction.")
    lines.append("     This evaluation validates plane-IDENTITY classification cross-device,")
    lines.append("     NOT the standard-vs-Other decision.")
    lines.append("  2. HC18 and UCL cover Head, Fetal_abdomen, Fetal_femur ONLY.")
    lines.append("     No 'Fetal_thorax' or 'Maternal_cervix' cross-device validation was")
    lines.append("     possible — no suitable external dataset covers those two classes.")
    lines.append("  3. FP and MULTICENTRE subfolders are EXCLUDED (overlap with FETAL_PLANES_DB")
    lines.append("     training data). See 00_PROJECT_OVERVIEW.md §5a.")
    lines.append("")
    lines.append("COLLAPSED-LABEL SCORING RULE:")
    lines.append("  Head row → CORRECT if argmax ∈ {Brain_Trans_cerebellum,")
    lines.append("             Brain_Trans_thalamic, Brain_Trans_ventricular}")
    lines.append("  Fetal_abdomen / Fetal_femur → exact-match")

    # ---- Overall accuracy ----
    section("1. OVERALL ACCURACY (HC18 + UCL combined, collapsed labels)")
    n_c = metrics["n_correct"]
    n_t = metrics["n_total"]
    acc = metrics["overall_accuracy"]
    lines.append(f"  {n_c} / {n_t} correct  →  {acc:.4f}  ({acc * 100:.1f}%)")

    # ---- Per-subset, per-label ----
    section("2. PER-SUBSET ACCURACY (HC18 vs UCL reported SEPARATELY — never blended)")
    per_subset = metrics["per_subset"]
    for subset in sorted(per_subset.keys()):
        lines.append(f"\n  [{subset}]")
        subset_correct = 0
        subset_total = 0
        for lbl in sorted(per_subset[subset].keys()):
            e = per_subset[subset][lbl]
            c, t, a = e["correct"], e["total"], e["accuracy"]
            subset_correct += c
            subset_total += t
            lines.append(f"    {lbl:<20s}  {c:>4d} / {t:>4d}  →  {a:.4f}  ({a * 100:.1f}%)")
        subset_acc = subset_correct / subset_total if subset_total > 0 else float("nan")
        lines.append(f"    {'SUBSET TOTAL':<20s}  {subset_correct:>4d} / {subset_total:>4d}  →  {subset_acc:.4f}  ({subset_acc * 100:.1f}%)")

    # ---- Generalization gap ----
    section("3. GENERALIZATION GAP vs IN-DISTRIBUTION BASELINE")
    lines.append("  (In-distribution baseline uses collapsed scoring on test.csv,")
    lines.append("   matching the cross-device collapsed-label task exactly.)")
    lines.append("")
    lines.append(f"  {'Label':<20s}  {'InDist Acc':>12s}  {'CrossDev Acc':>12s}  {'Gap':>10s}")
    lines.append(f"  {'-'*60}")

    all_labels = ["Head", "Fetal_abdomen", "Fetal_femur"]
    cross_dev_collapsed: dict[str, float] = {}

    for lbl in all_labels:
        # aggregate cross-device accuracy for this label across all subsets
        c_total, c_correct = 0, 0
        for subset in per_subset.values():
            if lbl in subset:
                c_total += subset[lbl]["total"]
                c_correct += subset[lbl]["correct"]
        cross_acc = c_correct / c_total if c_total > 0 else float("nan")
        cross_dev_collapsed[lbl] = cross_acc

        indist_acc = indist_baselines.get(lbl, float("nan"))
        gap = cross_acc - indist_acc

        indist_str = f"{indist_acc:.4f}" if not np.isnan(indist_acc) else "N/A"
        cross_str = f"{cross_acc:.4f}" if not np.isnan(cross_acc) else "N/A"
        gap_str = f"{gap:+.4f}" if (not np.isnan(gap)) else "N/A"
        lines.append(f"  {lbl:<20s}  {indist_str:>12s}  {cross_str:>12s}  {gap_str:>10s}")

    # ---- Head misclass breakdown ----
    section("4. HEAD MISCLASSIFICATION BREAKDOWN")
    lines.append("  When a Head image is wrong, which class does the model predict?")
    lines.append("  (Brain→Brain sub-plane confusions are NOT misclassifications under the")
    lines.append("   collapsed scoring rule — they all count as CORRECT.)")
    lines.append("")
    hm = metrics["head_misclass_breakdown"]
    if hm:
        for pred_cls, cnt in sorted(hm.items(), key=lambda x: -x[1]):
            flag = "  <-- DOMAIN-SHIFT RED FLAG" if pred_cls not in BRAIN_SUBCLASSES else ""
            lines.append(f"    {pred_cls:<30s}  {cnt:>4d}{flag}")
    else:
        lines.append("    No Head misclassifications.")

    # ---- Gate check ----
    section("5. GATE CHECK")
    head_c, head_t = 0, 0
    for subset in per_subset.values():
        if "Head" in subset:
            head_c += subset["Head"]["correct"]
            head_t += subset["Head"]["total"]
    head_overall_acc = head_c / head_t if head_t > 0 else float("nan")
    gate_pass = head_overall_acc >= 0.50
    lines.append(f"  Combined Head accuracy: {head_overall_acc:.4f} ({head_overall_acc * 100:.1f}%)")
    lines.append(f"  Gate threshold: >= 50%")
    lines.append(f"  Gate result: {'PASS ✓' if gate_pass else 'FAIL ✗ — STOP AND INSPECT (see Phase 6 spec §3)'}")
    if not gate_pass:
        lines.append("")
        lines.append("  WARNING: Head accuracy is below 50% (worse than random-among-3).")
        lines.append("  This likely indicates a collapsed-scoring implementation bug.")
        lines.append("  Do NOT write the EVAL_REPORT.md cross-device section until resolved.")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 8 Stage 4: Cross-device generalization evaluation (HC18 + UCL). "
                    "Supports any backbone checkpoint via --ckpt. Output filenames are "
                    "suffixed with the checkpoint directory stem so runs never overwrite each other."
    )
    parser.add_argument("--ckpt", default=CKPT_PATH, help="Checkpoint path (e.g. checkpoints/repvgg_a2/best.pt)")
    parser.add_argument("--manifest", default=CROSS_DEVICE_CSV)
    parser.add_argument("--test-csv", default=TEST_CSV)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    # Derive per-backbone output names from the checkpoint directory stem.
    # e.g. checkpoints/repvgg_a2/best.pt → backbone_stem = "repvgg_a2"
    # This ensures per-backbone logs, CSVs, and PNGs never collide.
    backbone_stem: str = Path(args.ckpt).parent.name

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # 1. Run cross-device inference
    # ------------------------------------------------------------------ #
    log.info("Running cross-device inference (manifest: %s) …", args.manifest)
    results = run_cross_device_inference(
        args.ckpt, args.manifest, args.batch_size, args.num_workers
    )
    log.info("Inference complete — %d images evaluated.", len(results))

    # ------------------------------------------------------------------ #
    # 2. Save per-image CSV  (backbone-suffixed filename)
    # ------------------------------------------------------------------ #
    results_df = pd.DataFrame(results)
    # Reorder columns to match spec: image_path, source_subset, true_label, pred_class_name, correct
    # image_path not stored in batch — re-read manifest to attach it
    manifest_df = pd.read_csv(args.manifest)
    assert len(manifest_df) == len(results_df), (
        f"Manifest row count ({len(manifest_df)}) != inference result count ({len(results_df)})"
    )
    results_df.insert(0, "image_path", manifest_df["image_path"].to_numpy())
    csv_out = OUT_DIR / f"{backbone_stem}_cross_device_results.csv"
    results_df.to_csv(csv_out, index=False, encoding="utf-8")
    log.info("Per-image CSV saved → %s", csv_out)

    # ------------------------------------------------------------------ #
    # 3. Compute summary metrics
    # ------------------------------------------------------------------ #
    log.info("Computing summary tables …")
    metrics = build_summary_tables(results)

    # Gate check immediately
    per_subset = metrics["per_subset"]
    head_c, head_t = 0, 0
    for subset in per_subset.values():
        if "Head" in subset:
            head_c += subset["Head"]["correct"]
            head_t += subset["Head"]["total"]
    head_overall_acc = head_c / head_t if head_t > 0 else float("nan")
    gate_pass = head_overall_acc >= 0.50
    if not gate_pass:
        log.error(
            "GATE CHECK FAILED: Overall Head accuracy = %.4f (< 50%%).\n"
            "Halting — likely a collapsed-scoring bug. See Phase 6 spec §3.",
            head_overall_acc,
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 4. In-distribution collapsed baselines (for generalization gap)
    # ------------------------------------------------------------------ #
    log.info("Computing in-distribution collapsed baselines from %s …", args.test_csv)
    indist_baselines = run_indist_collapsed_baselines(
        args.ckpt, args.test_csv, args.batch_size, args.num_workers
    )
    log.info(
        "InDist baselines: Head=%.4f  Abdomen=%.4f  Femur=%.4f",
        indist_baselines.get("Head", float("nan")),
        indist_baselines.get("Fetal_abdomen", float("nan")),
        indist_baselines.get("Fetal_femur", float("nan")),
    )

    # ------------------------------------------------------------------ #
    # 5. Save bar chart  (backbone-suffixed filename)
    # ------------------------------------------------------------------ #
    chart_path = OUT_DIR / f"{backbone_stem}_cross_device_confusion.png"
    save_head_misclass_bar(results, chart_path)

    # ------------------------------------------------------------------ #
    # 6. Write summary log  (backbone-suffixed filename)
    # ------------------------------------------------------------------ #
    summary = format_summary(metrics, indist_baselines)
    log_path = LOG_DIR / f"evaluate_cross_device_{backbone_stem}.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(summary)
    log.info("Summary log saved → %s", log_path)

    # Also print to stdout for immediate review
    print(summary)

    log.info("Task 2 complete. Gate check: %s", "PASS" if gate_pass else "FAIL")


if __name__ == "__main__":
    main()
