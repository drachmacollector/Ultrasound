"""
scripts/bootstrap_significance.py

Paired bootstrap test for backbone macro-F1 comparison on the test set.

Resamples the 5,271 test images (with replacement) N=2000 times, recomputes
macro-F1 for each backbone on each resample using saved logit predictions,
and reports the 95% CI (2.5th–97.5th percentile) for the pairwise differences:
  (convnext_tiny − tf_efficientnetv2_s)
  (convnext_tiny − efficientnet_lite0)
  (convnext_tiny − repvgg_a2)

If the 95% CI for a difference excludes zero → robust evidence convnext leads.
If it straddles zero → top models are statistically indistinguishable on this
  test set, and secondary criteria (speed, ONNX friendliness) should decide.

Usage:
    conda run -n fetalplane python scripts/bootstrap_significance.py

Step 1: Runs inference for the top-4 backbones and saves per-image predictions
        to a .npz cache (scripts/_bootstrap_preds_cache.npz) so re-runs are
        instantaneous.
Step 2: Performs bootstrap and prints results.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CANONICAL_CLASSES, NUM_CLASSES, FocalPlanesDataset
from src.data.transforms import get_eval_transform
from src.models.backbone import build_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

TEST_CSV = "data/splits/test.csv"
CACHE_PATH = Path("scripts/_bootstrap_preds_cache.npz")
N_BOOTSTRAP = 2000
SEED = 42

# Top-4 backbones by test F1 (the ones worth bootstrapping).
TOP4_CKPTS = [
    Path("checkpoints/convnext_tiny/best.pt"),
    Path("checkpoints/tf_efficientnetv2_s/best.pt"),
    Path("checkpoints/efficientnet_lite0/best.pt"),
    Path("checkpoints/repvgg_a2/best.pt"),
]

# ---------------------------------------------------------------------------
# Inference step
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(ckpt_path: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Returns (targets, preds) as int32 numpy arrays over the full test set."""
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    cfg = ckpt["config"]
    backbone_name: str = cfg["backbone"]
    img_size: int = cfg.get("image_size", 224)
    mean = tuple(cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
    std = tuple(cfg.get("normalize_std", [0.229, 0.224, 0.225]))
    batch_size: int = cfg.get("batch_size", 32)

    log.info("Inference: %s (img_size=%d)", backbone_name, img_size)
    model = build_model(backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    transform = get_eval_transform(img_size=img_size, mean=mean, std=std)
    loader = DataLoader(
        FocalPlanesDataset(TEST_CSV, transform=transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    targets_list, preds_list = [], []
    for images, labels in loader:
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            logits = model(images.to(device, non_blocking=True))
        preds_list.extend(logits.argmax(dim=1).cpu().tolist())
        targets_list.extend(labels.tolist())

    return np.array(targets_list, dtype=np.int32), np.array(preds_list, dtype=np.int32)


def build_or_load_cache(device: torch.device) -> dict[str, dict[str, np.ndarray]]:
    """Return {backbone_name: {targets, preds}} — loading from cache if available."""
    if CACHE_PATH.exists():
        log.info("Loading predictions from cache: %s", CACHE_PATH)
        raw = np.load(CACHE_PATH, allow_pickle=True)
        result: dict[str, dict[str, np.ndarray]] = {}
        for ckpt in TOP4_CKPTS:
            ckpt_data = torch.load(str(ckpt), map_location="cpu", weights_only=False)
            name = ckpt_data["config"]["backbone"]
            if f"{name}_targets" in raw and f"{name}_preds" in raw:
                result[name] = {
                    "targets": raw[f"{name}_targets"],
                    "preds": raw[f"{name}_preds"],
                }
            else:
                log.warning("Cache missing entries for %s — re-running inference", name)
        if len(result) == len(TOP4_CKPTS):
            return result

    log.info("Running inference for top-4 backbones (cached after first run)...")
    save_dict: dict[str, np.ndarray] = {}
    result = {}
    for ckpt_path in TOP4_CKPTS:
        ckpt_data = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        name = ckpt_data["config"]["backbone"]
        targets, preds = run_inference(ckpt_path, device)
        result[name] = {"targets": targets, "preds": preds}
        save_dict[f"{name}_targets"] = targets
        save_dict[f"{name}_preds"] = preds

    np.savez(str(CACHE_PATH), **save_dict)  # type: ignore
    log.info("Predictions cached → %s", CACHE_PATH)
    return result


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def macro_f1_from_preds(targets: np.ndarray, preds: np.ndarray) -> float:
    """Compute macro-F1 without sklearn (fast numpy version for bootstrap)."""
    f1s = []
    for cls in range(NUM_CLASSES):
        tp = int(((preds == cls) & (targets == cls)).sum())
        fp = int(((preds == cls) & (targets != cls)).sum())
        fn = int(((preds != cls) & (targets == cls)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall > 0:
            f1s.append(2 * precision * recall / (precision + recall))
        else:
            f1s.append(0.0)
    return float(np.mean(f1s))


def run_bootstrap(pred_data: dict[str, dict[str, np.ndarray]]) -> None:
    rng = np.random.default_rng(SEED)
    n = len(next(iter(pred_data.values()))["targets"])
    backbone_names = list(pred_data.keys())
    
    # Align targets (should be identical across all backbones since same test set)
    targets = next(iter(pred_data.values()))["targets"]

    # Store per-bootstrap macro-F1 for each backbone
    bootstrap_f1s: dict[str, list[float]] = {name: [] for name in backbone_names}

    log.info("Running %d bootstrap iterations (n=%d)...", N_BOOTSTRAP, n)
    for _ in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, size=n)
        t_resamp = targets[indices]
        for name in backbone_names:
            p_resamp = pred_data[name]["preds"][indices]
            bootstrap_f1s[name].append(macro_f1_from_preds(t_resamp, p_resamp))

    # Point estimates (on full test set)
    point_f1 = {
        name: macro_f1_from_preds(targets, pred_data[name]["preds"])
        for name in backbone_names
    }

    reference = backbone_names[0]  # convnext_tiny (first in TOP4_CKPTS)
    ref_f1s = np.array(bootstrap_f1s[reference])

    print("\n" + "=" * 72)
    print(f"BOOTSTRAP SIGNIFICANCE — 2,000 iterations, n={n} images")
    print(f"Reference (winner by point estimate): {reference}")
    print("=" * 72)
    print(f"\n{'Backbone':<42} {'Point F1':>9} {'Bootstrap 95% CI':>20}")
    print("-" * 72)

    for name in backbone_names:
        arr = np.array(bootstrap_f1s[name])
        ci_lo, ci_hi = np.percentile(arr, 2.5), np.percentile(arr, 97.5)
        print(f"  {name:<40} {point_f1[name]:>9.4f}   [{ci_lo:.4f}, {ci_hi:.4f}]")

    print()
    print(f"{'Difference (ref − other)':<42} {'Point Δ':>9} {'95% CI of Δ':>20}  {'Verdict':>18}")
    print("-" * 72)
    for name in backbone_names:
        if name == reference:
            continue
        diffs = ref_f1s - np.array(bootstrap_f1s[name])
        point_diff = point_f1[reference] - point_f1[name]
        d_lo, d_hi = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)
        if d_lo > 0:
            verdict = "robust (CI > 0)"
        elif d_hi < 0:
            verdict = "ref is WORSE"
        else:
            verdict = "indistinguishable"
        print(f"  {reference} vs {name}")
        print(f"  {'':40} {point_diff:>9.4f}   [{d_lo:.4f}, {d_hi:.4f}]  {verdict:>18}")
        print()

    print("=" * 72)
    print("Interpretation guide:")
    print("  CI > 0 (both bounds positive) → robust evidence reference leads")
    print("  CI straddles 0               → statistically indistinguishable;")
    print("                                  secondary criteria should decide")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)
    pred_data = build_or_load_cache(device)
    run_bootstrap(pred_data)
