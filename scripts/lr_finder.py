"""
scripts/lr_finder.py

Leslie Smith-style LR range test for a given backbone config.

Runs an exponentially increasing LR sweep from lr_start to lr_end over one
partial epoch (max_steps batches), plots loss vs. LR, and saves the plot to
logs/<backbone>/lr_finder.png.

No external torch-lr-finder dependency needed — this is a self-contained
manual implementation of the standard LR range test loop.

Usage:
    conda run -n fetalplane python scripts/lr_finder.py --config configs/repvgg_a1.yaml
    conda run -n fetalplane python scripts/lr_finder.py --config configs/tf_efficientnetv2_s.yaml --lr-start 1e-8 --lr-end 0.5
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from src.data.dataset import CANONICAL_CLASSES, FocalPlanesDataset
from src.data.transforms import get_eval_transform
from src.models.backbone import build_model
from src.train.train import build_class_weight_tensor, load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)



def run_lr_finder(
    cfg: dict,  # type: ignore[type-arg]
    lr_start: float = 1e-7,
    lr_end: float = 1.0,
    max_steps: int = 100,
    smoothing: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Run the LR range test.

    Args:
        cfg: Loaded YAML config dict.
        lr_start: Starting (lowest) LR.
        lr_end: Ending (highest) LR to sweep to.
        max_steps: Number of batches to run (one partial epoch).
        smoothing: Exponential smoothing beta for loss curve.

    Returns:
        (lrs, losses): Parallel lists of LR and smoothed loss values.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("LR finder using device: %s", device)

    backbone_name: str = cfg["backbone"]
    img_size: int = cfg.get("image_size", 224)
    mean: list[float] = cfg.get("normalize_mean", [0.485, 0.456, 0.406])
    std: list[float] = cfg.get("normalize_std", [0.229, 0.224, 0.225])

    model = build_model(backbone_name, num_classes=8, pretrained=cfg.get("pretrained", True))
    model = model.to(device)
    model.train()

    # Use eval-only transform (no augmentation) — augmentation would add noise
    # to the loss signal and obscure the true LR-vs-loss relationship.
    transform = get_eval_transform(
        img_size=img_size,
        mean=tuple(mean),
        std=tuple(std),
    )
    dataset = FocalPlanesDataset(csv_path=cfg["train_csv"], transform=transform)
    loader: DataLoader = DataLoader(  # type: ignore[type-arg]
        dataset,
        batch_size=cfg.get("batch_size", 32),
        shuffle=True,
        num_workers=cfg.get("num_workers", 4),
        drop_last=True,
    )

    weight_tensor = build_class_weight_tensor(cfg["class_weights_json"], device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = AdamW(model.parameters(), lr=lr_start, weight_decay=cfg.get("weight_decay", 0.01))
    use_amp: bool = cfg.get("amp", True) and device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Exponential LR multiplier per step
    lr_mult = (lr_end / lr_start) ** (1.0 / max_steps)

    lrs: list[float] = []
    losses: list[float] = []
    smoothed_loss: float = 0.0
    best_loss: float = float("inf")
    loader_iter = iter(loader)

    log.info(
        "Running LR range test: %d steps, LR %.2e → %.2e", max_steps, lr_start, lr_end
    )

    for step in range(max_steps):
        # Refill iterator if dataset is smaller than max_steps batches
        try:
            images, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            images, labels = next(loader_iter)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaled_loss = scaler.scale(loss)
        assert isinstance(scaled_loss, torch.Tensor)
        scaled_loss.backward()
        scaler.step(optimizer)
        scaler.update()

        raw_loss = loss.item()

        # Exponential smoothing
        if step == 0:
            smoothed_loss = raw_loss
        else:
            smoothed_loss = smoothing * raw_loss + (1 - smoothing) * smoothed_loss

        # Bias-corrected smoothed loss (like Adam's bias correction)
        bc_loss = smoothed_loss / (1 - (1 - smoothing) ** (step + 1))

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        losses.append(bc_loss)

        # Stop early if loss diverges significantly
        if bc_loss < best_loss:
            best_loss = bc_loss
        if bc_loss > 4 * best_loss:
            log.info("Loss diverging at LR=%.2e — stopping early at step %d.", current_lr, step)
            break

        # Advance LR for next step
        for pg in optimizer.param_groups:
            pg["lr"] = pg["lr"] * lr_mult

        if step % 10 == 0:
            log.info("Step %3d | LR=%.2e | loss=%.4f (smoothed=%.4f)", step, current_lr, raw_loss, bc_loss)

    return lrs, losses


def plot_lr_finder(
    lrs: list[float],
    losses: list[float],
    save_path: Path,
    backbone_name: str,
) -> None:
    """Plot and save the LR finder curve."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(lrs, losses, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Learning Rate (log scale)")
    ax.set_ylabel("Smoothed Loss")
    ax.set_title(f"LR Range Test — {backbone_name}")
    ax.grid(True, which="both", alpha=0.3)

# Annotate the steepest descent region
    if len(losses) > 20:
        # FIX: Skip the first 15 steps to ignore the initial bias-correction spike
        skip = 15
        log_lrs = np.log10(lrs[skip:])
        grads = np.gradient(losses[skip:], log_lrs)
        
        # Find the steepest downward slope in the valid region
        best_step = int(np.argmin(grads))
        best_lr = lrs[skip + best_step]
        
        ax.axvline(best_lr, color="red", linestyle="--", alpha=0.7,
                   label=f"Steepest descent ≈ {best_lr:.2e}")
        ax.legend()
        log.info("Suggested LR (steepest descent): %.2e", best_lr)

    fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("LR finder plot saved → %s", save_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LR range test for a fetal plane backbone.")
    parser.add_argument("--config", type=str, required=True, help="Path to backbone YAML config.")
    parser.add_argument("--lr-start", type=float, default=1e-7, help="Starting LR.")
    parser.add_argument("--lr-end", type=float, default=1.0, help="Ending LR.")
    parser.add_argument("--max-steps", type=int, default=100, help="Max batches to sweep.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = load_config(args.config)
    backbone_name: str = cfg["backbone"]

    log_dir = Path(cfg.get("log_dir", f"logs/{backbone_name}"))
    log_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_dir / "output.txt")
    file_handler.setFormatter(
        logging.Formatter("[%(levelname)s] %(message)s")
        )
    logging.getLogger().addHandler(file_handler)

    lrs, losses = run_lr_finder(
        cfg=cfg,
        lr_start=args.lr_start,
        lr_end=args.lr_end,
        max_steps=args.max_steps,
    )

    plot_lr_finder(lrs, losses, log_dir / "lr_finder.png", backbone_name)
