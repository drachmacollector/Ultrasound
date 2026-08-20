"""
src/train/losses.py

Custom loss functions for the fetal plane classifier.

FocalLoss:
    Standard focal loss (Lin et al., 2017) with optional class weights.
    FL(p_t) = -weight * (1 - p_t)^gamma * log(p_t)

    Class-weighting and focal reweighting *compose* here — the weight tensor
    from build_class_weight_tensor() is passed unchanged, so the focal term
    adds on top of the per-class imbalance correction rather than replacing it.

    gamma=2.0 is the default from the original paper and almost universally
    used in medical imaging applications.  Only touch it if ablation shows
    a clear benefit to a different value.

Usage in train.py:
    from src.train.losses import FocalLoss
    criterion = FocalLoss(weight=weight_tensor, gamma=cfg.get("focal_gamma", 2.0))
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss with optional per-class weights.

    Args:
        weight:  Per-class weight tensor of shape [num_classes], same as
                 nn.CrossEntropyLoss(weight=...).  Pass None for unweighted.
        gamma:   Focusing exponent.  0 → reduces to standard CE.  2.0 is the
                 value from the original paper and is the recommended default.
        reduction: 'mean' (default) or 'sum'.  'none' not supported.
    """

    def __init__(
        self,
        weight: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
        self.register_buffer("weight", weight)  # moves with .to(device)
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits:  [B, C] unnormalised logits.
            targets: [B] ground-truth class indices (int64).
                     Entries equal to ``ignore_index`` (default -100) are
                     excluded from the loss, matching nn.CrossEntropyLoss
                     semantics.  This is required for the multitask training
                     path where detection-only rows carry cls_label = -100.

        Returns:
            Scalar loss tensor.  Returns 0.0 if all targets are masked.
        """
        weight: torch.Tensor | None = getattr(self, "weight", None)  # type: ignore[assignment]

        # --- Filter out ignore_index entries before any indexing ---------------
        # Passing ignore_index into .gather() would silently index OOB on some
        # backends or crash outright.  Always mask first.
        valid_mask = targets != self.ignore_index
        if not valid_mask.any():
            return logits.sum() * 0.0  # preserve autograd graph, return zero

        logits  = logits[valid_mask]   # [V, C]
        targets = targets[valid_mask]  # [V]

        # log_softmax is numerically stable and required for gathering log-probs
        log_probs = F.log_softmax(logits, dim=1)                   # [V, C]
        probs     = log_probs.exp()                                 # [V, C]

        # Gather the log-prob and prob for each sample's true class
        log_pt = log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)  # [V]
        pt     = probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)      # [V]

        # Focal modulating factor
        focal_factor = (1.0 - pt) ** self.gamma  # [V]

        # Per-sample loss (without class weight applied yet)
        loss = -focal_factor * log_pt  # [V]

        # Apply per-class weights if provided
        sample_weight = None
        if weight is not None:
            # weight is [C]; index by target to get per-sample weight
            sample_weight = weight[targets]  # [V]
            loss = loss * sample_weight      # [V]

        if self.reduction == "mean":
            # When class weights are present, use weight-normalised mean
            # (same convention as nn.CrossEntropyLoss)
            if sample_weight is not None:
                return loss.sum() / sample_weight.sum()
            return loss.mean()
        return loss.sum()
