"""
src/models/backbone.py

Generic timm-based backbone wrapper for the 8-way fetal plane classifier.

Design rationale (from PHASE_4_KICKOFF_PROMPT.md §2):
- One wrapper for all four backbone candidates; timm.create_model handles
  the head replacement automatically for every architecture.
- get_pretrained_cfg() exposes input_size, mean, std, crop_pct so the
  training script can make an informed per-backbone normalization decision
  (§0.3) rather than blindly assuming 224×224 / ImageNet stats.

Supported backbone_name values (verify against timm.list_models):
    'repvgg_a1'
    'repvgg_a2'
    'mobilenetv3_large_100'
    'efficientnet_lite0'
    'tf_efficientnetv2_s'
"""
from __future__ import annotations

import timm
import torch.nn as nn


def build_model(
    backbone_name: str,
    num_classes: int = 8,
    pretrained: bool = True,
) -> nn.Module:
    """Create a timm model with the classifier head replaced for num_classes outputs.

    timm.create_model(..., num_classes=N) automatically replaces the final
    classification layer for every supported architecture — no manual .fc
    replacement needed (and none attempted, since RepVGG / MobileNetV3 /
    EfficientNet don't expose a uniform .fc attribute).

    Args:
        backbone_name: A timm model identifier string.  Validated by timm at
            call time — an unknown string raises a RuntimeError from timm.
        num_classes: Output classes for the classifier head.  Must be 8 for
            this project (CANONICAL_CLASSES has 8 entries).
        pretrained: Whether to load ImageNet pretrained weights.

    Returns:
        nn.Module ready for training / fine-tuning.

    Raises:
        RuntimeError: If backbone_name is not found in the installed timm version.
    """
    model: nn.Module = timm.create_model(
        backbone_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


def get_pretrained_cfg(backbone_name: str) -> dict:  # type: ignore[type-arg]
    """Return the pretrained config dict for a backbone without downloading weights.

    Loads the model with pretrained=False to avoid a network hit when we only
    need the config metadata (input_size, mean, std, crop_pct, crop_mode).

    Per PHASE_4_KICKOFF_PROMPT.md §0.3 — call this for each of the four
    candidates and record the results in EXPERIMENTS.md before any training run.

    Args:
        backbone_name: A timm model identifier string.

    Returns:
        dict-like pretrained_cfg object with at minimum the keys:
            input_size  (tuple[int, int, int] — C×H×W)
            mean        (tuple[float, ...])
            std         (tuple[float, ...])
            crop_pct    (float)
            crop_mode   (str, may be absent in older timm)
    """
    # pretrained=False avoids downloading weights; the cfg is baked into the class.
    model: nn.Module = timm.create_model(backbone_name, pretrained=False)
    return model.pretrained_cfg  # type: ignore[return-value]
