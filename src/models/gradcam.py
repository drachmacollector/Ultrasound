"""
src/models/gradcam.py

Grad-CAM module using the pytorch-grad-cam library (grad-cam package).

Design notes (PHASE_4_KICKOFF_PROMPT.md §11):
- Uses pytorch_grad_cam library; does NOT hand-roll hooks the way inference.py did.
- NO use_cuda= constructor argument — newer versions removed it; move the model
  to the correct device yourself before calling.
- get_target_layer() maps each known backbone to its last spatial conv block.
  This is not optional: layer4[-1] is a ResNet convention that does NOT
  generalize to RepVGG / MobileNetV3 / EfficientNet.
- Identify layers by printing model.named_modules() — documented per backbone.

Backbone → last spatial conv block mapping (verified 2026-07-26 with timm==1.0.28):

    repvgg_a1 / repvgg_a2:
        model.stages         → Sequential of 4 stages
        model.stages[3]      → Sequential (last stage = stage index 3)
        model.stages[3][-1]  → RepVggBlock (last block in last stage)

    mobilenetv3_large_100:
        model.blocks         → Sequential of 7 groups
        model.blocks[6]      → Sequential (last group)
        model.blocks[6][-1]  → ConvBnAct (last module in last group)

    efficientnet_lite0:
        model.blocks         → Sequential of 7 groups
        model.blocks[6]      → Sequential (last group)
        model.blocks[6][-1]  → InvertedResidual (last module in last group)

    tf_efficientnetv2_s:
        model.blocks         → Sequential of 6 groups
        model.blocks[5]      → Sequential (last group)
        model.blocks[5][-1]  → InvertedResidual (last module in last group)

Note: these paths are resolved at runtime via _get_layer_by_path() to guard
against timm version differences that might rename attributes.  A RuntimeError
with a clear message is raised if the path doesn't resolve, prompting the user
to re-inspect named_modules().

WARNING — RepVGG reparameterization invalidates the verified path:
_BACKBONE_LAYER_PATHS["repvgg_a1/a2"] was verified against the *training-time*
multi-branch RepVGG graph (each RepVggBlock has separate 3×3, 1×1, and identity
branches).  Phase 4 Step 6 (reparameterization) collapses that multi-branch
structure into a single plain conv stack, which changes the module tree.
After reparameterization, re-run:
    conda run -n fetalplane python scripts/_verify_gradcam_layers.py
against the reparameterized model and update _BACKBONE_LAYER_PATHS if the
path no longer resolves to the same layer type.  Do NOT assume the existing
path still works — the layer names and nesting can change after reparam.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backbone → target-layer attribute path
# ---------------------------------------------------------------------------
# Each value is a list of (str | int) steps to traverse from the model root
# to the target layer.  str → getattr, int → index into list(children()).
# Verified 2026-07-26 with timm==1.0.28 using scripts/_inspect_layers.py.

_BACKBONE_LAYER_PATHS: dict[str, list[str | int]] = {
    # model.stages (Sequential of 4 stages) → stage[3] → last RepVggBlock
    "repvgg_a1": ["stages", 3, -1],
    "repvgg_a2": ["stages", 3, -1],
    # model.blocks (Sequential of 7 groups) → group[6] → last ConvBnAct
    "mobilenetv3_large_100": ["blocks", 6, -1],
    # model.blocks (Sequential of 7 groups) → group[6] → last InvertedResidual
    "efficientnet_lite0": ["blocks", 6, -1],
    # model.blocks (Sequential of 6 groups) → group[5] → last InvertedResidual
    "tf_efficientnetv2_s.in21k_ft_in1k": ["blocks", 5, -1],
    # model.stages (Sequential of 4 stages) → stage[3] → blocks → last ConvNeXtBlock
    "convnext_tiny.fb_in22k_ft_in1k": ["stages", 3, "blocks", -1],
}



def _get_layer_by_path(model: nn.Module, path: list[str | int]) -> nn.Module:
    """Traverse a model attribute chain described by path.

    Each element of path is either:
    - a string  → resolved via getattr(current, step)
    - an integer → resolved via list(current.children())[step]

    Using list(children())[int] handles Sequential objects that don't support
    direct integer indexing via __getitem__ in all PyTorch versions.

    Raises:
        RuntimeError if the path does not resolve — caller should print
        list(model.named_modules()) and update _BACKBONE_LAYER_PATHS.
    """
    current: nn.Module = model
    for step in path:
        try:
            if isinstance(step, int):
                children_list = list(current.children())
                current = children_list[step]  # type: ignore[assignment]
            else:
                current = getattr(current, step)  # type: ignore[assignment]
        except (AttributeError, IndexError) as exc:
            raise RuntimeError(
                f"Could not resolve layer path {path!r} at step {step!r}.\n"
                "The backbone's internal attribute names may differ in your timm version.\n"
                "Run: list(model.named_modules()) and update _BACKBONE_LAYER_PATHS in "
                "src/models/gradcam.py accordingly.\n"
                f"Original error: {exc}"
            ) from exc
    return current


def get_target_layer(model: nn.Module, backbone_name: str) -> nn.Module:
    """Return the last spatial conv block for Grad-CAM targeting.

    Args:
        model: The trained/loaded nn.Module.
        backbone_name: One of the supported backbone identifier strings.

    Returns:
        The nn.Module that Grad-CAM should hook into.

    Raises:
        ValueError: If backbone_name is not in the supported set.
        RuntimeError: If the layer path does not resolve (indicates timm API change).
    """
    if backbone_name not in _BACKBONE_LAYER_PATHS:
        supported = sorted(_BACKBONE_LAYER_PATHS.keys())
        raise ValueError(
            f"backbone_name={backbone_name!r} not in supported set: {supported}.\n"
            "Add an entry to _BACKBONE_LAYER_PATHS in src/models/gradcam.py."
        )
    path = _BACKBONE_LAYER_PATHS[backbone_name]
    layer = _get_layer_by_path(model, path)
    log.debug("Grad-CAM target layer for %s: %s", backbone_name, type(layer).__name__)
    return layer


def run_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    class_idx: int,
    backbone_name: str,
    original_rgb_float: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute a Grad-CAM visualization overlay.

    Args:
        model: Trained nn.Module, already on the correct device and in eval mode.
        input_tensor: Preprocessed input [B, C, H, W] or [1, C, H, W] tensor,
            on the same device as model.
        class_idx: Target class index for Grad-CAM (0-7).
        backbone_name: Used to select the correct target layer.
        original_rgb_float: Optional HxWx3 float32 image in [0,1] range for
            overlay visualization.  If None, returns the raw grayscale CAM.

    Returns:
        If original_rgb_float is provided: HxWx3 uint8 overlay image.
        Otherwise: HxW float32 Grad-CAM heatmap in [0,1].
    """
    target_layers = [get_target_layer(model, backbone_name)]
    targets = [ClassifierOutputTarget(class_idx)]

    # GradCAM context manager cleans up hooks automatically.
    # NOTE: no use_cuda= argument — device is already set on model/tensor.
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam: np.ndarray = cam(
            input_tensor=input_tensor,
            targets=targets,
        )[0]  # shape: [H, W]

    if original_rgb_float is not None:
        # show_cam_on_image expects float32 in [0,1] and returns uint8 overlay
        overlay: np.ndarray = show_cam_on_image(
            original_rgb_float.astype(np.float32),
            grayscale_cam,
            use_rgb=True,
        )
        return overlay

    return grayscale_cam
