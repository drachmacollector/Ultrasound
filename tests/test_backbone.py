"""
tests/test_backbone.py

build_model() yields [B, 8] logits for every backbone used in the study, and
get_pretrained_cfg() exposes the keys the configs rely on. No weights downloaded.
"""
from __future__ import annotations

import pytest
import torch

from src.data.dataset import NUM_CLASSES
from src.models.backbone import build_model, get_pretrained_cfg

# backbone -> (input size, batch) at native pretrained resolution
BACKBONES: dict[str, tuple[int, int]] = {
    "repvgg_a1": (224, 2),
    "repvgg_a2": (224, 2),
    "mobilenetv3_large_100": (224, 2),
    "efficientnet_lite0": (224, 2),
    "tf_efficientnetv2_s.in21k_ft_in1k": (300, 1),
    "convnext_tiny.fb_in22k_ft_in1k": (224, 2),
}


@pytest.mark.parametrize("backbone_name,img_size,batch", [(k, *v) for k, v in BACKBONES.items()])
def test_build_model_output_shape(backbone_name: str, img_size: int, batch: int) -> None:
    model = build_model(backbone_name, num_classes=NUM_CLASSES, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.zeros(batch, 3, img_size, img_size))
    assert out.shape == (batch, NUM_CLASSES)


@pytest.mark.parametrize("backbone_name", list(BACKBONES))
def test_get_pretrained_cfg(backbone_name: str) -> None:
    cfg = get_pretrained_cfg(backbone_name)
    for key in ("input_size", "mean", "std", "crop_pct"):
        assert key in cfg, f"{backbone_name}: pretrained_cfg missing {key!r}"
