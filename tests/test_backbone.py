"""
src/models/test_backbone.py

Lightweight unit tests for src/models/backbone.py.

Tests: build_model() produces correct output shape [B, 8] for each backbone,
without requiring data or pretrained weights download.

Run:
    conda run -n fetalplane python -m pytest src/models/test_backbone.py -v
    # or directly:
    conda run -n fetalplane python src/models/test_backbone.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.backbone import build_model, get_pretrained_cfg
from src.data.dataset import NUM_CLASSES

# Map backbone → (input_size, batch_size) for forward pass test
# Use the native pretrained resolution per §0.3 decision
_BACKBONE_TEST_CONFIGS: dict[str, tuple[int, int]] = {
    "repvgg_a1": (224, 2),
    "repvgg_a2": (224, 2),
    "mobilenetv3_large_100": (224, 2),
    "efficientnet_lite0": (224, 2),
    "tf_efficientnetv2_s.in21k_ft_in1k": (300, 1),
    "convnext_tiny.fb_in22k_ft_in1k": (224, 2),
}

def test_build_model_output_shape(backbone_name: str, img_size: int, batch: int) -> None:
    """Assert that build_model produces logits of shape [B, NUM_CLASSES]."""
    print(f"Testing {backbone_name}...", end=" ")
    model = build_model(backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    model.eval()

    dummy_input = torch.zeros(batch, 3, img_size, img_size)
    with torch.no_grad():
        output = model(dummy_input)

    expected_shape = (batch, NUM_CLASSES)
    assert output.shape == expected_shape, (
        f"{backbone_name}: expected output shape {expected_shape}, got {output.shape}"
    )
    print(f"✓ output shape {output.shape}")


def test_get_pretrained_cfg(backbone_name: str) -> None:
    """Assert that get_pretrained_cfg returns a dict with required keys."""
    cfg = get_pretrained_cfg(backbone_name)
    for key in ("input_size", "mean", "std", "crop_pct"):
        assert key in cfg, f"{backbone_name}: pretrained_cfg missing key '{key}'"
    print(f"  pretrained_cfg OK: input_size={cfg['input_size']} mean={cfg['mean']}")


if __name__ == "__main__":
    all_pass = True
    print(f"\nBackbone unit tests (NUM_CLASSES={NUM_CLASSES})\n{'=' * 50}")

    for backbone_name, (img_size, batch) in _BACKBONE_TEST_CONFIGS.items():
        try:
            test_build_model_output_shape(backbone_name, img_size, batch)
            test_get_pretrained_cfg(backbone_name)
        except Exception as exc:  # noqa: BLE001
            print(f"✗ FAILED: {exc}")
            all_pass = False

    print()
    if all_pass:
        print("All backbone tests PASSED.")
        sys.exit(0)
    else:
        print("Some backbone tests FAILED.")
        sys.exit(1)
