"""
scripts/tests/test_acam.py

Unit tests for src/models/acam.py (ACAM — arXiv:2509.00808).

§ Cross-reference: Phase 8, Stage 5, Task 5.1
  (docs/instructions/08_FINAL_EVALUATION_AND_POLISH.md §6, Task 5.1)

Covers:
  1. test_output_shape            — [B,3,H,W] in → [B,3,H,W] out (no shape change)
  2. test_gradient_flow           — backward() completes; ACAM weights have grads
  3. test_alpha_range             — predicted α values all lie within [alpha_min, alpha_max]
  4. test_single_sample           — B=1 works (no batch-norm issues in eval mode)
  5. test_fusion_conv_init        — fusion conv initialised as approx. identity (avg of views)
  6. test_sequential_with_backbone — nn.Sequential(acam, backbone) forward/backward pass
  7. test_build_acam_from_cfg     — build_acam() reads K / alpha bounds from config dict

Usage:
    conda run -n fetalplane python -m pytest scripts/tests/test_acam.py -v \
        | Out-File -Encoding utf8 -FilePath logs/smoketest/test_acam_output.txt
    # or directly (writes its own log):
    conda run -n fetalplane python scripts/tests/test_acam.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make project root importable when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.acam import ACAM, build_acam
from src.models.backbone import build_model
from src.data.dataset import NUM_CLASSES

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
# Test parameters
# ---------------------------------------------------------------------------
_BATCH = 2
_K = 10
_H = _W = 224
_C = 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_acam(K: int = _K, train_mode: bool = True) -> ACAM:
    acam = ACAM(K=K, in_channels=_C)
    acam.train(train_mode)
    return acam


def _dummy(B: int = _BATCH) -> torch.Tensor:
    return torch.randn(B, _C, _H, _W)


# ---------------------------------------------------------------------------
# Test 1: Output shape
# ---------------------------------------------------------------------------

def test_output_shape() -> None:
    """Module output shape must exactly match input shape [B, C, H, W]."""
    acam = _make_acam()
    x = _dummy()
    out = acam(x)
    assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
    log.info("test_output_shape PASSED — output: %s", tuple(out.shape))


# ---------------------------------------------------------------------------
# Test 2: Gradient flow
# ---------------------------------------------------------------------------

def test_gradient_flow() -> None:
    """Gradients must flow through the entire ACAM (decision net + fusion conv)."""
    acam = _make_acam(train_mode=True)
    x = _dummy()
    out = acam(x)
    loss = out.mean()
    loss.backward()

    # Check decision network parameters
    for name, param in acam.decision_net.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"decision_net.{name} has None grad"
            assert not torch.isnan(param.grad).any(), f"decision_net.{name} has NaN grad"

    # Check fusion conv parameters
    assert acam.fusion_conv.weight.grad is not None, "fusion_conv.weight has None grad"
    assert not torch.isnan(acam.fusion_conv.weight.grad).any(), "fusion_conv.weight has NaN grad"

    log.info("test_gradient_flow PASSED — all ACAM parameters have valid gradients")


# ---------------------------------------------------------------------------
# Test 3: Alpha range
# ---------------------------------------------------------------------------

def test_alpha_range() -> None:
    """All predicted α values must lie within [alpha_min, alpha_max] = [1.0, 3.0]."""
    acam = _make_acam()
    acam.eval()
    x = _dummy()

    with torch.no_grad():
        alpha_raw = acam.decision_net(x)                                # [B, K] ∈ (0,1)
        alpha = acam.alpha_min + (acam.alpha_max - acam.alpha_min) * alpha_raw  # [B, K]

    assert alpha.min().item() >= acam.alpha_min - 1e-5, (
        f"Alpha below min: {alpha.min().item()} < {acam.alpha_min}"
    )
    assert alpha.max().item() <= acam.alpha_max + 1e-5, (
        f"Alpha above max: {alpha.max().item()} > {acam.alpha_max}"
    )
    log.info(
        "test_alpha_range PASSED — α ∈ [%.4f, %.4f] (expected [%.1f, %.1f])",
        alpha.min().item(), alpha.max().item(), acam.alpha_min, acam.alpha_max,
    )


# ---------------------------------------------------------------------------
# Test 4: Single sample (B=1)
# ---------------------------------------------------------------------------

def test_single_sample() -> None:
    """Module must handle batch size 1 without errors (BatchNorm in eval mode)."""
    acam = _make_acam()
    acam.eval()   # BatchNorm2d uses running stats in eval — B=1 is safe
    x = _dummy(B=1)
    with torch.no_grad():
        out = acam(x)
    assert out.shape == (1, _C, _H, _W), f"Single-sample shape wrong: {out.shape}"
    log.info("test_single_sample PASSED — output: %s", tuple(out.shape))


# ---------------------------------------------------------------------------
# Test 5: Fusion conv initialisation (approximate identity)
# ---------------------------------------------------------------------------

def test_fusion_conv_init() -> None:
    """Fusion conv weight should be initialised as 1/K (equal-weight average)."""
    K = 5
    acam = ACAM(K=K, in_channels=_C)
    expected_w = 1.0 / K
    actual_w = acam.fusion_conv.weight.data.mean().item()
    assert abs(actual_w - expected_w) < 1e-6, (
        f"fusion_conv weight mean {actual_w:.6f} != expected {expected_w:.6f}"
    )
    assert acam.fusion_conv.bias.data.abs().max().item() < 1e-8, (
        "fusion_conv bias not zero-initialised"
    )
    log.info(
        "test_fusion_conv_init PASSED — weight mean=%.6f (expected %.6f)",
        actual_w, expected_w,
    )


# ---------------------------------------------------------------------------
# Test 6: nn.Sequential(acam, backbone) — full stack forward + backward
# ---------------------------------------------------------------------------

def test_sequential_with_backbone() -> None:
    """ACAM + backbone as nn.Sequential must produce [B, NUM_CLASSES] logits
    and support backward (gradients flow into ACAM from classification loss)."""
    acam = ACAM(K=_K, in_channels=_C)
    backbone = build_model(
        "convnext_tiny.fb_in22k_ft_in1k",
        num_classes=NUM_CLASSES,
        pretrained=False,   # no network hit — weights not needed for shape test
    )
    combined: nn.Module = nn.Sequential(acam, backbone)
    combined.train()

    x = _dummy()
    logits = combined(x)
    assert logits.shape == (_BATCH, NUM_CLASSES), (
        f"Logits shape wrong: {logits.shape} != ({_BATCH}, {NUM_CLASSES})"
    )

    # Backward pass through full stack
    loss = logits.mean()
    loss.backward()

    # ACAM gradients must be non-None
    assert acam.fusion_conv.weight.grad is not None, (
        "ACAM fusion_conv.weight has no grad after full-stack backward"
    )
    log.info(
        "test_sequential_with_backbone PASSED — logits %s, grads flow into ACAM",
        tuple(logits.shape),
    )


# ---------------------------------------------------------------------------
# Test 7: build_acam reads config dict correctly
# ---------------------------------------------------------------------------

def test_build_acam_from_cfg() -> None:
    """build_acam() must read K, alpha_min, alpha_max from the config dict."""
    cfg = {"acam_K": 5, "acam_alpha_min": 0.5, "acam_alpha_max": 2.5}
    acam = build_acam(cfg)
    assert acam.K == 5
    assert acam.alpha_min == 0.5
    assert acam.alpha_max == 2.5
    log.info(
        "test_build_acam_from_cfg PASSED — K=%d α∈[%.1f,%.1f]",
        acam.K, acam.alpha_min, acam.alpha_max,
    )


# ---------------------------------------------------------------------------
# Runner (direct execution)
# ---------------------------------------------------------------------------

_TESTS = [
    test_output_shape,
    test_gradient_flow,
    test_alpha_range,
    test_single_sample,
    test_fusion_conv_init,
    test_sequential_with_backbone,
    test_build_acam_from_cfg,
]

if __name__ == "__main__":
    # Create log output directory
    log_out_dir = Path("logs/smoketest")
    log_out_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_out_dir / "test_acam_output.txt"

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    )
    logging.getLogger().addHandler(file_handler)

    log.info("=" * 60)
    log.info("ACAM Unit Tests  (src/models/acam.py)")
    log.info("Paper: arXiv:2509.00808 — Chen et al.")
    log.info("=" * 60)

    passed = 0
    failed = 0

    for test_fn in _TESTS:
        name = test_fn.__name__
        try:
            test_fn()
            passed += 1
        except Exception as exc:  # noqa: BLE001
            log.error("FAILED: %s — %s", name, exc)
            failed += 1

    log.info("-" * 60)
    log.info("Results: %d passed, %d failed", passed, failed)
    log.info("Full log → %s", log_path)

    sys.exit(0 if failed == 0 else 1)
