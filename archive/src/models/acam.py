"""
from __future__ import annotations

src/models/acam.py

Adaptive Contrast Adjustment Module (ACAM) — learnable plug-and-play
preprocessing sub-network inserted before the backbone classifier.

§ Cross-reference: Phase 8, Stage 5, Task 5.1
  (docs/instructions/08_FINAL_EVALUATION_AND_POLISH.md §6, Task 5.1)
  Paper: arXiv:2509.00808 — "Adaptive Contrast Adjustment Module: A
  Clinically-Inspired Plug-and-Play Approach for Enhanced Fetal Plane
  Classification" (Chen et al.)

Design (confirmed from §2.1–2.3 of the paper):
-------------------------------------------------
1. INPUT: [B, 3, H, W] normalized tensor (our pipeline already converts
   grayscale → 3-channel RGB; ACAM operates over the 3-ch representation,
   which is semantically equivalent to grayscale because all 3 channels are
   identical copies of the same grey values).

2. DECISION NETWORK (shallow CNN, predicts K contrast factors α):
       Conv2d(3→16, 3×3, stride=2, padding=1) → BatchNorm2d → ReLU
       Conv2d(16→32, 3×3, stride=2, padding=1) → BatchNorm2d → ReLU
       AdaptiveAvgPool2d(1) → Flatten → Linear(32 → K) → Sigmoid
   The sigmoid output is then linearly re-scaled to [α_min, α_max] = [1, 3],
   matching the paper's "sigmoid-mapped parameters reflecting real sonographer
   adjustment ranges (1–3)" (§5 Conclusion).

3. CONTRAST TRANSFORM (differentiable, per §2.1):
       I'(x,y) = α · (I(x,y) − μ)
   where μ = per-image, per-channel spatial mean.  Applied once per α_k
   for k = 1 … K, producing K contrast-adjusted views of the same image.

4. FUSION (implementation choice, documented explicitly per Task 5.1):
   The paper states the K contrast-enhanced views are "fused within downstream
   classifiers" (Abstract) but does not specify the exact fusion operator.  The
   most standard and defensible interpretation for a plug-and-play module that
   must produce an output compatible with an arbitrary downstream backbone is:
       torch.cat([view_1, ..., view_K], dim=1)      → [B, 3K, H, W]
       Conv2d(3K → 3, 1×1, bias=True)               → [B, 3, H, W]
   This is a linear weighted sum over contrast views (learned per-position
   weights via the 1×1 conv), is fully differentiable, and restores the
   original input shape so the module is a true drop-in replacement for the
   identity transform. Alternative (channel-attention weighted sum) was
   considered but requires a second decision-network forward pass and was
   not implied by the paper's language.

5. OUTPUT: [B, 3, H, W] — identical shape to the input.
   → Feeds directly into the timm backbone, no shape change required.

Hyperparameters (from paper §2.3, Table 1):
  K = 10  (n=10 in the paper's notation)
  α_min = 1.0, α_max = 3.0

Usage (smoke-test a single forward pass):
    conda run -n fetalplane python -m src.models.acam
"""
from __future__ import annotations

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (from arXiv:2509.00808, §2.3, Table 1 and §5 Conclusion)
# ---------------------------------------------------------------------------
_DEFAULT_K: int = 10          # number of contrast views
_ALPHA_MIN: float = 1.0       # minimum contrast factor (no reduction)
_ALPHA_MAX: float = 3.0       # maximum contrast factor (matches sonographer range)


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class ACAM(nn.Module):
    """Adaptive Contrast Adjustment Module (arXiv:2509.00808).

    Args:
        K:          Number of contrast-adjusted views to generate.  Default 10
                    (paper's n=10, Table 1).
        in_channels: Number of input channels.  Must be 3 for this project
                    (grayscale-to-RGB pipeline).
        alpha_min:  Lower bound of the sigmoid-remapped contrast range.
                    Default 1.0 (paper §5).
        alpha_max:  Upper bound of the sigmoid-remapped contrast range.
                    Default 3.0 (paper §5).
    """

    def __init__(
        self,
        K: int = _DEFAULT_K,
        in_channels: int = 3,
        alpha_min: float = _ALPHA_MIN,
        alpha_max: float = _ALPHA_MAX,
    ) -> None:
        super().__init__()
        if K < 1:
            raise ValueError(f"K must be ≥ 1, got {K}")
        if alpha_min >= alpha_max:
            raise ValueError(f"alpha_min ({alpha_min}) must be < alpha_max ({alpha_max})")

        self.K = K
        self.in_channels = in_channels
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        # ------------------------------------------------------------------
        # Decision network — shallow CNN (§2.2):
        # "a shallow network consisting of convolutional layers, a global
        # average pooling layer, and fully connected layers"
        # ------------------------------------------------------------------
        self.decision_net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),   # → [B, 32, 1, 1]
            nn.Flatten(),              # → [B, 32]
            nn.Linear(32, K),          # → [B, K]
            nn.Sigmoid(),              # → [B, K] in (0, 1)
        )

        # ------------------------------------------------------------------
        # Fusion: 1×1 conv collapses K·in_channels → in_channels
        # See module docstring §4 for rationale.
        # ------------------------------------------------------------------
        self.fusion_conv = nn.Conv2d(
            in_channels * K,
            in_channels,
            kernel_size=1,
            bias=True,
        )

        # Initialise fusion conv as equal-weight average so the module starts
        # as (approximately) the identity transform, giving stable early training.
        nn.init.constant_(self.fusion_conv.weight, 1.0 / K)
        nn.init.zeros_(self.fusion_conv.bias)

        log.debug(
            "ACAM initialised: K=%d, in_channels=%d, α∈[%.1f, %.1f]",
            K, in_channels, alpha_min, alpha_max,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [B, C, H, W], float32, already normalized.

        Returns:
            Fused contrast-adjusted tensor [B, C, H, W].
        """
        B, C, H, W = x.shape

        # 1. Predict K contrast factors α ∈ [alpha_min, alpha_max]  [B, K]
        alpha_raw = self.decision_net(x)                              # [B, K] ∈ (0,1)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_raw  # [B, K]

        # 2. Per-image, per-channel spatial mean (μ in the paper's formula)
        #    Shape: [B, C, 1, 1] — broadcast-friendly
        mu = x.mean(dim=(2, 3), keepdim=True)                        # [B, C, 1, 1]

        # 3. Contrast-adjust: I'(x,y) = α_k · (I(x,y) − μ)
        #    alpha: [B, K] → reshape to [B, K, 1, 1, 1] for broadcast over [B, C, H, W]
        #    We compute all K views in one vectorised step.
        alpha_bcast = alpha.view(B, self.K, 1, 1, 1)                # [B, K, 1, 1, 1]
        x_centered = (x - mu).unsqueeze(1)                           # [B, 1, C, H, W]
        views = alpha_bcast * x_centered                             # [B, K, C, H, W]

        # 4. Flatten K views along the channel dimension for the fusion conv
        views_flat = views.view(B, self.K * C, H, W)                # [B, K·C, H, W]

        # 5. 1×1 fusion conv → [B, C, H, W]
        out = self.fusion_conv(views_flat)

        return out


# ---------------------------------------------------------------------------
# Convenience constructor (matches naming used in train.py integration)
# ---------------------------------------------------------------------------

def build_acam(cfg: dict) -> ACAM:  # type: ignore[type-arg]
    """Build an ACAM instance from a training config dict.

    Reads optional keys: acam_K (default 10), acam_alpha_min (default 1.0),
    acam_alpha_max (default 3.0).

    Args:
        cfg: Training config dict (from YAML).

    Returns:
        ACAM instance.
    """
    return ACAM(
        K=int(cfg.get("acam_K", _DEFAULT_K)),
        in_channels=3,
        alpha_min=float(cfg.get("acam_alpha_min", _ALPHA_MIN)),
        alpha_max=float(cfg.get("acam_alpha_max", _ALPHA_MAX)),
    )


# ---------------------------------------------------------------------------
# Smoke-test entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("ACAM smoke test — forward pass on CPU")

    acam = ACAM(K=10)
    acam.eval()

    dummy = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out = acam(dummy)

    assert out.shape == dummy.shape, f"Shape mismatch: {out.shape} != {dummy.shape}"
    log.info("Output shape: %s  ✓", out.shape)

    # Parameter count
    n_params = sum(p.numel() for p in acam.parameters())
    log.info("ACAM parameter count: %d", n_params)

    log.info("Smoke test PASSED.")
    sys.exit(0)
