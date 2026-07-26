"""
scripts/inspect_pretrained_cfg.py

Per PHASE_4_KICKOFF_PROMPT.md §0.3 — run this BEFORE any training run.

Prints the pretrained_cfg for all four backbone candidates so you can make an
informed per-backbone resolution/normalization decision.  Results should be
recorded in EXPERIMENTS.md.

Usage:
    conda run -n fetalplane python scripts/inspect_pretrained_cfg.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.backbone import get_pretrained_cfg

BACKBONES = [
    "repvgg_a1",
    "repvgg_a2",
    "mobilenetv3_large_100",
    "efficientnet_lite0",
    "tf_efficientnetv2_s",
]

print(f"\n{'=' * 70}")
print("Per-backbone pretrained_cfg inspection  (§0.3 of PHASE_4_KICKOFF_PROMPT)")
print(f"{'=' * 70}\n")

for backbone_name in BACKBONES:
    print(f"── {backbone_name}")
    try:
        cfg = get_pretrained_cfg(backbone_name)
        print(f"   input_size : {cfg.get('input_size')}")
        print(f"   mean       : {cfg.get('mean')}")
        print(f"   std        : {cfg.get('std')}")
        print(f"   crop_pct   : {cfg.get('crop_pct')}")
        print(f"   crop_mode  : {cfg.get('crop_mode', 'N/A')}")
    except Exception as exc:  # noqa: BLE001
        print(f"   ERROR: {exc}")
    print()

print("Record these values in docs/EXPERIMENTS.md before training.\n")
