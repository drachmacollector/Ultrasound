"""
src/models/loading.py

Single entry point for loading a trained classifier checkpoint for evaluation.

The model architecture, input size, normalisation and preprocessing mode are
always read from ``checkpoint["config"]``; nothing is hard-coded here.

Checkpoints that must never be evaluated in this study are refused:
  * multitask detection checkpoints (trained on HC18/UCL -> external sets leaked),
  * ACAM checkpoints (module archived; tag legacy-fetscan-demo-v1),
  * any checkpoint whose training CSV looks like a test/external manifest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import albumentations as A
import torch
import torch.nn as nn

from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD, get_eval_transform
from src.models.backbone import build_model
from src.preprocess.pipeline import is_cached

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Substrings that must never appear in a training/validation CSV path.
FORBIDDEN_TRAIN_TOKENS: tuple[str, ...] = (
    "test",
    "cross_device",
    "external",
    "african",
    "landmark",
    "hc18",
    "ucl",
    "multitask",
)


class RefusedCheckpointError(RuntimeError):
    """Raised when a checkpoint is not admissible for this study."""


def forbidden_token_in(path: str | Path) -> str | None:
    """Return the first forbidden token in ``path`` (case-insensitive), else None.

    For files inside the repository the repo-relative path is inspected
    (``data/splits/test.csv``, ``data/processed/external/hc18.csv``); for files outside
    it only the file name, so unrelated ancestor folders (e.g. ``pytest-…`` temp dirs)
    cannot trigger false positives. Relative paths are resolved against the CWD, which
    is the repo root for every entry point.
    """
    p = Path(str(path).replace("\\", "/"))
    try:
        text = p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        text = p.name
    text = text.lower()
    for tok in FORBIDDEN_TRAIN_TOKENS:
        if tok in text:
            return tok
    return None


def check_admissible(state_dict_keys: list[str], cfg: dict[str, Any], ckpt_path: str | Path = "") -> None:
    """Raise RefusedCheckpointError if the checkpoint must not be used."""
    if any(k.startswith("retinanet.") for k in state_dict_keys):
        raise RefusedCheckpointError(
            f"{ckpt_path}: multitask detection checkpoint (trained on HC18/UCL) - refused."
        )
    if cfg.get("use_acam"):
        raise RefusedCheckpointError(
            f"{ckpt_path}: ACAM checkpoint - the ACAM module is archived (tag legacy-fetscan-demo-v1)."
        )
    for key in ("train_csv", "val_csv"):
        if key in cfg:
            tok = forbidden_token_in(cfg[key])
            if tok is not None:
                raise RefusedCheckpointError(
                    f"{ckpt_path}: config {key}={cfg[key]!r} contains forbidden token {tok!r}."
                )


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    backbone_name: str
    device: torch.device
    img_size: int
    normalize_mean: tuple[float, ...]
    normalize_std: tuple[float, ...]
    preprocess_mode: str
    ckpt_path: Path
    config: dict[str, Any] = field(repr=False)

    def eval_transform(self, mode: str | None = None) -> A.Compose:
        """Deterministic eval pipeline matching this checkpoint's training geometry.

        ``mode`` overrides the checkpoint's preprocess mode (cross-preprocessing analyses).
        """
        mode = self.preprocess_mode if mode is None else mode
        if mode != "resize" and not is_cached(mode):
            raise NotImplementedError(f"preprocess_mode={mode!r} is not implemented yet.")
        # cached modes are stored pre-resized to img_size, so the same pipeline applies (resize = identity)
        return get_eval_transform(img_size=self.img_size, mean=self.normalize_mean, std=self.normalize_std)


def load_model(ckpt_path: str | Path, device: torch.device | None = None) -> LoadedModel:
    """Load a classifier checkpoint (config read from the checkpoint) in eval mode."""
    ckpt_path = Path(ckpt_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load on CPU first: avoids transient CUDA errors on Windows; then move.
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg: dict[str, Any] = dict(ckpt["config"])
    state_dict = ckpt["model_state_dict"]
    check_admissible(list(state_dict.keys()), cfg, ckpt_path)

    backbone_name = cfg["backbone"]
    model = build_model(backbone_name, num_classes=int(cfg.get("num_classes", 8)), pretrained=False)
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    return LoadedModel(
        model=model,
        backbone_name=backbone_name,
        device=device,
        img_size=int(cfg.get("image_size", 224)),
        normalize_mean=tuple(cfg.get("normalize_mean", IMAGENET_MEAN)),
        normalize_std=tuple(cfg.get("normalize_std", IMAGENET_STD)),
        preprocess_mode=str(cfg.get("preprocess_mode", "resize")),
        ckpt_path=ckpt_path,
        config=cfg,
    )
