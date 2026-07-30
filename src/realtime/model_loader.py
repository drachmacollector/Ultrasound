"""
src/realtime/model_loader.py

Single source of truth for loading a trained checkpoint for real-time inference.
Mirrors src/eval/evaluate_test.py's loading pattern exactly — config comes from
the checkpoint, never from a hardcoded default, per PHASE_5_KICKOFF_PROMPT.md §0.2.
"""
from dataclasses import dataclass
from pathlib import Path
import torch
from src.models.backbone import build_model
from src.data.transforms import get_eval_transform

@dataclass
class LoadedModel:
    model: torch.nn.Module
    backbone_name: str
    device: torch.device
    img_size: int
    normalize_mean: tuple
    normalize_std: tuple
    transform: object  # albumentations.Compose

def load_inference_model(ckpt_path: str, device: torch.device | None = None) -> LoadedModel:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    backbone_name = cfg["backbone"]
    img_size = cfg.get("image_size", 224)
    mean = tuple(cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
    std = tuple(cfg.get("normalize_std", [0.229, 0.224, 0.225]))

    model = build_model(backbone_name, num_classes=8, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    transform = get_eval_transform(img_size=img_size, mean=mean, std=std)
    return LoadedModel(model, backbone_name, device, img_size, mean, std, transform)
