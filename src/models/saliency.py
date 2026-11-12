"""
src/models/saliency.py  (R4.1)

Saliency maps in model-input space for the localisation audit.

    METHODS = ("gradcam", "gradcampp", "occlusion")
    compute_saliency(model, x, targets, method, backbone_name) -> (B, H, W) float32

* ``gradcam`` / ``gradcampp`` use ``pytorch_grad_cam`` (GradCAM / GradCAMPlusPlus) on the layer
  given by ``src.models.gradcam.get_target_layer``. The library min-max normalises every map
  to [0, 1]; ``_RawCAM`` overrides that step, so maps are the raw ReLU'd CAM, bilinearly
  upsampled to the input size (same ``cv2.resize`` call the library uses). Metrics normalise
  as they need (EPG, a ratio of sums, is scale-invariant anyway).
* A target is a class index or a *set* of class indices. A set means the summed logits
  (``SumOfLogitsTarget``) for CAMs and the summed softmax probability for occlusion, so a Head
  image can target {trans-cerebellum, trans-thalamic, trans-ventricular} jointly.
* ``occlusion``: a ``patch`` x ``patch`` square slid with ``stride`` over the input and filled
  with a per-image value (normally the image's in-sector median grey, normalised). Saliency at
  a pixel = mean drop of the target probability over the patches covering it. One pass yields
  the drop for all classes, so every target set comes from the same forwards
  (``SaliencyEngine.occlusion_all``).

Create one ``SaliencyEngine`` per model and reuse it over batches (hooks are registered once).
"""
from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus

from src.models.gradcam import get_target_layer

METHODS = ("gradcam", "gradcampp", "occlusion")
Target = int | Sequence[int]

#: Occlusion geometry per input size: (patch, stride). 16/8 at 224 px (plan R4.1); at 300 px
#: (EfficientNetV2-S) stride 10 with patch = 2 x stride keeps the patch/stride ratio.
OCCLUSION_GEOMETRY = {224: (16, 8), 300: (20, 10)}


def as_index_list(t: Target) -> list[int]:
    return [int(t)] if isinstance(t, int | np.integer) else [int(i) for i in t]


class SumOfLogitsTarget:
    """Grad-CAM target = sum of the logits of ``idxs`` (one image's output, or a batch)."""

    def __init__(self, idxs: Target):
        self.idxs = as_index_list(idxs)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        if model_output.ndim == 1:
            return model_output[self.idxs].sum()
        return model_output[:, self.idxs].sum(dim=-1)


class _RawCAM:
    """Mixin: keep the raw ReLU'd CAM (no per-image min-max scaling), upsampled to the input."""

    def compute_cam_per_layer(self, input_tensor, targets, eigen_smooth):  # noqa: D401 (library API)
        acts = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
        grads = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        w, h = self.get_target_width_height(input_tensor)
        out = []
        for layer, a, g in zip(self.target_layers, acts, grads):
            cam = np.maximum(self.get_cam_image(input_tensor, layer, targets, a, g, eigen_smooth), 0)
            out.append(np.stack([cv2.resize(np.float32(c), (w, h)) for c in cam])[:, None])
        return out

    def aggregate_multi_layers(self, cam_per_target_layer):
        return np.mean(np.concatenate(cam_per_target_layer, axis=1), axis=1)


class RawGradCAM(_RawCAM, GradCAM):
    pass


class RawGradCAMPlusPlus(_RawCAM, GradCAMPlusPlus):
    pass


_CAMS = {"gradcam": RawGradCAM, "gradcampp": RawGradCAMPlusPlus}


def normalised_fill(grey: np.ndarray | float, mean: Sequence[float], std: Sequence[float]) -> np.ndarray:
    """Grey level(s) in [0, 255] -> (B, 3) values in normalised model-input space."""
    g = np.atleast_1d(np.asarray(grey, dtype=np.float64)) / 255.0
    return ((g[:, None] - np.asarray(mean)[None]) / np.asarray(std)[None]).astype(np.float32)


class SaliencyEngine:
    """One model, one method; reusable across batches.

    ``target_layer`` overrides the backbone's registered CAM layer (used by unit tests). CAM inputs are
    processed in fixed batches of ``cam_batch`` (padded), occlusion copies in fixed chunks of
    ``occlusion_batch``; occlusion runs under fp16 autocast (``occlusion_amp``).
    """

    def __init__(self, model: nn.Module, backbone_name: str, method: str, *,
                 occlusion_patch: int | None = None, occlusion_stride: int | None = None,
                 occlusion_batch: int = 256, occlusion_amp: bool = True,
                 target_layer: nn.Module | None = None, cam_batch: int = 32):
        if method not in METHODS:
            raise ValueError(f"unknown saliency method {method!r}; choose from {METHODS}")
        self.model, self.backbone_name, self.method = model.eval(), backbone_name, method
        self.device = next(model.parameters()).device
        self.cam = None
        self._requires_grad: dict[nn.Parameter, bool] = {}
        if method in _CAMS:
            layer = target_layer if target_layer is not None else get_target_layer(model, backbone_name)
            # Only the target layer's parameters need gradients: the backward pass then stops at that
            # layer instead of running through the whole network (CAMs are unchanged; restored on close).
            self._requires_grad = {p: p.requires_grad for p in model.parameters()}
            for p in model.parameters():
                p.requires_grad_(False)
            for p in layer.parameters():
                p.requires_grad_(True)
            self.cam = _CAMS[method](model=model, target_layers=[layer])
        self.patch, self.stride = occlusion_patch, occlusion_stride
        self.occ_batch, self.occ_amp, self.cam_batch = occlusion_batch, occlusion_amp, cam_batch
        self._occ_cache: dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------ CAMs
    def cam_maps(self, x: torch.Tensor, targets: Sequence[Target]) -> tuple[np.ndarray, np.ndarray]:
        """(B, H, W) raw CAM maps and the (B, C) logits of the same forward pass."""
        if self.cam is None:
            raise RuntimeError(f"{self.method} is not a CAM method")
        if len(targets) != len(x):
            raise ValueError("one target per image")
        maps, logits = [], []
        for k in range(0, len(x), self.cam_batch):
            xb, tb = x[k:k + self.cam_batch], list(targets[k:k + self.cam_batch])
            n = len(xb)
            if n < self.cam_batch:
                # Pad to a fixed batch size: every new input shape costs seconds of cuDNN set-up
                # (measured ~5 s per shape). Samples are independent in eval mode, so padding is exact.
                xb = torch.cat([xb, xb[-1:].expand(self.cam_batch - n, -1, -1, -1)])
                tb += [tb[-1]] * (self.cam_batch - n)
            with torch.enable_grad():
                m = self.cam(input_tensor=xb.to(self.device), targets=[SumOfLogitsTarget(t) for t in tb])
            maps.append(m[:n])
            logits.append(self.cam.outputs.detach().float().cpu().numpy()[:n])
            self.cam.outputs = None  # drop the retained autograd graph
        return np.concatenate(maps).astype(np.float32), np.concatenate(logits)

    # ------------------------------------------------------------------ occlusion
    def _geometry(self, size: int) -> tuple[int, int]:
        if self.patch is not None and self.stride is not None:
            return self.patch, self.stride
        if size not in OCCLUSION_GEOMETRY:
            raise ValueError(f"no default occlusion geometry for input size {size}; pass patch/stride")
        return OCCLUSION_GEOMETRY[size]

    @torch.no_grad()
    def _probs(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type=self.device.type, enabled=self.occ_amp and self.device.type == "cuda"):
            return torch.softmax(self.model(x).float(), dim=1)

    def _patches(self, H: int) -> tuple[torch.Tensor, torch.Tensor]:
        """(P, 1, H, W) bool patch masks and the (P, H*W) coverage-normalised scatter matrix."""
        patch, stride = self._geometry(H)
        key = (H, patch, stride)
        if key not in self._occ_cache:
            starts = list(range(0, H - patch + 1, stride))
            if starts[-1] != H - patch:  # cover the far border
                starts.append(H - patch)
            masks = torch.zeros(len(starts) ** 2, 1, H, H, dtype=torch.bool, device=self.device)
            for k, (r, c) in enumerate((r, c) for r in starts for c in starts):
                masks[k, 0, r:r + patch, c:c + patch] = True
            flat = masks.view(len(masks), -1).float()
            self._occ_cache[key] = (masks, flat / flat.sum(dim=0, keepdim=True))
        return self._occ_cache[key]

    @torch.no_grad()
    def occlusion_all(self, x: torch.Tensor, fill: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Per-class drop maps (B, C, H, W) and baseline probabilities (B, C).

        ``fill``: (B, 3) per-image fill in normalised input space (default 0 = dataset mean).
        Occluded copies of all images are evaluated in fixed-size chunks (the last one padded).
        """
        x = x.to(self.device)
        B, _, H, W = x.shape
        if H != W:
            raise ValueError("occlusion expects square inputs")
        masks, scatter = self._patches(H)
        P = len(masks)
        fill_t = (torch.zeros(B, 3, device=self.device) if fill is None
                  else torch.as_tensor(fill, dtype=x.dtype, device=self.device))
        p0 = self._probs(x)
        drops = torch.empty(B * P, p0.shape[1], device=self.device)
        for k in range(0, B * P, self.occ_batch):
            ids = torch.arange(k, k + self.occ_batch, device=self.device).clamp(max=B * P - 1)
            bi, pi = ids // P, ids % P
            xx = torch.where(masks[pi], fill_t[bi][:, :, None, None], x[bi])
            n = min(self.occ_batch, B * P - k)
            drops[k:k + n] = (p0[bi] - self._probs(xx))[:n]
        maps = torch.einsum("bpc,pq->bcq", drops.view(B, P, -1), scatter).view(B, -1, H, W)
        return maps.cpu().numpy().astype(np.float32), p0.cpu().numpy()

    @staticmethod
    def combine(class_maps: np.ndarray, targets: Sequence[Target]) -> np.ndarray:
        """(B, C, H, W) per-class maps -> (B, H, W) map of each image's target (sets are summed)."""
        return np.stack([class_maps[b, as_index_list(t)].sum(axis=0) for b, t in enumerate(targets)])

    # ------------------------------------------------------------------ generic
    def __call__(self, x: torch.Tensor, targets: Sequence[Target], fill: np.ndarray | None = None) -> np.ndarray:
        if self.method == "occlusion":
            maps, _ = self.occlusion_all(x, fill)
            return self.combine(maps, targets)
        return self.cam_maps(x, targets)[0]

    def close(self) -> None:
        if self.cam is not None:
            self.cam.activations_and_grads.release()
            self.cam = None
        for p, rg in self._requires_grad.items():
            p.requires_grad_(rg)
        self._requires_grad = {}
        self._occ_cache = {}


def compute_saliency(model: nn.Module, x: torch.Tensor, targets: Sequence[Target], method: str,
                     backbone_name: str, **kw) -> np.ndarray:
    """One-shot convenience wrapper; use ``SaliencyEngine`` for repeated batches."""
    fill = kw.pop("fill", None)
    eng = SaliencyEngine(model, backbone_name, method, **kw)
    try:
        return eng(x, targets, fill)
    finally:
        eng.close()
