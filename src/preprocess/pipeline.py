"""
src/preprocess/pipeline.py

Preprocessing modes. A mode turns the native-resolution greyscale image (plus its R1 masks)
into the image that enters the B0 geometric pipeline (resize to the model input, normalise).

    resize               the original image (B0)
    nonanat_only         E4: sector interior (``sector_clean``) filled with the image's in-sector
                         median grey, except marker pixels; everything outside the sector kept.
                         -> only non-anatomical content (frame, UI, text, ruler, markers) survives.
    overlay_only         E4: black canvas plus the original pixels under ``ui | marker``.
    sector_shape_only    E4: the binary sector mask, white on black (frame + sector geometry only).

Non-``resize`` modes are *cached*: ``scripts/research/build_cache.py`` renders every image once
and stores it already resized to the model input size (``A.Resize``, the first op of both
the train and eval transforms). Resizing a cached square image to the same size is the
identity, so training and evaluation see exactly the pixels an on-the-fly pipeline would
produce, without re-decoding large PNGs every epoch (tests/test_preprocess_pipeline.py).
"""
from __future__ import annotations

from pathlib import Path

import albumentations as A
import numpy as np

CACHE_ROOT = Path("data/processed/cache")
CACHED_MODES: tuple[str, ...] = ("nonanat_only", "overlay_only", "sector_shape_only")
ALL_MODES: tuple[str, ...] = ("resize", *CACHED_MODES)


def is_cached(mode: str) -> bool:
    if mode not in ALL_MODES:
        raise ValueError(f"unknown preprocess_mode {mode!r}; known: {ALL_MODES}")
    return mode in CACHED_MODES


def render(gray: np.ndarray, masks: dict[str, np.ndarray], mode: str) -> np.ndarray:
    """Native-resolution rendering of ``mode`` (uint8, same shape as ``gray``)."""
    if mode == "resize":
        return gray
    sector = masks["sector_clean"].astype(bool)
    marker = masks["marker"].astype(bool)
    if mode == "nonanat_only":
        out = gray.copy()
        fill = sector & ~marker
        if sector.any():
            out[fill] = np.uint8(np.median(gray[sector]))
        return out
    if mode == "overlay_only":
        keep = masks["ui"].astype(bool) | marker
        out = np.zeros_like(gray)
        out[keep] = gray[keep]
        return out
    if mode == "sector_shape_only":
        return np.where(masks["sector"].astype(bool), 255, 0).astype(np.uint8)
    raise ValueError(f"unknown preprocess_mode {mode!r}")


def cache_dir(mode: str, img_size: int) -> Path:
    return CACHE_ROOT / f"{mode}_{img_size}"


def cache_path(mode: str, img_size: int, dataset: str, image_id: str) -> Path:
    """data/processed/cache/<mode>_<size>/<dataset>/<image_id>.png"""
    return cache_dir(mode, img_size) / dataset / f"{image_id}.png"


def resize_for_cache(img: np.ndarray, img_size: int) -> np.ndarray:
    """The exact resize the train/eval transforms apply first (albumentations defaults)."""
    return A.Resize(img_size, img_size)(image=img)["image"]
