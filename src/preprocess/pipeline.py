"""
src/preprocess/pipeline.py

Preprocessing modes. A mode turns the native-resolution greyscale image (plus its R1 masks)
into the image that enters the geometric transform (to the model input), then normalisation.

    resize                     the original image, square resize (B0; legacy pipeline)
    sector_letterbox           R5 B1: zero outside the fan sector (``sector_clean``), crop to the
                               sector bounding box, then letterbox: longest side -> S, zero padding
                               to S x S, centred (aspect ratio kept)
    sector_letterbox_inpaint   R5 B2/AFP: as above, after Telea inpainting of the R1 ``marker``
                               mask (frozen R1.5 detector parameters)
    nonanat_only               R3 E4: sector interior filled with its median grey, outside kept
    overlay_only               R3 E4: black canvas plus the pixels under ``ui | marker``
    sector_shape_only          R3 E4: the binary sector mask, white on black

Caches (``scripts/research/build_cache.py``):
* E4 modes are stored **pre-resized** to the model input (``A.Resize`` is the first op of the
  train and eval transforms, and resizing a square image to its own size is the identity).
* Letterbox modes are stored at **native scale** (cropped image + cropped sector mask), because
  caliper augmentation (R5.2) draws markers in native pixels before the geometric transform.
  Every letterbox cache carries ``cache_manifest.json`` with the overlay-parameter hash and
  ``PIPELINE_VERSION``; a mismatch is an error (stale cache).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

from src.preprocess.overlays import inpaint_overlays

CACHE_ROOT = Path("data/processed/cache")
OVERLAY_PARAMS = Path("data/processed/audit/overlay_params.json")
PIPELINE_VERSION = "r5.1-v1"
CACHED_MODES: tuple[str, ...] = ("nonanat_only", "overlay_only", "sector_shape_only")
LETTERBOX_MODES: tuple[str, ...] = ("sector_letterbox", "sector_letterbox_inpaint")
ALL_MODES: tuple[str, ...] = ("resize", *LETTERBOX_MODES, *CACHED_MODES)


def _read_gray(path: str | Path) -> np.ndarray:
    """cv2 greyscale read (``src.data.transforms.load_grayscale`` imports this module)."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def check_mode(mode: str) -> None:
    if mode not in ALL_MODES:
        raise ValueError(f"unknown preprocess_mode {mode!r}; known: {ALL_MODES}")


def is_cached(mode: str) -> bool:
    """True for the R3 E4 modes, whose cache is stored pre-resized to the model input."""
    check_mode(mode)
    return mode in CACHED_MODES


def is_letterbox(mode: str) -> bool:
    check_mode(mode)
    return mode in LETTERBOX_MODES


# ----------------------------------------------------------------------------- rendering
def render(gray: np.ndarray, masks: dict[str, np.ndarray], mode: str) -> np.ndarray:
    """Native-resolution rendering of the R3 E4 modes (uint8, same shape as ``gray``)."""
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
    raise ValueError(f"render() handles resize and the E4 modes, not {mode!r}")


def sector_bbox(sector: np.ndarray) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1), x1/y1 exclusive; the whole frame if the sector is empty."""
    ys, xs = np.nonzero(sector)
    if xs.size == 0:
        return 0, 0, sector.shape[1], sector.shape[0]
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def crop_to_sector(arr: np.ndarray, sector: np.ndarray) -> np.ndarray:
    x0, y0, x1, y1 = sector_bbox(sector)
    return arr[y0:y1, x0:x1]


def prepare_native(gray: np.ndarray, masks: dict[str, np.ndarray], mode: str) -> tuple[np.ndarray, np.ndarray]:
    """(image_uint8, sector_mask_uint8) at native scale for ``mode``.

    Letterbox modes return the sector crop (zero outside the sector); other modes return the
    full frame and its ``sector_clean`` mask.
    """
    check_mode(mode)
    sector = masks["sector_clean"].astype(np.uint8)
    if mode not in LETTERBOX_MODES:
        return render(gray, masks, mode), sector
    img = inpaint_overlays(gray, masks["marker"]) if mode == "sector_letterbox_inpaint" else gray.copy()
    img[sector == 0] = 0
    return crop_to_sector(img, sector), crop_to_sector(sector, sector)


# ----------------------------------------------------------------------------- geometry
def geometric(img_size: int, mode: str) -> list[A.BasicTransform]:
    """The deterministic geometric transform to the model input; shared by train and eval."""
    check_mode(mode)
    if mode in LETTERBOX_MODES:
        return [A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT,
                              fill=0, position="center")]
    return [A.Resize(img_size, img_size)]


def to_input_geometry(arr: np.ndarray, img_size: int, mode: str) -> np.ndarray:
    """Apply ``geometric`` alone (e.g. to a float mask)."""
    return A.Compose(geometric(img_size, mode))(image=arr)["image"]


# ----------------------------------------------------------------------------- caches
def cache_dir(mode: str, img_size: int | None = None) -> Path:
    """E4 modes: ``<mode>_<size>``; letterbox modes: ``<mode>`` (native scale, size-independent)."""
    return CACHE_ROOT / (mode if is_letterbox(mode) else f"{mode}_{img_size}")


def cache_path(mode: str, img_size: int | None, dataset: str, image_id: str) -> Path:
    """data/processed/cache/<mode>[_<size>]/<dataset>/<image_id>.png"""
    return cache_dir(mode, img_size) / dataset / f"{image_id}.png"


def sector_cache_path(mode: str, dataset: str, image_id: str) -> Path:
    return cache_dir(mode) / dataset / f"{image_id}_sector.png"


def resize_for_cache(img: np.ndarray, img_size: int) -> np.ndarray:
    """The exact resize the train/eval transforms apply first (albumentations defaults)."""
    return A.Resize(img_size, img_size)(image=img)["image"]


def cache_key() -> dict:
    return {"overlay_params_sha256": hashlib.sha256(OVERLAY_PARAMS.read_bytes()).hexdigest(),
            "pipeline_version": PIPELINE_VERSION}


def write_cache_manifest(mode: str) -> None:
    d = cache_dir(mode)
    d.mkdir(parents=True, exist_ok=True)
    (d / "cache_manifest.json").write_text(json.dumps(cache_key(), indent=1), encoding="utf-8")


def check_cache(mode: str) -> None:
    """Raise unless the letterbox cache for ``mode`` exists and matches the current key."""
    m = cache_dir(mode) / "cache_manifest.json"
    if not m.exists():
        raise FileNotFoundError(f"no cache for {mode!r}: run scripts/research/build_cache.py --mode {mode}")
    if json.loads(m.read_text(encoding="utf-8")) != cache_key():
        raise RuntimeError(f"stale cache for {mode!r} (overlay params or pipeline version changed); rebuild it")


def load_prepared(mode: str, img_size: int, dataset: str, image_id: str,
                  image_path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """(greyscale image, sector mask or None) as it enters the geometric transform."""
    if mode == "resize":
        return _read_gray(image_path), None
    if is_cached(mode):
        return _read_gray(cache_path(mode, img_size, dataset, image_id)), None
    img = _read_gray(cache_path(mode, None, dataset, image_id))
    sector = _read_gray(sector_cache_path(mode, dataset, image_id))
    return img, (sector > 0).astype(np.uint8)
