"""
src/preprocess/clean.py  (R7 E11: the B2 input cleaning as a reusable, label-free function)

``clean_image`` turns one 2-D B-mode greyscale frame into the input the R5 B2 arm sees:

    1. segment the imaging sector (``segment_sector``; no device consensus, as for the external sets),
    2. detect UI/text glyphs and in-sector measurement markers with the frozen R1.5 overlay parameters,
    3. Telea-inpaint the marker mask, zero everything outside the cleaned sector,
    4. crop to the sector bounding box (native scale)  -> ``native``
    5. optionally letterbox to ``size`` x ``size`` (longest side -> size, zero padding, centred)  -> ``letterbox``

Steps 1-4 are exactly ``audit.metadata.image_features`` + ``pipeline.prepare_native(..., "sector_letterbox_inpaint")``,
the code that built the B2 cache for HC18, UCL and the African set (tests/test_clean.py checks this on cached images).
FPDB images in the study additionally used a train-only device-geometry consensus sector; a new user has no such
consensus, so the tool uses the per-image path that the external sets used.

No labels, no model, no torch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.audit.metadata import image_features
from src.preprocess.overlays import DEFAULT_PARAMS, OverlayParams
from src.preprocess.pipeline import OVERLAY_PARAMS, prepare_native, to_input_geometry

MODE = "sector_letterbox_inpaint"


@dataclass
class Cleaned:
    native: np.ndarray            # uint8 sector crop, markers inpainted, zero outside the sector
    sector: np.ndarray            # uint8 {0,1} sector mask of ``native``
    letterbox: np.ndarray | None  # uint8 size x size, or None
    masks: dict[str, np.ndarray]  # full-frame masks: sector, sector_clean, ui, marker
    info: dict                    # label-free features (sector shape, caliper count, overlay area, ...)


def load_params(path: Path | None = OVERLAY_PARAMS) -> OverlayParams:
    """The frozen study parameters if available, else the package defaults."""
    return OverlayParams.from_json(path) if path is not None and Path(path).exists() else DEFAULT_PARAMS


def clean_image(gray: np.ndarray, size: int | None = 224, params: OverlayParams | None = None) -> Cleaned:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        raise ValueError("expected a uint8 image")
    info, masks = image_features(gray, params or load_params())
    native, sector = prepare_native(gray, masks, MODE)
    letterbox = to_input_geometry(native, size, MODE) if size else None
    return Cleaned(native=native, sector=sector, letterbox=letterbox, masks=masks, info=info)
