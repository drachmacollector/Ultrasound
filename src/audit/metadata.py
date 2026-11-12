"""
src/audit/metadata.py

Per-image acquisition metadata and overlay/sector masks (R1.6).

``image_features(gray)`` returns label-free geometry / intensity / sector /
overlay features. ``save_masks`` / ``load_masks`` store the sector, cleaned
sector, UI and marker masks bit-packed in a compressed npz, so they can be
reused by perturbations (R3 E5) and preprocessing (R5) without recomputation.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.preprocess.overlays import DEFAULT_PARAMS, OverlayParams, detect_overlays
from src.preprocess.sector import refine_with_consensus, segment_sector

MASK_ROOT = Path("data/processed/masks")
MASK_KEYS = ("sector", "sector_clean", "ui", "marker")
CONSENSUS_ROOT = MASK_ROOT / "consensus"


def mask_path(image_id: str, dataset: str, root: Path = MASK_ROOT) -> Path:
    """data/processed/masks/<dataset>/<image_id>.npz ('/' in ids become subfolders)."""
    return root / dataset / f"{image_id}.npz"


def save_masks(path: Path, masks: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    shape = next(iter(masks.values())).shape
    np.savez_compressed(path, shape=np.array(shape), **{k: np.packbits(masks[k].astype(bool)) for k in MASK_KEYS})


def load_masks(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        shape = tuple(int(v) for v in z["shape"])
        n = shape[0] * shape[1]
        return {k: np.unpackbits(z[k])[:n].reshape(shape).astype(np.uint8) for k in MASK_KEYS}


CONSENSUS_ERODE_PX = 15


def consensus_key(machine: str, width: int, height: int) -> str:
    """Key of a device-geometry consensus sector, e.g. ``Voluson_E6_959x661``."""
    return f"{machine.replace(' ', '_')}_{int(width)}x{int(height)}"


def load_consensus_index(root: Path = CONSENSUS_ROOT) -> dict[str, np.ndarray]:
    """key -> train-only consensus core sector (empty dict if never built)."""
    idx = root / "consensus_index.csv"
    if not idx.exists():
        return {}
    out = {}
    for key in pd.read_csv(idx)["key"]:
        with np.load(root / f"{key}.npz") as z:
            out[key] = z["mask"].astype(np.uint8)
    return out


def image_features(gray: np.ndarray, params: OverlayParams = DEFAULT_PARAMS,
                   consensus: np.ndarray | None = None) -> tuple[dict, dict[str, np.ndarray]]:
    """Label-free features of one greyscale uint8 image, plus its masks.

    ``consensus``: optional train-only device-geometry consensus sector for this image's
    (machine, width, height) group (FPDB only; see scripts/build_sector_consensus.py).
    """
    h, w = gray.shape
    sec = segment_sector(gray)
    protect = None
    if consensus is not None:
        sec = refine_with_consensus(sec, consensus)
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * CONSENSUS_ERODE_PX + 1,) * 2)
        protect = cv2.erode(consensus, k)
    ov = detect_overlays(gray, sec, params, protect)
    x0, y0, x1, y1 = sec.bbox
    inside = sec.mask > 0
    feats = {
        "height": h, "width": w, "aspect": w / h,
        "mean_intensity": float(gray.mean()), "std_intensity": float(gray.std()),
        "frac_nonzero": float((gray > 0).mean()),
        "sector_area_frac": sec.area_frac, "sector_shape_class": sec.shape_class,
        "sector_width_ratio": sec.width_ratio,
        "sector_x0": x0 / w, "sector_y0": y0 / h, "sector_x1": x1 / w, "sector_y1": y1 / h,
        "sector_touch_top": sec.touches_border[0], "sector_touch_right": sec.touches_border[1],
        "sector_touch_bottom": sec.touches_border[2], "sector_touch_left": sec.touches_border[3],
        "sector_mean_intensity": float(gray[inside].mean()) if inside.any() else 0.0,
        "ui_area_frac": float(ov.ui_mask.mean()), "n_ui_components": ov.n_ui_components,
        "n_crosses": ov.n_crosses, "n_dotted_lines": ov.n_dotted_lines, "caliper_score": ov.caliper_score,
        "marker_area_frac": float(ov.marker_mask.mean()),
        "consensus_used": consensus is not None,
    }
    masks = {"sector": sec.mask, "sector_clean": ov.sector_clean, "ui": ov.ui_mask, "marker": ov.marker_mask}
    return feats, masks
