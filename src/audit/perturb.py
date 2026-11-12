"""
src/audit/perturb.py

Test-time perturbations for the reliance experiments (R3 E5). Each is a picklable callable
``perturb(gray_uint8, meta) -> gray_uint8`` for ``src.eval.predict.predict(..., perturb=...)``.
They act on the native-resolution greyscale image using the R1 masks (``meta["masks"]``) and
change **only** their declared region (``region(...)``), so everything else is pixel-identical.

    a_text           Telea-inpaint the UI/text mask (dilated 2 px), restricted to outside the
                     anatomical sector (``sector_clean``)
    b_zero           everything outside ``sector_clean`` set to 0
    b_mean           ... set to the image's out-of-sector median grey
    b_blur           ... replaced by a heavily blurred copy of the image (Gaussian, sigma = max(h, w) / 40)
    e_ui_transplant  (exploratory) everything outside the recipient's ``sector_clean`` replaced by the
                     out-of-sector content of a donor image from another machine (donor's own sector
                     blanked to 0), resized to the recipient frame

Gate G1 made ``a_marker`` / ``c_control`` (marker inpainting vs equal-area control) infeasible
(caliper detector recall 0.20; no detector∧audit-positive test images), so they are not implemented.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.audit.metadata import load_masks, mask_path
from src.data.transforms import load_grayscale

KINDS = ("a_text", "b_zero", "b_mean", "b_blur")
UI_DILATE_PX = 2
INPAINT_RADIUS = 3
BLUR_DIVISOR = 40


def _masks(meta: dict) -> dict[str, np.ndarray]:
    m = meta.get("masks")
    if m is None:
        raise ValueError(f"no R1 masks for {meta.get('image_id')}")
    return m


def outside(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Out-of-sector region: everything that is not anatomical sector (bool)."""
    return ~masks["sector_clean"].astype(bool)


def region(kind: str, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Pixels a perturbation may change (bool)."""
    if kind == "a_text":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * UI_DILATE_PX + 1,) * 2)
        return cv2.dilate(masks["ui"].astype(np.uint8), k).astype(bool) & outside(masks)
    if kind in ("b_zero", "b_mean", "b_blur", "e_ui_transplant"):
        return outside(masks)
    raise ValueError(f"unknown perturbation {kind!r}")


@dataclass(frozen=True)
class Perturbation:
    kind: str

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown perturbation {self.kind!r}; known: {KINDS}")

    @property
    def variant(self) -> str:
        return f"perturb-{self.kind}"

    def __call__(self, gray: np.ndarray, meta: dict) -> np.ndarray:
        masks = _masks(meta)
        reg = region(self.kind, masks)
        out = gray.copy()
        if not reg.any():
            return out
        if self.kind == "a_text":
            painted = cv2.inpaint(gray, reg.astype(np.uint8), INPAINT_RADIUS, cv2.INPAINT_TELEA)
            out[reg] = painted[reg]
        elif self.kind == "b_zero":
            out[reg] = 0
        elif self.kind == "b_mean":
            out[reg] = np.uint8(np.median(gray[reg]))
        elif self.kind == "b_blur":
            sigma = max(gray.shape) / BLUR_DIVISOR
            out[reg] = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)[reg]
        return out


@dataclass(frozen=True)
class UITransplant:
    """Exploratory: give each recipient the out-of-sector frame of a donor from another machine.

    ``donors`` maps recipient image_id -> (donor image_path, donor image_id, donor dataset).
    """
    donors: dict[str, tuple[str, str, str]] = field(hash=False)

    @property
    def variant(self) -> str:
        return "perturb-e_ui_transplant"

    def __call__(self, gray: np.ndarray, meta: dict) -> np.ndarray:
        masks = _masks(meta)
        if meta["image_id"] not in self.donors:
            return gray.copy()
        path, did, dds = self.donors[meta["image_id"]]
        dgray = load_grayscale(path)
        dsec = load_masks(mask_path(did, dds))["sector_clean"].astype(bool)
        dframe = np.where(dsec, 0, dgray).astype(np.uint8)          # donor anatomy removed
        h, w = gray.shape
        dframe = cv2.resize(dframe, (w, h), interpolation=cv2.INTER_LINEAR)
        reg = outside(masks)
        out = gray.copy()
        out[reg] = dframe[reg]
        return out


def pick_donors(recipients: "pd.DataFrame", pool: "pd.DataFrame", seed: int = 0) -> dict:  # noqa: F821
    """Deterministic donor per recipient: a random image from ``pool`` on a *different* machine pair.

    Both frames need ``image_id, image_path, source_machine``; ``pool`` also ``dataset``.
    Recipients on Aloka get Voluson E6 donors and vice versa; others are skipped.
    """
    rng = np.random.default_rng(seed)
    swap = {"Aloka": "Voluson E6", "Voluson E6": "Aloka"}
    out = {}
    for r in recipients.itertuples(index=False):
        if r.source_machine not in swap:
            continue
        cand = pool[pool["source_machine"] == swap[r.source_machine]]
        d = cand.iloc[int(rng.integers(0, len(cand)))]
        out[r.image_id] = (str(d["image_path"]), str(d["image_id"]), str(d["dataset"]))
    return out
