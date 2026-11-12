"""
src/data/paths.py

Path and identifier helpers.

* Split CSVs store Windows backslash paths (``data\\raw\\...``); ``norm_path`` turns
  any stored path into a repo-relative ``Path``.
* FPDB filenames encode patient and acquisition: ``Patient00168_Plane2_1_of_2``.
  ``PlaneN`` is a per-patient acquisition index, NOT a class.
* Patient IDs are dataset-qualified strings so clusters never collide across datasets.
"""
from __future__ import annotations

import re
from pathlib import Path

FPDB_STEM_RE = re.compile(r"^Patient(\d+)_Plane(\d+)_(\d+)_of_(\d+)$")
NUM_PREFIX_RE = re.compile(r"^(\d+)_")


def norm_path(p: str | Path) -> Path:
    """Normalise a stored path (backslashes or slashes) to a pathlib Path."""
    return Path(str(p).replace("\\", "/"))


def fpdb_image_id(path: str | Path) -> str:
    """FPDB image id = filename stem, e.g. ``Patient00168_Plane2_1_of_2``."""
    return norm_path(path).stem


def parse_fpdb_stem(stem: str) -> dict[str, int]:
    """Parse ``Patient{n}_Plane{p}_{i}_of_{k}`` into ints; ValueError if malformed."""
    m = FPDB_STEM_RE.match(stem)
    if m is None:
        raise ValueError(f"not an FPDB image stem: {stem!r}")
    n, p, i, k = (int(g) for g in m.groups())
    return {"patient_num": n, "plane_n": p, "idx": i, "of_k": k}


def fpdb_patient_id(patient_num: int) -> str:
    return f"fpdb_{int(patient_num)}"


def numeric_prefix(name: str) -> int:
    """Leading integer of an HC18/UCL filename (``783_4HC.png`` -> 783)."""
    m = NUM_PREFIX_RE.match(name)
    if m is None:
        raise ValueError(f"filename has no numeric prefix: {name!r}")
    return int(m.group(1))
