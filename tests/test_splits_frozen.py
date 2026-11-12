"""The FPDB patient-disjoint splits are frozen: their SHA-256 must never change.

Hashes are computed on LF-normalised bytes so they do not depend on git core.autocrlf.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

SPLITS_DIR = Path("data/splits")
MANIFEST = SPLITS_DIR / "SPLITS_SHA256.txt"


def _read_manifest() -> dict[str, str]:
    out = {}
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split()
            out[name] = digest
    return out


def test_manifest_lists_the_three_splits() -> None:
    assert set(_read_manifest()) == {"train.csv", "val.csv", "test.csv"}


def test_split_hashes_unchanged() -> None:
    for name, digest in _read_manifest().items():
        actual = hashlib.sha256((SPLITS_DIR / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        assert actual == digest, f"{name} changed (sha256 {actual} != frozen {digest})"
