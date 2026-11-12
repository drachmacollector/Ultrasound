"""Shared pytest configuration: repo root on sys.path, data/gpu skip markers."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# Tests use repo-relative paths (data/splits/..., configs/...), like every script.
os.chdir(REPO_ROOT)

FPDB_IMAGES = REPO_ROOT / "data" / "raw" / "fetal_planes_db" / "Images"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    has_data = FPDB_IMAGES.is_dir()
    try:
        import torch

        has_gpu = torch.cuda.is_available()
    except ImportError:  # pragma: no cover
        has_gpu = False
    skip_data = pytest.mark.skip(reason="local datasets not available")
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "data" in item.keywords and not has_data:
            item.add_marker(skip_data)
        if "gpu" in item.keywords and not has_gpu:
            item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT
