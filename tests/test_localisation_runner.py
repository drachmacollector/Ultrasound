"""src/localisation/runner.py helpers that need no data: threaded batching and per-map metrics."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.localisation import runner as R


class Toy(Dataset):
    def __len__(self) -> int:
        return 10

    def __getitem__(self, i: int) -> dict:
        return {"idx": i, "x": torch.full((2,), float(i))}


def test_threaded_batches_keep_order_and_last_partial_batch() -> None:
    batches = list(R.threaded_batches(Toy(), 4, workers=3))
    assert [b["idx"].tolist() for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert batches[2]["x"].shape == (2, 2)
    assert list(R.threaded_batches(Toy(), 20, workers=1))[0]["idx"].tolist() == list(range(10))


def test_map_metrics_columns() -> None:
    S = 32
    sal = np.zeros((S, S), np.float32)
    sal[10:20, 10:20] = 1
    sector = np.zeros((S, S), np.float32)
    sector[4:28, 4:28] = 1
    ui = np.zeros((S, S), np.float32)
    ui[0:2, 0:5] = 1
    anat = np.zeros((3, S, S), np.float32)
    anat[:, 8:22, 8:22] = 1
    m = R.map_metrics(sal, sector, ui, anat, femur=True)
    assert m["out_sector_frac"] == 0.0 and m["overlay_frac"] == 0.0 and m["epg"] == 1.0 and m["hit"] == 1.0
    assert {"epg_minus_area_w05", "epg_minus_area_w15"} <= set(m)
    assert set(R.map_metrics(sal, sector, ui, None, femur=False)) == {"out_sector_frac", "overlay_frac", "sal_energy"}
