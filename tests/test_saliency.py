"""src/models/saliency.py: maps peak where the evidence is; shapes; set targets; raw (unscaled) CAMs."""
from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.saliency import METHODS, SaliencyEngine, SumOfLogitsTarget, compute_saliency, normalised_fill

S = 32


def region_masks() -> torch.Tensor:
    """Class 0 evidence = top-left 10x10 block, class 1 = bottom-right 10x10 block."""
    m = torch.zeros(2, S, S)
    m[0, 4:14, 4:14] = 1
    m[1, 18:28, 18:28] = 1
    return m


class RegionFeatures(nn.Module):
    """Channel k = relu(mean of RGB) restricted to region k (a hand-built 1x1 conv + fixed mask)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 1, bias=False)
        with torch.no_grad():
            self.conv.weight.fill_(1 / 3)
        self.register_buffer("mask", region_masks())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x)) * self.mask[None]


class RegionNet(nn.Module):
    """logit_k = mean over the image of channel k = (area-scaled) mean intensity of region k."""

    def __init__(self) -> None:
        super().__init__()
        self.feat = RegionFeatures()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feat(x).mean(dim=(2, 3)) * 10


def inputs(n: int = 3, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 0.5 + 0.5 * torch.rand(n, 3, S, S, generator=g)


def engine(method: str) -> SaliencyEngine:
    model = RegionNet().eval()
    return SaliencyEngine(model, "toy", method, target_layer=model.feat,
                          occlusion_patch=4, occlusion_stride=2, occlusion_amp=False)


def peak_in(sal: np.ndarray, k: int) -> bool:
    m = region_masks()[k].numpy().astype(bool)
    return bool(m[np.unravel_index(np.argmax(sal), sal.shape)])


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("target", [0, 1])
def test_map_peaks_inside_evidence_region(method: str, target: int) -> None:
    x = inputs()
    eng = engine(method)
    maps = eng(x, [target] * len(x))
    eng.close()
    assert maps.shape == (len(x), S, S) and maps.dtype == np.float32
    for m in maps:
        assert peak_in(m, target)
        inside = region_masks()[target].numpy().astype(bool)
        assert m[inside].mean() > m[~inside].mean()


def test_cam_is_raw_not_minmax_scaled() -> None:
    """Doubling the input doubles the (linear) toy CAM; a [0, 1]-normalised map would not change."""
    eng = engine("gradcam")
    x = inputs(2)
    a, b = eng(x, [0, 0]), eng(2 * x, [0, 0])
    eng.close()
    assert a.max() > 0
    np.testing.assert_allclose(b, 2 * a, rtol=1e-4, atol=1e-7)


def test_sum_of_logits_target_equals_sum_of_single_targets() -> None:
    out = torch.randn(4, 8)
    t = SumOfLogitsTarget([0, 1, 2])
    expected = sum(SumOfLogitsTarget(i)(out) for i in (0, 1, 2))
    torch.testing.assert_close(t(out), expected)
    torch.testing.assert_close(t(out[1]), out[1, :3].sum())


def test_set_target_cam_is_sum_of_member_cams_for_linear_layer() -> None:
    """For this model Grad-CAM is linear in the target, so CAM({0,1}) = CAM(0) + CAM(1)."""
    eng = engine("gradcam")
    x = inputs(2)
    both = eng(x, [[0, 1]] * 2)
    parts = eng(x, [0, 0]) + eng(x, [1, 1])
    eng.close()
    np.testing.assert_allclose(both, parts, rtol=1e-4, atol=1e-7)


def test_occlusion_set_target_is_sum_of_class_drops() -> None:
    eng = engine("occlusion")
    x = inputs(2)
    maps, p0 = eng.occlusion_all(x)
    assert maps.shape == (2, 2, S, S) and p0.shape == (2, 2)
    np.testing.assert_allclose(eng.combine(maps, [[0, 1], [0, 1]]), maps.sum(axis=1), atol=1e-6)
    # with two classes the drops of p0 and p1 cancel exactly
    np.testing.assert_allclose(maps.sum(axis=1), 0, atol=1e-5)


def test_compute_saliency_wrapper_and_fill() -> None:
    model = RegionNet().eval()
    x = inputs(1)
    fill = normalised_fill([0.0], (0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
    assert fill.shape == (1, 3)
    m = compute_saliency(model, x, [1], "occlusion", "toy", occlusion_patch=4, occlusion_stride=2,
                         occlusion_amp=False, fill=fill)
    assert m.shape == (1, S, S) and peak_in(m[0], 1)
    with pytest.raises(ValueError):
        SaliencyEngine(model, "toy", "lime")


def test_cam_engine_restores_requires_grad_on_close() -> None:
    model = RegionNet().eval()
    eng = SaliencyEngine(model, "toy", "gradcam", target_layer=model.feat)
    eng(inputs(1), [0])
    eng.close()
    assert all(p.requires_grad for p in model.parameters())


def test_occlusion_chunking_does_not_change_maps() -> None:
    """Chunk size (and the padded last chunk) must not change the result."""
    x = inputs(3)
    a = engine("occlusion").occlusion_all(x)[0]
    model = RegionNet().eval()
    small = SaliencyEngine(model, "toy", "occlusion", occlusion_patch=4, occlusion_stride=2,
                           occlusion_amp=False, occlusion_batch=7)
    np.testing.assert_allclose(small.occlusion_all(x)[0], a, atol=1e-6)


def test_cam_padding_and_chunking_do_not_change_maps() -> None:
    x = inputs(5)
    model = RegionNet().eval()
    ref = SaliencyEngine(model, "toy", "gradcampp", target_layer=model.feat, cam_batch=5)(x, [0, 1, 0, 1, 1])
    for cb in (2, 8):
        eng = SaliencyEngine(model, "toy", "gradcampp", target_layer=model.feat, cam_batch=cb)
        np.testing.assert_allclose(eng(x, [0, 1, 0, 1, 1]), ref, rtol=1e-5, atol=1e-7)
        eng.close()
