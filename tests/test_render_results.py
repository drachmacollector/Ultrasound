"""scripts/research/render_results.py: toy template + toy JSON -> expected markdown; missing keys raise."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("render_results", Path("scripts/research/render_results.py"))
rr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rr)  # type: ignore[union-attr]


@pytest.fixture()
def store(tmp_path: Path):
    (tmp_path / "E2.json").write_text(json.dumps({
        "prior": {"auroc": {"point": 0.7351, "lo": 0.714, "hi": 0.7512, "n_images": 4153, "n_clusters": 712}},
        "delta": {"delta": 0.137, "lo": 0.122, "hi": 0.154, "p_two_sided": 0.0005},
        "share": 0.5163, "a|b": 2.0, "n": 1071, "ok": True, "methods": ["gradcam", "occlusion"], "seq": [{"x": 1.5}],
    }), encoding="utf-8")
    return rr.Store(tmp_path)


def test_render_formats(store, tmp_path: Path) -> None:
    log = tmp_path / "log.md"
    log.write_text("## A\noutcome: x\ndeviation: one\n## B\ndeviation: two\ndeviation: three\n", encoding="utf-8")
    tpl = ("AUROC {{E2.prior.auroc}}; bare {{ E2.prior.auroc | bare, dp=2 }}; Δ {{E2.delta}}; "
           "pct {{E2.share|pct}}; n {{E2.n}}; ok {{E2.ok}}; m {{E2.methods|join}}; idx {{E2.seq.0.x|dp=1}}\n{{@deviations}}")
    out, missing = rr.render(tpl, store, log=log)
    assert missing == []
    assert "AUROC 0.735 [0.714, 0.751] (n = 4,153 images, 712 patients)" in out
    assert "bare 0.74 [0.71, 0.75];" in out
    assert "Δ +0.137 [+0.122, +0.154], p = 0.0005" in out
    assert "pct 51.6%" in out and "n 1,071" in out and "ok yes" in out
    assert "m gradcam, occlusion" in out and "idx 1.5" in out
    assert rr.render("{{E2.a~b|dp=1}}", store)[0] == "2.0"
    assert "**A**\n\n- one" in out and "- two\n- three" in out
    assert set(store.used) == {(store.root / "E2.json").as_posix()}


def test_missing_key_raises_unless_allowed(store) -> None:
    with pytest.raises(rr.MissingKey):
        rr.render("{{E2.nope}}", store)
    with pytest.raises(rr.MissingKey):
        rr.render("{{E9.anything}}", store)
    out, missing = rr.render("x {{E2.nope}} y", store, allow_missing=True)
    assert out == f"x {rr.PENDING} y" and len(missing) == 1


def test_paired_percent() -> None:
    v = {"delta": -0.048, "lo": -0.06, "hi": -0.03, "p_two_sided": 0.001}
    assert rr.fmt_value(v, {"pct": ""}) == "-4.8 pp [-6.0, -3.0], p = 0.001"
