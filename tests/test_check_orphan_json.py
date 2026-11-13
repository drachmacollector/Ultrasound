"""scripts/research/check_orphan_json.py: placeholder and figure-source references."""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("orph", Path("scripts/research/check_orphan_json.py"))
orph = importlib.util.module_from_spec(spec)
spec.loader.exec_module(orph)  # type: ignore[union-attr]


def test_orphans(tmp_path: Path):
    j = tmp_path / "json"
    j.mkdir()
    for stem in ("E1", "E2", "R4_subsets", "lonely", "R4"):
        (j / f"{stem}.json").write_text("{}", encoding="utf-8")
    tpl = tmp_path / "t.md"
    tpl.write_text("a {{E1.x.y|bare}} b {{R4_subsets.seed}}", encoding="utf-8")
    fig = tmp_path / "figs.md"
    fig.write_text("| 2 | ROC | E2.json |\n| 9 | other | not_R4.json |", encoding="utf-8")
    # 'R4' is not referenced: 'R4_subsets' and 'not_R4.json' must not count as references to it
    assert set(orph.orphans(j, tpl, (fig, tmp_path / "missing.py"))) == {"R4.json", "lonely.json"}


def test_repository_has_no_orphans():
    assert orph.orphans() == []
