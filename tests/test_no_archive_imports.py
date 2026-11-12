"""Live code must never import archived legacy modules (archive/, realtime, smoothing, ACAM, multitask)."""
from __future__ import annotations

import ast
from pathlib import Path

LIVE_DIRS = ("src", "scripts", "tests")
FORBIDDEN_PREFIXES = (
    "archive",
    "src.realtime",
    "src.smoothing",
    "src.models.acam",
    "src.models.multitask_model",
    "src.data.synthetic_video",
    "src.eval.evaluate_test",
    "streamlit",
)


def _imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    mods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.append(node.module)
    return mods


def test_no_archived_imports() -> None:
    offenders = []
    for d in LIVE_DIRS:
        for py in Path(d).rglob("*.py"):
            for mod in _imported_modules(py):
                if any(mod == p or mod.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                    offenders.append(f"{py}: {mod}")
    assert not offenders, "archived modules imported:\n" + "\n".join(offenders)


def test_archived_paths_gone_from_live_tree() -> None:
    for p in ("src/realtime", "src/smoothing", "src/models/acam.py", "src/models/multitask_model.py",
              "app_streamlit.py", "docs/phases", "docs/instructions"):
        assert not Path(p).exists(), f"{p} should live under archive/"
