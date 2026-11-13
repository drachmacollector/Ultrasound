"""
scripts/research/check_orphan_json.py  (R6 D.3: every results JSON is used somewhere)

A file results/json/<stem>.json counts as used when its stem appears in the RESULTS template as a
placeholder (``{{<stem>.``) or as ``<stem>.json`` in a figure/report source:
results/figures/STUDY_FIGURES.md, make_figures.py, make_review_figures.py or render_box1.py.
Exit code 1 lists the orphans.

Usage:  python scripts/research/check_orphan_json.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

JSON_DIR = Path("results/json")
TEMPLATE = Path("docs/research/templates/RESULTS.template.md")
FIGURE_SOURCES = (Path("results/figures/STUDY_FIGURES.md"), Path("scripts/research/make_figures.py"),
                  Path("scripts/research/make_review_figures.py"), Path("scripts/research/render_box1.py"))


def orphans(json_dir: Path = JSON_DIR, template: Path = TEMPLATE,
            figure_sources: tuple[Path, ...] = FIGURE_SOURCES) -> list[str]:
    tpl = template.read_text(encoding="utf-8")
    figs = "\n".join(p.read_text(encoding="utf-8") for p in figure_sources if p.exists())
    out = []
    for p in sorted(json_dir.glob("*.json")):
        stem = re.escape(p.stem)
        if re.search(r"\{\{" + stem + r"\.", tpl) or re.search(r"(?<![\w-])" + stem + r"\.json", figs):
            continue
        out.append(p.name)
    return out


def main() -> None:
    o = orphans()
    n = len(list(JSON_DIR.glob("*.json")))
    if o:
        print(f"{len(o)} of {n} JSON files are not used by RESULTS.md or any figure:")
        for name in o:
            print(f"  {name}")
        sys.exit(1)
    print(f"all {n} JSON files in {JSON_DIR} are used by RESULTS.md or a figure")


if __name__ == "__main__":
    main()
