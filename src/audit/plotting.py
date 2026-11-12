"""
src/audit/plotting.py

Shared figure conventions for R3+ (matplotlib, Agg).

* Collapsed classes use one fixed colour each, in a fixed order (validated categorical palette:
  CVD and normal-vision separation pass for adjacent pairs; low-contrast slots are always shown
  with white segment gaps and a matching table in results/tables).
* Every figure is written as PDF + PNG (``save``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR = Path("results/figures")
CLASS_COLORS: dict[str, str] = {
    "Head": "#2a78d6",
    "Abdomen": "#eb6834",
    "Femur": "#1baf7a",
    "Thorax": "#eda100",
    "Cervix": "#e87ba4",
    "Other": "#6250d6",
}
#: series colours for up to 3 compared things (first three palette slots validate all-pairs)
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
NEUTRAL = "#8a8984"
INK = "#2b2b2a"

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#9a9993", "axes.labelcolor": INK, "xtick.color": INK,
    "ytick.color": INK, "axes.grid": False, "legend.frameon": False, "pdf.fonttype": 42,
})


def save(fig: plt.Figure, stem: str) -> Path:
    """Write results/figures/<stem>.pdf and .png (200 dpi); close the figure."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf = FIG_DIR / f"{stem}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return pdf
