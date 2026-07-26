"""
scripts/run_sanity_checks.py

Produces all Phase 3 sanity-check visualizations and saves them to
data/processed/sanity_checks/ as PNG files.

Per 03_DATA_PIPELINE.md Step 7 and PHASE_3_KICKOFF_PROMPT.md Task 8:

1. Class distribution bar charts:
   - train / val / test (FETAL_PLANES_DB) — stacked/grouped bar chart
   - cross-device manifest broken down by source_subset (HC18 vs UCL)

2. Sample image grids:
   - ~5 images per class from train.csv (FETAL_PLANES_DB)
   - ~5 images per class from cross_device_manifest.csv

3. Patient count per class per split (printed as table + saved as CSV)

4. Image resolution / aspect-ratio histogram (FETAL_PLANES_DB only —
   confirms whether plain 224x224 resize is reasonable)

[MANUAL]: Review every chart before Phase 4 starts.
"""
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # Non-interactive backend -- no display required
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SPLITS_DIR = Path("data/splits")
PROCESSED_DIR = Path("data/processed")
OUT_DIR = Path("data/processed/sanity_checks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
CROSS_DEVICE_PATH = PROCESSED_DIR / "cross_device_manifest.csv"

CANONICAL_CLASSES = [
    "Brain_Trans_cerebellum", "Brain_Trans_thalamic", "Brain_Trans_ventricular",
    "Fetal_abdomen", "Fetal_femur", "Fetal_thorax", "Maternal_cervix", "Other",
]

# Shortened display names for cleaner chart axes
SHORT_NAMES = {
    "Brain_Trans_cerebellum": "B_cerebellum",
    "Brain_Trans_thalamic": "B_thalamic",
    "Brain_Trans_ventricular": "B_ventricular",
    "Fetal_abdomen": "F_abdomen",
    "Fetal_femur": "F_femur",
    "Fetal_thorax": "F_thorax",
    "Maternal_cervix": "M_cervix",
    "Other": "Other",
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Class distribution bar charts
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_distributions():
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    splits = {"Train": train_df, "Val": val_df, "Test": test_df}
    counts = {}
    for name, df in splits.items():
        counts[name] = df["plane_label"].value_counts().reindex(CANONICAL_CLASSES).fillna(0)

    dist_df = pd.DataFrame(counts)
    dist_df.index = [SHORT_NAMES.get(c, c) for c in dist_df.index]

    _fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(dist_df))
    width = 0.28
    colors = ["#4C72B0", "#55A868", "#C44E52"]

    for i, (split_name, color) in enumerate(zip(["Train", "Val", "Test"], colors)):
        bars = ax.bar(x + (i - 1) * width, dist_df[split_name], width, label=split_name,
                      color=color, alpha=0.85, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 15,
                        f"{int(h)}", ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(x)
    ax.set_xticklabels(dist_df.index, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Image count")
    ax.set_title("FETAL_PLANES_DB — Class distribution per split", fontsize=13, fontweight="bold")
    ax.legend(title="Split")
    ax.set_ylim(0, dist_df.max().max() * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "01_class_distribution_splits.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


def plot_cross_device_distribution():
    if not CROSS_DEVICE_PATH.exists():
        print("  [SKIP] cross_device_manifest.csv not found — skipping cross-device chart")
        return

    df = pd.read_csv(CROSS_DEVICE_PATH)
    counts = df.groupby(["source_subset", "plane_label"]).size().unstack(fill_value=0)

    _fig, ax = plt.subplots(figsize=(8, 5))
    counts.T.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", width=0.6)
    ax.set_xlabel("Plane label")
    ax.set_ylabel("Image count")
    ax.set_title("Cross-device manifest — Image count by label & source subset\n"
                 "(HC18 = Netherlands/Head only; UCL = UCLH Head+Abdomen+Femur)",
                 fontsize=11, fontweight="bold")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.legend(title="Source subset")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = OUT_DIR / "02_cross_device_distribution.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sample image grids
# ─────────────────────────────────────────────────────────────────────────────

def _load_gray_rgb(path: str) -> np.ndarray | None:
    """Load image as RGB array regardless of whether it's stored as gray or color."""
    img = cv2.imread(str(path))  # type: ignore
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # type: ignore


def plot_sample_grid_fetal_planes(n_per_class: int = 5):
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    classes = CANONICAL_CLASSES
    n_cols = n_per_class
    n_rows = len(classes)

    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.2))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.05)
    fig.suptitle("FETAL_PLANES_DB — Sample images per class (train split)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row_i, cls in enumerate(classes):
        cls_df = train_df[train_df["plane_label"] == cls]
        sampled = cls_df.sample(n=min(n_per_class, len(cls_df)), random_state=42)
        for col_i, (_, row) in enumerate(sampled.iterrows()):
            ax = fig.add_subplot(gs[row_i, col_i])
            img = _load_gray_rgb(row["image_path"])
            if img is not None:
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None, aspect="auto")
            ax.axis("off")
            if col_i == 0:
                ax.set_ylabel(SHORT_NAMES.get(cls, cls), rotation=0, ha="right",
                              va="center", fontsize=8, labelpad=60)

    plt.tight_layout()
    out = OUT_DIR / "03_sample_grid_fetal_planes.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


def plot_sample_grid_cross_device(n_per_label: int = 5):
    if not CROSS_DEVICE_PATH.exists():
        print("  [SKIP] cross_device_manifest.csv not found — skipping cross-device grid")
        return

    df = pd.read_csv(CROSS_DEVICE_PATH)
    groups = df.groupby(["source_subset", "plane_label"])
    keys_raw = list(groups.groups.keys())
    keys_tuple = [k for k in keys_raw if isinstance(k, tuple) and len(k) >= 2]
    keys = sorted(keys_tuple, key=lambda x: (str(x[0]), str(x[1])))

    n_rows = len(keys)
    n_cols = n_per_label

    fig = plt.figure(figsize=(n_cols * 2.2, n_rows * 2.2))
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig, hspace=0.35, wspace=0.05)
    fig.suptitle("Cross-device dataset — Sample images per (subset, label)",
                 fontsize=13, fontweight="bold", y=1.01)

    for row_i, (subset, label) in enumerate(keys):  # type: ignore
        grp = groups.get_group((subset, label))
        sampled = grp.sample(n=min(n_per_label, len(grp)), random_state=42)
        for col_i, (_, row) in enumerate(sampled.iterrows()):
            ax = fig.add_subplot(gs[row_i, col_i])
            img = _load_gray_rgb(row["image_path"])
            if img is not None:
                ax.imshow(img, cmap="gray" if img.ndim == 2 else None, aspect="auto")
            ax.axis("off")
            if col_i == 0:
                ax.set_ylabel(f"{subset}/{label}", rotation=0, ha="right",
                              va="center", fontsize=7, labelpad=80)

    plt.tight_layout()
    out = OUT_DIR / "04_sample_grid_cross_device.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Patient count per class per split (table)
# ─────────────────────────────────────────────────────────────────────────────

def save_patient_count_table():
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "val.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")

    def count_patients(df):
        return df.groupby("plane_label")["patient_id"].nunique().reindex(CANONICAL_CLASSES).fillna(0).astype(int)

    table = pd.DataFrame({
        "Train_patients": count_patients(train_df),
        "Val_patients": count_patients(val_df),
        "Test_patients": count_patients(test_df),
        "Train_images": train_df["plane_label"].value_counts().reindex(CANONICAL_CLASSES).fillna(0).astype(int),
        "Val_images": val_df["plane_label"].value_counts().reindex(CANONICAL_CLASSES).fillna(0).astype(int),
        "Test_images": test_df["plane_label"].value_counts().reindex(CANONICAL_CLASSES).fillna(0).astype(int),
    })

    out_csv = OUT_DIR / "05_patient_image_counts_per_split.csv"
    table.to_csv(out_csv)
    print(f"  Saved: {out_csv}")
    print("\n--- Patient & image counts per class per split ---")
    print(table.to_string())
    print("---\n")

    # Plot as heatmap for easy visual review
    _fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, cols, title in [
        (axes[0], ["Train_patients", "Val_patients", "Test_patients"], "Patients per class per split"),
        (axes[1], ["Train_images", "Val_images", "Test_images"], "Images per class per split"),
    ]:
        sub = table[cols].copy()
        sub.index = [SHORT_NAMES.get(c, c) for c in sub.index]
        sub.columns = [c.replace("_patients", "").replace("_images", "") for c in sub.columns]
        sns.heatmap(sub, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5,
                    ax=ax, cbar_kws={"shrink": 0.7})
        ax.set_title(title, fontweight="bold")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.suptitle("FETAL_PLANES_DB — Per-class coverage by split", fontsize=13, y=1.02)
    plt.tight_layout()
    out_png = OUT_DIR / "05_patient_count_heatmap.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_png}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Image resolution / aspect-ratio histogram
# ─────────────────────────────────────────────────────────────────────────────

def plot_resolution_histogram(max_sample: int = 1000):
    """Sample up to max_sample images from manifest.csv to check resolution distribution."""
    manifest = pd.read_csv(MANIFEST_PATH)
    sample = manifest.sample(n=min(max_sample, len(manifest)), random_state=42)

    widths, heights, aspects = [], [], []
    skipped = 0
    for _, row in sample.iterrows():
        img = cv2.imread(str(row["image_path"]))  # type: ignore
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]
        heights.append(h)
        widths.append(w)
        aspects.append(w / h)

    if skipped:
        print(f"  [WARN] Could not read {skipped} images during resolution scan (skipped)")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].hist(widths, bins=30, color="#4C72B0", edgecolor="white", alpha=0.85)
    axes[0].set_xlabel("Width (px)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Image widths")

    axes[1].hist(heights, bins=30, color="#55A868", edgecolor="white", alpha=0.85)
    axes[1].set_xlabel("Height (px)")
    axes[1].set_title("Image heights")

    axes[2].hist(aspects, bins=30, color="#C44E52", edgecolor="white", alpha=0.85)
    axes[2].set_xlabel("Aspect ratio (W/H)")
    axes[2].set_title("Aspect ratios")
    axes[2].axvline(1.0, color="black", linestyle="--", linewidth=1, label="Square (1:1)")
    axes[2].legend()

    w_arr = np.array(widths)
    h_arr = np.array(heights)
    a_arr = np.array(aspects)
    stats_text = (
        f"W: {w_arr.min()}–{w_arr.max()} px (median {int(np.median(w_arr))})\n"
        f"H: {h_arr.min()}–{h_arr.max()} px (median {int(np.median(h_arr))})\n"
        f"AR: {a_arr.min():.2f}–{a_arr.max():.2f} (median {np.median(a_arr):.2f})"
    )
    fig.text(0.5, -0.05, stats_text, ha="center", fontsize=9, color="#555555")

    plt.suptitle(f"FETAL_PLANES_DB — Image resolution distribution (n={len(widths)} sampled)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = OUT_DIR / "06_resolution_histogram.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n=== Sanity check 1: Class distribution charts ===")
    plot_class_distributions()
    plot_cross_device_distribution()

    print("\n=== Sanity check 2: Sample image grids ===")
    plot_sample_grid_fetal_planes()
    plot_sample_grid_cross_device()

    print("\n=== Sanity check 3: Patient count per class per split ===")
    save_patient_count_table()

    print("\n=== Sanity check 4: Resolution histogram ===")
    plot_resolution_histogram()

    print(f"\nAll sanity-check visualizations saved to {OUT_DIR}/")
    print("\n[MANUAL REVIEW REQUIRED] Open the sanity_checks/ folder and review every chart")
    print("before starting Phase 4. This is the cheapest point to catch labeling/leakage bugs.")


if __name__ == "__main__":
    main()
