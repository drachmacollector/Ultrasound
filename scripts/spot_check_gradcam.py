"""
scripts/spot_check_gradcam.py

Generates Grad-CAM overlay PNGs for convnext_tiny on 10 hand-picked test-set
images — 2 per class for: Fetal_femur, Maternal_cervix, Brain_Trans_cerebellum,
Brain_Trans_thalamic, Brain_Trans_ventricular.

Femur and cervix are the well-separated "easy" classes — activations should
land clearly on the relevant anatomy. Brain sub-planes are the hard cluster;
the goal is to confirm the heatmap at least localises to the fetal skull/brain
region even when sub-plane prediction is wrong.

Outputs:
    checkpoints/convnext_tiny/gradcam_spotcheck/<class>_<filename>_overlay.png

Usage:
    conda run -n fetalplane python scripts/spot_check_gradcam.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dataset import CANONICAL_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
from src.data.transforms import get_eval_transform, load_and_prep_grayscale_to_rgb
from src.models.backbone import build_model
from src.models.gradcam import run_gradcam

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

CKPT_PATH = Path("checkpoints/convnext_tiny/best.pt")
TEST_CSV = "data/splits/test.csv"
OUT_DIR = Path("checkpoints/convnext_tiny/gradcam_spotcheck")

# 2 samples per class for these 5 classes.  Seed controls which rows are picked.
TARGET_CLASSES = [
    "Fetal_femur",
    "Maternal_cervix",
    "Brain_Trans_cerebellum",
    "Brain_Trans_thalamic",
    "Brain_Trans_ventricular",
]
SAMPLES_PER_CLASS = 2
SEED = 0


def pick_samples(df: pd.DataFrame, target_classes: list[str], n: int, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for cls in target_classes:
        class_df = df[df["plane_label"] == cls]
        chosen = class_df.sample(n=min(n, len(class_df)), random_state=int(rng.integers(0, 9999)))
        rows.append(chosen)
    return pd.concat(rows).reset_index(drop=True)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Using device: %s", device)

    # --- Load model ---
    log.info("Loading checkpoint: %s", CKPT_PATH)
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    cfg = ckpt["config"]
    backbone_name: str = cfg["backbone"]
    img_size: int = cfg.get("image_size", 224)
    mean = tuple(cfg.get("normalize_mean", [0.485, 0.456, 0.406]))
    std = tuple(cfg.get("normalize_std", [0.229, 0.224, 0.225]))

    model = build_model(backbone_name, num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()
    log.info("Model loaded: %s", backbone_name)

    transform = get_eval_transform(img_size=img_size, mean=mean, std=std)

    # --- Pick test images ---
    df = pd.read_csv(TEST_CSV)
    rng = np.random.default_rng(SEED)
    sample_df = pick_samples(df, TARGET_CLASSES, SAMPLES_PER_CLASS, rng)
    log.info("Selected %d images for spot-check", len(sample_df))

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in sample_df.iterrows():
        image_path = str(row["image_path"])
        true_label = row["plane_label"]
        class_idx = CLASS_TO_IDX[true_label]

        # Load image for model input
        img_rgb = load_and_prep_grayscale_to_rgb(image_path)
        augmented = transform(image=img_rgb)
        tensor = augmented["image"].unsqueeze(0).to(device)

        # Prepare float [0,1] image for overlay (resize to match model input)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        img_float = img_resized.astype(np.float32) / 255.0

        # Get prediction
        with torch.no_grad():
            logits = model(tensor)
        pred_idx = int(logits.argmax(dim=1).item())
        pred_label = IDX_TO_CLASS[pred_idx]

        # Run Grad-CAM targeting the TRUE class
        overlay = run_gradcam(
            model=model,
            input_tensor=tensor,
            class_idx=class_idx,
            backbone_name=backbone_name,
            original_rgb_float=img_float,
        )

        # Save: class__filename__pred_<predicted>.png
        stem = Path(image_path).stem
        correct_str = "CORRECT" if pred_idx == class_idx else f"WRONG_pred_{pred_label}"
        out_name = f"{true_label}__{stem}__{correct_str}.png"
        out_path = OUT_DIR / out_name

        # overlay is RGB uint8 — save as BGR for cv2
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        log.info("  Saved → %s  [true=%s, pred=%s]", out_name, true_label, pred_label)

    log.info("Done. Spot-check images → %s", OUT_DIR)
    log.info("[MANUAL] Open those PNGs and verify activation lands on plausible anatomy.")
    log.info("  Femur/cervix: should clearly highlight bone shaft / lower uterine segment.")
    log.info("  Brain sub-planes: check heatmap localises to fetal skull region even on errors.")


if __name__ == "__main__":
    main()
