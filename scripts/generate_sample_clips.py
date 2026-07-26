"""
scripts/generate_sample_clips.py

Driver script: picks ~10 random images (a few per class) from train.csv,
generates a 16-frame synthetic ego-motion clip for each via
src.data.synthetic_video.generate_ego_motion_clip, and saves them as MP4s
to data/processed/synthetic_clips/.

These clips should be manually reviewed (Phase 3 checkpoint #2) before Phase 5
smoothing development. The motion should look like plausible small probe wobble,
NOT jittery per-frame noise.
"""
import random
import sys
from pathlib import Path

import pandas as pd

# Allow imports from project root when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic_video import generate_ego_motion_clip, save_clip_as_mp4
from src.data.transforms import load_and_prep_grayscale_to_rgb

TRAIN_CSV = Path("data/splits/train.csv")
OUT_DIR = Path("data/processed/synthetic_clips")
CLIPS_PER_CLASS = 2          # 2 × 8 classes = up to 16 clips (a few may be missing for rare classes)
N_FRAMES = 16
FPS = 24
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)
    df = pd.read_csv(TRAIN_CSV)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    classes = df["plane_label"].unique().tolist()
    clips_written = 0

    for cls in sorted(classes):
        cls_df = df[df["plane_label"] == cls]
        n_pick = min(CLIPS_PER_CLASS, len(cls_df))
        sampled = cls_df.sample(n=n_pick, random_state=RANDOM_SEED)

        for i, (_, row) in enumerate(sampled.iterrows()):
            img_path = row["image_path"]
            try:
                img = load_and_prep_grayscale_to_rgb(img_path)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            frames = generate_ego_motion_clip(img, n_frames=N_FRAMES, seed=RANDOM_SEED + i)

            # Safe filename: replace slashes/spaces
            safe_cls = cls.replace(" ", "_")
            out_name = f"{safe_cls}_clip{i+1:02d}.mp4"
            out_path = str(OUT_DIR / out_name)
            save_clip_as_mp4(frames, out_path, fps=FPS)
            print(f"  Saved {out_path}  ({N_FRAMES} frames @ {FPS}fps)")
            clips_written += 1

    print(f"\nDone. {clips_written} clips written to {OUT_DIR}/")
    print("ACTION REQUIRED: Watch a handful of these clips and confirm the motion looks like")
    print("plausible small probe wobble (not jittery per-frame noise) before Phase 5 starts.")


if __name__ == "__main__":
    main()
