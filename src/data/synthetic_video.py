"""
src/data/synthetic_video.py

Generates an N-frame synthetic clip from a single static frame by applying a
SMOOTHLY DRIFTING sequence of small affine transforms (never independently
randomized per frame) plus mild speckle noise and blur jitter. Mimics probe
wobble/hand tremor for temporal-smoothing validation (05_TEMPORAL_SMOOTHING...).

Does NOT synthesize class transitions -- only within-class motion.

Key design principle: _SmoothRandomWalk uses a momentum-based 1D random walk
so consecutive frames are continuous. Independent per-frame randomization
would produce jittery noise that looks nothing like real hand tremor and would
make Tier-1 smoothing validation in Phase 5 meaningless.
"""
import cv2
import numpy as np


class _SmoothRandomWalk:
    """1D bounded random walk with momentum, for continuous parameter drift."""
    def __init__(self, low: float, high: float, start=None, momentum=0.85, step_std=None, rng=None):
        self.low, self.high = low, high
        self.momentum = momentum
        self.step_std = step_std if step_std is not None else (high - low) * 0.08
        self.rng = rng or np.random.default_rng()
        self.value = start if start is not None else (low + high) / 2
        self.velocity = 0.0

    def step(self) -> float:
        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * self.rng.normal(0, self.step_std)
        self.value = float(np.clip(self.value + self.velocity, self.low, self.high))
        return self.value


def generate_ego_motion_clip(image: np.ndarray, n_frames: int = 16, seed: int | None = None) -> list:
    """
    Generate a synthetic clip mimicking probe hand tremor/wobble.

    Args:
        image: HxWx3 uint8 RGB frame.
        n_frames: Number of frames to generate.
        seed: Optional random seed for reproducibility.

    Returns:
        List of n_frames HxWx3 uint8 RGB frames.

    Motion parameters (all bounded and smoothly drifting):
    - Pan X/Y: <= ~3% of frame dimension per axis
    - Zoom: <= ~2% scale change
    - Rotation: <= ~2 degrees
    - Speckle: multiplicative noise sigma=0.04 (ultrasound-appropriate)
    - Blur: occasional mild Gaussian blur (ksize 3 or 5), p=0.3
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    pan_x_walk = _SmoothRandomWalk(-0.03 * w, 0.03 * w, start=0.0, rng=rng)   # <= ~3% frame width
    pan_y_walk = _SmoothRandomWalk(-0.03 * h, 0.03 * h, start=0.0, rng=rng)
    zoom_walk = _SmoothRandomWalk(-0.02, 0.02, start=0.0, rng=rng)             # <= ~2% zoom
    rot_walk = _SmoothRandomWalk(-2.0, 2.0, start=0.0, rng=rng)                # <= ~2 degrees

    frames = []
    for _ in range(n_frames):
        dx, dy = pan_x_walk.step(), pan_y_walk.step()
        zoom, rot = zoom_walk.step(), rot_walk.step()

        M = cv2.getRotationMatrix2D((cx, cy), angle=rot, scale=1.0 + zoom)
        M[0, 2] += dx
        M[1, 2] += dy
        warped = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

        # mild speckle (multiplicative) noise, ultrasound-appropriate
        speckle_sigma = 0.04
        noise_raw = rng.normal(1.0, speckle_sigma, warped.shape)
        noise = np.asarray(noise_raw, dtype=np.float32)
        speckled = np.clip(warped.astype(np.float32) * noise, 0, 255).astype(np.uint8)

        # slow blur jitter -- occasionally soften slightly, never sharpen
        if rng.random() < 0.3:
            k = rng.choice([3, 5])
            speckled = cv2.GaussianBlur(speckled, (k, k), 0)

        frames.append(speckled)

    return frames


def save_clip_as_mp4(frames: list, out_path: str, fps: int = 24):
    """Write a list of HxWx3 RGB frames as an MP4 file.

    Args:
        frames: List of HxWx3 uint8 RGB numpy arrays.
        out_path: Destination .mp4 path.
        fps: Frames per second for the output video.
    """
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
