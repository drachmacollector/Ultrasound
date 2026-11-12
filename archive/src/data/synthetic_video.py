"""
src/data/synthetic_video.py

Generates an N-frame synthetic clip from a single static frame using a
DEPTH-LAYERED PARALLAX approach — not a single rigid affine warp.

WHY THE SINGLE-WARP ("PHOTOCOPIER") APPROACH FAILS
---------------------------------------------------
The naive approach (one cv2.warpAffine call per frame, same M for the whole
image) produces what the user correctly called a "photocopier effect": every
pixel in the image translates/rotates by exactly the same amount, like sliding
a piece of paper on a scanner bed. This looks wrong immediately because:

  1. Real ultrasound contains multiple tissue layers at different depths.
     Near-field structures (e.g. maternal skin/fat) move MORE relative to the
     probe than far-field structures (deeper anatomy) for the same probe
     displacement. This depth-dependent parallax is what makes video "feel 3D."

  2. Specular highlights and acoustic shadows are fixed relative to the anatomy
     being insonated — they do not rigidly follow the transducer frame.

  3. A rigid affine warp preserves ALL spatial relationships: background and
     foreground move as one piece. This has the visual signature of a flat
     photograph being slid under a camera, not a probe scanning tissue.

DEPTH-LAYERED PARALLAX SOLUTION
--------------------------------
We decompose the frame into two virtual "depth planes":

  Layer A — "Far field" (deep tissue / background):
    Extracted via a strong Gaussian blur (sigma ~ 0.03 * image width).
    Warped with a REDUCED motion vector (parallax_far = 0.35 × full motion).
    Represents slowly-shifting deep anatomy.

  Layer B — "Near field" (local texture / specular highlights):
    Extracted as the difference between the original and the blurred version
    (i.e. the high-spatial-frequency residual).
    Warped with a LARGER motion vector (parallax_near = 1.0 × full motion).
    Represents fast-moving near-field tissue.

  Composite = far_warped + near_warped (clamped to [0, 255]).

This is a well-known trick in 2.5D parallax photography (also called
"Ken Burns with depth separation") and produces a convincing tissue-like
layered motion from a single static frame. It is NOT physically accurate
ultrasound simulation — but it is sufficient to validate temporal smoothing
behaviour (the goal of these clips) and looks far more convincing than the
rigid warp on a real ultrasound image.

MOTION MODEL
------------
All motion parameters use a momentum-based bounded random walk (_SmoothRandomWalk)
so consecutive frames are continuous. Independent per-frame randomization would
produce jittery noise indistinguishable from inference noise, which would make
Tier-1 smoothing validation meaningless.

Parameters:
  - Pan X/Y:  ≤ ~3% of frame dimension per axis
  - Zoom:     ≤ ~2% scale change
  - Rotation: ≤ ~2 degrees
  - Speckle:  multiplicative noise sigma=0.04 (ultrasound-appropriate)
  - Blur:     occasional mild Gaussian blur (ksize 3 or 5), p=0.2
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
        new_value = self.value + self.velocity

        # Zero out velocity if we hit a boundary to prevent "sticking"
        if new_value <= self.low or new_value >= self.high:
            self.velocity = 0.0

        self.value = float(np.clip(new_value, self.low, self.high))
        return self.value


def _warp(img: np.ndarray, M: np.ndarray, w: int, h: int) -> np.ndarray:
    """Apply affine warp M to img, replicating borders."""
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)


def generate_ego_motion_clip(
    image: np.ndarray,
    n_frames: int = 120,
    seed: int | None = None,
    wobble_scale: float = 1.0,
) -> list:
    """
    Generate a synthetic clip mimicking probe hand tremor/wobble.

    Uses depth-layered parallax compositing (see module docstring) to avoid
    the flat "photocopier effect" of a single rigid affine warp.

    Args:
        image:    HxWx3 uint8 RGB frame.
        n_frames: Number of frames to generate.
        seed:     Optional random seed for reproducibility.
        wobble_scale: Multiplier on pan/zoom/rotation walk bounds and speckle/blur
            intensity. 1.0 = original calibrated defaults. Added for
            scripts/generate_multiplane_clips.py, which varies this per clip to
            produce a mix of calm and handheld-shaky footage. Backward compatible:
            every existing call site omits this arg and gets identical behavior
            to before this change.

    Returns:
        List of n_frames HxWx3 uint8 RGB frames.
    """
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    # ------------------------------------------------------------------
    # Decompose into two depth layers (done once, outside the frame loop)
    # ------------------------------------------------------------------
    # sigma ~ 3% of image width gives a good "deep tissue blur radius"
    blur_sigma = max(3.0, w * 0.03)
    ksize = int(blur_sigma * 6) | 1   # always odd

    img_f32 = image.astype(np.float32)

    # Far-field: heavy blur — low-frequency, slowly-varying background anatomy
    far_layer = cv2.GaussianBlur(img_f32, (ksize, ksize), blur_sigma)

    # Near-field: high-frequency residual — specular highlights, fine texture
    near_layer = img_f32 - far_layer      # can be negative: range roughly [-127, 127]

    # ------------------------------------------------------------------
    # Motion walks (bounded, momentum-based — see module docstring)
    # ------------------------------------------------------------------
    pan_x_walk  = _SmoothRandomWalk(-0.03 * w * wobble_scale, 0.03 * w * wobble_scale, start=0.0, rng=rng)
    pan_y_walk  = _SmoothRandomWalk(-0.03 * h * wobble_scale, 0.03 * h * wobble_scale, start=0.0, rng=rng)
    zoom_walk   = _SmoothRandomWalk(-0.02 * wobble_scale, 0.02 * wobble_scale, start=0.0, rng=rng)
    rot_walk    = _SmoothRandomWalk(-2.0 * wobble_scale, 2.0 * wobble_scale, start=0.0, rng=rng)

    # ------------------------------------------------------------------
    # Parallax scaling factors
    #   far:  0.35 × full motion  → deep tissue drifts slowly
    #   near: 1.00 × full motion  → surface texture moves at full probe speed
    # These values are perceptually tuned; adjust if clips look wrong.
    # ------------------------------------------------------------------
    PARALLAX_FAR  = 0.35
    PARALLAX_NEAR = 1.00

    frames = []
    for _ in range(n_frames):
        dx, dy = pan_x_walk.step(), pan_y_walk.step()
        zoom, rot = zoom_walk.step(), rot_walk.step()

        # Base affine matrix (full motion)
        M_full = cv2.getRotationMatrix2D((cx, cy), angle=rot, scale=1.0 + zoom)
        M_full[0, 2] += dx
        M_full[1, 2] += dy

        # Far-field matrix: scale down translation AND rotation
        M_far = M_full.copy()
        # Scale the rotation/zoom part toward identity and translation toward 0
        M_far[:, :2] = np.eye(2) + (M_full[:, :2] - np.eye(2)) * PARALLAX_FAR
        M_far[:, 2]  = M_full[:, 2] * PARALLAX_FAR

        # Near-field matrix: full motion (or slightly beyond, but 1.0 is clean)
        M_near = M_full.copy()

        # Warp each layer separately
        far_warped  = _warp(far_layer,  M_far,  w, h)
        near_warped = _warp(near_layer, M_near, w, h)

        # Composite: far background + near residual, clamped to [0, 255]
        composite_f32 = np.clip(far_warped + near_warped, 0.0, 255.0)

        # ------------------------------------------------------------------
        # Ultrasound-appropriate speckle (multiplicative noise)
        # Applied to the composite so noise rides on the already-warped image
        # ------------------------------------------------------------------
        speckle_sigma = 0.04 * (0.5 + 0.5 * wobble_scale)
        noise = rng.normal(1.0, speckle_sigma, composite_f32.shape).astype(np.float32)
        speckled = np.clip(composite_f32 * noise, 0.0, 255.0).astype(np.uint8)

        # Occasional mild blur (simulates slight focus jitter) — more frequent on shakier clips
        if rng.random() < min(0.6, 0.2 * wobble_scale):
            k = int(rng.choice([3, 5]))
            speckled = cv2.GaussianBlur(speckled, (k, k), 0)

        frames.append(speckled)

    return frames


def save_clip_as_mp4(frames: list, out_path: str, fps: int = 24):
    """Write a list of HxWx3 RGB frames as an MP4 file using imageio-ffmpeg
    for browser-compatible H.264 output.

    Falls back to OpenCV mp4v if imageio is unavailable (not recommended for
    browser playback — install imageio[ffmpeg] for full compatibility).

    Note: libx264 requires even width and height. If the source frames have odd
    dimensions (common with some ultrasound image crops), we crop 1 pixel from
    the right/bottom edge. This is imperceptible and far cheaper than padding.

    Args:
        frames:   List of HxWx3 uint8 RGB numpy arrays.
        out_path: Destination .mp4 path.
        fps:      Frames per second for the output video.
    """
    # Enforce even dimensions — libx264 hard requirement
    h, w = frames[0].shape[:2]
    even_w = w - (w % 2)
    even_h = h - (h % 2)
    if even_w != w or even_h != h:
        frames = [f[:even_h, :even_w] for f in frames]

    try:
        import imageio
        writer = imageio.get_writer(
            out_path,
            fps=fps,
            codec="libx264",
            quality=None,
            pixelformat="yuv420p",
            ffmpeg_params=["-crf", "23"],
            macro_block_size=1,
        )
        for frame in frames:
            writer.append_data(frame)   # imageio expects RGB
        writer.close()
    except ImportError:
        # Fallback: OpenCV mp4v (may not play inline in browsers)
        h2, w2 = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
        writer_cv = cv2.VideoWriter(out_path, fourcc, fps, (w2, h2))
        for frame in frames:
            writer_cv.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer_cv.release()
