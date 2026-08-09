"""
src/data/transforms.py

Defines train and eval preprocessing pipelines using albumentations.
Uses A.MultiplicativeNoise for speckle augmentation (ultrasound-appropriate
multiplicative noise model, not additive Gaussian).

Grayscale→RGB replication is done in load_and_prep_grayscale_to_rgb()
BEFORE any transform is applied, so the albumentations pipeline always
receives a 3-channel uint8 image and ToTensorV2 produces a [3, H, W] tensor.

Both get_train_transform() and get_eval_transform() accept optional mean/std
parameters so callers (train.py, lr_finder.py) can supply per-backbone
normalization values without duplicating the Compose pipeline definition.
Default values are IMAGENET_MEAN / IMAGENET_STD, which are correct for all
candidates except tf_efficientnetv2_s (whose config supplies 0.5/0.5).
"""
import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform(
    img_size: int = IMG_SIZE,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> A.Compose:
    """Full augmentation pipeline for training.

    Args:
        img_size: Square spatial dimension for resize.  Per-backbone native
            resolution should be passed here (e.g. 300 for tf_efficientnetv2_s).
        mean: Normalisation mean per channel.  Defaults to ImageNet values;
            override to (0.5, 0.5, 0.5) for tf_efficientnetv2_s.in21k_ft_in1k.
        std:  Normalisation std per channel.  Same override rules as mean.

    Augmentation choices:
    - HorizontalFlip: safe for all 8 classes (none are laterality-defined)
    - Affine: mild rotation/scale/translate to simulate probe placement variability
    - RandomBrightnessContrast: accounts for gain/TGC variability across machines
    - MultiplicativeNoise: ultrasound speckle is multiplicative (Rayleigh distributed),
      not additive -- using A.MultiplicativeNoise here is physically correct
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),  # verified safe: none of our 8 classes are laterality-defined
        A.Affine(rotate=(-10, 10), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=True, p=0.3),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def get_eval_transform(
    img_size: int = IMG_SIZE,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> A.Compose:
    """Minimal pipeline for val/test/inference: resize + normalize only.

    Args:
        img_size: Square spatial dimension for resize.
        mean: Normalisation mean per channel.  Defaults to ImageNet values.
        std:  Normalisation std per channel.
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])


def load_and_prep_grayscale_to_rgb(image_path: str) -> np.ndarray:
    """Load an ultrasound image and replicate the single channel to 3 channels.

    Ultrasound images are effectively single-channel (greyscale); replicating to
    3ch is the simplest and most compatible approach for ImageNet-pretrained
    backbones. Returns HxWx3 uint8 numpy array.

    Raises FileNotFoundError if the file does not exist.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img_rgb


def prep_frame_grayscale_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    """In-memory equivalent of load_and_prep_grayscale_to_rgb() for live video/webcam
    frames (cv2.VideoCapture.read() output), which arrive as BGR arrays, not file paths.
    Must produce bit-identical output to the path-based function for the same underlying
    image, to guarantee training/serving preprocessing parity.

    Implementation note: uses COLOR_BGR2GRAY → COLOR_GRAY2RGB, which is the direct
    in-memory equivalent of imread(IMREAD_GRAYSCALE) → cvtColor(GRAY2RGB).  Both
    apply OpenCV's standard luminance formula (0.114·B + 0.587·G + 0.299·R), so
    the outputs are bit-identical for any input image.  A previous implementation
    round-tripped through imencode/imdecode, which was functionally equivalent but
    unnecessarily slow for a per-frame hot path.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    return img_rgb
