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
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

from src.preprocess.pipeline import geometric

IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def style_augmentations() -> list[A.BasicTransform]:
    """R5.4 device-style augmentation (arm A4), albumentations 2.0.8 argument names."""
    return [
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.MultiplicativeNoise(multiplier=(0.8, 1.2), per_channel=False, elementwise=True, p=0.3),
        A.Downscale(scale_range=(0.5, 0.9), p=0.2),
        A.CLAHE(clip_limit=2.0, p=0.2),
    ]


def get_train_transform(
    img_size: int = IMG_SIZE,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    mode: str = "resize",
    style_aug: bool = False,
    augment: bool = True,
) -> A.Compose:
    """Training pipeline = the mode's geometric transform + augmentation + normalisation.

    B0 (``mode="resize"``, ``style_aug=False``) is the legacy recipe. ``style_aug`` replaces the
    0.2/0.2 brightness-contrast and the 0.9-1.1 speckle entries by the R5.4 list; flip and affine
    are unchanged. ``augment=False`` drops every random op (parity check against eval).
    """
    aug: list[A.BasicTransform] = []
    if augment:
        aug = [A.HorizontalFlip(p=0.5),
               A.Affine(rotate=(-10, 10), scale=(0.9, 1.1), translate_percent=(-0.1, 0.1), p=0.7)]
        if style_aug:
            aug += style_augmentations()
        else:
            aug += [A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), per_channel=False, elementwise=True, p=0.3)]
    return A.Compose([*geometric(img_size, mode), *aug, A.Normalize(mean=mean, std=std), ToTensorV2()])


def get_eval_transform(
    img_size: int = IMG_SIZE,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
    mode: str = "resize",
) -> A.Compose:
    """Deterministic pipeline for val/test/inference: the mode's geometry + normalisation."""
    return A.Compose([*geometric(img_size, mode), A.Normalize(mean=mean, std=std), ToTensorV2()])


def load_grayscale(image_path: str | Path) -> np.ndarray:
    """Load an image as HxW uint8 greyscale.

    IMREAD_GRAYSCALE also converts RGB sources (e.g. African/Egypt PNGs) with the
    standard luminance formula. Raises FileNotFoundError if the file cannot be read.
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    return img


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Replicate a HxW uint8 image to HxWx3 (the input format of every backbone)."""
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def load_and_prep_grayscale_to_rgb(image_path: str | Path) -> np.ndarray:
    """Load an ultrasound image and replicate the single channel to 3 channels.

    Ultrasound images are effectively single-channel (greyscale); replicating to
    3ch is the simplest and most compatible approach for ImageNet-pretrained
    backbones. Returns HxWx3 uint8 numpy array.

    Raises FileNotFoundError if the file does not exist.
    """
    return gray_to_rgb(load_grayscale(image_path))
