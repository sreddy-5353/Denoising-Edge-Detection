"""
preprocessing.py
================
Basic preprocessing steps applied to every image before the noise /
denoising / edge-detection pipeline: grayscale conversion and
(conditional) resizing.
"""

import cv2
import numpy as np

import config


def to_grayscale(image):
    """
    Ensure an image is single-channel grayscale (8-bit).
    If it is already grayscale, it is returned unchanged.
    """
    if image is None:
        return None
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def resize_if_needed(image, max_dimension=None):
    """
    Resize the image ONLY if its longer side exceeds `max_dimension`.
    Aspect ratio is preserved. If max_dimension is None, the image is
    returned unchanged (quality is preserved as much as possible).
    """
    if image is None:
        return None
    max_dimension = config.MAX_IMAGE_DIMENSION if max_dimension is None else max_dimension
    if max_dimension is None:
        return image

    h, w = image.shape[:2]
    longer_side = max(h, w)
    if longer_side <= max_dimension:
        return image

    scale = max_dimension / float(longer_side)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    interpolation = cv2.INTER_AREA  # best for shrinking
    return cv2.resize(image, (new_w, new_h), interpolation=interpolation)


def preprocess_image(image):
    """Full preprocessing pipeline: grayscale -> conditional resize."""
    gray = to_grayscale(image)
    resized = resize_if_needed(gray)
    return resized.astype(np.uint8)
