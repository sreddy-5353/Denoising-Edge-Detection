"""
edge_detection.py
==================
Canny edge detection wrapper with configurable thresholds, plus a helper
to build the "ground truth" edge map from the original, clean (noise-free)
image -- this is the reference every noisy/denoised pipeline is compared
against.
"""

import cv2

import config


def canny_edge_detection(image, low_threshold=None, high_threshold=None):
    """
    Run Canny edge detection on a grayscale image.

    Returns a binary image (0 = not an edge, 255 = edge), matching the
    native output of cv2.Canny.
    """
    low_threshold = config.CANNY_LOW_THRESHOLD if low_threshold is None else low_threshold
    high_threshold = config.CANNY_HIGH_THRESHOLD if high_threshold is None else high_threshold
    edges = cv2.Canny(image, low_threshold, high_threshold)
    return edges


def generate_ground_truth_edges(clean_image, low_threshold=None, high_threshold=None):
    """
    Generate the reference/ground-truth edge map from the ORIGINAL clean
    (noise-free) image. Every pipeline (noisy-only, noisy+denoised) is
    evaluated against this reference.
    """
    return canny_edge_detection(clean_image, low_threshold, high_threshold)
