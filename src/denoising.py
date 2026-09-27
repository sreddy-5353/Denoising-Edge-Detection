"""
denoising.py
============
Denoising methods compared in this project:

- "none"              : baseline, no denoising applied
- "gaussian_blur"      : cv2.GaussianBlur - fast linear smoothing, blurs edges too
- "median_filter"      : cv2.medianBlur - excellent for salt & pepper noise
- "bilateral_filter"   : cv2.bilateralFilter - smooths while preserving edges
- "nlm" (Non-Local Means) : cv2.fastNlMeansDenoising - strong denoising that
                             exploits repeating patterns in the image

All functions take and return 8-bit single-channel (grayscale) images.
"""

import cv2
import numpy as np

import config


def estimate_noise_sigma(image):
    """
    Blind (reference-free) estimation of the noise standard deviation in an
    image, using the fast Laplacian-based estimator of Immerkaer (1996).

    This lets Non-Local Means automatically scale its filtering strength
    (`h`) to match how noisy the input actually is, instead of using one
    fixed `h` that would only work well for a single noise level. A larger
    estimated sigma -> more aggressive smoothing; a smaller sigma -> gentler
    smoothing that preserves detail.
    """
    laplacian_kernel = np.array([[1, -2, 1],
                                  [-2, 4, -2],
                                  [1, -2, 1]], dtype=np.float64)
    h, w = image.shape
    conv = cv2.filter2D(image.astype(np.float64), -1, laplacian_kernel)
    sigma = np.sum(np.abs(conv)) * np.sqrt(0.5 * np.pi) / (6 * (w - 2) * (h - 2))
    return sigma


def apply_gaussian_blur(image, ksize=None, sigmaX=None):
    params = config.DENOISING_PARAMS["gaussian_blur"]
    ksize = ksize or params["ksize"]
    sigmaX = params["sigmaX"] if sigmaX is None else sigmaX
    if ksize % 2 == 0:
        ksize += 1  # kernel size must be odd
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX)


def apply_median_filter(image, ksize=None):
    params = config.DENOISING_PARAMS["median_filter"]
    ksize = ksize or params["ksize"]
    if ksize % 2 == 0:
        ksize += 1
    return cv2.medianBlur(image, ksize)


def apply_bilateral_filter(image, d=None, sigmaColor=None, sigmaSpace=None):
    params = config.DENOISING_PARAMS["bilateral_filter"]
    d = d or params["d"]
    sigmaColor = sigmaColor or params["sigmaColor"]
    sigmaSpace = sigmaSpace or params["sigmaSpace"]
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)


def apply_nlm(image, h=None, templateWindowSize=None, searchWindowSize=None, adaptive=True):
    """
    Non-Local Means denoising. If `h` is not explicitly provided and
    `adaptive` is True (default), `h` is automatically derived from the
    image's own estimated noise level (see estimate_noise_sigma), so NLM
    scales its strength to low/medium/high noise instead of using one
    fixed setting for all of them.
    """
    params = config.DENOISING_PARAMS["nlm"]
    templateWindowSize = templateWindowSize or params["templateWindowSize"]
    searchWindowSize = searchWindowSize or params["searchWindowSize"]

    if h is None:
        if adaptive:
            sigma_estimate = estimate_noise_sigma(image)
            # Scale factor found empirically to work well across the
            # low/medium/high noise levels used in this project.
            h = float(np.clip(sigma_estimate * 1.4, params["h"], 45))
        else:
            h = params["h"]

    return cv2.fastNlMeansDenoising(
        image, None, h=h,
        templateWindowSize=templateWindowSize,
        searchWindowSize=searchWindowSize,
    )


def apply_denoising(image, method):
    """
    Dispatcher: apply the named denoising method to an image.

    Parameters
    ----------
    method : str
        One of config.DENOISING_METHODS: "none", "gaussian_blur",
        "median_filter", "bilateral_filter", "nlm".
    """
    if method == "none":
        return image.copy()
    elif method == "gaussian_blur":
        return apply_gaussian_blur(image)
    elif method == "median_filter":
        return apply_median_filter(image)
    elif method == "bilateral_filter":
        return apply_bilateral_filter(image)
    elif method == "nlm":
        return apply_nlm(image)
    else:
        raise ValueError(f"Unknown denoising method: {method}")
