"""
noise.py
========
Synthetic noise generators used to simulate degraded images.

Two noise models are implemented:

1. Gaussian noise   - continuous, sensor-like noise. Controlled by `sigma`
                       (standard deviation of the noise added to each pixel).
2. Salt & pepper     - impulse noise where a fraction of pixels are randomly
                       set to pure black (0) or pure white (255). Controlled
                       by `amount` (fraction of pixels affected).

All functions operate on 8-bit single-channel (grayscale) images and
return 8-bit images clipped to the valid [0, 255] range.
"""

import numpy as np

import config


def add_gaussian_noise(image, sigma, seed=None):
    """
    Add zero-mean Gaussian noise with standard deviation `sigma` to an image.

    Parameters
    ----------
    image : np.ndarray (uint8, grayscale)
    sigma : float
        Standard deviation of the noise (on the 0-255 pixel scale).
    seed : int or None
        Optional RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=image.shape)
    noisy = image.astype(np.float64) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(image, amount, salt_vs_pepper=0.5, seed=None):
    """
    Add salt-and-pepper (impulse) noise to an image.

    Parameters
    ----------
    image : np.ndarray (uint8, grayscale)
    amount : float
        Fraction of total pixels to corrupt (0.0 - 1.0).
    salt_vs_pepper : float
        Fraction of the corrupted pixels that become "salt" (white, 255)
        as opposed to "pepper" (black, 0). Default 0.5 = balanced.
    seed : int or None
        Optional RNG seed for reproducibility.
    """
    rng = np.random.default_rng(seed)
    noisy = image.copy()
    total_pixels = image.size
    num_corrupt = int(np.ceil(amount * total_pixels))

    num_salt = int(np.ceil(num_corrupt * salt_vs_pepper))
    num_pepper = num_corrupt - num_salt

    flat_indices = rng.choice(total_pixels, size=num_salt + num_pepper, replace=False)
    salt_indices = flat_indices[:num_salt]
    pepper_indices = flat_indices[num_salt:]

    noisy_flat = noisy.reshape(-1)
    noisy_flat[salt_indices] = 255
    noisy_flat[pepper_indices] = 0
    return noisy_flat.reshape(image.shape)


def apply_noise(image, noise_type, level, seed=None):
    """
    Dispatcher: apply the configured noise type/level to an image.

    Parameters
    ----------
    noise_type : str  ("gaussian" or "salt_pepper")
    level : str        ("low", "medium", "high")
    """
    if noise_type == "gaussian":
        sigma = config.GAUSSIAN_NOISE_LEVELS[level]
        return add_gaussian_noise(image, sigma, seed=seed)
    elif noise_type == "salt_pepper":
        amount = config.SALT_PEPPER_NOISE_LEVELS[level]
        return add_salt_pepper_noise(image, amount, seed=seed)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")
