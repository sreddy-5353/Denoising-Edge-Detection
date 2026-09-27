"""
config.py
=========
Central configuration for the "Denoising an Image Prior to Edge Detection:
Impact on Accuracy" project.

Change values here (dataset path, noise levels, Canny thresholds, etc.)
instead of editing code in src/. All paths are relative to the project
root, so the project works after being extracted / moved to any computer.
"""

import os

# ---------------------------------------------------------------------------
# 1. PATHS  (all relative to this file's location -> project is portable)
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Put your dataset anywhere inside this folder (subfolders are supported,
# e.g. dataset/BSDS500/images/train/, dataset/BSDS500/images/val/, ...).
# The loader searches this folder RECURSIVELY, so you do not need a fixed
# folder layout -- just change DATASET_DIR if you keep images elsewhere.
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
RESULTS_IMAGES_DIR = os.path.join(RESULTS_DIR, "images")
RESULTS_GRAPHS_DIR = os.path.join(RESULTS_DIR, "graphs")
RESULTS_CSV_DIR = os.path.join(RESULTS_DIR, "csv")

for _d in (RESULTS_DIR, RESULTS_IMAGES_DIR, RESULTS_GRAPHS_DIR, RESULTS_CSV_DIR, DATASET_DIR):
    os.makedirs(_d, exist_ok=True)

RESULTS_CSV_PATH = os.path.join(RESULTS_CSV_DIR, "experiment_results.csv")
BEST_METHODS_CSV_PATH = os.path.join(RESULTS_CSV_DIR, "best_methods.csv")

# ---------------------------------------------------------------------------
# 2. SUPPORTED IMAGE FILE TYPES
# ---------------------------------------------------------------------------
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------
# Images are only resized DOWN if they exceed this size on the longer side,
# so quality is preserved for smaller images. Set to None to disable resizing.
MAX_IMAGE_DIMENSION = 321  # BSDS500 images are 481x321 / 321x481 -> fits as-is

# ---------------------------------------------------------------------------
# 4. NOISE CONFIGURATION
# ---------------------------------------------------------------------------
# Gaussian noise: sigma = standard deviation of noise added to pixel values (0-255 scale)
GAUSSIAN_NOISE_LEVELS = {
    "low": 10,
    "medium": 25,
    "high": 50,
}

# Salt & pepper noise: amount = fraction of pixels replaced with salt/pepper
SALT_PEPPER_NOISE_LEVELS = {
    "low": 0.02,
    "medium": 0.08,
    "high": 0.18,
}

NOISE_TYPES = ["gaussian", "salt_pepper"]
NOISE_LEVELS = ["low", "medium", "high"]

# ---------------------------------------------------------------------------
# 5. DENOISING METHODS
# ---------------------------------------------------------------------------
DENOISING_METHODS = [
    "none",             # baseline: no denoising
    "gaussian_blur",
    "median_filter",
    "bilateral_filter",
    "nlm",              # Non-Local Means
]

DENOISING_PARAMS = {
    "gaussian_blur": {"ksize": 5, "sigmaX": 0},
    "median_filter": {"ksize": 5},
    "bilateral_filter": {"d": 9, "sigmaColor": 75, "sigmaSpace": 75},
    "nlm": {"h": 10, "templateWindowSize": 7, "searchWindowSize": 21},
}

# ---------------------------------------------------------------------------
# 6. EDGE DETECTION (Canny)
# ---------------------------------------------------------------------------
CANNY_LOW_THRESHOLD = 100
CANNY_HIGH_THRESHOLD = 200

# ---------------------------------------------------------------------------
# 7. EXPERIMENT SETTINGS
# ---------------------------------------------------------------------------
# Number of images to use for the batch experiment. Set to None to use the
# entire dataset (slower). Keep this modest for quick iteration; increase
# for a more statistically robust study.
SAMPLE_SIZE = 40

# Reproducibility
RANDOM_SEED = 42

# How many (image, noise, level) combinations get their visual comparison
# panel saved to results/images (saving all of them would create thousands
# of files). Panels are chosen evenly across the sampled images.
NUM_VISUAL_SAMPLES = 4
