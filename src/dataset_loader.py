"""
dataset_loader.py
==================
Locates and loads images from the user-provided dataset directory.

The dataset directory structure is NOT assumed to be fixed. This module
walks the directory tree recursively and collects every file with a
supported extension (.jpg, .jpeg, .png), regardless of how deeply nested
it is in subfolders. This means it automatically adapts to datasets such
as:

    dataset/BSDS500/images/train/*.jpg
    dataset/BSDS500/images/val/*.jpg
    dataset/BSDS500/images/test/*.jpg

or a flat folder of images, or any other subfolder arrangement.
"""

import os
import cv2
import numpy as np

import config


def find_images(root_dir=None, extensions=None):
    """
    Recursively find all image files under root_dir.

    Parameters
    ----------
    root_dir : str
        Directory to search. Defaults to config.DATASET_DIR.
    extensions : tuple
        Allowed file extensions (case-insensitive).

    Returns
    -------
    list[str]
        Sorted list of absolute file paths.
    """
    root_dir = root_dir or config.DATASET_DIR
    extensions = extensions or config.SUPPORTED_EXTENSIONS

    image_paths = []
    for dirpath, _dirnames, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(extensions):
                image_paths.append(os.path.join(dirpath, fname))

    image_paths.sort()
    return image_paths


def inspect_dataset(root_dir=None):
    """
    Print / return a short summary of the dataset structure: how many
    images were found and which subfolders they came from. Useful for
    quickly verifying that a newly-provided dataset was detected correctly.
    """
    root_dir = root_dir or config.DATASET_DIR
    image_paths = find_images(root_dir)

    subfolder_counts = {}
    for path in image_paths:
        rel_dir = os.path.relpath(os.path.dirname(path), root_dir)
        subfolder_counts[rel_dir] = subfolder_counts.get(rel_dir, 0) + 1

    summary = {
        "root_dir": root_dir,
        "total_images": len(image_paths),
        "subfolder_counts": subfolder_counts,
    }
    return summary


def load_image_grayscale(path):
    """
    Load an image from disk as an 8-bit single-channel grayscale array.

    Handles corrupt / unreadable files gracefully by returning None
    instead of raising, so a single bad file does not crash a batch run.
    """
    try:
        # imdecode via numpy buffer is more robust to unicode paths than imread
        with open(path, "rb") as f:
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        return image
    except Exception:
        return None


if __name__ == "__main__":
    info = inspect_dataset()
    print(f"Dataset root : {info['root_dir']}")
    print(f"Total images : {info['total_images']}")
    print("Images per subfolder:")
    for folder, count in sorted(info["subfolder_counts"].items()):
        print(f"  {folder}: {count}")
