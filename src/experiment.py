"""
experiment.py
=============
Orchestrates the full study:

    For each sampled image:
        1. Preprocess (grayscale + resize).
        2. Generate ground-truth edges from the CLEAN image.
        3. For each noise type x noise level:
              a. Add noise to the clean image.
              b. For each denoising method (including "none" baseline):
                    - Denoise the noisy image (skip for "none").
                    - Run Canny edge detection.
                    - Compare detected edges against ground truth.
                    - Record all metrics.
        4. Save a handful of visual comparison panels.

Results are aggregated into a pandas DataFrame, written to CSV, and used
to compute the best-performing method per noise condition and to generate
comparison graphs.
"""

import os
import time

import numpy as np
import pandas as pd

import config
from src import dataset_loader, preprocessing, noise, denoising, edge_detection, metrics, visualization


def process_single_condition(clean_image, ground_truth_edges, noise_type, noise_level,
                              method, seed=None):
    """
    Run ONE (noise_type, noise_level, method) combination on an already
    preprocessed clean image and return the metrics dict plus the
    intermediate noisy/denoised/detected images (useful for visualization).
    """
    noisy_image = noise.apply_noise(clean_image, noise_type, noise_level, seed=seed)
    denoised_image = denoising.apply_denoising(noisy_image, method)
    detected_edges = edge_detection.canny_edge_detection(denoised_image)
    result_metrics = metrics.compute_metrics(ground_truth_edges, detected_edges)

    return {
        "noisy_image": noisy_image,
        "denoised_image": denoised_image,
        "detected_edges": detected_edges,
        "metrics": result_metrics,
    }


def run_full_experiment(sample_size=None, verbose=True, save_visuals=True):
    """
    Run the complete experiment over a sample of the dataset and every
    combination of noise type, noise level and denoising method.

    Returns
    -------
    pd.DataFrame with one row per (image, noise_type, noise_level, method).
    """
    sample_size = config.SAMPLE_SIZE if sample_size is None else sample_size

    image_paths = dataset_loader.find_images()
    if len(image_paths) == 0:
        raise FileNotFoundError(
            f"No images found under '{config.DATASET_DIR}'. "
            f"Place your dataset there (any subfolder layout is fine) and try again."
        )

    rng = np.random.default_rng(config.RANDOM_SEED)
    if sample_size is not None and sample_size < len(image_paths):
        chosen_indices = rng.choice(len(image_paths), size=sample_size, replace=False)
        chosen_paths = [image_paths[i] for i in sorted(chosen_indices)]
    else:
        chosen_paths = image_paths

    if verbose:
        print(f"Found {len(image_paths)} images total. Using {len(chosen_paths)} for this experiment run.")

    visual_sample_paths = set(
        chosen_paths[:: max(1, len(chosen_paths) // config.NUM_VISUAL_SAMPLES)][: config.NUM_VISUAL_SAMPLES]
    ) if save_visuals else set()

    records = []
    start_time = time.time()

    for img_idx, path in enumerate(chosen_paths):
        raw_image = dataset_loader.load_image_grayscale(path)
        if raw_image is None:
            if verbose:
                print(f"  [skip] Could not read image: {path}")
            continue

        clean_image = preprocessing.preprocess_image(raw_image)
        ground_truth_edges = edge_detection.generate_ground_truth_edges(clean_image)
        image_name = os.path.splitext(os.path.basename(path))[0]

        for noise_type in config.NOISE_TYPES:
            for noise_level in config.NOISE_LEVELS:
                seed = abs(hash((image_name, noise_type, noise_level))) % (2 ** 31)

                for method in config.DENOISING_METHODS:
                    outcome = process_single_condition(
                        clean_image, ground_truth_edges, noise_type, noise_level, method, seed=seed
                    )
                    row = {
                        "image": image_name,
                        "noise_type": noise_type,
                        "noise_level": noise_level,
                        "method": method,
                        **outcome["metrics"],
                    }
                    records.append(row)

                    # Save one visual comparison panel per (image, noise combo),
                    # using the baseline pass ("none") position to also capture
                    # the noisy image, and additionally save each method's panel
                    # only for the chosen visual-sample images to avoid clutter.
                    if save_visuals and path in visual_sample_paths:
                        panel_name = f"{image_name}_{noise_type}_{noise_level}_{method}.png"
                        panel_path = os.path.join(config.RESULTS_IMAGES_DIR, panel_name)
                        visualization.save_comparison_panel(
                            original=clean_image,
                            noisy=outcome["noisy_image"],
                            denoised=outcome["denoised_image"],
                            ground_truth_edges=ground_truth_edges,
                            detected_edges=outcome["detected_edges"],
                            title=f"{image_name} | {noise_type} ({noise_level}) | {method}",
                            save_path=panel_path,
                        )

        if verbose:
            elapsed = time.time() - start_time
            print(f"  [{img_idx + 1}/{len(chosen_paths)}] processed '{image_name}'  (elapsed {elapsed:.1f}s)")

    results_df = pd.DataFrame.from_records(records)
    return results_df


def compute_best_methods(results_df):
    """
    For every (noise_type, noise_level) condition, determine the
    best-performing denoising method by mean F1 Score (a balanced metric
    that accounts for both precision and recall, unlike raw accuracy).

    Returns
    -------
    pd.DataFrame with columns: noise_type, noise_level, best_method,
                                mean_f1_score, mean_iou, mean_accuracy
    """
    grouped = (
        results_df
        .groupby(["noise_type", "noise_level", "method"])[["f1_score", "iou", "accuracy", "precision", "recall"]]
        .mean()
        .reset_index()
    )

    best_rows = []
    for (noise_type, noise_level), group in grouped.groupby(["noise_type", "noise_level"]):
        best_row = group.loc[group["f1_score"].idxmax()]
        best_rows.append({
            "noise_type": noise_type,
            "noise_level": noise_level,
            "best_method": best_row["method"],
            "mean_f1_score": best_row["f1_score"],
            "mean_iou": best_row["iou"],
            "mean_accuracy": best_row["accuracy"],
            "mean_precision": best_row["precision"],
            "mean_recall": best_row["recall"],
        })

    return pd.DataFrame(best_rows)


def save_results(results_df, best_methods_df):
    results_df.to_csv(config.RESULTS_CSV_PATH, index=False)
    best_methods_df.to_csv(config.BEST_METHODS_CSV_PATH, index=False)


if __name__ == "__main__":
    df = run_full_experiment()
    best = compute_best_methods(df)
    save_results(df, best)
    print("\nSaved results to:", config.RESULTS_CSV_PATH)
    print("Saved best-methods summary to:", config.BEST_METHODS_CSV_PATH)
    print("\nBest method per noise condition:\n", best)
