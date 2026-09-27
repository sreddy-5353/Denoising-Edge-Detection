"""
main.py
=======
Command-line entry point for the full experiment.

Usage
-----
    python main.py
    python main.py --sample-size 100
    python main.py --dataset-dir "dataset/my_other_dataset" --no-visuals

This will:
    1. Inspect the dataset (auto-detecting its folder structure).
    2. Run every combination of noise type x noise level x denoising method
       over a sample of images.
    3. Save all metrics to results/csv/experiment_results.csv
    4. Compute and save the best method per noise condition to
       results/csv/best_methods.csv
    5. Save visual comparison panels to results/images/
    6. Save comparison graphs to results/graphs/
"""

import argparse
import sys

import config
from src import dataset_loader, experiment, visualization


def parse_args():
    parser = argparse.ArgumentParser(
        description="Denoising an Image Prior to Edge Detection: Impact on Accuracy"
    )
    parser.add_argument(
        "--dataset-dir", type=str, default=None,
        help="Override the dataset directory (default: config.DATASET_DIR)",
    )
    parser.add_argument(
        "--sample-size", type=int, default=None,
        help="Number of images to sample for the experiment (default: config.SAMPLE_SIZE). "
             "Use 0 to run on the ENTIRE dataset.",
    )
    parser.add_argument(
        "--no-visuals", action="store_true",
        help="Skip saving visual comparison panels (faster).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dataset_dir:
        config.DATASET_DIR = args.dataset_dir

    print("=" * 70)
    print("Denoising an Image Prior to Edge Detection: Impact on Accuracy")
    print("=" * 70)

    # Step 1: Inspect dataset
    info = dataset_loader.inspect_dataset()
    print(f"\nDataset directory : {info['root_dir']}")
    print(f"Total images found: {info['total_images']}")
    if info["total_images"] == 0:
        print(
            "\nERROR: No images found. Please place your dataset inside the "
            f"'{config.DATASET_DIR}' folder (any subfolder structure is fine) "
            "and run this script again."
        )
        sys.exit(1)

    print("Images per subfolder:")
    for folder, count in sorted(info["subfolder_counts"].items()):
        print(f"  {folder}: {count}")

    sample_size = args.sample_size
    if sample_size == 0:
        sample_size = None  # use entire dataset

    # Step 2: Run experiment
    print("\nRunning experiment...\n")
    results_df = experiment.run_full_experiment(
        sample_size=sample_size,
        verbose=True,
        save_visuals=not args.no_visuals,
    )

    # Step 3 & 4: Save CSV + best methods
    best_methods_df = experiment.compute_best_methods(results_df)
    experiment.save_results(results_df, best_methods_df)

    print(f"\nSaved raw results to   : {config.RESULTS_CSV_PATH}")
    print(f"Saved best-methods to  : {config.BEST_METHODS_CSV_PATH}")

    # Step 5: Graphs
    print("\nGenerating comparison graphs...")
    graph_paths = visualization.generate_all_metric_graphs(results_df)
    for p in graph_paths:
        print(f"  saved: {p}")

    # Summary
    print("\n" + "=" * 70)
    print("BEST DENOISING METHOD PER NOISE CONDITION (by mean F1 Score)")
    print("=" * 70)
    print(best_methods_df.to_string(index=False))
    print("\nDone! Explore results/csv, results/images and results/graphs.")
    print("Run 'streamlit run app.py' for the interactive application.")


if __name__ == "__main__":
    main()
