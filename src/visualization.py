"""
visualization.py
=================
Matplotlib-based helpers to save:

1. Five-panel comparison images: Original | Noisy | Denoised |
   Ground-Truth Edges | Detected Edges.
2. Bar/line graphs comparing Accuracy, Precision, Recall, F1 and IoU
   across denoising methods, noise types and noise levels.
"""

import os

import matplotlib
matplotlib.use("Agg")  # headless backend, safe for scripts/servers
import matplotlib.pyplot as plt
import pandas as pd

import config


def save_comparison_panel(original, noisy, denoised, ground_truth_edges,
                           detected_edges, title, save_path):
    """
    Save a single figure with 5 panels side by side:
    Original, Noisy, Denoised, Ground Truth Edges, Detected Edges.
    """
    fig, axes = plt.subplots(1, 5, figsize=(15, 3.4))
    panels = [
        (original, "Original"),
        (noisy, "Noisy"),
        (denoised, "Denoised"),
        (ground_truth_edges, "Ground Truth Edges"),
        (detected_edges, "Detected Edges"),
    ]
    for ax, (img, label) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(label, fontsize=11)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=80)
    plt.close(fig)


def plot_metric_comparison(results_df, metric, save_path,
                            group_col="method", facet_col="noise_type"):
    """
    Plot a grouped bar chart comparing `metric` across denoising methods,
    with one subplot per noise type, and bars grouped by noise level.

    Parameters
    ----------
    results_df : pd.DataFrame
        Must contain columns: method, noise_type, noise_level, <metric>.
    metric : str
        Column name to plot, e.g. "accuracy", "f1_score", "iou".
    """
    noise_types = sorted(results_df[facet_col].unique())
    methods = list(config.DENOISING_METHODS)
    levels = list(config.NOISE_LEVELS)

    fig, axes = plt.subplots(1, len(noise_types), figsize=(7 * len(noise_types), 5), squeeze=False)
    axes = axes[0]

    bar_width = 0.8 / len(levels)
    x_positions = range(len(methods))

    for ax, noise_type in zip(axes, noise_types):
        subset = results_df[results_df[facet_col] == noise_type]
        agg = subset.groupby([group_col, "noise_level"])[metric].mean().reset_index()

        for i, level in enumerate(levels):
            level_data = agg[agg["noise_level"] == level]
            values = [
                level_data[level_data[group_col] == m][metric].values[0]
                if m in level_data[group_col].values else 0
                for m in methods
            ]
            offsets = [x + i * bar_width for x in x_positions]
            ax.bar(offsets, values, width=bar_width, label=level)

        ax.set_xticks([x + bar_width * (len(levels) - 1) / 2 for x in x_positions])
        ax.set_xticklabels(methods, rotation=30, ha="right")
        ax.set_title(f"{metric.replace('_', ' ').title()} — {noise_type}")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_ylim(0, 1.0)
        ax.legend(title="Noise level")
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=110)
    plt.close(fig)


def generate_all_metric_graphs(results_df, output_dir=None):
    """Generate and save comparison graphs for every key metric."""
    output_dir = output_dir or config.RESULTS_GRAPHS_DIR
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score", "iou"]
    saved_paths = []
    for metric in metrics_to_plot:
        save_path = os.path.join(output_dir, f"comparison_{metric}.png")
        plot_metric_comparison(results_df, metric, save_path)
        saved_paths.append(save_path)
    return saved_paths
