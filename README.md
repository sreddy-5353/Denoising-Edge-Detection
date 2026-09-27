# Denoising an Image Prior to Edge Detection: Impact on Accuracy

A complete, beginner-friendly Computer Vision project that investigates a
simple but important question:

> **Does denoising a noisy image before running edge detection actually
> improve the accuracy of the detected edges — and if so, by how much, and
> under which conditions?**

Dataset used: **[BSDS500](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)**
(Berkeley Segmentation Dataset), 500 natural images (train/val/test splits),
included in `dataset/BSDS500/images/`.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Objectives](#objectives)
3. [Dataset Description](#dataset-description)
4. [Methodology](#methodology)
5. [Workflow](#workflow)
6. [Project Structure](#project-structure)
7. [Technologies Used](#technologies-used)
8. [Installation](#installation)
9. [Dataset Placement](#dataset-placement)
10. [How to Run the Experiment](#how-to-run-the-experiment)
11. [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
12. [How to Run the Notebook](#how-to-run-the-notebook)
13. [Explanation: Noise Types](#explanation-noise-types)
14. [Explanation: Denoising Methods](#explanation-denoising-methods)
15. [Explanation: Canny Edge Detection](#explanation-canny-edge-detection)
16. [Explanation: Evaluation Metrics](#explanation-evaluation-metrics)
17. [Results Interpretation](#results-interpretation)
18. [Limitations](#limitations)
19. [Future Scope](#future-scope)
20. [Conclusion](#conclusion)

---

## Problem Statement

Real-world images are rarely perfectly clean — sensor noise, low light, and
compression artifacts all introduce noise. Since classic edge detectors like
Canny rely on **image gradients**, noise (which creates fake, sharp
brightness changes) can easily be mistaken for real edges, or can obscure
genuine ones. A common assumption in image processing pipelines is:
*"denoise first, then detect edges."* This project rigorously **tests that
assumption** instead of just assuming it, across multiple noise types,
severities, and denoising algorithms.

## Objectives

- Simulate realistic image degradation using **Gaussian** and **Salt &
  Pepper** noise at **low / medium / high** severities.
- Compare edge detection **with** and **without** a denoising step beforehand.
- Quantify the difference using rigorous, class-imbalance-aware metrics
  (not just raw pixel accuracy).
- Identify which denoising method works best for which noise type/severity.
- Package everything (code, notebook, app, docs) into one reproducible project.

## Dataset Description

This project ships with the **BSDS500** dataset that was provided:

```
dataset/BSDS500/images/train/   -> 200 images
dataset/BSDS500/images/val/     -> 100 images
dataset/BSDS500/images/test/    -> 200 images
```

All images are natural photographs (animals, people, landscapes, objects) in
`.jpg` format, mostly `481x321` or `321x481` pixels. The dataset loader
(`src/dataset_loader.py`) does **not** assume this exact layout — it
recursively scans whatever folder you point `config.DATASET_DIR` to and
picks up every `.jpg` / `.jpeg` / `.png` file it finds, at any depth. This
means you can swap in a different dataset (flat folder, nested folders,
anything) without touching any code — see
[Dataset Placement](#dataset-placement).

## Methodology

For every image, the pipeline is:

```
                      ┌────────────────────────────────────────┐
                      │            GROUND TRUTH PATH             │
  Original (clean) ──▶│  Canny Edge Detection  ──▶ Reference Edges │
                      └────────────────────────────────────────┘

                      ┌───────────────────────────────────────────────────┐
                      │                 BASELINE PATH                      │
  Original ──▶ Add Noise ──▶ Canny Edge Detection ──▶ "Noisy" Edges       │
                      └───────────────────────────────────────────────────┘

                      ┌───────────────────────────────────────────────────────────────┐
                      │                     DENOISED PATH                                │
  Original ──▶ Add Noise ──▶ Denoise ──▶ Canny Edge Detection ──▶ "Denoised" Edges       │
                      └───────────────────────────────────────────────────────────────┘
```

Both the "Noisy" edges and the "Denoised" edges are compared, pixel-by-pixel,
against the **Reference Edges** (Canny run on the original clean image),
using the metric suite described below.

## Workflow

1. **Dataset Handling** (`src/dataset_loader.py`) — recursively discover images.
2. **Preprocessing** (`src/preprocessing.py`) — grayscale conversion, conditional resize.
3. **Noise Generation** (`src/noise.py`) — Gaussian & Salt-and-Pepper, 3 levels each.
4. **Denoising** (`src/denoising.py`) — 4 methods + baseline.
5. **Edge Detection** (`src/edge_detection.py`) — Canny, configurable thresholds.
6. **Metrics** (`src/metrics.py`) — Accuracy, Precision, Recall, F1, IoU, FPR, FNR.
7. **Experiment Orchestration** (`src/experiment.py`) — runs every combination, saves CSV + visuals.
8. **Visualization** (`src/visualization.py`) — comparison panels + metric graphs.
9. **Reporting** — `main.py` (CLI), `app.py` (Streamlit), `notebooks/experiment.ipynb`.

## Project Structure

```
Denoising_Edge_Detection_Project/
│
├── app.py                     # Streamlit application
├── main.py                    # CLI entry point for the full experiment
├── config.py                  # ALL configuration lives here
├── requirements.txt
├── README.md
├── build_notebook.py           # (dev utility) regenerates the notebook programmatically
│
├── src/
│   ├── dataset_loader.py       # Recursive dataset discovery + safe image loading
│   ├── preprocessing.py        # Grayscale + conditional resize
│   ├── noise.py                # Gaussian & Salt-and-Pepper noise generators
│   ├── denoising.py            # Gaussian Blur, Median, Bilateral, NLM
│   ├── edge_detection.py       # Canny wrapper + ground-truth generator
│   ├── metrics.py              # Accuracy/Precision/Recall/F1/IoU/FPR/FNR
│   ├── visualization.py        # Comparison panels + metric graphs
│   └── experiment.py           # Orchestrates the full batch experiment
│
├── dataset/
│   └── BSDS500/images/{train,val,test}/*.jpg   # <- the provided dataset
│
├── results/
│   ├── images/                 # Saved 5-panel visual comparisons
│   ├── graphs/                 # Saved metric comparison bar charts
│   └── csv/                    # experiment_results.csv, best_methods.csv
│
└── notebooks/
    └── experiment.ipynb        # Step-by-step, already-executed walkthrough
```

## Technologies Used

- **Python 3.9+**
- **OpenCV** — noise simulation, filtering, Canny edge detection
- **NumPy** — array operations
- **Pandas** — results aggregation & CSV export
- **Matplotlib** — visualizations
- **scikit-learn** — confusion-matrix computation for metrics
- **Streamlit** — interactive web application
- **Jupyter** — step-by-step notebook walkthrough

## Installation

```bash
# 1. Extract the ZIP, then move into the project folder
cd Denoising_Edge_Detection_Project

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Dataset Placement

The dataset is **already included** in `dataset/BSDS500/images/`, so the
project works immediately after extracting the ZIP — no download needed.

If you want to use a **different** dataset instead:

1. Put your images anywhere inside the `dataset/` folder (subfolders are
   fine, any depth, any mix of `.jpg` / `.jpeg` / `.png`).
2. Either leave `config.DATASET_DIR` pointing at `dataset/`, or change it to
   a completely different path if your images live elsewhere.
3. Run `python -m src.dataset_loader` to verify how many images were found
   and how they're distributed across subfolders.

No code changes are required — the loader adapts automatically.

## How to Run the Experiment

```bash
python main.py
```

Optional flags:

```bash
python main.py --sample-size 100      # use 100 randomly sampled images instead of the default (config.SAMPLE_SIZE)
python main.py --sample-size 0        # use the ENTIRE dataset (slower)
python main.py --no-visuals           # skip saving comparison panel images (faster)
python main.py --dataset-dir path/to/other/dataset
```

This will populate:
- `results/csv/experiment_results.csv` — one row per (image, noise type, noise level, method)
- `results/csv/best_methods.csv` — best method per noise condition
- `results/images/*.png` — 5-panel visual comparisons for a handful of sample images
- `results/graphs/*.png` — bar charts comparing every metric across methods

## How to Run the Streamlit App

```bash
streamlit run app.py
```

Then, in the browser tab that opens:
- **Interactive Demo tab** — pick a dataset image (or upload your own),
  choose noise type/level, denoising method, and Canny thresholds, then
  click **Run Experiment** to see every stage of the pipeline plus a live
  comparison table across all denoising methods.
- **Batch Experiment Results tab** — browse the results of the full
  `main.py` run: summary tables, the best-method table, comparison graphs,
  and sample visual panels.

## How to Run the Notebook

```bash
jupyter notebook notebooks/experiment.ipynb
```

The notebook is already executed and contains output (plots, tables) from a
real run, so you can also just read through it without re-running anything.
To regenerate it from scratch: `python build_notebook.py` then re-execute.

---

## Explanation: Noise Types

**Gaussian Noise** — Random noise drawn from a normal (bell-curve)
distribution is added to every pixel independently. It models sensor/thermal
noise common in real cameras, especially in low light. Controlled by
`sigma` (standard deviation) — higher sigma = noisier image.

**Salt & Pepper Noise** — A fraction of pixels are randomly replaced with
pure black (0, "pepper") or pure white (255, "salt"), leaving the rest of
the image untouched. It models transmission errors, dead sensor pixels, or
bit errors. Controlled by `amount` (the fraction of pixels corrupted).

## Explanation: Denoising Methods

| Method | Idea | Strengths | Weaknesses |
|---|---|---|---|
| **None (baseline)** | No denoising applied | Fastest, no risk of blurring edges | Fully exposed to noise |
| **Gaussian Blur** | Averages each pixel with its neighbors, weighted by a Gaussian kernel | Fast, smooths continuous noise well | Blurs real edges too |
| **Median Filter** | Replaces each pixel with the median of its neighborhood | Excellent at removing salt & pepper outliers | Can round off fine corners |
| **Bilateral Filter** | Like Gaussian blur, but only averages pixels that are also similar in *intensity* | Smooths while better preserving edges | Slower, sensitive to parameter tuning |
| **Non-Local Means (NLM)** | Compares small patches across the whole image and averages similar ones, wherever they are | Very strong denoising, especially for Gaussian noise | Slowest; less effective on impulse noise |

NLM's strength parameter (`h`) is automatically scaled to the image's own
estimated noise level (via a fast blind noise estimator), so it adapts
sensibly to low/medium/high noise rather than using one fixed setting.

## Explanation: Canny Edge Detection

Canny edge detection finds edges by:
1. Smoothing the image slightly (built into the algorithm).
2. Computing intensity gradients (magnitude + direction).
3. Thinning wide gradient responses down to 1-pixel-wide lines ("non-maximum suppression").
4. Applying **two thresholds** — pixels above the *upper* threshold are
   immediately marked as edges; pixels below the *lower* threshold are
   discarded; pixels in between are only kept if they connect to a
   confirmed edge pixel ("hysteresis").

Both thresholds are configurable in `config.py` (`CANNY_LOW_THRESHOLD`,
`CANNY_HIGH_THRESHOLD`) and in the Streamlit app.

## Explanation: Evaluation Metrics

Every detected edge map is compared, pixel by pixel, against the ground
truth edge map (Canny on the clean image). Relative to the ground truth:

- **TP** (True Positive) — pixel correctly detected as an edge
- **FP** (False Positive) — pixel incorrectly detected as an edge
- **FN** (False Negative) — real edge pixel that was missed
- **TN** (True Negative) — pixel correctly identified as *not* an edge

From these:

| Metric | Formula | Meaning |
|---|---|---|
| **Accuracy** | (TP+TN) / Total | Fraction of all pixels classified correctly |
| **Precision** | TP / (TP+FP) | Of the pixels flagged as edges, how many really are edges |
| **Recall** | TP / (TP+FN) | Of the real edges, how many were actually found |
| **F1 Score** | 2·P·R / (P+R) | Balance of Precision and Recall |
| **IoU** | TP / (TP+FP+FN) | Overlap between predicted and true edge pixels |
| **False Positive Rate** | FP / (FP+TN) | Of all non-edge pixels, how many were wrongly flagged |
| **False Negative Rate** | FN / (FN+TP) | Of all real edges, how many were missed |

> ⚠️ **Why accuracy alone can mislead**: in a typical image, edge pixels are
> a small minority (often under 10% of all pixels). A method that predicts
> "no edges anywhere" can still score 90%+ accuracy just by being right
> about all the non-edge pixels, while being completely useless as an edge
> detector. **Precision, Recall, F1 and IoU** specifically measure
> performance on the edge pixels themselves, so they are much more reliable
> indicators of true edge-detection quality and should be weighed more
> heavily than accuracy when interpreting results.

## Results Interpretation

A sample run (40 images, `config.SAMPLE_SIZE`) produced (see
`results/csv/best_methods.csv` for your own run's exact numbers):

| Noise Type | Level | Best Method (by F1) | Mean F1 | Mean IoU |
|---|---|---|---|---|
| Gaussian | low | none (baseline) | ~0.66 | ~0.50 |
| Gaussian | medium | gaussian_blur | ~0.43 | ~0.28 |
| Gaussian | high | gaussian_blur | ~0.32 | ~0.19 |
| Salt & Pepper | low | none (baseline) | ~0.60 | ~0.45 |
| Salt & Pepper | medium | median_filter | ~0.37 | ~0.24 |
| Salt & Pepper | high | median_filter | ~0.36 | ~0.22 |

**Key takeaways:**
- **At low noise**, denoising can actually *hurt* — the noise barely
  disturbs Canny's gradients, but smoothing softens real edges too, so the
  no-denoising baseline often wins.
- **At medium/high noise**, denoising **clearly helps**: without it, Canny
  produces a flood of spurious edges from noise speckle, tanking precision.
  Denoising cleans this up substantially, even though F1/IoU are naturally
  lower overall at higher noise (some information is genuinely lost).
- **Gaussian noise** responds best to **Gaussian Blur** / **Non-Local
  Means** (they target smooth, continuous noise).
- **Salt & Pepper noise** responds best to the **Median Filter** (built
  specifically to reject outlier/impulse pixels).
- This confirms the intuition "denoise before edge detection" is
  **conditionally true** — it depends heavily on noise severity and type,
  not a universal rule.

## Limitations

- Ground truth is derived from Canny on the *clean* image, not a
  human-annotated edge map — it inherits any weaknesses of Canny itself.
- Only Gaussian and Salt & Pepper noise are modeled; real sensor noise is
  often more complex (e.g., Poisson-Gaussian, JPEG compression artifacts).
- Denoising parameters (kernel sizes, `h`, etc.) are fixed/adaptive
  heuristics, not individually tuned per image.
- The default experiment samples a subset of the dataset for speed; results
  may shift slightly with a larger sample or the full dataset.

## Future Scope

- Compare against learned/deep-denoisers (e.g., DnCNN) — intentionally
  excluded here to keep the project classical and lightweight per spec.
- Add more noise models (Poisson, speckle, JPEG compression).
- Tune denoising hyperparameters per noise level via grid search.
- Evaluate against human-annotated ground-truth edges (BSDS500 ships these
  in `.mat` format) instead of Canny-on-clean-image as the reference.
- Extend to color edge detection.

## Conclusion

Denoising before edge detection is **not universally beneficial** — its
value depends on how noisy the image actually is and what kind of noise is
present. For clean-ish images, skipping denoising preserves more true edge
detail. For meaningfully noisy images, an appropriately-matched denoising
step (Gaussian Blur/NLM for Gaussian noise, Median Filter for Salt & Pepper
noise) meaningfully improves edge-detection quality, primarily by
suppressing the false positives that noise would otherwise create.
