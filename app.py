"""
app.py
======
Streamlit application for "Denoising an Image Prior to Edge Detection:
Impact on Accuracy".

Run with:
    streamlit run app.py

Two tabs:
  1. Interactive Demo - pick/upload an image, choose noise + denoising
     settings, and see the full pipeline + metrics live.
  2. Batch Results - browse the results of the full experiment (run via
     `python main.py`), including the comparison table and graphs.
"""

import os

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import config
from src import dataset_loader, preprocessing, noise, denoising, edge_detection, metrics, experiment


st.set_page_config(
    page_title="Denoising vs Edge Detection",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset_images():
    return dataset_loader.find_images()


def load_and_preprocess(path_or_bytes, is_bytes=False):
    if is_bytes:
        file_bytes = np.frombuffer(path_or_bytes, dtype=np.uint8)
        raw = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    else:
        raw = dataset_loader.load_image_grayscale(path_or_bytes)
    if raw is None:
        return None
    return preprocessing.preprocess_image(raw)


def metrics_to_display(m):
    return {
        "Accuracy": round(m["accuracy"], 4),
        "Precision": round(m["precision"], 4),
        "Recall": round(m["recall"], 4),
        "F1 Score": round(m["f1_score"], 4),
        "IoU": round(m["iou"], 4),
        "False Positive Rate": round(m["false_positive_rate"], 4),
        "False Negative Rate": round(m["false_negative_rate"], 4),
    }


# ---------------------------------------------------------------------------
# Sidebar / Header
# ---------------------------------------------------------------------------
st.title("🔍 Denoising an Image Prior to Edge Detection: Impact on Accuracy")
st.caption(
    "Compares **Noisy → Canny** against **Noisy → Denoising → Canny**, "
    "using the clean image's own edges as ground truth."
)

tab_demo, tab_batch = st.tabs(["🖼️ Interactive Demo", "📊 Batch Experiment Results"])

# ---------------------------------------------------------------------------
# TAB 1: Interactive demo
# ---------------------------------------------------------------------------
with tab_demo:
    left_col, right_col = st.columns([1, 2])

    with left_col:
        st.subheader("1. Choose an image")
        source = st.radio("Image source", ["From dataset", "Upload my own"], horizontal=True)

        clean_image = None
        image_label = ""

        if source == "From dataset":
            dataset_images = get_dataset_images()
            if len(dataset_images) == 0:
                st.error(
                    f"No images found in '{config.DATASET_DIR}'. "
                    "Place your dataset there and refresh the page."
                )
            else:
                rel_paths = [os.path.relpath(p, config.DATASET_DIR) for p in dataset_images]
                choice = st.selectbox("Pick an image", rel_paths, index=0)
                full_path = os.path.join(config.DATASET_DIR, choice)
                clean_image = load_and_preprocess(full_path)
                image_label = choice
        else:
            uploaded = st.file_uploader("Upload a JPG/PNG image", type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                clean_image = load_and_preprocess(uploaded.read(), is_bytes=True)
                image_label = uploaded.name

        st.subheader("2. Noise settings")
        noise_type = st.selectbox("Noise type", config.NOISE_TYPES, format_func=lambda x: x.replace("_", " ").title())
        noise_level = st.select_slider("Noise level", options=config.NOISE_LEVELS, value="medium")

        st.subheader("3. Denoising method")
        method = st.selectbox(
            "Method", config.DENOISING_METHODS,
            format_func=lambda x: "No denoising (baseline)" if x == "none" else x.replace("_", " ").title(),
        )

        st.subheader("4. Canny thresholds")
        low_thresh = st.slider("Lower threshold", 0, 255, config.CANNY_LOW_THRESHOLD)
        high_thresh = st.slider("Upper threshold", 0, 255, config.CANNY_HIGH_THRESHOLD)

        run_clicked = st.button("▶ Run Experiment", type="primary", use_container_width=True)

    with right_col:
        if clean_image is None:
            st.info("Choose or upload an image on the left, then click **Run Experiment**.")
        elif run_clicked:
            ground_truth_edges = edge_detection.generate_ground_truth_edges(clean_image, low_thresh, high_thresh)
            noisy_image = noise.apply_noise(clean_image, noise_type, noise_level, seed=42)
            denoised_image = denoising.apply_denoising(noisy_image, method)
            detected_edges = edge_detection.canny_edge_detection(denoised_image, low_thresh, high_thresh)
            result_metrics = metrics.compute_metrics(ground_truth_edges, detected_edges)

            st.subheader(f"Results for: `{image_label}`")

            img_cols = st.columns(5)
            panels = [
                (clean_image, "Original"),
                (noisy_image, "Noisy"),
                (denoised_image, "Denoised"),
                (ground_truth_edges, "Ground Truth Edges"),
                (detected_edges, "Detected Edges"),
            ]
            for col, (img, caption) in zip(img_cols, panels):
                col.image(img, caption=caption, use_container_width=True, clamp=True)

            st.subheader("Evaluation Metrics")
            display_metrics = metrics_to_display(result_metrics)
            metric_cols = st.columns(len(display_metrics))
            for col, (name, value) in zip(metric_cols, display_metrics.items()):
                col.metric(name, value)

            st.divider()
            st.subheader("Compare ALL Denoising Methods on This Image")
            compare_rows = []
            for m in config.DENOISING_METHODS:
                den = denoising.apply_denoising(noisy_image, m)
                det = edge_detection.canny_edge_detection(den, low_thresh, high_thresh)
                mm = metrics.compute_metrics(ground_truth_edges, det)
                compare_rows.append({
                    "Method": "No denoising (baseline)" if m == "none" else m.replace("_", " ").title(),
                    **metrics_to_display(mm),
                })
            compare_df = pd.DataFrame(compare_rows).set_index("Method")
            st.dataframe(
               compare_df,
               use_container_width=True
             )
            best_method_row = compare_df["F1 Score"].idxmax()
            st.success(f"✅ Best method for this image/noise setting (by F1 Score): **{best_method_row}**")
        else:
            st.image(clean_image, caption=f"Preview: {image_label}", use_container_width=True, clamp=True)
            st.info("Click **Run Experiment** to see the full pipeline and metrics.")

# ---------------------------------------------------------------------------
# TAB 2: Batch experiment results
# ---------------------------------------------------------------------------
with tab_batch:
    st.subheader("Full Batch Experiment")
    st.write(
        "Run `python main.py` from the project root to generate/update these "
        "results across many images at once."
    )

    if not os.path.exists(config.RESULTS_CSV_PATH):
        st.warning(
            "No batch results found yet. Run `python main.py` in a terminal, "
            "then refresh this page."
        )
        if st.button("Run a quick experiment now (small sample)"):
            with st.spinner("Running experiment on a small sample of images..."):
                df = experiment.run_full_experiment(sample_size=10, verbose=False, save_visuals=False)
                best_df = experiment.compute_best_methods(df)
                experiment.save_results(df, best_df)
            st.rerun()
    else:
        results_df = pd.read_csv(config.RESULTS_CSV_PATH)
        best_df = pd.read_csv(config.BEST_METHODS_CSV_PATH) if os.path.exists(config.BEST_METHODS_CSV_PATH) else None

        st.markdown(f"**{results_df['image'].nunique()}** images × "
                    f"**{results_df['noise_type'].nunique()}** noise types × "
                    f"**{results_df['noise_level'].nunique()}** levels × "
                    f"**{results_df['method'].nunique()}** methods "
                    f"= **{len(results_df)}** total runs")

        st.markdown("#### Mean metrics by method / noise type / noise level")
        summary = (
            results_df
            .groupby(["noise_type", "noise_level", "method"])[
                ["accuracy", "precision", "recall", "f1_score", "iou"]
            ]
            .mean()
            .round(4)
            .reset_index()
        )
        st.dataframe(summary, use_container_width=True, height=350)

        if best_df is not None:
            st.markdown("#### 🏆 Best method per noise condition (by mean F1 Score)")
            st.dataframe(best_df.round(4), use_container_width=True)

        st.markdown("#### Comparison Graphs")
        graph_files = [
            "comparison_accuracy.png", "comparison_precision.png", "comparison_recall.png",
            "comparison_f1_score.png", "comparison_iou.png",
        ]
        for gf in graph_files:
            gpath = os.path.join(config.RESULTS_GRAPHS_DIR, gf)
            if os.path.exists(gpath):
                st.image(gpath, use_container_width=True)

        st.markdown("#### Sample Visual Comparisons")
        image_dir = config.RESULTS_IMAGES_DIR
        if os.path.isdir(image_dir):
            sample_images = sorted(os.listdir(image_dir))[:12]
            if sample_images:
                img_cols = st.columns(3)
                for i, fname in enumerate(sample_images):
                    img_cols[i % 3].image(os.path.join(image_dir, fname), caption=fname, use_container_width=True)
            else:
                st.info("No saved comparison panels yet. Run `python main.py` (without --no-visuals).")
