"""
metrics.py
==========
Evaluation metrics for comparing a predicted (detected) binary edge map
against the ground-truth binary edge map (generated from the clean,
noise-free image).

Every edge map is a 2D array where a pixel value > 0 means "edge" and
0 means "not edge" (this matches cv2.Canny's 0/255 output).

IMPORTANT NOTE ON PIXEL ACCURACY
---------------------------------
In edge-detection images, edge pixels are typically a SMALL minority of
the total pixels (most of an image is flat, non-edge background/texture).
Because of this severe class imbalance, plain pixel accuracy can look
deceptively high even for a poor edge detector -- a method that predicts
"no edges anywhere" can still score >90% accuracy simply because most
pixels genuinely are non-edges. For this reason, Precision, Recall, F1
Score and IoU (which focus specifically on how well the actual edge
pixels were recovered) are much more informative for judging real
edge-detection quality, and should be weighed more heavily than accuracy
alone when interpreting results.

Metrics computed here
----------------------
- Accuracy              : (TP + TN) / Total
- Precision             : TP / (TP + FP)
- Recall (Sensitivity)  : TP / (TP + FN)
- F1 Score              : harmonic mean of Precision and Recall
- IoU (Jaccard Index)   : TP / (TP + FP + FN)
- False Positive Rate   : FP / (FP + TN)
- False Negative Rate   : FN / (FN + TP)

where, relative to the ground-truth edge map:
  TP = predicted edge pixel that is also a ground-truth edge pixel
  FP = predicted edge pixel that is NOT a ground-truth edge pixel
  FN = ground-truth edge pixel that was NOT predicted as an edge
  TN = pixel correctly predicted as NOT an edge
"""

import numpy as np
from sklearn.metrics import confusion_matrix


def _binarize(edge_map):
    """Convert a 0/255 (or any nonzero) edge map into a 0/1 integer array."""
    return (edge_map > 0).astype(int)


def compute_confusion(ground_truth_edges, predicted_edges):
    """
    Compute the pixel-level confusion matrix components between a
    predicted edge map and the ground-truth edge map, using
    scikit-learn's confusion_matrix on the flattened, binarized pixel
    arrays (1 = edge pixel, 0 = non-edge pixel).

    Returns
    -------
    dict with keys: tp, fp, fn, tn
    """
    gt = _binarize(ground_truth_edges)
    pred = _binarize(predicted_edges)

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch between ground truth {gt.shape} and prediction {pred.shape}"
        )

    # labels=[0, 1] guarantees a full 2x2 matrix even if one class is
    # entirely absent (e.g. an image with zero ground-truth edge pixels).
    tn, fp, fn, tp = confusion_matrix(
        gt.flatten(), pred.flatten(), labels=[0, 1]
    ).ravel()

    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def _safe_divide(numerator, denominator):
    return float(numerator) / denominator if denominator > 0 else 0.0


def compute_metrics(ground_truth_edges, predicted_edges):
    """
    Compute the full metric suite for one predicted edge map against one
    ground-truth edge map.

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1_score, iou,
                    false_positive_rate, false_negative_rate,
                    tp, fp, fn, tn
    """
    c = compute_confusion(ground_truth_edges, predicted_edges)
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    total = tp + fp + fn + tn

    accuracy = _safe_divide(tp + tn, total)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1_score = _safe_divide(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    iou = _safe_divide(tp, tp + fp + fn)
    false_positive_rate = _safe_divide(fp, fp + tn)
    false_negative_rate = _safe_divide(fn, fn + tp)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "iou": iou,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
