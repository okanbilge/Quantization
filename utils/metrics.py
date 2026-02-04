"""Metrics utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional
import numpy as np


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray],
    num_classes: int,
) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        precision_recall_fscore_support,
        roc_auc_score,
        cohen_kappa_score,
        confusion_matrix,
        log_loss,
    )

    metrics: Dict[str, Any] = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    metrics["precision_macro"] = float(prec)
    metrics["recall_macro"] = float(rec)
    metrics["f1_macro"] = float(f1)

    metrics["cohen_kappa"] = float(cohen_kappa_score(y_true, y_pred))
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    if y_prob is not None:
        try:
            metrics["auc_roc_ovr_macro"] = float(
                roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            )
        except Exception:
            metrics["auc_roc_ovr_macro"] = float("nan")

        try:
            metrics["log_loss"] = float(
                log_loss(y_true, y_prob, labels=list(range(num_classes)))
            )
        except Exception:
            metrics["log_loss"] = float("nan")
    else:
        metrics["auc_roc_ovr_macro"] = float("nan")
        metrics["log_loss"] = float("nan")

    return metrics


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(accuracies[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        ece += (np.sum(mask) / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score_multiclass(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    y_onehot = np.eye(num_classes, dtype=np.float32)[y_true]
    return float(np.mean(np.sum((y_onehot - y_prob) ** 2, axis=1)))
