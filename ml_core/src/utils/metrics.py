"""
QEGVD — Metrics Utility
=======================
Centralised metric computation for binary vulnerability classification.
Supports per-epoch tracking, threshold sweeping, and JSON serialisation.
"""

import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation (pure numpy — no sklearn dependency at import time)
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5,
    dataset_name: str = "unknown",
) -> "MetricBundle":
    """
    Compute full binary classification metrics.

    Parameters
    ----------
    y_true       : Ground-truth binary labels (0/1), shape (N,)
    y_pred_prob  : Predicted probabilities for class 1, shape (N,)
    threshold    : Decision threshold (default 0.5)
    dataset_name : Tag for logging/reporting

    Returns
    -------
    MetricBundle dataclass
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, roc_auc_score, matthews_corrcoef,
        confusion_matrix, average_precision_score,
    )

    y_pred = (y_pred_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy   = float(accuracy_score(y_true, y_pred))
    precision  = float(precision_score(y_true, y_pred, zero_division=0))
    recall     = float(recall_score(y_true, y_pred, zero_division=0))
    f1         = float(f1_score(y_true, y_pred, zero_division=0))
    mcc        = float(matthews_corrcoef(y_true, y_pred))
    roc_auc    = float(roc_auc_score(y_true, y_pred_prob))
    pr_auc     = float(average_precision_score(y_true, y_pred_prob))

    # False Positive Rate, False Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return MetricBundle(
        dataset_name=dataset_name,
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        mcc=mcc,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        fpr=float(fpr),
        fnr=float(fnr),
        tp=int(tp), fp=int(fp), tn=int(tn), fn=int(fn),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    metric: str = "youden",
    steps: int = 200,
) -> float:
    """
    Find the optimal decision threshold using one of three strategies:

    "youden"   (default) -- Youden J statistic: J = Sensitivity + Specificity - 1
                            Maximises the geometric balance between TPR and TNR.
                            Reduces false positives compared to F1-only optimisation.

    "balanced" -- Balanced accuracy: (TPR + TNR) / 2
                  Alternative to Youden, equivalent for binary classification.

    "f1"       -- Maximise F1 score (original behaviour, kept for compatibility).
    "mcc"      -- Maximise Matthews Correlation Coefficient.

    Parameters
    ----------
    y_true       : Ground-truth binary labels (0/1)
    y_pred_prob  : Predicted probabilities for class 1
    metric       : Threshold strategy ("youden" | "balanced" | "f1" | "mcc")
    steps        : Number of threshold candidates in [0.05, 0.95]

    Returns
    -------
    Optimal threshold as float.
    """
    from sklearn.metrics import (
        roc_curve, f1_score, matthews_corrcoef, balanced_accuracy_score
    )

    if metric in ("youden", "balanced"):
        # Youden J = TPR + TNR - 1 = sensitivity + specificity - 1
        # roc_curve gives us fpr, tpr at all thresholds efficiently
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
        # TNR = 1 - FPR
        j_scores = tpr + (1 - fpr) - 1   # Youden J
        best_idx = int(np.argmax(j_scores))
        best_t   = float(thresholds[best_idx])
        # Clamp to sensible range
        return float(np.clip(best_t, 0.05, 0.95))

    # F1 / MCC path (legacy)
    from sklearn.metrics import accuracy_score
    metric_fns = {
        "f1":       lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "mcc":      lambda yt, yp: matthews_corrcoef(yt, yp),
        "accuracy": lambda yt, yp: accuracy_score(yt, yp),
    }
    fn = metric_fns.get(metric, metric_fns["f1"])
    thresholds = np.linspace(0.05, 0.95, steps)
    best_t, best_score = 0.5, -np.inf
    for t in thresholds:
        yp = (y_pred_prob >= t).astype(int)
        score = fn(y_true, yp)
        if score > best_score:
            best_score, best_t = score, t
    return float(best_t)


# ---------------------------------------------------------------------------
# MetricBundle dataclass
# ---------------------------------------------------------------------------

@dataclass
class MetricBundle:
    dataset_name: str
    threshold:   float
    accuracy:    float
    precision:   float
    recall:      float
    f1:          float
    mcc:         float
    roc_auc:     float
    pr_auc:      float
    fpr:         float   # false positive rate
    fnr:         float   # false negative rate
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    epoch: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Metrics saved → {path}")

    def pretty_print(self) -> None:
        lines = [
            f"\n{'─' * 45}",
            f"  Metrics — {self.dataset_name}"
            + (f"  [epoch {self.epoch}]" if self.epoch is not None else ""),
            f"{'─' * 45}",
            f"  Accuracy   : {self.accuracy:.4f}",
            f"  Precision  : {self.precision:.4f}",
            f"  Recall     : {self.recall:.4f}",
            f"  F1 Score   : {self.f1:.4f}",
            f"  MCC        : {self.mcc:.4f}",
            f"  ROC-AUC    : {self.roc_auc:.4f}",
            f"  PR-AUC     : {self.pr_auc:.4f}",
            f"  FPR        : {self.fpr:.4f}",
            f"  FNR        : {self.fnr:.4f}",
            f"  Threshold  : {self.threshold:.3f}",
            f"  TP={self.tp}  FP={self.fp}  TN={self.tn}  FN={self.fn}",
            f"{'─' * 45}",
        ]
        text = "\n".join(lines)
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode("ascii", errors="replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Epoch tracker — accumulates metrics over training
# ---------------------------------------------------------------------------

@dataclass
class EpochTracker:
    """Accumulates per-epoch metrics for train/val/test."""
    history: dict = field(default_factory=lambda: {"train": [], "val": [], "test": []})

    def log(self, split: str, bundle: MetricBundle) -> None:
        self.history[split].append(bundle.to_dict())

    def best_val_epoch(self, metric: str = "f1") -> int:
        if not self.history["val"]:
            return 0
        scores = [e.get(metric, 0) for e in self.history["val"]]
        return int(np.argmax(scores))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"Epoch history saved → {path}")