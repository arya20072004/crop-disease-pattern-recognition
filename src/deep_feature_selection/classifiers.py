"""
classifiers.py
==============
Downstream classifier implementations for evaluating ANOVA-selected deep embeddings:
- Experiment A: WeightSlicedClassifier (Analytical zero-retraining boundary evaluation).
- Experiment B: RetrainedLinearHead (Convex linear evaluation head trained strictly on training embeddings).
"""

import sys
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 38) -> Dict[str, float]:
    """Computes Top-1 Accuracy, Macro/Weighted Precision, Recall, and F1."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_wt, rec_wt, f1_wt, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": acc,
        "macro_precision": float(prec_macro),
        "macro_recall": float(rec_macro),
        "macro_f1": float(f1_macro),
        "weighted_f1": float(f1_wt),
        "num_samples": int(len(y_true)),
        "num_classes": int(num_classes)
    }


class WeightSlicedClassifier:
    """
    Experiment A: Direct Weight-Slicing on the Frozen Classifier.
    Slices corresponding weight columns: W_reduced = W[:, I].
    Evaluates retained decision boundary information without retraining.
    """
    def __init__(
        self,
        original_weights: Union[torch.Tensor, np.ndarray],
        original_bias: Union[torch.Tensor, np.ndarray],
        selected_indices: Union[List[int], np.ndarray]
    ):
        if isinstance(original_weights, torch.Tensor):
            w = original_weights.detach().cpu().numpy()
        else:
            w = np.array(original_weights)

        if isinstance(original_bias, torch.Tensor):
            b = original_bias.detach().cpu().numpy()
        else:
            b = np.array(original_bias)

        self.selected_indices = np.array(selected_indices, dtype=np.int64)
        self.k = len(self.selected_indices)
        self.num_classes = w.shape[0]
        self.original_dim = w.shape[1]

        assert self.k <= self.original_dim, f"k={self.k} exceeds orig_dim={self.original_dim}"

        # Slicing: W_reduced = W[:, I]
        self.w_reduced = w[:, self.selected_indices].astype(np.float32)  # (38, k)
        self.b_reduced = b.astype(np.float32)                           # (38,)

    def predict_logits(self, X_k: np.ndarray) -> np.ndarray:
        """Computes logits: X_k @ W_reduced.T + b."""
        if X_k.shape[1] != self.k:
            raise ValueError(f"Feature dimension mismatch: expected {self.k}, got {X_k.shape[1]}")
        return np.dot(X_k, self.w_reduced.T) + self.b_reduced

    def predict(self, X_k: np.ndarray) -> np.ndarray:
        """Returns predicted class indices."""
        logits = self.predict_logits(X_k)
        return np.argmax(logits, axis=1)

    def evaluate(self, X_k: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Evaluates classification metrics and inference timing."""
        t0 = time.perf_counter()
        y_pred = self.predict(X_k)
        elapsed = time.perf_counter() - t0

        metrics = compute_classification_metrics(y_true, y_pred, num_classes=self.num_classes)
        metrics["paradigm"] = "Experiment_A_Weight_Sliced"
        metrics["selected_dimensions"] = self.k
        metrics["original_dimensions"] = self.original_dim
        metrics["inference_time_sec"] = elapsed
        metrics["throughput_img_per_sec"] = float(len(y_true) / elapsed) if elapsed > 0 else 0.0
        metrics["weights_storage_bytes"] = int(self.w_reduced.nbytes + self.b_reduced.nbytes)
        return metrics


class RetrainedLinearHead:
    """
    Experiment B: Retrained Linear Evaluation Head.
    Fits an optimal linear classifier strictly on reduced training embeddings.
    Measures performance recovery in the reduced subspace.
    """
    def __init__(
        self,
        k: int,
        num_classes: int = 38,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42
    ):
        self.k = k
        self.num_classes = num_classes
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver="lbfgs"
        )
        self.is_fitted = False
        self.fit_duration_sec = 0.0

    def fit(self, X_train_k: np.ndarray, y_train: np.ndarray) -> "RetrainedLinearHead":
        """Fits classifier strictly on training embeddings."""
        if X_train_k.shape[1] != self.k:
            raise ValueError(f"Feature dimension mismatch: expected {self.k}, got {X_train_k.shape[1]}")

        t0 = time.time()
        self.model.fit(X_train_k, y_train)
        self.fit_duration_sec = time.time() - t0
        self.is_fitted = True
        return self

    def predict(self, X_k: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("RetrainedLinearHead is not fitted! Call fit() first.")
        return self.model.predict(X_k)

    def evaluate(self, X_k: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """Evaluates classification metrics and timing."""
        if not self.is_fitted:
            raise RuntimeError("RetrainedLinearHead is not fitted! Call fit() first.")

        t0 = time.perf_counter()
        y_pred = self.predict(X_k)
        elapsed = time.perf_counter() - t0

        metrics = compute_classification_metrics(y_true, y_pred, num_classes=self.num_classes)
        metrics["paradigm"] = "Experiment_B_Retrained_Linear"
        metrics["selected_dimensions"] = self.k
        metrics["classifier_train_time_sec"] = self.fit_duration_sec
        metrics["inference_time_sec"] = elapsed
        metrics["throughput_img_per_sec"] = float(len(y_true) / elapsed) if elapsed > 0 else 0.0
        
        coef_bytes = int(self.model.coef_.nbytes + self.model.intercept_.nbytes)
        metrics["weights_storage_bytes"] = coef_bytes
        return metrics

