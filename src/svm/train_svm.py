"""
train_svm.py
============
CCA 1 Pattern Recognition: Linear SVM Pipeline using SGDClassifier.

Replaces the single-threaded primal LinearSVC with an SGD-trained linear
hinge-loss classifier with native multi-core parallelism across all 38 One-vs-Rest
classes.

Regularisation Mapping:
-----------------------
In standard SVM, the primal objective is:
    min_w  (1/2) * ||w||^2 + C * sum_{i=1}^N max(0, 1 - y_i * w^T x_i)

In SGDClassifier(loss='hinge', penalty='l2', alpha=alpha):
    min_w  (alpha/2) * ||w||^2 + (1/N) * sum_{i=1}^N max(0, 1 - y_i * w^T x_i)

Mathematical equivalence:
    alpha = 1 / (N * C)  <=>  C = 1 / (N * alpha)
"""

import sys
import os
import json
import time
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from src.features.feature_extraction import (
    FeatureConfig,
    FeatureScaler,
    extract_manifest_features
)
from src.features.dataset_protocol import load_class_names


def calculate_alpha(N: int, C: float) -> float:
    """Calculates alpha for SGDClassifier to match SVM regularisation parameter C."""
    return 1.0 / (float(N) * float(C))


def train_and_evaluate_sgd_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    C: float,
    class_names: List[str],
    max_iter: int = 1000,
    tol: float = 1e-3,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict[str, Any]:
    """Trains a multi-class One-vs-Rest Linear SVM via SGDClassifier and returns metrics."""
    N = len(X_train)
    alpha = calculate_alpha(N=N, C=C)

    clf = SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs,
        random_state=random_state
    )

    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        clf.fit(X_train, y_train)
    train_time = time.time() - t0

    converged = True
    for w in caught_warnings:
        if issubclass(w.category, ConvergenceWarning):
            converged = False
            break

    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0
    )
    weighted_p, weighted_r, weighted_f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="weighted", zero_division=0
    )
    cls_report = classification_report(
        y_val, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )

    return {
        "model": clf,
        "C_equivalent": C,
        "alpha": alpha,
        "train_time_sec": train_time,
        "converged": converged,
        "n_iter": int(clf.n_iter_),
        "validation_accuracy": float(acc),
        "validation_macro_precision": float(macro_p),
        "validation_macro_recall": float(macro_r),
        "validation_macro_f1": float(macro_f1),
        "validation_weighted_f1": float(weighted_f1),
        "classification_report": cls_report
    }


if __name__ == "__main__":
    print("Linear SVM training module loaded successfully.")
