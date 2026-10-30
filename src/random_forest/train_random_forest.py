"""
train_random_forest.py
======================
CCA 1 Pattern Recognition: Random Forest Implementation and Evaluation.

Evaluates 4 bounded configurations:
- RF-1: 200 trees, max_depth=None, min_samples_split=2, min_samples_leaf=1
- RF-2: 300 trees, max_depth=30,   min_samples_split=2, min_samples_leaf=1 (Primary Selected)
- RF-3: 300 trees, max_depth=50,   min_samples_split=2, min_samples_leaf=1
- RF-4: 300 trees, max_depth=None, min_samples_split=5, min_samples_leaf=2

Includes Gini feature importance analysis (HSV vs HOG modality breakdown).
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Tuple, List

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from src.features.dataset_protocol import load_class_names


def analyze_feature_importance(
    rf_model: RandomForestClassifier,
    hsv_dim: int = 96,
    hog_dim: int = 8100
) -> Dict[str, Any]:
    """Analyzes Gini feature importance across HSV and HOG modalities."""
    importances = rf_model.feature_importances_
    total_dim = len(importances)
    assert total_dim == hsv_dim + hog_dim, f"Expected {hsv_dim + hog_dim} features, got {total_dim}"

    hsv_imp = importances[:hsv_dim]
    hog_imp = importances[hsv_dim:]

    sum_hsv = float(np.sum(hsv_imp))
    sum_hog = float(np.sum(hog_imp))
    mean_hsv = float(np.mean(hsv_imp))
    mean_hog = float(np.mean(hog_imp))

    sorted_indices = np.argsort(importances)[::-1]
    top_20 = []
    for rank, idx in enumerate(sorted_indices[:20], 1):
        is_hsv = idx < hsv_dim
        modality = "HSV_Colour" if is_hsv else "HOG_Shape"
        local_idx = int(idx) if is_hsv else int(idx - hsv_dim)
        top_20.append({
            "rank": rank,
            "global_index": int(idx),
            "modality": modality,
            "modality_index": local_idx,
            "importance": float(importances[idx])
        })

    return {
        "hsv_sum_importance": sum_hsv,
        "hog_sum_importance": sum_hog,
        "hsv_percentage": sum_hsv * 100.0,
        "hog_percentage": sum_hog * 100.0,
        "hsv_mean_importance": mean_hsv,
        "hog_mean_importance": mean_hog,
        "hsv_feature_count": hsv_dim,
        "hog_feature_count": hog_dim,
        "top_20_features": top_20
    }


if __name__ == "__main__":
    print("Random Forest training module loaded successfully.")
