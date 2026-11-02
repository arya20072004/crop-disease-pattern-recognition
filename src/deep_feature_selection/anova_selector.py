"""
anova_selector.py
=================
ANOVA F-test Feature Selector for Deep Visual Embeddings.
Enforces strict train-only fitting, deterministic 64-bit ranking, secondary tie-breaking,
and zero-leakage transformation of validation and test embeddings.
"""

import sys
import hashlib
import warnings
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
from sklearn.feature_selection import f_classif

REPO_ROOT = Path = None


def deterministic_rank_scores(scores: np.ndarray) -> np.ndarray:
    """
    Ranks feature dimensions in descending order of score with deterministic
    ascending index secondary tie-breaking.
    """
    n_features = len(scores)
    # Negate scores for descending sort; use feature index as secondary ascending key
    # Using structured array for exact bit-level tie breaking
    dtype = [('neg_score', np.float64), ('orig_idx', np.int64)]
    struct_arr = np.empty(n_features, dtype=dtype)
    struct_arr['neg_score'] = -scores.astype(np.float64)
    struct_arr['orig_idx'] = np.arange(n_features, dtype=np.int64)

    sorted_indices = np.argsort(struct_arr, order=['neg_score', 'orig_idx'])
    return sorted_indices.astype(np.int64)


class DeepANOVASelector:
    """
    One-Way ANOVA F-Statistic Selector for Deep Visual Embeddings.
    """
    def __init__(self, k: int = 256):
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k
        self.is_fitted = False
        self.f_scores_: Optional[np.ndarray] = None
        self.p_values_: Optional[np.ndarray] = None
        self.ranking_: Optional[np.ndarray] = None
        self.selected_indices_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.train_samples_seen_: Optional[int] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "DeepANOVASelector":
        """
        Fits ANOVA F-statistic strictly on training embeddings and labels.
        """
        if X_train.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_train.shape}")
        if len(X_train) != len(y_train):
            raise ValueError(f"Sample count mismatch: X={len(X_train)}, y={len(y_train)}")

        n_samples, n_features = X_train.shape
        if self.k > n_features:
            raise ValueError(f"k ({self.k}) cannot exceed total feature dimensions ({n_features})")

        # Compute ANOVA F-statistics (catching constant-feature divide-by-zero warnings)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f_scores, p_values = f_classif(X_train, y_train)
        
        # Replace NaN / Inf scores with 0.0 for numerical safety
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

        # Deterministic ranking
        ranking = deterministic_rank_scores(f_scores)
        selected_indices = np.sort(ranking[:self.k])  # Sorted ascending for memory locality

        self.f_scores_ = f_scores.astype(np.float64)
        self.p_values_ = p_values.astype(np.float64)
        self.ranking_ = ranking
        self.selected_indices_ = selected_indices
        self.n_features_in_ = n_features
        self.train_samples_seen_ = n_samples
        self.is_fitted = True

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Applies fitted feature selection to an embedding matrix without refitting.
        """
        if not self.is_fitted:
            raise RuntimeError("DeepANOVASelector is not fitted! Call fit() with training data first.")
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch! Expected (N, {self.n_features_in_}), got {X.shape}"
            )
        return X[:, self.selected_indices_]

    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        return self.fit(X_train, y_train).transform(X_train)

    def get_selected_indices(self) -> List[int]:
        if not self.is_fitted:
            raise RuntimeError("Selector is not fitted.")
        return self.selected_indices_.tolist()

    def get_indices_sha256(self) -> str:
        """Computes SHA-256 checksum of the selected indices array."""
        if not self.is_fitted:
            raise RuntimeError("Selector is not fitted.")
        arr_bytes = self.selected_indices_.astype(np.int64).tobytes()
        return hashlib.sha256(arr_bytes).hexdigest()

    def get_summary(self) -> Dict[str, Any]:
        """Returns selector metadata and summary statistics."""
        if not self.is_fitted:
            raise RuntimeError("Selector is not fitted.")
        return {
            "selector_type": "SelectKBest_ANOVA",
            "k": self.k,
            "original_dim": self.n_features_in_,
            "train_samples": self.train_samples_seen_,
            "selected_indices_sha256": self.get_indices_sha256(),
            "f_score_min": float(np.min(self.f_scores_)),
            "f_score_max": float(np.max(self.f_scores_)),
            "f_score_mean": float(np.mean(self.f_scores_)),
            "reduction_percentage": float((self.n_features_in_ - self.k) / self.n_features_in_) * 100.0
        }

