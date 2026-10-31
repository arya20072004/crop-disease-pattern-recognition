"""
pipeline.py
===========
Leakage-Free Preprocessing and Feature Selection Pipeline Wrappers.

Enforces strict train-only contracts:
- TrainOnlyStandardScaler: Ensures StandardScaler is fit exclusively on training data.
- FeatureSelectionPipeline: Coordinates selector fitting, index extraction, and subset transformation.
- EvaluatorPrep: Prepares train and validation subsets for Linear SVM and Random Forest.
"""

from typing import Tuple, Optional, Union, Dict, Any, List
from pathlib import Path
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

from .selectors import BaseFeatureSelector


class TrainOnlyStandardScaler:
    """
    StandardScaler wrapper that guarantees zero leakage by enforcing that
    transformations can only occur after an explicit training fit.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "TrainOnlyStandardScaler":
        if len(X_train) == 0:
            raise ValueError("Cannot fit scaler on empty array.")
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "TrainOnlyStandardScaler must be fitted on training data before transforming!"
            )
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)

    def save(self, filepath: Union[str, Path]) -> None:
        if not self.is_fitted:
            raise RuntimeError("Cannot save an unfitted scaler.")
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "TrainOnlyStandardScaler":
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Scaler artifact not found at {p}")
        obj = joblib.load(p)
        if not isinstance(obj, TrainOnlyStandardScaler):
            raise TypeError(f"Loaded object is not a TrainOnlyStandardScaler: {type(obj)}")
        return obj


class FeatureSelectionPipeline:
    """
    Coordinates feature selection fitting and dataset transformations adhering to:
    - Preprocessing Sequence 1 (SelectKBest ANOVA): Direct fit on raw X_train.
    - Preprocessing Sequence 2 (SelectFromModel SVM): Train-only scaling -> SVM fit.
    - Preprocessing Sequence 3 (SelectFromModel RF): Direct fit on raw X_train.
    - Preprocessing Sequence 4 (Tiered RFE): Train-only scaling -> RFE fit.
    """

    def __init__(self, selector: BaseFeatureSelector, requires_scaling: bool = False):
        self.selector = selector
        self.requires_scaling = requires_scaling
        self.scaler: Optional[TrainOnlyStandardScaler] = TrainOnlyStandardScaler() if requires_scaling else None
        self.is_fitted = False

    def fit(self, X_train_raw: np.ndarray, y_train: np.ndarray) -> "FeatureSelectionPipeline":
        if self.requires_scaling:
            assert self.scaler is not None
            X_train_proc = self.scaler.fit_transform(X_train_raw)
        else:
            X_train_proc = X_train_raw

        self.selector.fit(X_train_proc, y_train)
        self.is_fitted = True
        return self

    @property
    def selected_indices(self) -> np.ndarray:
        if not self.is_fitted or self.selector.selected_indices_ is None:
            raise RuntimeError("Pipeline must be fitted before accessing selected indices.")
        return self.selector.selected_indices_

    def prepare_subsets_for_svm(
        self,
        X_train_raw: np.ndarray,
        X_val_raw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, TrainOnlyStandardScaler]:
        """
        Prepares standardized subsets for Linear SVM evaluation:
        1. Slices raw X_train and X_val to selected indices.
        2. Fits an evaluation scaler strictly on X_train_sub.
        3. Transforms both X_train_sub and X_val_sub.
        """
        indices = self.selected_indices
        X_tr_sub = X_train_raw[:, indices]
        X_val_sub = X_val_raw[:, indices]

        eval_scaler = TrainOnlyStandardScaler()
        X_tr_scaled = eval_scaler.fit_transform(X_tr_sub)
        X_val_scaled = eval_scaler.transform(X_val_sub)

        return X_tr_scaled, X_val_scaled, eval_scaler

    def prepare_subsets_for_rf(
        self,
        X_train_raw: np.ndarray,
        X_val_raw: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares raw unscaled subsets for Random Forest evaluation:
        1. Slices raw X_train and X_val to selected indices.
        2. Trees are scale-invariant; no scaling is applied.
        """
        indices = self.selected_indices
        X_tr_sub = X_train_raw[:, indices]
        X_val_sub = X_val_raw[:, indices]
        return X_tr_sub, X_val_sub

