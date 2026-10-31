"""
selectors.py
============
CCA 1 Pattern Recognition Feature Selection Algorithms:
1. SelectKBestANOVA: Filter selection using one-way ANOVA F-statistic (f_classif).
2. SelectFromModelSVM: Embedded selection using Linear SVM (SGDClassifier) L2 coefficient norms.
3. SelectFromModelRF: Embedded selection using Random Forest Gini MDI feature importances.
4. TieredRFE: Wrapper-based Recursive Feature Elimination using Linear SVM with exact 20-step schedule.

All selectors enforce:
- Strict fitting on training data only.
- Deterministic feature ranking using exact 64-bit floating point scores.
- Deterministic secondary tie-breaking by ascending original feature index.
- Serialization and deserialization via joblib and JSON index exports.
"""

import os
import json
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import joblib
from sklearn.feature_selection import f_classif
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning

INITIAL_FEATURE_DIM: int = 8196
TARGET_FEATURE_DIMS: List[int] = [4096, 2048, 1024, 512, 256]

# Exact 20-iteration schedule of intermediate active dimensions:
# Stage 1 (8196 -> 4096): 4 steps of 1,025 features pruned each
# Stage 2 (4096 -> 2048): 4 steps of 512 features pruned each
# Stage 3 (2048 -> 1024): 4 steps of 256 features pruned each
# Stage 4 (1024 -> 512):  4 steps of 128 features pruned each
# Stage 5 (512  -> 256):  4 steps of 64 features pruned each
RFE_SCHEDULE: List[int] = [
    7171, 6146, 5121, 4096,
    3584, 3072, 2560, 2048,
    1792, 1536, 1280, 1024,
    896,  768,  640,  512,
    448,  384,  320,  256
]

RFE_PRUNING_STEPS: List[int] = [
    1025, 1025, 1025, 1025,
    512,  512,  512,  512,
    256,  256,  256,  256,
    128,  128,  128,  128,
    64,   64,   64,   64
]


def verify_rfe_schedule_integrity(
    initial_dim: int = INITIAL_FEATURE_DIM,
    schedule: Optional[List[int]] = None,
    steps: Optional[List[int]] = None,
    targets: Optional[List[int]] = None
) -> bool:
    """
    Automated invariant assertion verifying the mathematical integrity of the RFE schedule.
    Asserts exact subtraction at every step and exact arrival at each target dimension.
    """
    sched = schedule if schedule is not None else RFE_SCHEDULE
    stps = steps if steps is not None else RFE_PRUNING_STEPS
    tgts = targets if targets is not None else TARGET_FEATURE_DIMS

    assert len(sched) == 20, f"Expected 20 iterations, got {len(sched)}"
    assert len(stps) == 20, f"Expected 20 step counts, got {len(stps)}"
    assert sum(stps) == initial_dim - tgts[-1], (
        f"Cumulative pruning sum mismatch: expected {initial_dim - tgts[-1]}, got {sum(stps)}"
    )

    current_dim = initial_dim
    for iter_idx, (expected_dim, prune_count) in enumerate(zip(sched, stps), start=1):
        assert current_dim - prune_count == expected_dim, (
            f"Arithmetic step violation at iteration {iter_idx}: "
            f"{current_dim} - {prune_count} != {expected_dim}"
        )
        current_dim = expected_dim

    for target in tgts:
        assert target in sched, f"Target dimension {target} missing from programmatic schedule"

    return True


# Enforce schedule verification at module load time
assert verify_rfe_schedule_integrity()


def deterministic_rank_scores(scores: np.ndarray) -> np.ndarray:
    """
    Ranks feature indices by descending score using exact 64-bit float values.
    Ties in exact floating-point value are resolved deterministically by ascending original feature index.

    Specification details:
    - Tolerance-based score grouping is NOT performed. Exact float values are compared.
    - Determinism guarantee: Identical score arrays and identical indices will produce identical rankings.
    - Cross-platform note: Bit-identical numerical equivalence across differing CPU architectures,
      compilers, or BLAS/LAPACK builds cannot be assumed.
    """
    exact_scores = np.asarray(scores, dtype=np.float64)
    n_features = len(exact_scores)
    # np.lexsort evaluates keys in reverse order:
    # secondary key = original index ascending: np.arange(n_features)
    # primary key   = score descending: -exact_scores
    ranking = np.lexsort((np.arange(n_features), -exact_scores))
    return ranking


class BaseFeatureSelector:
    """Base class for all feature selection strategies in CCA 1."""

    def __init__(self, k: int):
        self.k: int = k
        self.is_fitted: bool = False
        self.selected_indices_: Optional[np.ndarray] = None
        self.feature_scores_: Optional[np.ndarray] = None
        self.fit_time_sec_: float = 0.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseFeatureSelector":
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.selected_indices_ is None:
            raise RuntimeError("Selector must be fitted on training data before transforming.")
        return X[:, self.selected_indices_]

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serializes fitted selector and index mask."""
        if not self.is_fitted or self.selected_indices_ is None:
            raise RuntimeError("Cannot save an unfitted selector.")
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, p)

        # Also export raw indices as JSON for maximum portability and auditability
        json_path = p.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "k": self.k,
                "fit_time_sec": self.fit_time_sec_,
                "selected_indices": self.selected_indices_.tolist()
            }, f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "BaseFeatureSelector":
        p = Path(filepath)
        if not p.exists():
            raise FileNotFoundError(f"Selector artifact not found at {p}")
        obj = joblib.load(p)
        if not isinstance(obj, BaseFeatureSelector):
            raise TypeError(f"Loaded object is not a BaseFeatureSelector instance: {type(obj)}")
        return obj


class SelectKBestANOVA(BaseFeatureSelector):
    """
    Filter feature selection using one-way ANOVA F-statistic (f_classif).
    Mathematically scale-invariant to per-feature linear shifts and scaling.
    Operates directly on training features.
    """

    def __init__(self, k: int):
        super().__init__(k=k)
        self.p_values_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SelectKBestANOVA":
        t0 = time.perf_counter()
        f_scores, p_vals = f_classif(X, y)
        # Clean any potential NaN or Inf values from zero-variance features
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

        self.feature_scores_ = f_scores.astype(np.float64)
        self.p_values_ = p_vals

        ranked_indices = deterministic_rank_scores(self.feature_scores_)
        # Take top-k and sort indices in ascending order for cache-friendly slicing
        self.selected_indices_ = np.sort(ranked_indices[:self.k])
        self.fit_time_sec_ = time.perf_counter() - t0
        self.is_fitted = True
        return self


class SelectFromModelSVM(BaseFeatureSelector):
    """
    Embedded feature selection using Linear SVM (SGDClassifier with hinge loss).
    Ranks features by the L2 norm of hyperplane weights across all multiclass OvR hyperplanes:
        I_j = sqrt(sum_{c=1}^{C} w_{c, j}^2)

    Requires standardized features to ensure fair regularisation penalties.
    """

    def __init__(
        self,
        k: int,
        C: float = 10.0,
        alpha: Optional[float] = None,
        max_iter: int = 1000,
        tol: float = 1e-3,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        super().__init__(k=k)
        self.C = C
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimator_: Optional[SGDClassifier] = None
        self.converged_: bool = True
        self.n_iter_: int = 0
        self.convergence_warning_count_: int = 0

    def fit(self, X_scaled: np.ndarray, y: np.ndarray) -> "SelectFromModelSVM":
        t0 = time.perf_counter()
        n_samples = len(X_scaled)
        eff_alpha = self.alpha if self.alpha is not None else 1.0 / (float(n_samples) * self.C)

        clf = SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=eff_alpha,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always", ConvergenceWarning)
            clf.fit(X_scaled, y)
            for w in caught_warnings:
                if issubclass(w.category, ConvergenceWarning):
                    self.converged_ = False
                    self.convergence_warning_count_ += 1

        self.estimator_ = clf
        self.n_iter_ = int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else 0

        # clf.coef_ shape is (n_classes, n_features)
        # Compute L2 norm across classes for each feature column:
        l2_norms = np.sqrt(np.sum(clf.coef_ ** 2, axis=0)).astype(np.float64)
        self.feature_scores_ = l2_norms

        ranked_indices = deterministic_rank_scores(self.feature_scores_)
        self.selected_indices_ = np.sort(ranked_indices[:self.k])
        self.fit_time_sec_ = time.perf_counter() - t0
        self.is_fitted = True
        return self


class SelectFromModelRF(BaseFeatureSelector):
    """
    Embedded feature selection using Random Forest (RF-2 architecture: 300 trees, depth 30).
    Ranks features by Gini impurity feature importance (Mean Decrease in Impurity, MDI).

    Uses conservative, explicitly configured worker allocation (n_jobs=4 by default)
    to prevent memory buffer duplication. Operates on raw features (tree-invariant).
    """

    def __init__(
        self,
        k: int,
        n_estimators: int = 300,
        max_depth: Optional[int] = 30,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        random_state: int = 42,
        n_jobs: int = 4
    ):
        super().__init__(k=k)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.estimator_: Optional[RandomForestClassifier] = None

    def fit(self, X_raw: np.ndarray, y: np.ndarray) -> "SelectFromModelRF":
        t0 = time.perf_counter()
        rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )
        rf.fit(X_raw, y)
        self.estimator_ = rf

        importances = rf.feature_importances_.astype(np.float64)
        self.feature_scores_ = importances

        ranked_indices = deterministic_rank_scores(self.feature_scores_)
        self.selected_indices_ = np.sort(ranked_indices[:self.k])
        self.fit_time_sec_ = time.perf_counter() - t0
        self.is_fitted = True
        return self


class TieredRFE(BaseFeatureSelector):
    """
    Tiered Recursive Feature Elimination using Linear SVM (SGDClassifier).
    Follows the verified exact 20-step schedule to reach all target dimensions.

    Features:
    - Pre-iteration elapsed-time checking against max_total_seconds (900s).
    - Caching of selected indices at every intermediate target dimension (4096, 2048, 1024, 512, 256).
    - Per-iteration telemetry recording.
    """

    def __init__(
        self,
        target_k: int = 256,
        schedule: Optional[List[int]] = None,
        pruning_steps: Optional[List[int]] = None,
        max_iter_per_step: int = 250,
        tol: float = 1e-3,
        random_state: int = 42,
        n_jobs: int = -1,
        max_total_seconds: float = 900.0
    ):
        super().__init__(k=target_k)
        self.schedule: List[int] = schedule if schedule is not None else list(RFE_SCHEDULE)
        self.pruning_steps: List[int] = pruning_steps if pruning_steps is not None else list(RFE_PRUNING_STEPS)
        self.max_iter_per_step = max_iter_per_step
        self.tol = tol
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.max_total_seconds = max_total_seconds

        # Intermediate target index masks: target_dim -> np.ndarray of indices
        self.intermediate_masks_: Dict[int, np.ndarray] = {}
        self.iteration_telemetry_: List[Dict[str, Any]] = []
        self.halted_due_to_timeout_: bool = False

    def fit(self, X_scaled: np.ndarray, y: np.ndarray) -> "TieredRFE":
        t_start = time.perf_counter()
        n_samples, initial_dim = X_scaled.shape
        active_indices = np.arange(initial_dim, dtype=np.int64)

        target_set = set(TARGET_FEATURE_DIMS)

        for step_idx, (target_dim, prune_count) in enumerate(zip(self.schedule, self.pruning_steps), start=1):
            # Pre-iteration elapsed-time check
            elapsed_so_far = time.perf_counter() - t_start
            if elapsed_so_far >= self.max_total_seconds:
                self.halted_due_to_timeout_ = True
                warnings.warn(
                    f"RFE reached elapsed-time limit ({elapsed_so_far:.2f}s >= {self.max_total_seconds}s) "
                    f"before starting iteration {step_idx}. Halting safely."
                )
                break

            step_t0 = time.perf_counter()
            X_curr = X_scaled[:, active_indices]

            eff_alpha = 1.0 / (float(n_samples) * 10.0)
            clf = SGDClassifier(
                loss="hinge",
                penalty="l2",
                alpha=eff_alpha,
                max_iter=self.max_iter_per_step,
                tol=self.tol,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )

            converged = True
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", ConvergenceWarning)
                clf.fit(X_curr, y)
                for w in caught_warnings:
                    if issubclass(w.category, ConvergenceWarning):
                        converged = False

            # Compute L2 norms on active subspace
            l2_norms = np.sqrt(np.sum(clf.coef_ ** 2, axis=0)).astype(np.float64)

            # Deterministic ranking within active subspace
            local_ranked = deterministic_rank_scores(l2_norms)
            # Prune the bottom prune_count features: retain the top (len - prune_count)
            n_retain = len(active_indices) - prune_count
            assert n_retain == target_dim, f"Subspace dimension mismatch: expected {target_dim}, got {n_retain}"

            surviving_local_indices = np.sort(local_ranked[:n_retain])
            active_indices = active_indices[surviving_local_indices]

            step_duration = time.perf_counter() - step_t0
            self.iteration_telemetry_.append({
                "iteration": step_idx,
                "active_dimension": int(target_dim),
                "pruned_features": int(prune_count),
                "step_duration_sec": round(step_duration, 3),
                "cumulative_duration_sec": round(time.perf_counter() - t_start, 3),
                "converged": converged,
                "n_iter": int(np.max(clf.n_iter_)) if hasattr(clf, "n_iter_") else 0
            })

            # If this intermediate dimension is a study target, record its mask
            if target_dim in target_set:
                self.intermediate_masks_[int(target_dim)] = np.sort(active_indices.copy())

            # If we've reached user's target_k, we can break early
            if target_dim == self.k:
                break

        self.fit_time_sec_ = time.perf_counter() - t_start
        self.selected_indices_ = np.sort(active_indices)
        self.is_fitted = True
        return self

    def get_indices_for_dimension(self, k: int) -> np.ndarray:
        """Retrieves selected feature indices for any intermediate target dimension."""
        if not self.is_fitted:
            raise RuntimeError("RFE selector must be fitted first.")
        if k in self.intermediate_masks_:
            return self.intermediate_masks_[k]
        if k == self.k and self.selected_indices_ is not None:
            return self.selected_indices_
        raise KeyError(f"Target dimension {k} not captured in intermediate RFE masks.")

