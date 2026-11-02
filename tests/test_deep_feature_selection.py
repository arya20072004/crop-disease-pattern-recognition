"""
test_deep_feature_selection.py
==============================
Comprehensive automated test suite for Phase 2 deep visual embedding feature selection:
- Checkpoint integrity and EmbeddingExtractor verification.
- Train-only ANOVA selector fitting, determinism, and zero-leakage invariant.
- Experiment A: WeightSlicedClassifier analytical projection and evaluation.
- Experiment B: RetrainedLinearHead fitting, convergence, and evaluation.
- Experiment C: AdaptiveCrossAttentionFusion tensor shape and projection adaptation.
"""

import sys
import hashlib
from pathlib import Path
import pytest
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deep_feature_selection.extractor import (
    EmbeddingExtractor,
    CHECKPOINT_V1_EXPECTED_SHA256,
    compute_file_sha256,
)
from src.deep_feature_selection.anova_selector import (
    DeepANOVASelector,
    deterministic_rank_scores,
)
from src.deep_feature_selection.classifiers import (
    WeightSlicedClassifier,
    RetrainedLinearHead,
    compute_classification_metrics,
)
from src.deep_feature_selection.multimodal import AdaptiveCrossAttentionFusion


# ============================================================================
# 1. Extractor & Checkpoint Integrity Tests
# ============================================================================

def test_checkpoint_v1_integrity():
    """Verifies that CNN v1 checkpoint exists and strictly matches expected SHA-256."""
    checkpoint_path = REPO_ROOT / "models" / "weights" / "best_model.pth"
    fallback_path = REPO_ROOT.parent / "crop-disease-detection" / "archive" / "increment_1_cnn" / "best_model.pth"
    
    target_path = checkpoint_path if checkpoint_path.exists() else fallback_path
    assert target_path.exists(), f"CNN v1 checkpoint not found at {target_path}"
    
    actual_sha = compute_file_sha256(target_path)
    assert actual_sha == CHECKPOINT_V1_EXPECTED_SHA256, (
        f"Integrity violation! Expected {CHECKPOINT_V1_EXPECTED_SHA256}, got {actual_sha}"
    )


def test_embedding_extractor_initialization_and_forward():
    """Verifies model loading, frozen weights, and forward tensor shapes."""
    extractor = EmbeddingExtractor(device="cpu")
    assert extractor.checkpoint_sha256 == CHECKPOINT_V1_EXPECTED_SHA256
    assert extractor.num_classes == 38
    
    # Verify all parameters are frozen
    for p in extractor.model.parameters():
        assert not p.requires_grad
        
    # Verify embedding extraction on dummy input
    dummy_input = torch.zeros(2, 3, 380, 380, dtype=torch.float32)
    with torch.no_grad():
        emb = extractor.model.extract_embedding(dummy_input)
        logits = extractor.model(dummy_input)
        
    assert emb.shape == (2, 512), f"Expected embedding shape (2, 512), got {emb.shape}"
    assert logits.shape == (2, 38), f"Expected logits shape (2, 38), got {logits.shape}"


# ============================================================================
# 2. ANOVA Feature Selection & Zero-Leakage Tests
# ============================================================================

def test_anova_selector_unfitted_guard():
    """Verifies selector prevents transformation prior to fitting."""
    selector = DeepANOVASelector(k=256)
    dummy_x = np.random.randn(10, 512).astype(np.float32)
    with pytest.raises(RuntimeError, match="not fitted"):
        selector.transform(dummy_x)
        
    with pytest.raises(RuntimeError, match="not fitted"):
        selector.get_selected_indices()


def test_anova_selector_dimension_guard():
    """Verifies selector rejects invalid k dimensions or mismatched feature shapes."""
    with pytest.raises(ValueError, match="k must be positive"):
        DeepANOVASelector(k=0)
        
    selector = DeepANOVASelector(k=600)
    X = np.random.randn(50, 512).astype(np.float32)
    y = np.random.randint(0, 5, size=50)
    with pytest.raises(ValueError, match="cannot exceed total feature dimensions"):
        selector.fit(X, y)


def test_anova_selector_fit_and_zero_leakage():
    """Verifies train-only fitting and transformation without data leakage."""
    rng = np.random.RandomState(42)
    n_train = 100
    n_val = 50
    n_features = 512
    n_classes = 5

    X_train = rng.randn(n_train, n_features).astype(np.float32)
    y_train = rng.randint(0, n_classes, size=n_train)
    X_val = rng.randn(n_val, n_features).astype(np.float32)

    k = 128
    selector = DeepANOVASelector(k=k)
    selector.fit(X_train, y_train)

    assert selector.is_fitted
    assert len(selector.ranking_) == n_features
    assert len(selector.selected_indices_) == k
    assert selector.train_samples_seen_ == n_train

    # Transform train and val
    X_train_k = selector.transform(X_train)
    X_val_k = selector.transform(X_val)

    assert X_train_k.shape == (n_train, k)
    assert X_val_k.shape == (n_val, k)

    # Invariant: Transforming val must not alter fitted indices or scores
    f_scores_before = selector.f_scores_.copy()
    indices_before = selector.selected_indices_.copy()

    _ = selector.transform(X_val)

    np.testing.assert_array_equal(selector.f_scores_, f_scores_before)
    np.testing.assert_array_equal(selector.selected_indices_, indices_before)


def test_anova_selector_determinism_and_tie_breaking():
    """Verifies bit-level determinism and ascending index tie-breaking."""
    # Synthetic scores with identical values
    scores = np.array([10.5, 20.0, 5.0, 20.0, 15.0], dtype=np.float64)
    ranking = deterministic_rank_scores(scores)
    
    # Feature 1 and Feature 3 both have score 20.0; tie breaker must prefer index 1 over index 3
    assert ranking[0] == 1, f"Expected index 1 first, got {ranking[0]}"
    assert ranking[1] == 3, f"Expected index 3 second, got {ranking[1]}"
    assert ranking[2] == 4  # 15.0
    assert ranking[3] == 0  # 10.5
    assert ranking[4] == 2  # 5.0


# ============================================================================
# 3. Experiment A: Weight-Sliced Classifier Tests
# ============================================================================

def test_weight_sliced_classifier():
    """Verifies weight slicing without retraining and metric evaluation."""
    rng = np.random.RandomState(42)
    num_classes = 38
    d = 512
    k = 128
    n_samples = 30

    orig_w = rng.randn(num_classes, d).astype(np.float32)
    orig_b = rng.randn(num_classes).astype(np.float32)
    selected_indices = np.sort(rng.choice(d, size=k, replace=False))

    clf_a = WeightSlicedClassifier(orig_w, orig_b, selected_indices)
    assert clf_a.w_reduced.shape == (num_classes, k)
    assert clf_a.b_reduced.shape == (num_classes,)

    X_k = rng.randn(n_samples, k).astype(np.float32)
    logits = clf_a.predict_logits(X_k)
    assert logits.shape == (n_samples, num_classes)

    preds = clf_a.predict(X_k)
    assert len(preds) == n_samples
    assert all(0 <= p < num_classes for p in preds)

    y_true = rng.randint(0, num_classes, size=n_samples)
    metrics = clf_a.evaluate(X_k, y_true)

    assert metrics["paradigm"] == "Experiment_A_Weight_Sliced"
    assert metrics["selected_dimensions"] == k
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["inference_time_sec"] >= 0.0


# ============================================================================
# 4. Experiment B: Retrained Linear Head Tests
# ============================================================================

def test_retrained_linear_head():
    """Verifies fitting convex linear head on reduced features."""
    rng = np.random.RandomState(42)
    k = 64
    num_classes = 10
    n_train = 80
    n_val = 30

    X_train_k = rng.randn(n_train, k).astype(np.float32)
    y_train = rng.randint(0, num_classes, size=n_train)
    X_val_k = rng.randn(n_val, k).astype(np.float32)
    y_val = rng.randint(0, num_classes, size=n_val)

    head = RetrainedLinearHead(k=k, num_classes=num_classes, max_iter=200, random_state=42)
    assert not head.is_fitted

    with pytest.raises(RuntimeError, match="not fitted"):
        head.predict(X_val_k)

    head.fit(X_train_k, y_train)
    assert head.is_fitted
    assert head.fit_duration_sec >= 0.0

    metrics = head.evaluate(X_val_k, y_val)
    assert metrics["paradigm"] == "Experiment_B_Retrained_Linear"
    assert metrics["selected_dimensions"] == k
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert metrics["classifier_train_time_sec"] >= 0.0


# ============================================================================
# 5. Experiment C: Multimodal Cross-Attention Tests
# ============================================================================

@pytest.mark.parametrize("k_dim", [512, 256, 64])
def test_multimodal_cross_attention_forward(k_dim):
    """Verifies cross-attention projection adaptation, tensor shapes, and forward pass."""
    batch_size = 4
    weather_dim = 256
    meta_dim = 21
    n_diseases = 26
    attn_dim = 256

    model = AdaptiveCrossAttentionFusion(
        cnn_dim=k_dim,
        weather_dim=weather_dim,
        meta_dim=meta_dim,
        n_diseases=n_diseases,
        attn_dim=attn_dim,
        n_heads=8
    )
    model.eval()

    summary = model.get_structure_summary()
    assert summary["k_visual_dim"] == k_dim
    assert summary["cnn_proj_weight_shape"] == [attn_dim, k_dim]
    assert summary["total_parameters"] > 0

    cnn_feat = torch.randn(batch_size, k_dim)
    weather_feat = torch.randn(batch_size, weather_dim)
    meta_feat = torch.randn(batch_size, meta_dim)

    with torch.no_grad():
        logits, cross_weights = model(cnn_feat, weather_feat, meta_feat)

    assert logits.shape == (batch_size, n_diseases), f"Expected ({batch_size}, {n_diseases}), got {logits.shape}"
    assert cross_weights.shape == (batch_size, 1, 1), f"Expected ({batch_size}, 1, 1), got {cross_weights.shape}"
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values"
    assert torch.isfinite(cross_weights).all(), "Attention weights contain NaN or Inf values"


def test_multimodal_cross_attention_dim_mismatch():
    """Verifies that passing incorrect feature dimensions raises a ValueError."""
    model = AdaptiveCrossAttentionFusion(cnn_dim=256)
    bad_cnn = torch.randn(2, 128)  # Mismatch: 128 != 256
    good_weather = torch.randn(2, 256)
    good_meta = torch.randn(2, 21)

    with pytest.raises(ValueError, match="Visual feature dimension mismatch"):
        model(bad_cnn, good_weather, good_meta)

