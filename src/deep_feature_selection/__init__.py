"""
deep_feature_selection
======================
Phase 2 Deep Visual Embedding ANOVA Feature Selection Package.
Supports EfficientNet-B4 visual embeddings (512-d), train-only ANOVA selection,
weight-slicing evaluation, retrained linear head evaluation, and multimodal cross-attention integration.
"""

from .extractor import EmbeddingExtractor
from .anova_selector import DeepANOVASelector
from .classifiers import WeightSlicedClassifier, RetrainedLinearHead
from .multimodal import AdaptiveCrossAttentionFusion

__all__ = [
    "EmbeddingExtractor",
    "DeepANOVASelector",
    "WeightSlicedClassifier",
    "RetrainedLinearHead",
    "AdaptiveCrossAttentionFusion",
]

