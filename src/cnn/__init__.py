"""EfficientNet-B4 Convolutional Neural Network inference and evaluation package."""
from .cnn_adapter import CNNAdapter, preprocess_image

__all__ = ["CNNAdapter", "preprocess_image"]
