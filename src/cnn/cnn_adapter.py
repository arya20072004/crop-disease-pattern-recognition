"""
cnn_adapter.py
==============
CCA 1 Pattern Recognition CNN Inference Adapter.

Encapsulates CropDiseaseClassifier (EfficientNet-B4) for evaluation:
- Loads CNN v1 (primary zero-shot field benchmark model).
- Uses exact 380x380 resolution and ImageNet normalisation.
- Provides batch inference (logits, probabilities, predicted indices and class names).
- Provides access to the 512-dimensional embedding layer.
- Instantiates without downloading ImageNet weights (weights=None).
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4
import numpy as np
from PIL import Image
import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_CLASS_NAMES = REPO_ROOT / "configs" / "class_names.json"
CHECKPOINT_V1 = REPO_ROOT / "models" / "weights" / "best_model.pth"

IMG_SIZE = 380
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class CropDiseaseClassifier(nn.Module):
    """
    EfficientNet-B4 Crop Disease Classifier matching increment 1 architecture.
    Instantiated with weights=None to prevent downloading ImageNet weights.
    """
    def __init__(self, num_classes: int = 38, dropout: float = 0.4):
        super().__init__()
        backbone = efficientnet_b4(weights=None)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.embedding = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1792, 512),
            nn.ReLU()
        )
        self.classifier = nn.Linear(512, num_classes)

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts 512-dimensional feature embedding."""
        feat = self.features(x)
        pooled = self.avgpool(feat).flatten(1)
        return self.embedding(pooled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.extract_embedding(x)
        return self.classifier(emb)


def preprocess_image(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """
    Deterministically preprocesses an image to (3, 380, 380) float32 array
    normalized by ImageNet mean and std.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        with Image.open(p) as img:
            rgb = np.array(img.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, Image.Image):
        rgb = np.array(image_input.convert("RGB"), dtype=np.uint8)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        elif image_input.ndim == 3 and image_input.shape[2] == 4:
            rgb = cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        else:
            rgb = image_input.copy()
        if rgb.dtype != np.uint8:
            rgb = (rgb * 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.astype(np.uint8)
    else:
        raise TypeError(f"Unsupported image type: {type(image_input)}")

    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    norm = resized.astype(np.float32) / 255.0
    norm = (norm - IMAGENET_MEAN) / IMAGENET_STD
    return norm.transpose(2, 0, 1)


class CNNAdapter:
    """
    Evaluation adapter for EfficientNet-B4 CNN models.
    """
    def __init__(
        self,
        model: CropDiseaseClassifier,
        class_names: List[str],
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.class_names = class_names
        self.num_classes = len(class_names)

        for param in self.model.parameters():
            param.requires_grad = False

    @classmethod
    def load(
        cls,
        checkpoint_path: Union[str, Path],
        class_names_path: Optional[Union[str, Path]] = None,
        device: str = "cpu"
    ) -> "CNNAdapter":
        """Loads model from checkpoint."""
        c_path = Path(checkpoint_path)
        if not c_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {c_path}")

        cn_path = Path(class_names_path) if class_names_path else CONFIG_CLASS_NAMES
        if not cn_path.exists():
            raise FileNotFoundError(f"Class names config not found at {cn_path}")

        with open(cn_path, "r", encoding="utf-8") as f:
            class_names = json.load(f)

        model = CropDiseaseClassifier(num_classes=len(class_names))
        state_dict = torch.load(c_path, map_location=device)
        model.load_state_dict(state_dict)

        return cls(model=model, class_names=class_names, device=device)

    def predict_single(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image]
    ) -> Dict[str, Any]:
        """Runs single-image inference."""
        chw = preprocess_image(image_input)
        tensor = torch.from_numpy(chw).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        return {
            "predicted_index": pred_idx,
            "predicted_class": self.class_names[pred_idx],
            "confidence": float(probs[pred_idx]),
            "probabilities": probs
        }


if __name__ == "__main__":
    print("CNN Adapter module loaded successfully.")
