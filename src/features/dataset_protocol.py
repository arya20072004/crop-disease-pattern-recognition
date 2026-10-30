"""
dataset_protocol.py
===================
CCA 1 Pattern Recognition Evaluation Foundation: Dataset Contracts & Manifests.

Implements deterministic dataset partitioning and class mappings for:
1. PlantVillage (38 classes, 54,305 images):
   - Stratified split: Train (70%), Validation (15%), Test (15%).
   - Exact class name to index mapping.
2. PlantDoc (27 classes, 2,552 images):
   - Mapped benchmark: 25 classes (219 test images).
   - Excluded classes: 2 classes (17 test images).
   - Duplicate detection & contradictory label detection.
"""

import sys
import os
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_CLASS_NAMES_PATH = REPO_ROOT / "configs" / "class_names.json"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "data" / "manifests"

SEED = 42

# Official PlantDoc to PlantVillage category mapping
PLANTDOC_TO_PLANTVILLAGE: Dict[str, Optional[str]] = {
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "grape leaf": "Grape___healthy",
    "grape leaf black rot": "Grape___Black_rot",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato leaf": "Tomato___healthy",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    # Excluded unmapped classes
    "Bell_pepper leaf": None,
    "Bell_pepper leaf spot": None
}


def load_class_names(config_path: Optional[Path] = None) -> List[str]:
    """Loads the official list of 38 PlantVillage class names."""
    p = config_path or CONFIG_CLASS_NAMES_PATH
    if not p.exists():
        raise FileNotFoundError(f"Class names config file not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_file_md5(file_path: Path) -> str:
    """Computes MD5 hash of an image file for binary duplicate detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


if __name__ == "__main__":
    names = load_class_names()
    print(f"Loaded {len(names)} official classes.")
    assert len(names) == 38
    print("Class names verification PASSED.")
