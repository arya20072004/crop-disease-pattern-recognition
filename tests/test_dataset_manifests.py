"""
test_dataset_manifests.py
=========================
Validates integrity of dataset manifests and official class names.
"""

import sys
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MANIFEST_DIR = REPO_ROOT / "data" / "manifests"
CLASS_NAMES_PATH = REPO_ROOT / "configs" / "class_names.json"


def test_class_names():
    assert CLASS_NAMES_PATH.exists()
    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        names = json.load(f)
    assert len(names) == 38
    assert names[0] == "Apple___Apple_scab"
    assert names[37] == "Tomato___healthy"


def test_plantvillage_val_manifest():
    p = MANIFEST_DIR / "plantvillage_val.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["split"] == "val"
    assert len(data["samples"]) == 8146
    classes = set(s["class_name"] for s in data["samples"])
    assert len(classes) == 38


def test_plantdoc_test_manifest():
    p = MANIFEST_DIR / "plantdoc_test.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    assert len(samples) == 236
    mapped = [s for s in samples if s["is_mapped"]]
    assert len(mapped) == 219
    unmapped = [s for s in samples if not s["is_mapped"]]
    assert len(unmapped) == 17
