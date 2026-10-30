"""
test_metrics_consistency.py
===========================
Verifies consistency across metrics configs and summary tables.
"""

import sys
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CONFIG_DIR = REPO_ROOT / "configs"
RESULTS_DIR = REPO_ROOT / "results"


def test_svm_metrics():
    p = CONFIG_DIR / "svm" / "svm_metrics.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert abs(data["validation_accuracy"] - 0.640805) < 1e-4
    assert abs(data["validation_macro_f1"] - 0.596078) < 1e-4


def test_rf_metrics():
    p = CONFIG_DIR / "random_forest" / "random_forest_metrics.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["best_model_name"] == "RF-2"
    assert abs(data["validation_accuracy"] - 0.814633) < 1e-4
    assert abs(data["validation_macro_f1"] - 0.767366) < 1e-4


def test_cnn_metrics():
    p = CONFIG_DIR / "cnn" / "cnn_v1_plantvillage_metrics.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert abs(data["val_accuracy"] - 0.995090) < 1e-4
    assert abs(data["val_macro_f1"] - 0.992803) < 1e-4


def test_summary_table():
    p = RESULTS_DIR / "metrics" / "summary_metrics_table.json"
    assert p.exists()
    with open(p, "r", encoding="utf-8") as f:
        table = json.load(f)
    assert len(table) == 7
    assert table[0]["model"] == "Linear SVM"
    assert table[1]["model"] == "Random Forest (RF-2)"
    assert table[2]["model"] == "EfficientNet-B4 CNN v1"
