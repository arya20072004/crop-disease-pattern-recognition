"""
test_benchmark_integrity.py
===========================
Tests paired intra-iteration timing arithmetic decomposition and schema compliance.
"""

import sys
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BENCHMARK_V2_PATH = REPO_ROOT / "configs" / "benchmarks" / "standardized_inference_benchmark_v2.json"
BENCHMARK_HISTORICAL_PATH = REPO_ROOT / "configs" / "benchmarks" / "standardized_inference_benchmark.json"


def test_benchmark_files_exist():
    assert BENCHMARK_V2_PATH.exists(), f"Missing benchmark v2: {BENCHMARK_V2_PATH}"
    assert BENCHMARK_HISTORICAL_PATH.exists(), f"Missing historical benchmark: {BENCHMARK_HISTORICAL_PATH}"


def test_paired_timing_arithmetic():
    with open(BENCHMARK_V2_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. Linear SVM
    svm = data["cpu_benchmarks"]["linear_svm"]
    s_e2e = svm["single_image"]["end_to_end"]["mean_ms"]
    s_pre = svm["single_image"]["preprocessing"]["mean_ms"]
    s_mod = svm["single_image"]["model_only"]["mean_ms"]
    assert abs(s_e2e - (s_pre + s_mod)) <= 0.002

    # 2. Random Forest
    rf = data["cpu_benchmarks"]["random_forest_rf2"]
    rf_s1_e2e = rf["single_image_production"]["end_to_end"]["mean_ms"]
    rf_s1_pre = rf["single_image_production"]["preprocessing"]["mean_ms"]
    rf_s1_mod = rf["single_image_production"]["model_only"]["mean_ms"]
    assert abs(rf_s1_e2e - (rf_s1_pre + rf_s1_mod)) <= 0.002

    rf_diag_e2e = rf["single_image_multicore_diagnostic"]["end_to_end"]["mean_ms"]
    rf_diag_pre = rf["single_image_multicore_diagnostic"]["preprocessing"]["mean_ms"]
    rf_diag_mod = rf["single_image_multicore_diagnostic"]["model_only"]["mean_ms"]
    assert abs(rf_diag_e2e - (rf_diag_pre + rf_diag_mod)) <= 0.002

    # 3. EfficientNet-B4 CPU
    cnn = data["cpu_benchmarks"]["efficientnet_b4_cnn_v1"]
    cnn_e2e = cnn["single_image"]["end_to_end"]["mean_ms"]
    cnn_pre = cnn["single_image"]["preprocessing"]["mean_ms"]
    cnn_mod = cnn["single_image"]["model_only"]["mean_ms"]
    assert abs(cnn_e2e - (cnn_pre + cnn_mod)) <= 0.002
