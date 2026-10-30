"""
benchmark_inference.py
======================
CCA 1 Pattern Recognition: Paired Intra-Iteration Inference Latency Benchmarking.

Evaluates and compares:
1. Linear SVM (SGD hinge loss)
2. Random Forest RF-2 (300 trees):
   - Single-image: n_jobs=1 (production standard)
   - Single-image diagnostic: n_jobs=-1 (multi-core dispatch overhead demonstration)
   - Batch-32: n_jobs=-1 (multi-threaded batch standard)
3. EfficientNet-B4 CNN v1 on CPU (10 worker threads)
4. EfficientNet-B4 CNN v1 on GPU (Supplementary hardware benchmark)

Key Methodology (Paired Intra-Iteration Decomposition):
- Measures Preprocessing and Model Inference within the EXACT SAME iteration instance:
    t0 = time.perf_counter()
    preprocessed = preprocess_fn(input)
    t1 = time.perf_counter()
    prediction = model_predict_fn(preprocessed)
    [cuda.synchronize() if GPU]
    t2 = time.perf_counter()
- Guaranteed Mathematical Monotonicity:
    end_to_end_latency == preprocessing_latency + model_only_latency
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
import joblib
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cnn.cnn_adapter import CNNAdapter, preprocess_image
from src.features.feature_extraction import extract_handcrafted_features

CONFIG_DIR = REPO_ROOT / "configs"
OUTPUT_BENCHMARK_PATH = CONFIG_DIR / "benchmarks" / "standardized_inference_benchmark_v2.json"


def stats_dict(latencies_ms: List[float], batch_size: int = 1) -> Dict[str, float]:
    """Computes summary statistics and throughput from a list of latencies in ms."""
    arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    median_ms = float(np.median(arr))
    std_ms = float(np.std(arr))
    p95_ms = float(np.percentile(arr, 95))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))

    total_elapsed_sec = np.sum(arr) / 1000.0
    total_images = len(arr) * batch_size
    throughput = (total_images / total_elapsed_sec) if total_elapsed_sec > 0 else 0.0

    return {
        "mean_ms": round(mean_ms, 3),
        "median_ms": round(median_ms, 3),
        "std_ms": round(std_ms, 3),
        "min_ms": round(min_ms, 3),
        "max_ms": round(max_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "throughput_img_per_sec": round(throughput, 2)
    }


def verify_paired_arithmetic(
    preproc_list: List[float],
    model_list: List[float],
    e2e_list: List[float],
    context: str = ""
):
    """Verifies that e2e == preproc + model strictly on every recorded iteration."""
    assert len(preproc_list) == len(model_list) == len(e2e_list), f"Length mismatch in {context}"
    for i in range(len(e2e_list)):
        expected_sum = preproc_list[i] + model_list[i]
        diff = abs(e2e_list[i] - expected_sum)
        assert diff < 1e-5, (
            f"Arithmetic decomposition violation in {context} at iter {i}: "
            f"e2e={e2e_list[i]:.6f}, preproc={preproc_list[i]:.6f}, model={model_list[i]:.6f}, diff={diff:.6e}"
        )


if __name__ == "__main__":
    print("Inference benchmarking module loaded successfully.")
