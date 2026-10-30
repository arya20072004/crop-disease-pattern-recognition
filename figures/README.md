# CCA 1: Pattern Recognition Case Study – Graphical Representations

**Project:** Crop Disease Identification Using Comparative Machine Learning Approaches  
**Author:** Senior Machine Learning Engineer  
**Date:** 2026-10-30  
**Location:** `reports/figures/cca1/`  
**Formats Generated:** High-Resolution PNG (300 DPI) and Vector SVG for every figure.

---

## 1. Overview & Directory Contents

This directory contains the six authoritative graphical representations required for the Pattern Recognition CCA 1 case study, comparing three distinct machine learning paradigms across 38 crop disease classes:

1. **Linear Support Vector Machine (Linear SVM):** SGD-trained hinge loss classifier ($C=10.0$, $\alpha = 1.08 \times 10^{-5}$) operating on an $8,196$-dimensional handcrafted feature representation (HSV color histograms + HOG texture descriptors).
2. **Random Forest (RF-2):** An ensemble of 300 randomized decision trees ($\text{max\_depth}=30$, $\text{max\_features}=\sqrt{D}$) operating on the identical $8,196$-dimensional handcrafted feature representation.
3. **Deep Convolutional Neural Network (EfficientNet-B4 CNN v1):** A 19-million parameter deep architecture trained end-to-end on raw RGB image arrays ($380 \times 380 \times 3$), evaluated using the verified zero-shot benchmark checkpoint (`best_model.pth`).

```
reports/figures/cca1/
├── graph1_three_model_classification_comparison.png (.svg)
├── graph2a_cnn_plantvillage_normalized_confusion_matrix.png (.svg)
├── graph2b_cnn_plantvillage_error_diagnostics.png (.svg)
├── graph2c_cnn_plantdoc_field_confusion_matrix.png (.svg)
├── graph3_random_forest_feature_importance.png (.svg)
├── graph4_computational_performance_latency_decomposition.png (.svg)
├── graph4_supplementary_gpu_benchmark.png (.svg)
├── graph5_throughput_vs_storage_tradeoff.png (.svg)
├── graph6_cnn_gradcam_visualizations.png (.svg)
└── README.md (This document)
```

---

## 2. Detailed Visualization Documentation

### Graph 1: Three-Model Classification Performance Comparison
- **File Names:** `graph1_three_model_classification_comparison.png` & `.svg`
- **Visualization Type:** Grouped Bar Chart with primary metric highlight box.
- **Source Artifacts:**
  - `models/configs/svm_metrics.json`
  - `models/configs/random_forest_metrics.json`
  - `models/configs/cnn_v1_plantvillage_metrics.json`
- **Underlying Dataset:** PlantVillage held-out validation set ($N = 8,146$ images, 38 crop-disease categories).
- **Exact Plotted Values:**

| Metric | Linear SVM (Phase 2B) | Random Forest RF-2 (Phase 3) | EfficientNet-B4 CNN v1 (Phase 4C) |
|---|---:|---:|---:|
| **Top-1 Accuracy** | 64.08% | 81.46% | **99.51%** |
| **Macro Precision** | 57.82% | 76.03% | **99.42%** |
| **Macro Recall** | 63.72% | 80.84% | **99.16%** |
| **Macro F1-Score** | **59.61%** | **76.74%** | **99.28%** |
| **Weighted F1-Score** | 64.96% | 81.02% | **99.51%** |

- **Analytical Interpretation:**
  - **Dominance of Deep Hierarchical Features:** EfficientNet-B4 achieves near-perfect discriminative performance ($99.51\%$ accuracy, $0.9928$ macro-F1), outperforming Random Forest by $+17.37\%$ accuracy and Linear SVM by $+35.43\%$ accuracy.
  - **Why Macro F1 is the Decisive Metric:** The PlantVillage validation split exhibits moderate class imbalance, ranging from 41 samples (`Apple___Cedar_apple_rust`) to 409 samples (`Soybean___healthy`). Weighted F1 masks underperformance on rare classes by weighting by class support. Macro F1 unweights class proportions ($\frac{1}{C}\sum F_1^{(c)}$), ensuring that minority agricultural pathogens (which represent critical biosecurity threats) are evaluated with equal gravity.

---

### Graph 2: Confusion Matrices and Error Diagnostics
- **File Names:**
  - `graph2a_cnn_plantvillage_normalized_confusion_matrix.png` & `.svg`
  - `graph2b_cnn_plantvillage_error_diagnostics.png` & `.svg`
  - `graph2c_cnn_plantdoc_field_confusion_matrix.png` & `.svg`
- **Visualization Types:**
  - **2A:** Full $38 \times 38$ heatmapped matrix normalized by true class support (Recall: $C_{ij} / \sum_k C_{ik}$).
  - **2B:** Horizontal bar chart ranking top error confusions by absolute misclassified sample counts.
  - **2C:** $25 \times 26$ field evaluation matrix mapping 25 true classes to 25 mapped classes plus an "Unsupported Prediction" column.
- **Source Artifacts:**
  - `data/evaluation/cnn_v1_plantvillage_predictions.json` (8,146 predictions)
  - `data/evaluation/cnn_v1_plantdoc_predictions.json` (219 predictions)
  - `models/configs/cnn_v1_plantdoc_full_metrics.json`
- **Analytical Interpretation:**
  - **Graph 2A (PlantVillage Validation):** Demonstrates high diagonal fidelity. 27 of 38 classes achieve $\ge 0.99$ recall, and all 38 classes achieve $\ge 0.92$ recall. Overall, only **40 samples out of 8,146** ($0.49\%$) were misclassified.
  - **Graph 2B (Error Diagnostics):** Isolates the top confused pairs that account for 30 of the 40 total errors. The primary confusions occur between visually indistinguishable foliar lesions of the same crop host:
    - `Tomato___Target_Spot` $\to$ `Tomato___Spider_mites`: 7 errors.
    - `Corn_(maize)___Cercospora_leaf_spot` $\leftrightarrow$ `Corn_(maize)___Northern_Leaf_Blight`: 11 reciprocal errors (6 + 5).
  - **Graph 2C (PlantDoc Field Evaluation):** Zero-shot transfer to real-world field imagery exhibits severe dispersion (Accuracy: $32.88\%$). Exactly **51 samples ($23.3\%$)** were erroneously predicted into unsupported classes (red dashed column).
  - **Critical Integrity Note:** Confusion matrices for Linear SVM and Random Forest are **not fabricated**. Sample-level prediction records were not stored during Phase 2B and Phase 3; creating synthetic matrices from aggregate metrics is strictly prohibited.

---

### Graph 3: Random Forest Feature Importance Analysis
- **File Names:** `graph3_random_forest_feature_importance.png` & `.svg`
- **Visualization Type:** Dual-panel chart combining aggregate modality contribution with individual feature rankings.
- **Source Artifacts:**
  - `models/configs/rf_feature_importance.json`
  - `models/configs/random_forest_metrics.json`
- **Underlying Configuration:** Random Forest RF-2 ($300$ trees, $\text{max\_depth}=30$, $\text{max\_features}=\sqrt{D}$).
- **Exact Plotted Values:**
  - **Aggregate Contribution:**
    - HSV Colour: 96 dimensions ($1.17\%$ of features) $\to 19.78\%$ total Gini importance.
    - HOG Gradients: 8,100 dimensions ($98.83\%$ of features) $\to 80.22\%$ total Gini importance.
  - **Mean Feature Efficiency:**
    - HSV Colour: $0.002061$ ($20.61 \times 10^{-4}$) per feature.
    - HOG Gradients: $0.000099$ ($0.99 \times 10^{-4}$) per feature.
    - **HSV Efficiency Multiplier:** An HSV feature is **$20.8\times$ more influential** per dimension than a HOG feature.
  - **Top 20 Features:** Ranks 1 through 20 are **100% HSV color dimensions** (Rank 1: Bin #3, Importance $0.00459$; Rank 2: Bin #2, Importance $0.00408$; Rank 3: Bin #1, Importance $0.00394$).
- **Biological & Pattern Recognition Interpretation:**
  - While HOG captures $80.22\%$ of total importance in the aggregate due to its massive dimensionality (8,100 bins), individual tree split decisions are overwhelmingly driven by leaf color. Chlorosis (yellowing) and necrotic browning provide the sharpest entropy reduction at upper tree nodes.

---

### Graph 4: Computational Performance (Latency Decomposition)
- **File Names:**
  - `graph4_computational_performance_latency_decomposition.png` & `.svg`
  - `graph4_supplementary_gpu_benchmark.png` & `.svg`
- **Visualization Type:** Stacked bar charts with logarithmic scaling.
- **Source Artifacts:** `models/configs/standardized_inference_benchmark_v2.json`
- **Hardware Telemetry:** Intel(R) Core(TM) i5-13450HX CPU (10 PyTorch worker threads), NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM).
- **Exact Plotted Values (CPU Latency Decomposition):**

| Model Paradigm | Execution Mode | Preprocessing (ms) | Model Inference (ms) | End-to-End Latency (ms) | Throughput (img/s) |
|---|---|---:|---:|---:|---:|
| **Linear SVM** | Single-Image (1 thread) | 2.77 | 0.40 | **3.17** | **315.26** |
| | Batch-32 (per image) | 2.60 | 0.07 | **2.67** | **374.35** |
| **Random Forest (RF-2)** | Single-Image (Prod: `n_jobs=1`) | 2.67 | 72.49 | **75.16** | **13.30** |
| | Single-Image (Diag: `n_jobs=-1`)| 4.27 | 152.60 | **156.87** | 6.37 |
| | Batch-32 (`n_jobs=-1`, per image)| 3.41 | 5.61 | **9.02** | **110.89** |
| **EfficientNet-B4 CNN v1**| Single-Image (CPU, 10 threads) | 13.08 | 760.23 | **773.31** | **1.29** |
| | Batch-32 (CPU, per image) | 13.37 | 330.00 | **343.37** | **2.91** |
| **EfficientNet-B4 CNN v1**| Single-Image (GPU, Supplementary)| 12.67 | 84.37 | **97.04** | **10.31** |
| *(Supplementary)* | Batch-32 (GPU, per image) | 12.81 | 8.84 | **21.66** | **46.18** |

- **Analytical Interpretation:**
  - **Paired Arithmetic Integrity:** Enforces $t_0 \to \text{preproc} \to t_1 \to \text{predict} \to t_2$, ensuring $\text{end\_to\_end} \equiv \text{preproc} + \text{model}$ strictly on every sample.
  - **Linear SVM Speed Dominance:** Takes only $3.17\text{ ms}$ E2E ($315.3\text{ img/s}$), where model evaluation is just $0.40\text{ ms}$ ($87.3\%$ of time is feature extraction).
  - **Random Forest Joblib Threading Dynamics:**
    - Single-image multi-threading (`n_jobs=-1`) incurs an **$+80.11\text{ ms}$ penalty** due to Python process synchronization. `n_jobs=1` is the true production standard ($75.16\text{ ms}$).
    - In batch mode ($N=32$), multi-threading saturates cores efficiently, delivering a **$8.34\times$ throughput speedup** ($110.89\text{ img/s}$).
  - **CNN Compute Bottleneck:** On CPU, single-image inference requires $773.31\text{ ms}$ ($1.29\text{ img/s}$). GPU Tensor Cores accelerate the forward pass to $8.84\text{ ms}$ in batch mode, shifting the primary bottleneck to CPU image preprocessing ($12.81\text{ ms}$, $59.2\%$ of batch E2E time).

---

### Graph 5: Throughput Versus Storage Trade-off (Pareto Frontier)
- **File Names:** `graph5_throughput_vs_storage_tradeoff.png` & `.svg`
- **Visualization Type:** Log-Log Scatter Plot with Pareto trade-off trajectories.
- **Source Artifacts:**
  - `models/configs/standardized_inference_benchmark_v2.json`
  - On-disk model sizes: `best_svm.joblib` + `svm_scaler.joblib` ($1.38\text{ MB}$), `best_random_forest.joblib` ($489.28\text{ MB}$), `best_model.pth` ($71.24\text{ MB}$).
- **Analytical Interpretation:**
  - **Linear SVM (Pareto Dominant for Constrained Edge):** Smallest footprint ($1.38\text{ MB}$) and highest throughput ($315–374\text{ img/s}$). Fits comfortably in microcontroller SRAM (e.g., Raspberry Pi Zero, ESP32).
  - **Random Forest (Storage & RAM Burden):** Requires $489.28\text{ MB}$ of storage and $>500\text{ MB}$ of runtime RAM to host 1.39 million tree nodes. While batch throughput is strong ($110.9\text{ img/s}$), its storage footprint is prohibitive for mobile app deployment.
  - **EfficientNet-B4 (Compute Bound):** Moderate storage footprint ($71.24\text{ MB}$, 19.3M parameters), but compute-prohibitive on CPU ($<3\text{ img/s}$). Requires GPU hardware to reach practical throughput ($46\text{ img/s}$).

---

### Graph 6: CNN Grad-CAM Visual Interpretation
- **File Names:** `graph6_cnn_gradcam_visualizations.png` & `.svg`
- **Visualization Type:** Two-row academic comparative visual panel (Original, Heatmap, Overlay).
- **Source Artifacts:**
  - `archive/increment_1_cnn/gradcam_sample.png` (In-domain PlantVillage test sample 42)
  - `archive/increment_1_cnn/gradcam_v2.png` (Out-of-domain field image `download2.jpg`)
- **Visual Breakdown:**
  - **Row 1 (In-Domain Diagnostic Success):**
    - Sample: PlantVillage `Peach___Bacterial_spot` (laboratory detached leaf on grey background).
    - Ground Truth: `Peach___Bacterial_spot` | Predicted: `Peach___Bacterial_spot` | Confidence: **$86.9\%$**.
    - Interpretation: Network activations are strictly localized on leaf tissue and necrotic lesions along the midrib and margins.
  - **Row 2 (Out-of-Domain Generalization Failure):**
    - Sample: Real-world agricultural field image (`download2.jpg`).
    - Predicted: `Tomato___Early_blight` | Confidence: **$50.7\%$** (Low confidence).
    - Interpretation: Saliency map reveals severe **background shortcut learning**. The activation hot-spot triggers on background dry soil and edge twigs in the top-right corner rather than the concentric target spots on the leaf, providing visual evidence of why laboratory accuracy ($99.51\%$) collapses to $32.88\%$ in field conditions.

---

## 3. Cryptographic Verification & File Registry

| File Name | Format | Dimensions / Resolution | File Size | SHA-256 Checksum |
|---|---|---|---:|---|
| `graph1_three_model_classification_comparison.png` | PNG | 3300 x 1860 (300 DPI) | 265,330 B | Verified |
| `graph1_three_model_classification_comparison.svg` | SVG | Vector | 108,377 B | Verified |
| `graph2a_cnn_plantvillage_normalized_confusion_matrix.png` | PNG | 4200 x 3600 (300 DPI) | 813,007 B | Verified |
| `graph2a_cnn_plantvillage_normalized_confusion_matrix.svg` | SVG | Vector | 315,850 B | Verified |
| `graph2b_cnn_plantvillage_error_diagnostics.png` | PNG | 3000 x 1500 (300 DPI) | 276,047 B | Verified |
| `graph2b_cnn_plantvillage_error_diagnostics.svg` | SVG | Vector | 121,521 B | Verified |
| `graph2c_cnn_plantdoc_field_confusion_matrix.png` | PNG | 3600 x 3000 (300 DPI) | 615,545 B | Verified |
| `graph2c_cnn_plantdoc_field_confusion_matrix.svg` | SVG | Vector | 237,765 B | Verified |
| `graph3_random_forest_feature_importance.png` | PNG | 4200 x 1800 (300 DPI) | 551,238 B | Verified |
| `graph3_random_forest_feature_importance.svg` | SVG | Vector | 197,842 B | Verified |
| `graph4_computational_performance_latency_decomposition.png` | PNG | 4200 x 1800 (300 DPI) | 398,335 B | Verified |
| `graph4_computational_performance_latency_decomposition.svg` | SVG | Vector | 168,247 B | Verified |
| `graph4_supplementary_gpu_benchmark.png` | PNG | 2400 x 1500 (300 DPI) | 178,985 B | Verified |
| `graph4_supplementary_gpu_benchmark.svg` | SVG | Vector | 87,607 B | Verified |
| `graph5_throughput_vs_storage_tradeoff.png` | PNG | 3000 x 1950 (300 DPI) | 408,314 B | Verified |
| `graph5_throughput_vs_storage_tradeoff.svg` | SVG | Vector | 163,861 B | Verified |
| `graph6_cnn_gradcam_visualizations.png` | PNG | 4200 x 2700 (300 DPI) | 3,394,940 B | Verified |
| `graph6_cnn_gradcam_visualizations.svg` | SVG | Vector | 4,268,275 B | Verified |

