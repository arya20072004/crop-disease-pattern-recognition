# Repository Packaging & Integrity Audit

**Project:** Crop Disease Identification Using Comparative Machine Learning Approaches  
**Repository:** `crop-disease-pattern-recognition`  
**Date:** 2026-10-30  
**Audit Status:** PASSED — Verified Standalone Academic Repository  

---

## 1. Inventory of Included Files and Directories

### 1.1 Documentation (`docs/`)
- `docs/project_overview.md`: High-level motivation, academic context, and research questions.
- `docs/methodology.md`: Standardized evaluation protocol, feature extraction, and benchmark rules.
- `docs/dataset_documentation.md`: PlantVillage and PlantDoc partitioning and provenance.
- `docs/limitations.md`: Explicit documentation of threats to validity and scope limits.
- `docs/phase_reports/`: Complete technical reports for Phases 1, 2B, 2C, 3, 4, 4C-R, and 4D.
- `docs/repository_packaging_audit.md`: This comprehensive packaging audit document.

### 1.2 Source Code (`src/`)
- `src/features/feature_extraction.py`: 8,196-dim HSV (96) + HOG (8,100) feature extraction pipeline.
- `src/features/dataset_protocol.py`: Class mappings (38 classes), PlantDoc mapping, and manifest utilities.
- `src/svm/train_svm.py`: Linear SVM training and evaluation via `SGDClassifier(loss="hinge")`.
- `src/random_forest/train_random_forest.py`: Random Forest RF-2 training, hyperparameter sweep, and Gini importance.
- `src/cnn/cnn_adapter.py`: Inference adapter for EfficientNet-B4 CNN v1.
- `src/evaluation/benchmark_inference.py`: Paired intra-iteration latency benchmarking harness.

### 1.3 Configurations (`configs/`)
- `configs/class_names.json`: Official 38 PlantVillage crop-disease classes.
- `configs/svm/`: Metric records for SVM sweeps ($C=0.1, 1.0, 10.0$) and `svm_metrics.json`.
- `configs/random_forest/`: Metric records for RF sweeps (RF-1 to RF-4) and `random_forest_metrics.json`.
- `configs/cnn/`: In-domain and out-of-domain evaluation metrics for CNN v1 across all variants.
- `configs/benchmarks/`: Authoritative (`v2`) and historical (`v1`) inference benchmark JSONs.

### 1.4 Data Manifests (`data/manifests/`)
- `data/manifests/plantvillage_val.json`: Exactly 8,146 validation samples across 38 classes.
- `data/manifests/plantdoc_test.json`: Exactly 236 test samples (219 mapped, 17 unmapped).
- `data/manifests/plantdoc_excluded_samples.json`: Explicit records for the 17 excluded samples.
- `data/README.md`: Dataset provenance, licensing, and access guide.

### 1.5 Results & Predictions (`results/`)
- `results/metrics/summary_metrics_table.csv` & `.json`: Comparative metrics table across all approaches.
- `results/predictions/cnn_v1_plantvillage_predictions.json`: 8,146 sample-level prediction records.
- `results/predictions/cnn_v1_plantdoc_predictions.json`: 219 sample-level prediction records.
- `results/feature_importance/rf_feature_importance.json`: Full Gini importance scores.
- `results/benchmarks/standardized_inference_benchmark_v2.json`: Paired benchmark measurements.

### 1.6 Graphical Representations (`figures/`)
All figures generated in high-resolution PNG (300 DPI) and vector SVG:
- `figures/classification/`: Graph 1 Three-Model Classification Comparison.
- `figures/confusion_matrices/`: Graph 2A (PV $38\times 38$), Graph 2B (Error Diagnostics), Graph 2C (PlantDoc $25\times 26$).
- `figures/feature_importance/`: Graph 3 Random Forest Feature Importance (Aggregate vs Top 20).
- `figures/computational_performance/`: Graph 4 Latency Decomposition and Graph 4 (Supplementary) GPU.
- `figures/throughput_storage/`: Graph 5 Throughput vs. Storage Trade-off (Pareto Frontier).
- `figures/gradcam/`: Graph 6 CNN Grad-CAM Saliency Maps.
- `figures/README.md`: Technical guide to all figures.

### 1.7 Verification Tests & Scripts (`tests/` & `scripts/`)
- `tests/test_benchmark_integrity.py`: Paired arithmetic monotonicity verification.
- `tests/test_dataset_manifests.py`: Manifest sample count and mapping validation.
- `tests/test_metrics_consistency.py`: Numeric bounds and config agreement validation.
- `scripts/generate_all_figures.py`: Standalone figure generation utility.
- `scripts/run_evaluation_suite.py`: Automated repository verification harness.

---

## 2. Excluded Files and Architectural Rationale

| Excluded Component | Location in Original Project | Reason for Exclusion | Alternative Access in New Repository |
|---|---|---|---|
| **Raw Image Datasets** | `archive/.../color/`, `archive/.../PlantDoc/` | > 2 GB raw volume exceeds Git best practices. | Manifests in `data/manifests/`; download instructions in `data/README.md`. |
| **Random Forest Checkpoint** | `models/weights/best_random_forest.joblib` | 513 MB exceeds GitHub's 100 MB hard limit. | Documented in `models/README.md`; reproducible in 197 s via `src/random_forest/train_random_forest.py`. |
| **CNN Checkpoint v1** | `archive/increment_1_cnn/best_model.pth` | 74.7 MB binary file. | Documented with SHA-256 in `models/README.md`. |
| **Production Backend / Flutter App** | `backend/`, `frontend/` | Application code is outside the scope of the CCA 1 Pattern Recognition case study. | Maintained in original application repository. |
| **Personal Windows Paths** | In-code fallback lists (`C:\Users\aryab\...`) | Violates repository hygiene and cross-platform portability. | Replaced with standard relative imports and `requirements.txt`. |
| **Contaminated Checkpoint v2**| `archive/increment_1_cnn/best_model_v2.pth` | Disqualified from zero-shot benchmark due to PlantDoc test leakage. | Documented in `models/README.md`. |

---

## 3. Cryptographic Verification & Integrity Checks

All numerical metrics and configuration files match the original repository bit-for-bit:

| Artifact | Original Path | New Repository Path | Verification Status |
|---|---|---|---|
| `class_names.json` | `models/configs/class_names.json` | `configs/class_names.json` | **Identical (1,215 B)** |
| `svm_metrics.json` | `models/configs/svm_metrics.json` | `configs/svm/svm_metrics.json` | **Identical (9,363 B)** |
| `random_forest_metrics.json` | `models/configs/random_forest_metrics.json` | `configs/random_forest/random_forest_metrics.json` | **Identical (30,932 B)** |
| `cnn_v1_plantvillage_metrics.json` | `models/configs/cnn_v1_plantvillage_metrics.json` | `configs/cnn/cnn_v1_plantvillage_metrics.json` | **Identical (22,140 B)** |
| `plantvillage_val.json` | `data/manifests/plantvillage_val.json` | `data/manifests/plantvillage_val.json` | **Identical (2,910,895 B)** |
| `plantdoc_test.json` | `data/manifests/plantdoc_test.json` | `data/manifests/plantdoc_test.json` | **Identical (143,253 B)** |
| `benchmark_v2.json` | `models/configs/standardized_inference_benchmark_v2.json` | `configs/benchmarks/standardized_inference_benchmark_v2.json` | **Identical (11,157 B)** |

---

## 4. Secret and Large-File Scan
- **Secret Scan:** Scanned for API keys, passwords, `.env` files, and personal user tokens. Result: **0 secrets found**.
- **Large-File Scan:** Scanned all tracked files in `crop-disease-pattern-recognition/`. Result: **No file exceeds 5 MB**. The largest files are `cnn_v1_plantvillage_predictions.json` (3.96 MB) and `graph6_cnn_gradcam_visualizations.png` (3.39 MB). Total repository size is **< 20 MB**, well within GitHub repository guidelines.

---

## 5. Confirmation of Original Project Isolation
- Original project directory: `c:\Users\aryab\Coding\crop-disease-detection`
- **Verification:**
  - No files were deleted, moved, or renamed in `crop-disease-detection`.
  - The working tree of `crop-disease-detection` remains untouched on branch `main`.
  - Checkpoints, caches, backend, and Flutter code remain 100% intact.
