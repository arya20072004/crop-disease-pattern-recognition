# Model Weights and Checkpoints Registry

This directory contains the cryptographic specifications, storage requirements, and provenance records for all trained pattern recognition models evaluated in this case study.

In accordance with GitHub storage policies (which enforce a 100 MB per-file upload limit and recommend repositories remain under 1 GB), serialized binary weights exceeding these limits are **not committed to this Git repository**.

---

## 1. Checkpoint Registry and Cryptographic Signatures

Every model evaluated in this project is permanently linked to an immutable SHA-256 digest:

| Model Paradigm | Checkpoint File | Storage Size | Parameter / Structure Count | Cryptographic SHA-256 Digest | Status in Study |
|---|---|---:|---|---|---|
| **Linear SVM** | `best_svm.joblib` | 1.19 MB (1,247,661 B) | 38 hyperplanes ($38 \times 8,196$ weights) | `c65f68ea2f70b5fb66606085a53907c12660d5bfa780d6ae3cb255d03ee61fd9` | **Accepted Reference Model (Phase 2B)** |
| **Linear SVM Scaler** | `svm_scaler.joblib` | 0.19 MB (197,366 B) | Train-only mean & std ($2 \times 8,196$) | `59f2a51aa229786dc79e1ea1ba6e680e66eb13253cf95bb18258e7458f0e33cc` | **Accepted Scaler (Phase 1/2B)** |
| **Random Forest (RF-2)** | `best_random_forest.joblib` | 489.28 MB (513,046,281 B) | 300 decision trees (1,393,582 nodes) | `415ad5ce00dbb61fbeee440cf6238b6fb1f2d2948658a36fa5e74c83a7b83501` | **Accepted Reference Model (Phase 3)** |
| **EfficientNet-B4 CNN v1**| `best_model.pth` | 71.24 MB (74,703,317 B) | 19,341,616 float32 weights | `e48a4d2f0562282d930bc0583017a45ad243083b38e102aa5eacfd401b6fcabb` | **Accepted Zero-Shot Benchmark (Phase 4C)** |
| **EfficientNet-B4 CNN v2**| `best_model_v2.pth` | 71.24 MB (74,705,459 B) | 19,341,616 float32 weights | `6de4b1b834d44103e0515bd2f3726c058011199985fa7b3eea909690d6a99737` | **Disqualified (PlantDoc Test Set Exposure)** |

---

## 2. Checkpoint Provenance Audit Summary

1. **EfficientNet-B4 CNN v1 (`best_model.pth`):**
   - Trained exclusively on the 38,013 PlantVillage training split.
   - Zero exposure to PlantDoc field images during training or hyperparameter selection.
   - Verified as an authentic zero-shot out-of-domain benchmark.
2. **EfficientNet-B4 CNN v2 (`best_model_v2.pth`):**
   - Exposed directly to the PlantDoc test set during incremental fine-tuning (`increment_1_cnn.py` lines 144–145, 207, 320–321).
   - Formally disqualified from the zero-shot field benchmark to prevent data contamination.
3. **Random Forest (RF-2) Git Exclusion:**
   - At 489.28 MB, `best_random_forest.joblib` exceeds GitHub's 100 MB hard limit. It can be fully re-trained in 197 seconds using `src/random_forest/train_random_forest.py` with random seed 42.

---

## 3. How to Obtain and Verify Weights

1. Place downloaded checkpoint files into `models/weights/`.
2. Verify their cryptographic integrity using Python:
```python
import hashlib
from pathlib import Path

def check_sha256(filepath, expected):
    h = hashlib.sha256(Path(filepath).read_bytes()).hexdigest()
    assert h == expected, f"Hash mismatch for {filepath}!"
    print(f"Verified {filepath}: {h[:16]}...")

check_sha256("models/weights/best_model.pth", "e48a4d2f0562282d930bc0583017a45ad243083b38e102aa5eacfd401b6fcabb")
check_sha256("models/weights/best_svm.joblib", "c65f68ea2f70b5fb66606085a53907c12660d5bfa780d6ae3cb255d03ee61fd9")
```
All experimental evaluation metrics, prediction records, and confusion matrices in `configs/`, `results/`, and `figures/` can be fully inspected and audited without downloading the binary weights.
