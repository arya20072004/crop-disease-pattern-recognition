# Phase 4: Controlled CNN Benchmarking and Field Evaluation

**Repository:** `crop-disease-pattern-recognition`  
**Phase:** 4 (CNN Audit, Evaluation & Field Generalization)  
**Status:** Completed & Accepted Benchmark  

---

## 1. Objective & Checkpoint Provenance Audit (Stage 4A)
Audit existing deep convolutional checkpoints and evaluate the verified zero-shot benchmark model (EfficientNet-B4 CNN v1) against the identical PlantVillage validation split and the independent PlantDoc field dataset.

### Checkpoint Provenance Findings:
1. **CNN v1 (`best_model.pth`, SHA-256 `e48a4d...`):**
   - Trained exclusively on the 38,013 PlantVillage training split.
   - Zero exposure to PlantDoc field data during training or parameter tuning.
   - Eligible as the primary zero-shot out-of-domain benchmark.
2. **CNN v2 (`best_model_v2.pth`, SHA-256 `6de4b1...`):**
   - Exposed directly to the PlantDoc test set during incremental fine-tuning (`increment_1_cnn.py`).
   - Disqualified from independent zero-shot field evaluation to prevent data leakage.

---

## 2. In-Domain PlantVillage Validation Results ($N = 8,146$)
- **Top-1 Accuracy:** 99.51%
- **Macro Precision:** 0.9942
- **Macro Recall:** 0.9916
- **Macro F1-Score:** 0.9928
- **Weighted F1-Score:** 0.9951
- **Total Misclassifications:** Exactly 40 errors out of 8,146 images ($0.49\%$).

---

## 3. Real-World PlantDoc Field Evaluation Results

| Variant | Samples ($N$) | Evaluated Classes | Correct | Unsupported Predictions | Top-1 Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full Mapped** | 219 | 25 | 72 | 51 (23.29%) | **32.88%** | 0.4774 | 0.3256 | **0.3186** |
| **Conflict-Excluded** | 211 | 25 | 70 | 50 (23.70%) | **33.18%** | 0.4841 | 0.3280 | **0.3191** |
| **Fully Deduplicated**| 208 | 25 | 69 | 49 (23.56%) | **33.17%** | 0.4797 | 0.3257 | **0.3158** |

**Domain Shift Collapse:** CNN v1 experiences an unprecedented **$66.6\%$ accuracy drop** ($99.51\% \to 32.88\%$). Saliency maps (Grad-CAM) reveal that the network relies on background soil, lighting, and edge artifacts rather than purely foliar lesions, exposing severe vulnerability to real-world domain shifts.
