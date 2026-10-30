# Phase 2B: Optimised Linear SVM Using SGDClassifier

**Repository:** `crop-disease-pattern-recognition`  
**Phase:** 2B (Linear SVM Implementation & Evaluation)  
**Status:** Completed & Accepted Reference Model  

---

## 1. Objective
Implement and evaluate an efficient, reproducible linear hinge-loss Support Vector Machine across the 38-class PlantVillage dataset using the 8,196-dimensional handcrafted feature representation.

---

## 2. Mathematical Formulation & Regularisation Mapping
The primal linear SVM problem is solved using `SGDClassifier` with hinge loss and L2 penalty.

Equivalence formula:
$$\alpha = \frac{1}{N \cdot C} \iff C = \frac{1}{N \cdot \alpha}$$

Where $N = 9,299$ training samples:
- $C = 0.1 \implies \alpha = 1.075 \times 10^{-3}$
- $C = 1.0 \implies \alpha = 1.075 \times 10^{-4}$
- $C = 10.0 \implies \alpha = 1.075 \times 10^{-5}$

---

## 3. Experimental Validation Results ($N = 8,146$)

| Configuration | $C$ Equivalent | $\alpha$ | Convergence | Training Time | Validation Accuracy | Validation Macro-F1 | Validation Weighted-F1 |
|---|---:|---:|---|---:|---:|---:|---:|
| Configuration 1 | 0.1 | $1.075 \times 10^{-3}$ | True (254 iters) | 60.75 s | 63.39% | 0.5882 | 0.6420 |
| Configuration 2 | 1.0 | $1.075 \times 10^{-4}$ | True (229 iters) | 52.63 s | 63.56% | 0.5899 | 0.6444 |
| **Configuration 3 (Best)** | **10.0** | **$1.075 \times 10^{-5}$** | **True (133 iters)** | **46.44 s** | **64.08%** | **0.5961** | **0.6496** |

**Selection:** Configuration 3 ($C=10.0$) selected as the primary reference model based on highest validation Macro-F1 ($0.5961$).
- Model storage: 1.19 MB (`models/weights/best_svm.joblib`).
- Scaler storage: 0.19 MB (`models/weights/svm_scaler.joblib`).
