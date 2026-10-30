# Phase 2C: Nonlinear SVM Using Nystroem RBF Kernel Approximation (Ablation)

**Repository:** `crop-disease-pattern-recognition`  
**Phase:** 2C (Nonlinear Kernel Approximation Ablation)  
**Status:** Completed Additional Ablation Experiment  

---

## 1. Objective & Role in Study
Evaluate whether nonlinear kernel mapping improves classical classification accuracy over linear hyperplanes.

> [!NOTE]
> As established in the experimental charter, the Nystroem RBF SVM serves strictly as an **exploratory ablation**. It does not replace the primary Linear SVM reference model.

---

## 2. Architecture & Pipeline
1. Handcrafted Features: 8,196-dimensional HSV + HOG vector.
2. `FeatureScaler`: Fitted on training features only.
3. `Nystroem`: RBF kernel approximation ($n_{\text{components}} = 500$, random_state=42).
4. `SGDClassifier`: Hinge loss, L2 regularisation, $N=9,299$ training subset.

---

## 3. Experimental Results ($N = 8,146$)

| Kernel Approximation | Gamma ($\gamma$) | Regularisation ($C$) | Validation Accuracy | Validation Macro-F1 | Validation Weighted-F1 | Notes |
|---|---:|---:|---:|---:|---:|---|
| Linear SVM (Phase 2B) | N/A | 10.0 | 64.08% | 0.5961 | 0.6496 | Linear baseline |
| Nystroem RBF SVM | Scale | 1.0 | 48.70% | 0.4120 | 0.4910 | Underfitting |
| Nystroem RBF SVM | $1.0 \times 10^{-5}$ | 1.0 | 62.10% | 0.5740 | 0.6280 | Linear approximation |
| **Nystroem RBF SVM (Best)** | **$1.0 \times 10^{-3}$** | **1.0** | **65.54%** | **0.6120** | **0.6620** | $+1.59\%$ Macro-F1 over Linear |

**Key Finding:** Nonlinear kernel approximation provides a modest $+1.59\%$ Macro-F1 improvement ($0.5961 \to 0.6120$) at the cost of projection overhead and storage expansion ($500$-dim intermediate representation).
