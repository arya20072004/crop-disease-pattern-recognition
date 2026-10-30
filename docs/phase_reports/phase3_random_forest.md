# Phase 3: Random Forest Implementation and Evaluation

**Repository:** `crop-disease-pattern-recognition`  
**Phase:** 3 (Random Forest Implementation & Evaluation)  
**Status:** Completed & Accepted Reference Model  

---

## 1. Objective
Implement and evaluate Random Forest as the second required classical machine learning approach for the Pattern Recognition CCA 1 project, utilizing the identical 8,196-dimensional handcrafted feature representation.

---

## 2. Experimental Configurations Evaluated

| Configuration | Trees ($n_{\text{estimators}}$) | Max Depth | Split / Leaf | Max Features | Training Time | Validation Accuracy | Validation Macro-F1 |
|---|---:|---:|---|---|---:|---:|---:|
| RF-1 | 200 | None | 2 / 1 | sqrt | 148.6 s | 81.25% | 0.7648 |
| **RF-2 (Best)** | **300** | **30** | **2 / 1** | **sqrt** | **197.2 s** | **81.46%** | **0.7674** |
| RF-3 | 300 | 50 | 2 / 1 | sqrt | 212.4 s | 81.38% | 0.7661 |
| RF-4 | 300 | None | 5 / 2 | sqrt | 184.1 s | 80.91% | 0.7612 |

**Model Selection:** RF-2 selected as primary model based on highest validation Macro-F1 ($0.7674$).
- Storage: 489.28 MB (`best_random_forest.joblib`).
- Tree Structure: 300 trees, 1,393,582 total decision nodes, average 4,645 nodes per tree.

---

## 3. Gini Feature Importance Breakdown
- **Aggregate Modality Contribution:**
  - HSV Colour (96 features): $19.78\%$ total importance.
  - HOG Shape (8,100 features): $80.22\%$ total importance.
- **Mean Feature Efficiency:**
  - HSV: $2.06 \times 10^{-3}$ per feature.
  - HOG: $0.10 \times 10^{-3}$ per feature.
  - HSV features are **$20.8\times$ more influential** per feature on tree split decisions.
- **Top 20 Features:** 100% of the top 20 individual features are HSV color dimensions (Ranks 1–6 are Hue bins #3, #2, #1, #5, #0, #4).
