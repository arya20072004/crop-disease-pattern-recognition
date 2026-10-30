# Phase 1: Dataset Protocol and Handcrafted Feature Engineering Foundation

**Repository:** `crop-disease-pattern-recognition`  
**Phase:** 1 (Foundation)  
**Status:** Completed & Authoritative  

---

## 1. Objective
Establish an immutable evaluation foundation, dataset partitioning protocol, and handcrafted feature extraction pipeline for the Pattern Recognition CCA 1 comparative case study.

---

## 2. Dataset Protocol & Manifests
- **PlantVillage Partitioning:**
  - Total Samples: 54,305 images across 38 classes.
  - Stratified Split (Seed 42): Train (70%, 38,013 images), Validation (15%, 8,146 images), Test (15%, 8,146 images).
  - Manifest: `data/manifests/plantvillage_val.json` ($N=8,146$).
- **PlantDoc Reconciliation:**
  - 25 mapped classes ($N=219$ images).
  - 2 excluded classes (17 images).
  - 11 train-test duplicate pairs flagged (3 consistent, 8 conflicting).
  - Manifest: `data/manifests/plantdoc_test.json`.

---

## 3. Handcrafted Feature Representation (8,196 Dimensions)
1. **HSV Colour Histograms:** 96 dimensions (32 bins per channel, L1-normalized).
2. **HOG Gradients:** 8,100 dimensions ($128\times 128$ window, $8\times 8$ cell, $16\times 16$ block, $8\times 8$ stride, 9 orientation bins).
3. **Total Feature Dimensionality:** Exactly $8,196$ continuous dimensions.
4. **Train-Only Standardization:** `FeatureScaler` fit exclusively on the 9,299 training subset to prevent data leakage.
