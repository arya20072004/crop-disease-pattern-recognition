# Known Limitations & Threats to Validity

## 1. Real-World Field Baseline Incompleteness
- **Scope Limitation:** Prediction-level evaluation on the PlantDoc field dataset was conducted exclusively for EfficientNet-B4 CNN v1.
- **Traditional Models:** Prediction-level records and evaluated metrics for Linear SVM and Random Forest on PlantDoc **do not exist**. Consequently, comparative claims regarding the relative generalization ability of traditional models in field environments are strictly withheld to maintain scientific integrity.

---

## 2. Laboratory Dataset Confounders
- **Shortcut Learning:** PlantVillage leaves were detached and photographed against plain grey or black backing paper under studio lighting.
- **Confounding Effect:** As demonstrated by Grad-CAM saliency visualizations, deep neural networks easily learn background and border correlations, causing significant performance degradation when transferred to in-the-wild agricultural environments.

---

## 3. Computational Benchmark Specificity
- **Hardware Architecture:** All latency measurements were recorded on an Intel 13th Gen Core i5-13450HX mobile workstation processor with 10 PyTorch worker threads and an NVIDIA RTX 4050 Laptop GPU.
- **Architecture Differences:** While relative scaling behaviors and intra-iteration arithmetic hold universally, absolute latencies will differ on low-power ARM architectures (e.g., Raspberry Pi, Apple Silicon, Qualcomm Snapdragon).

---

## 4. Handcrafted Feature Subspace
- The 8,196-dimensional handcrafted feature space represents a fixed engineering choice (HSV color histograms + HOG spatial gradients). While effective, it does not incorporate color covariance, local binary patterns (LBP), or multi-scale wavelet representations.
