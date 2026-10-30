# Project Overview: Crop Disease Identification

## 1. Academic Context & Motivation
Crop diseases represent a persistent threat to global food security, causing an estimated 20–40% yield loss across major agricultural staples annually. Smallholder farmers often lack immediate access to certified agronomists or plant pathologists, resulting in delayed interventions, inappropriate pesticide applications, or total harvest loss.

The **Pattern Recognition CCA 1 Case Study** evaluates automated computer vision and statistical pattern recognition approaches to classify crop diseases across **38 distinct categories** covering 14 economic crops (including apple, corn, grape, potato, tomato, and soybean) under both laboratory and field conditions.

---

## 2. Research Objectives
The central objective of this research is to perform an empirical, scientifically controlled comparison across three foundational pattern recognition paradigms:

1. **Linear Discriminative Classification (Linear SVM):** Convex empirical risk minimization using SGD-optimized hinge loss operating on explicit handcrafted color and texture representations.
2. **Non-Linear Ensemble Decision Boundaries (Random Forest):** Non-parametric partitioning via randomized orthogonal decision trees capable of capturing high-order non-linear feature interactions.
3. **Deep Representation Learning (Convolutional Neural Networks):** End-to-end hierarchical visual feature extraction using an ImageNet-pretrained EfficientNet-B4 architecture.

---

## 3. Comparative Research Questions
- **RQ1 (Representation Fidelity):** How much discriminative accuracy is lost when substituting high-capacity deep convolutional features with fixed 8,196-dimensional handcrafted color-texture descriptors?
- **RQ2 (Computational Efficiency & Edge Feasibility):** Can classical pattern recognition models provide viable low-latency inference on energy-constrained agricultural edge devices (e.g., IoT microcontrollers, field scouts) where deep neural networks encounter compute bottlenecks?
- **RQ3 (Real-World Domain Generalization):** Does near-perfect laboratory validation performance ($>99\%$) survive when models are evaluated on unconstrained in-the-wild agricultural imagery (PlantDoc) containing natural backgrounds, shadows, and occlusions?
