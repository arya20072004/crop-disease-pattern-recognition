# Experimental Methodology & Protocol

## 1. Experimental Design Framework
To guarantee scientific comparability and eliminate confounding factors, all three paradigms adhered to a unified experimental protocol established during Phase 1:

```
+---------------------------------------------------------------------------------------------------+
|                                  STANDARDIZED EVALUATION PROTOCOL                                 |
+------------------------------+----------------------------------+---------------------------------+
| Dimension                    | Classical Models (SVM & RF)      | Deep CNN (EfficientNet-B4)      |
+------------------------------+----------------------------------+---------------------------------+
| Input Modality               | Decoded RGB Image (uint8)        | Decoded RGB Image (uint8)       |
| Preprocessing Pipeline       | 1. Resize & uint8 decode         | 1. Resize to 380x380            |
|                              | 2. HSV Histogram (96 dims)       | 2. Float32 normalization (0-1)  |
|                              | 3. HOG Gradients (8,100 dims)    | 3. ImageNet mean/std scaling    |
|                              | 4. Train-only StandardScaler     | 4. CHW tensor transposition     |
| Feature Representation       | 8,196-dimensional fixed vector   | 1,792-dim deep spatial features |
| Training Sample Protocol     | 9,299 PlantVillage subset        | 38,013 full PlantVillage train  |
| Validation Sample Protocol   | 8,146 PlantVillage samples       | 8,146 PlantVillage samples      |
| Number of Output Classes     | 38 Classes                       | 38 Classes                      |
| Inference Timing Harness     | Paired intra-iteration (v2)      | Paired intra-iteration (v2)     |
+------------------------------+----------------------------------+---------------------------------+
```

---

## 2. Feature Extraction Pipeline (8,196 Dimensions)

### 2.1 HSV Colour Histograms (96 Dimensions)
- **Rationale:** Foliar fungal and bacterial diseases manifest as distinct visual discolourations (chlorosis, necrotic brown/black lesions, powdery white mildew). The HSV (Hue, Saturation, Value) color space separates chromaticity from luminance, providing illumination invariance.
- **Formulation:** 32 bins per channel:
  - Hue: $[0, 180]$
  - Saturation: $[0, 256]$
  - Value: $[0, 256]$
- **Normalization:** Each 32-bin channel histogram is L1-normalized ($\sum h_i = 1.0$), ensuring that image resolution does not alter relative color representation.

### 2.2 Histogram of Oriented Gradients (8,100 Dimensions)
- **Rationale:** Captures structural contours, lesion margin sharpness, vein degradation, and spot morphology.
- **Standardized Window Parameters:**
  - Window Size: $128 \times 128$ pixels
  - Orientation Bins: 9 bins ($0^\circ$ to $180^\circ$)
  - Cell Size: $8 \times 8$ pixels ($16 \times 16$ cells across image)
  - Block Size: $16 \times 16$ pixels ($2 \times 2$ cells per block, 36 values per block)
  - Block Stride: $8 \times 8$ pixels ($50\%$ spatial overlap)
- **Block Layout Arithmetic:**
  - Number of horizontal blocks: $(128 - 16) / 8 + 1 = 15$
  - Number of vertical blocks: $(128 - 16) / 8 + 1 = 15$
  - Total Blocks: $15 \times 15 = 225$ blocks
  - Feature Dimensions: $225 \times 36 = 8,100$ features.

---

## 3. Strict Train-Only Feature Standardization
Data leakage was rigorously prevented using an enforced `FeatureScaler`:
- $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ were computed **exclusively** on the $9,299$ training subset.
- Validation ($N = 8,146$) and test data were transformed strictly using the stored training statistics:
  $$x' = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}} + \epsilon}$$
- Scaler weights are persisted in `models/weights/svm_scaler.joblib`.

---

## 4. Paired Intra-Iteration Computational Benchmarking
To eliminate thermal downclocking artifacts (Intel PL2-to-PL1 power throttling) and Joblib worker contention, benchmarking follows the strict paired intra-iteration protocol:

```python
t0 = time.perf_counter()
preprocessed = preprocess_fn(image)
t1 = time.perf_counter()
prediction = model_predict_fn(preprocessed)
if is_gpu:
    torch.cuda.synchronize()
t2 = time.perf_counter()

preproc_ms = (t1 - t0) * 1000.0
model_ms   = (t2 - t1) * 1000.0
e2e_ms     = (t2 - t0) * 1000.0
```

This guarantees $\text{end\_to\_end} \equiv \text{preprocessing} + \text{model\_only}$ strictly on every sample.
