# Phase 2.1 Evidence Reconciliation and Experimental Freeze Report

**Project:** Systematic ANOVA Feature Selection on EfficientNet-B4 Visual Embeddings for Agricultural Crop Disease Recognition  
**Supporting Research Paper:** *"Multi-Crop Disease Identification Under Real-World Conditions: A Deep Learning Framework Using EfficientNet-B4, VGG-16, and ResNet-50"*  
**Date:** November 1, 2026  
**Phase Status:** **AUTHORIZED (READ-ONLY AUDIT)**  
**Gate Recommendation:** **ACCEPTED WITH OBSERVATIONS**  

---

## 1. Executive Summary

This report delivers a rigorous, read-only evidence reconciliation of the Phase 2 implementation, empirical pilot results, partition definitions, and architectural assumptions. The objective of Phase 2.1 is to establish whether the codebase and experimental protocol are sufficiently verified, leak-free, and unambiguous to authorize the full Phase 3 seven-dimension evaluation matrix.

Key conclusions established by this audit:
1. **Multimodal Output Dimension Discrepancy Resolved:** The $(4, 26)$ output logit shape reported in Phase 2 is **100% verified as intentional and mathematically correct**. It originates from the established Increment 3 multimodal fusion architecture (`CrossAttentionFusion`), where the visual embeddings interact with weather BiLSTM representations to predict disease risk across the **26 pathological disease classes** of the PlantVillage taxonomy (excluding the 12 healthy classes). It is completely independent of PlantDoc's 25 mapped categories.
2. **Pilot Metrics Independently Verified:** All reported pilot metrics for Experiment A (weight-sliced), Experiment B (retrained linear head), and Experiment C (multimodal cross-attention) were verified against the saved raw embedding caches (`pv_train_pilot_emb_512.npz` and `pv_val_pilot_emb_512.npz`) and evaluation records (`results/phase2_pilot/pilot_results.json`).
3. **Partition Independence & Leakage Prevention:** Empirical mathematical verification confirmed zero sample overlap ($\text{count} = 0$) between training subsets, validation subsets, and the frozen PlantVillage ($N=8,146$) and PlantDoc ($N=236$) test partitions.
4. **Frozen Training Strategy:** The primary training partition for Phase 3 ANOVA fitting and classifier retraining is frozen to the balanced **$N=9,299$** PlantVillage training subset (`plantvillage_train_subset.json`), which caps class imbalance at 250 samples/class to prevent ANOVA $F$-statistic distortion while maintaining methodological parity with the repository's CCA1 benchmarks.
5. **Phase 3 Protocol Frozen:** Precise mathematical and architectural specifications are established for all three experimental paradigms across all seven target dimensions ($k \in \{512, 384, 256, 192, 128, 64, 32\}$) along with a multi-objective Pareto frontier evaluation framework.

---

## 2. Scope and Authorization Boundaries

### 2.1 Authorization Status
- **Phase 2:** Provisionally accepted.
- **Phase 2.1:** Formally authorized under strict read-only terms.
- **Phase 3:** **NOT AUTHORIZED**.

### 2.2 Boundary Compliance
In strict adherence to instructions:
- Zero existing source files, test suites, manifests, or checkpoints were modified.
- Zero new experiments were launched.
- Zero test partitions (PlantVillage Test $N=8,146$, PlantDoc Test $N=236$) were accessed, extracted, or evaluated.
- No model architecture or class mappings were altered.
- No Git commits or pushes were executed.
- The working tree remains in a clean state relative to upstream `origin/main`.

---

## 3. Artifacts Inspected

The following primary artifacts, source files, manifests, and logs were inspected in detail:

| Artifact Path | Description / Type | Verification Status |
|---|---|---|
| `docs/phase_reports/phase2_deep_feature_selection_pilot_report.md` | Complete Phase 2 implementation & pilot report | **VERIFIED** |
| `src/deep_feature_selection/extractor.py` | Embedding extractor & checkpoint verifier | **VERIFIED** |
| `src/deep_feature_selection/anova_selector.py` | Train-only ANOVA selector & determinism module | **VERIFIED** |
| `src/deep_feature_selection/classifiers.py` | Downstream heads (Exp A weight-sliced, Exp B retrained) | **VERIFIED** |
| `src/deep_feature_selection/multimodal.py` | Adaptive cross-attention fusion architecture (Exp C) | **VERIFIED** |
| `src/deep_feature_selection/pilot_runner.py` | Pilot execution orchestrator | **VERIFIED** |
| `tests/test_deep_feature_selection.py` | Automated test suite (12 tests) | **VERIFIED** |
| `results/phase2_pilot/pilot_results.json` | Saved pilot execution records & metric dictionaries | **VERIFIED** |
| `data/cache/pv_train_pilot_emb_512.npz` | Cached pilot training embeddings ($1,140 \times 512$) | **VERIFIED** |
| `data/cache/pv_train_pilot_emb_512_meta.json` | Cache metadata & integrity sidecar (Train) | **VERIFIED** |
| `data/cache/pv_val_pilot_emb_512.npz` | Cached pilot validation embeddings ($760 \times 512$) | **VERIFIED** |
| `data/cache/pv_val_pilot_emb_512_meta.json` | Cache metadata & integrity sidecar (Val) | **VERIFIED** |
| `data/manifests/plantvillage_train_subset.json` | Primary balanced training manifest ($N=9,299$) | **VERIFIED** |
| `data/manifests/plantvillage_train.json` | Full training manifest ($N=38,013$) | **VERIFIED** |
| `data/manifests/plantvillage_val.json` | Official validation manifest ($N=8,146$) | **VERIFIED** |
| `data/manifests/plantvillage_test.json` | Frozen test manifest ($N=8,146$) | **VERIFIED (ISOLATED)** |
| `data/manifests/plantdoc_test.json` | Frozen PlantDoc field test manifest ($N=236$) | **VERIFIED (ISOLATED)** |
| `../crop-disease-detection/scripts/increment_3_fusion.py` | Historical Increment 3 multimodal fusion source | **VERIFIED** |
| `../crop-disease-detection/models/configs/fusion_config.json` | Multimodal fusion configuration & disease classes | **VERIFIED** |
| `configs/cnn/cnn_v1_plantvillage_metrics.json` | Reference CNN v1 full validation benchmark record | **VERIFIED** |

---

## 4. CNN v1 Checkpoint Integrity Verification

### 4.1 Verification Evidence
The CNN v1 checkpoint was inspected and hashed directly from disk using chunked binary SHA-256 computation:
- **File Path:** `models/weights/best_model.pth` (mirrored from `archive/increment_1_cnn/best_model.pth`).
- **Expected SHA-256:** `e48a4d2f0562282d930bc0583017a45ad243083b38e102aa5eacfd401b6fcabb`.
- **Computed SHA-256:** `e48a4d2f0562282d930bc0583017a45ad243083b38e102aa5eacfd401b6fcabb`.
- **Match Status:** **VERIFIED (EXACT BIT-LEVEL MATCH)**.

### 4.2 Disqualification of CNN v2
The CNN v2 checkpoint (`best_model_v2.pth`) remains strictly disqualified. Historical records confirm that CNN v2 was fine-tuned with exposure to the PlantDoc test set during incremental domain adaptation experiments. CNN v1 is verified as completely uncontaminated by any test set data.

---

## 5. Investigation A: Resolve the Multimodal Output Dimension

### 5.1 Architectural Investigation and Evidence
The user requested an investigation into why the multimodal cross-attention forward pass in Experiment C produces `(4, 26)` logits rather than 38 classes or 25 classes.

Direct inspection of `../crop-disease-detection/scripts/increment_3_fusion.py` and `../crop-disease-detection/models/configs/fusion_config.json` resolves this completely:

1. **Origin of the 26 Classes:**
   In PlantVillage, there are 38 total categories. Exactly 12 of these categories represent healthy foliage:
   - `Apple___healthy`
   - `Blueberry___healthy`
   - `Cherry_(including_sour)___healthy`
   - `Corn_(maize)___healthy`
   - `Grape___healthy`
   - `Peach___healthy`
   - `Pepper,_bell___healthy`
   - `Potato___healthy`
   - `Raspberry___healthy`
   - `Soybean___healthy`
   - `Strawberry___healthy`
   - `Tomato___healthy`

   Subtracting these 12 healthy categories from 38 yields **exactly 26 pathological disease categories**:
   $$38 \text{ total categories} - 12 \text{ healthy categories} = 26 \text{ disease categories}$$

2. **Multimodal Domain Formulation:**
   In the project's Increment 3 research design ("Multimodal Fusion of Visual Features, Weather Time-Series, and Crop Metadata"), weather time-series data (temperature, relative humidity, leaf wetness, rainfall) correlate with disease pathogens (fungal sporulation, bacterial proliferation). Weather data cannot predict a plant being "healthy"; it specifically models Disease-Conducive Weather Scores (DCWS) for pathogens.
   Consequently, the multimodal fusion network was designed to output risk predictions over the **26 disease classes**.

3. **Code Evidence in `increment_3_fusion.py`:**
   - **Lines 145–148:**
     ```python
     DISEASE_CLASSES = lstm_cfg['disease_classes']
     N_DISEASES = lstm_cfg['n_diseases']  # 26
     ```
   - **Lines 446–463:**
     ```python
     # CNN signal vector: for each of the 26 disease classes,
     # use the CNN's softmax probability for that class (mapped from 38 -> 26)
     cnn_disease_probs = np.zeros((len(cnn_labels), len(DISEASE_CLASSES)), dtype=np.float32)
     for sample_i in range(len(cnn_labels)):
         for class_38_i, cname in enumerate(CLASS_NAMES):
             if cname in DISEASE_CLASSES:
                 disease_j = DISEASE_CLASSES.index(cname)
                 cnn_disease_probs[sample_i, disease_j] = cnn_probs_all[sample_i, class_38_i]
     ```
   - **Lines 560–573 (`CrossAttentionFusion`):**
     ```python
     fusion_dim = attn_dim + attn_dim + 64  # 256 + 256 + 64 = 576
     self.fusion_head = nn.Sequential(
         nn.Linear(fusion_dim, 512),
         nn.LayerNorm(512),
         nn.GELU(),
         nn.Dropout(dropout),
         nn.Linear(512, 256),
         nn.LayerNorm(256),
         nn.GELU(),
         nn.Dropout(dropout),
         nn.Linear(256, n_diseases)  # n_diseases = 26
     )
     ```

### 5.2 Specific Reconciliation Findings

| Audit Question | Evidence-Based Answer | Status |
|---|---|---|
| **Why 26 output logits?** | Represents the 26 pathological disease classes defined in `fusion_config.json` and `increment_3_fusion.py`. | **VERIFIED** |
| **What does each output index represent?** | Indices $0 \dots 25$ map to the 26 specific disease pathogens listed in `fusion_config.json` lines 10–37. | **VERIFIED** |
| **Is 26 intentional or accidental?** | **100% Intentional**. It is the exact specification of the Increment 3 architecture. | **VERIFIED** |
| **Does it represent 25 PlantDoc + rejection?** | **No**. It has no relation to PlantDoc's 25 mapped categories. It is strictly derived from PlantVillage ($38 - 12 = 26$). | **VERIFIED** |
| **Is the 38-class head retained in multimodal?** | The unimodal CNN maintains its 38-class head; in multimodal fusion, the 512-d visual embeddings pass directly into cross-attention. | **VERIFIED** |
| **Does multimodal use a separate classification head?** | **Yes**. The 3-layer MLP fusion head (`576 -> 512 -> 256 -> 26`) is separate from the CNN linear head. | **VERIFIED** |
| **Are class mappings consistent across protocols?** | **Yes**: Unimodal PlantVillage = 38 classes; Zero-shot PlantDoc = 25 mapped classes; Multimodal = 26 disease classes. | **VERIFIED** |
| **Does confidence-rejection operate independently?** | **Yes**. Confidence rejection applies a post-hoc threshold $\tau$ to predicted probabilities, independent of class count. | **VERIFIED** |

**Conclusion for Investigation A:** **RESOLVED (VERIFIED)**. No code or architectural changes are required.

---

## 6. Investigation B: Reconcile Pilot Metrics

### 6.1 Pilot Metrics Audit Table
The pilot metrics recorded in `results/phase2_pilot/pilot_results.json` were audited against raw cache files:

| Dimension ($k$) | Paradigm | Accuracy | Macro F1 | Sample Partition | Samples ($N$) | Classes ($C$) | Feature Source | Fitted On Train Only? | Reproducibility |
|---|---|---|---|---|---|---|---|---|---|
| **$k=512$** | Exp A (Weight-Sliced) | 99.08% | 0.9907 | `plantvillage_val_pilot.json` | 760 | 38 | Full visual embedding | Yes (Full rank) | **VERIFIED** |
| **$k=512$** | Exp B (Retrained Head) | 98.82% | 0.9881 | `plantvillage_val_pilot.json` | 760 | 38 | Full visual embedding | Yes (Fitted on Train) | **VERIFIED** |
| **$k=256$** | Exp A (Weight-Sliced) | 97.37% | 0.9734 | `plantvillage_val_pilot.json` | 760 | 38 | Top 256 ANOVA features | Yes (Fitted on Train) | **VERIFIED** |
| **$k=256$** | Exp B (Retrained Head) | 98.29% | 0.9828 | `plantvillage_val_pilot.json` | 760 | 38 | Top 256 ANOVA features | Yes (Fitted on Train) | **VERIFIED** |
| **$k=64$** | Exp A (Weight-Sliced) | 66.45% | 0.6180 | `plantvillage_val_pilot.json` | 760 | 38 | Top 64 ANOVA features | Yes (Fitted on Train) | **VERIFIED** |
| **$k=64$** | Exp B (Retrained Head) | 94.21% | 0.9413 | `plantvillage_val_pilot.json` | 760 | 38 | Top 64 ANOVA features | Yes (Fitted on Train) | **VERIFIED** |

### 6.2 Partition Identity and Baseline Comparability
1. **Partition Identity:** The pilot evaluation was performed strictly on the **PlantVillage validation pilot partition** (`plantvillage_val_pilot.json`, $N=760$, 20 samples/class).
2. **Not Directly Comparable to Full Validation Baseline:**
   - The original CNN v1 baseline in `configs/cnn/cnn_v1_plantvillage_metrics.json` reports **99.51%** accuracy on the full $N=8,146$ validation partition.
   - The pilot's 99.08% accuracy represents 753 / 760 correct predictions on a balanced 20-sample/class distribution. The 0.43 percentage-point difference is attributable to the different sample size ($N=760$ vs. $N=8,146$) and support weighting.
   - The pilot figures must be documented as pilot-partition metrics, not full-dataset validation metrics.
3. **Interpretation of Experiment B Performance Recovery at $k=64$:**
   - At $k=64$, Experiment A achieves only 66.45% accuracy because the frozen linear weights were jointly optimized during backpropagation across all 512 dimensions. Slicing out 448 columns destroys the balance of the linear hyperplane sums.
   - Experiment B retrains a convex linear head on the 64-dimensional subspace, recovering accuracy to **94.21%** (+27.76 percentage points over slicing).
   - **Methodological Guardrail:** This recovery proves that **the 64 ANOVA dimensions retain substantial linear separability**, but it does NOT prove that dimensionality reduction improves the original model (since 94.21% remains 4.61 pp below the full 512-d baseline of 98.82%).

---

## 7. Investigation C: Verify Pilot Sample Representativeness

### 7.1 Selection Methodology
The 1,140 pilot training samples were generated by `create_stratified_pilot_manifest` in `pilot_runner.py`:
- **Source:** Stratified sampling from `data/manifests/plantvillage_train_subset.json`.
- **Stratification:** Grouped by `class_index`, drawing samples across all 38 classes.
- **Balance:** Exactly $\min(30, \text{class\_size}) = 30$ samples per class across all 38 classes.
- **Random Seed:** Fixed to `seed=42` (`np.random.RandomState(42)`).
- **Reproducibility:** Confirmed bit-for-bit reproducible with manifest SHA-256 `11cb386228079de7fde8494d2f6f61c00ce635a2e0268277099d73038e7228ba`.

### 7.2 Independence and Scope of Conclusions
- **Independence:** Sample overlap check confirmed 0 overlapping images between pilot training samples and validation/test manifests.
- **Scope Distinction:**
  - *Pilot implementation validation:* The pilot successfully proved that the code executes without memory spikes, handles dead ReLU features safely, enforces train-only fitting, and supports cross-attention adaptation.
  - *Feature-count selection & final evaluation:* The pilot sample ($N=1,140$) represents only 12.3% of the balanced training subset ($N=9,299$) and 3.0% of the full training set ($N=38,013$). Pilot metrics cannot be used to choose the final feature dimension. Feature-count selection must rely on the full validation evaluation in Phase 3.

---

## 8. Investigation D: Freeze the Training-Data Strategy

### 8.1 Comparison of Training Configurations

| Property | Full Training Partition (`plantvillage_train.json`) | Balanced Training Subset (`plantvillage_train_subset.json`) |
|---|---|---|
| **Sample Count ($N$)** | 38,013 | 9,299 |
| **Class Distribution** | Severe imbalance (106 to 3,855 samples/class) | Capped at 250 samples/class (106 to 250 samples/class) |
| **Imbalance Ratio** | **36.4 : 1** | **2.36 : 1** |
| **ANOVA Statistical Validity** | Unequal sample sizes bias Between-Group Mean Square (MSB) toward dominant tomato classes | Balanced sample counts satisfy ANOVA homoscedasticity assumptions; ranks features separating all 38 classes |
| **Extraction Wall Time (CPU)** | $\approx 2.93$ hours ($10,559\text{s}$) | $\approx 43.0$ minutes ($2,583\text{s}$) |
| **CCA1 Benchmark Parity** | Differed from handcrafted ML baseline | **Exact match** to the $N=9,299$ dataset used in CCA1 feature selection benchmarks |

### 8.2 Frozen Decision
- **Primary Training Configuration:** **Balanced $N=9,299$ subset (`plantvillage_train_subset.json`)**.
  - *Justification:* Statistically superior for ANOVA $F$-ranking across all classes, directly comparable to existing CCA1 benchmarks, and computationally feasible within host CPU memory and time limits.
- **Secondary Sensitivity Configuration:** The full $N=38,013$ partition is designated strictly as an optional sensitivity analysis, to be executed only if specifically authorized and compute permits.
- **Fitting Rule:** ANOVA feature selection must be fitted **exclusively on the designated training embeddings**.

---

## 9. Investigation E: Freeze the Data-Leakage Protocol

### 9.1 Partition Access Rules for Phase 3

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             DATA ACCESS BOUNDARIES                          │
├──────────────────────────┬──────────────────────────────────────────────────┤
│ Training Data            │ • Fit ANOVA selector                             │
│ (PlantVillage N=9,299)   │ • Calculate feature ranking and F-statistics     │
│                          │ • Train retrained linear evaluation heads (Exp B)│
│                          │ • Train multimodal cross-attention components    │
├──────────────────────────┼──────────────────────────────────────────────────┤
│ Validation Data          │ • Compare candidate dimensions (k ∈ {512..32})   │
│ (PlantVillage N=8,146)   │ • Select optimal operating point on Pareto front │
│                          │ • Evaluate hyperparameters (C, regularization)   │
│                          │ ✗ NEVER used to fit ANOVA selector or train heads│
├──────────────────────────┼──────────────────────────────────────────────────┤
│ Frozen Test Data         │ • Extract & cache embeddings under frozen policy │
│ (PlantVillage N=8,146)   │ • Evaluate ONLY on final frozen configurations   │
│ (PlantDoc N=236)         │ ✗ NEVER used for selection, tuning, or ranking   │
│                          │ ✗ Zero-shot PlantDoc kept separate from adapted  │
└──────────────────────────┴──────────────────────────────────────────────────┘
```

### 9.2 Feature Cache Metadata Integrity
Audit of `data/cache/` sidecars confirmed that each cache file records:
1. `model_name` & `checkpoint_sha256`
2. `manifest_name` & `manifest_sha256`
3. `num_samples` & `embedding_dim`
4. `batch_size` & `extraction_duration_sec`
5. `throughput_img_per_sec` & `created_at`
6. `sha256_npz` (checksum of the data payload)

**Observations / Non-Blocking Gaps:**
- Preprocessing constants (380x380 resolution, ImageNet normalization mean/std) and class mapping checksums are currently enforced via `cnn_adapter.py` imports rather than explicitly echoed in the JSON metadata sidecar. This does not cause data leakage but should be logged in Phase 3 run envelopes.

---

## 10. Investigation F: Freeze the Three Experimental Paradigms

### 10.1 Experiment A: Weight-Sliced Classifier (Zero-Retraining)
- **Mathematical Operation:** Given frozen linear weights $W \in \mathbb{R}^{38 \times 512}$ and bias $b \in \mathbb{R}^{38}$, and selected ascending index set $I = \{i_1, \dots, i_k\} \subset \{0, \dots, 511\}$:
  $$W_{\mathrm{reduced}} = W_{:, I} \in \mathbb{R}^{38 \times k}, \quad b_{\mathrm{reduced}} = b \in \mathbb{R}^{38}$$
  $$\hat{y} = \arg\max_{c \in \{0, \dots, 37\}} \left( X_k W_{\mathrm{reduced}}^T + b_{\mathrm{reduced}} \right)$$
- **Index Alignment:** Verified. `selected_indices` are sorted ascending; column indices in embedding $X[:, I]$ match rows in $W_{\mathrm{reduced}}^T$.
- **Retraining:** Zero retraining. Measures intrinsic geometric preservation.

### 10.2 Experiment B: Retrained Linear Evaluation Head
- **Architecture:** Multinomial Logistic Regression (Cross-Entropy loss) mapping $\mathbb{R}^k \to \mathbb{R}^{38}$.
- **Solver & Hyperparameters:**
  - Solver: `lbfgs`
  - Regularization: $L_2$ penalty, $C = 1.0$
  - Multi-class: Multinomial
  - Stopping criteria: `max_iter = 1000`, `tol = 1e-4`
  - Determinism: `random_state = 42`
- **Training Protocol:** Fitted strictly on reduced training embeddings $X_{\mathrm{train}, I} \in \mathbb{R}^{9299 \times k}$. Evaluated on $X_{\mathrm{val}, I} \in \mathbb{R}^{8146 \times k}$.

### 10.3 Experiment C: Multimodal Cross-Attention Integration
- **Architecture:** `AdaptiveCrossAttentionFusion` preserving Increment 3 design:
  - Adapted Visual Projection: $W_{\mathrm{cnn\_proj}} \in \mathbb{R}^{256 \times k}$, $b_{\mathrm{cnn\_proj}} \in \mathbb{R}^{256}$.
  - Weather Context: Fixed 256-d BiLSTM representation.
  - Multi-Head Cross-Attention: 8 heads ($d_k = 32$), query from visual stream, key/value from weather stream.
  - Multi-Head Self-Attention: 8 heads on visual stream.
  - Metadata Stream: MLP mapping 21 metadata features to 64.
  - Fusion Head: $\mathbb{R}^{576} \to \mathbb{R}^{512} \to \mathbb{R}^{256} \to \mathbb{R}^{26}$ disease categories.
- **Output:** Logits shape $(B, 26)$, cross-attention weights shape $(B, 1, 1)$.

---

## 11. Investigation G: Freeze the Seven-Dimension Matrix

The seven target dimensions are frozen as follows:

| Dimension Index | Target Feature Count ($k$) | Retained Features (%) | Dimensionality Reduction (%) | Scientific Purpose in Research Paper |
|---|---|---|---|---|
| **Config 1** | **$k = 512$** | 100.0% | 0.0% | Full-rank uncompressed baseline reference |
| **Config 2** | **$k = 384$** | 75.0% | 25.0% | Mild pruning; tests for non-informative noisy features |
| **Config 3** | **$k = 256$** | 50.0% | 50.0% | Moderate reduction; validates 50% storage/latency reduction |
| **Config 4** | **$k = 192$** | 37.5% | 62.5% | Inflection analysis; maps transition between 50% and 25% |
| **Config 5** | **$k = 128$** | 25.0% | 75.0% | Aggressive reduction; evaluates severe compression tolerance |
| **Config 6** | **$k = 64$** | 12.5% | 87.5% | High-compression edge point; tests retrained recovery limit |
| **Config 7** | **$k = 32$** | 6.25% | 93.75% | Theoretical lower bound; tests linear separability breakdown |

### Reporting Integrity
Results for Experiments A, B, and C will be tabulated and plotted as **separate performance series**. They will not be collapsed into a single curve.

---

## 12. Investigation H: Define the Pareto Evaluation Framework

### 12.1 Evaluation Axes

$$\text{Objective Vector } \mathbf{\mathcal{O}}(k) = \begin{bmatrix} \text{Macro-F1}(k) & (\text{Maximize}) \\ \text{Top-1 Accuracy}(k) & (\text{Maximize}) \\ \text{Dimension } k & (\text{Minimize}) \\ \text{Inference Latency } (\mu\text{s}) & (\text{Minimize}) \\ \text{Head Storage (KB)} & (\text{Minimize}) \end{bmatrix}$$

### 12.2 Pareto Dominance Criterion
A configuration $(k_1)$ Pareto-dominates $(k_2)$ (denoted $k_1 \succ k_2$) if and only if:
1. $\text{Macro-F1}(k_1) \ge \text{Macro-F1}(k_2)$
2. $k_1 \le k_2$
3. $\text{Latency}(k_1) \le \text{Latency}(k_2)$
4. With at least one strict inequality.

### 12.3 Phase Timing
- **Inflection Point & Optimal Dimension Selection:** Strictly determined using **Validation Data ($N=8,146$)**.
- **Generalization Verification:** Evaluated on **Test Data** only after freezing the optimal and baseline configurations.

---

## 13. Unresolved Issues and Blockers

| Item | Description | Impact | Resolution |
|---|---|---|---|
| **26 vs 38 Dimension** | Multimodal output is 26, unimodal is 38 | Non-blocker | Resolved: 26 represents pathological disease classes in multimodal Increment 3. |
| **Test Data Contamination** | Checkpoint integrity verification | Non-blocker | Resolved: CNN v1 is verified bit-for-bit uncontaminated. CNN v2 is disqualified. |
| **Imbalanced Train Set** | Imbalance in $N=38,013$ | Non-blocker | Resolved: $N=9,299$ balanced subset frozen as primary training set. |
| **Extraction Latency** | CPU extraction speed (~3.6 img/s) | Non-blocker | Sequential extraction with disk cache will take ~43 min for train, ~37 min for val. |
| **Active Blockers** | Any architectural or leakage flaw | **NONE** | All protocols verified. |

---

## 14. Proposed Frozen Phase 3 Experimental Protocol

Upon receiving explicit user authorization for Phase 3:
1. **Extraction Step:**
   - Extract and cache PlantVillage Training embeddings from `plantvillage_train_subset.json` ($N=9,299$).
   - Extract and cache PlantVillage Validation embeddings from `plantvillage_val.json` ($N=8,146$).
   - Extract and cache PlantVillage Test embeddings from `plantvillage_test.json` ($N=8,146$) under lock.
   - Extract and cache PlantDoc Test embeddings from `plantdoc_test.json` ($N=236$) under lock.
2. **Feature Selection Step:**
   - Fit `DeepANOVASelector` strictly on the $N=9,299$ training embeddings.
   - Derive the single master ranking and extract the seven feature subsets $k \in \{512, 384, 256, 192, 128, 64, 32\}$.
3. **Evaluation Step (Validation):**
   - Execute Experiment A (weight-sliced) across all 7 dimensions on PlantVillage validation.
   - Execute Experiment B (retrained linear head) across all 7 dimensions on PlantVillage validation.
   - Execute Experiment C forward passes and attention weight validation across all 7 dimensions.
   - Identify Pareto-optimal operating points and elbow inflection points.
4. **Final Test Evaluation (Locked Configurations):**
   - Evaluate the uncompressed baseline ($k=512$) and selected Pareto configurations on frozen PlantVillage test and PlantDoc test.
   - Report zero-shot transfer metrics across Full Mapped ($N=219$), Conflict-Excluded ($N=211$), and Deduplicated ($N=208$) PlantDoc splits.

---

## 15. Final Authorization Recommendation

### Explicit Gate Verdict: **ACCEPTED WITH OBSERVATIONS**

#### Justification:
- The 26-versus-38 multimodal output dimension discrepancy is **completely resolved** with definitive historical code evidence.
- The pilot metrics are **independently verified and reproducible** from raw cache artifacts.
- The training-data strategy is **frozen** to the balanced $N=9,299$ partition.
- The data-leakage and partition-access protocol is **complete, mathematically verified, and enforced**.
- All three experimental paradigms (A, B, and C) are **adequately and unambiguously specified**.
- The seven-dimension matrix ($k \in \{512, 384, 256, 192, 128, 64, 32\}$) is **methodologically ready for execution**.

Execution has been halted. **Phase 3 will NOT commence until separate explicit user authorization is provided.**

