# Phase 4.1: Research Paper Revision Guidance Correction and Evidence Verification Report

**Supporting Research Paper:** *"Multi-Crop Disease Identification Under Real-World Conditions: A Deep Learning Framework Using EfficientNet-B4, VGG-16, and ResNet-50"*  
**Authoritative Documents Audited:**  
- `docs/phase_reports/phase4_paper_revision_guidance_report.md` (Primary Guidance under Audit)  
- `DLM_Research_Paper_v6.pdf` (Existing 8-Page Draft)  
- `DLM_Research_Paper_v5.pdf` (Prior Draft)  
- `docs/phase_reports/phase3_1_final_results_audit_report.md` (Accepted Audit Authority)  
- `docs/phase_reports/phase3_full_matrix_evaluation_report.md` (Empirical Matrix Authority)  
- `docs/phase_reports/phase3_experimental_freeze_record.md` (Cryptographic Freeze)  
- `results/phase3/phase3_evaluation_summary.json` (Raw Results Authority)  
- `results/phase3/anova_feature_ranking.json` (Binary Feature Ranking)  
**Date:** November 1, 2026  
**Auditor Role:** Critical Academic Reviewer, Technical Editor, and Research Methodology Auditor  
**Deliverable Type:** Evidence Reconciliation & Revision Guidance Correction (Manual Application Guide)  
**Authorization Status:** **REVISION GUIDANCE AUDIT ONLY — SOURCE FILES AND ORIGINAL PHASE 4 REPORT REMAIN UNTOUCHED**  

---

## 1. Executive Summary and Final Verdict

### Final Verdict: ACCEPTED WITH OBSERVATIONS

This targeted Phase 4.1 audit evaluated the 15 Change IDs (`CHG-01` through `CHG-15`) formulated in the Phase 4 report (`phase4_paper_revision_guidance_report.md`). While the underlying empirical foundation established in Phase 3 remains completely intact, this audit identified several critical scientific, statistical, methodological, numerical, and bibliographic overstatements that must be corrected before the author manually applies the recommendations in Overleaf.

### Summary of Change ID Audit Outcomes:
- **RETAINED:** 4 Change IDs (`CHG-04`, `CHG-05`, `CHG-13`, `CHG-14`) — Formulations, manifests, and figure repositioning guidance are verified and unchanged.
- **CORRECTED:** 10 Change IDs (`CHG-01`, `CHG-02`, `CHG-03`, `CHG-06`, `CHG-07`, `CHG-08`, `CHG-10`, `CHG-11`, `CHG-12`, `CHG-15`) — Revised to eliminate prohibited statistical claims, correct bibliographic records, accurately scope projection metrics, clarify PlantDoc unsupported-class semantics, and remove speculative rank boundaries.
- **REQUIRES AUTHOR DECISION:** 1 Change ID (`CHG-09`) — Historical 99.42% attribution resolved; author must choose whether to remove the unverified historical multimodal row or retain it with explicit protocol qualification.
- **PARTIALLY CORRECTED:** 0
- **BLOCKED BY MISSING EVIDENCE:** 0

### Key Corrections Established:
1. **Statistical Language De-escalation (Area 3.1):** Replaced all instances of "statistical parity", "statistical equivalence", and claims of "proving 50% feature redundancy" with strictly permissible null-hypothesis language: *"No statistically significant difference was detected between the two configurations using the paired McNemar test ($p = 0.2636$)."*
2. **PlantDoc Unsupported-Prediction Semantics (Area 3.2):** Verified against `src/deep_feature_selection/phase3_runner.py` that model predictions into any of the 13 unsupported PlantVillage categories are **strictly counted as incorrect predictions** (evaluating to `False`) against the 25 mapped ground-truth categories. They are neither discarded nor excluded from the denominator ($N=219$).
3. **Forensic Reconciliation of the Historical 99.42% Result (Area 3.3):** Proved conclusively that the historical paper attributed 99.42% to the multimodal "Proposed Framework" on PlantVillage, while standalone EfficientNet-B4 was reported at 97.8%. In the repository, the verified standalone CNN v1 checkpoint achieved **99.51% validation accuracy** and **99.42% Macro Precision** ($N=8,146$). CHG-09 had mistakenly reassigned 99.42% as standalone CNN v1 accuracy. Two clear options are provided for Table I.
4. **Elimination of the "Intrinsic Rank Boundary" Claim (Area 3.4):** Removed all assertions that $k=64$ represents an "intrinsic rank" or "minimal linear separability boundary," replacing them with empirical statements regarding the progressive loss of discriminative subspace information.
5. **Separation of Multimodal Pipeline from Visual Feature Selection (Area 3.5):** Clarified that the multimodal cross-attention architecture (26 pathological classes, BiLSTM weather branch) is the contextual framework, while ANOVA feature selection was evaluated empirically on visual heads and projection layers. Corrected the Figure 1 caption and text to prevent readers from inferring that end-to-end multimodal classification was validated at $k=256$.
6. **Numerical Precision and Conflation Resolution (Area 3.6):** Resolved a critical conflation where the Phase 4 text mixed the 93.57% projection parameter reduction at $k=32$ with the 45.72% projection latency reduction measured at $k=256$. Added unambiguous labeling to Table II indicating that storage metrics represent classifier-head weights alone.
7. **Bibliographic Audit and De-hallucination (Area 3.7):** Identified that `hollier1997` is a phantom citation, `khaki2019` models crop yield via feed-forward DNNs (not BiLSTM weather disease forecasting), `garg2021` contained a corrupted DOI pointing to an unrelated image dehazing paper, and `li2017feature` was misattributed to IEEE TKDE instead of *ACM Computing Surveys*. Corrected all BibTeX entries and citation contexts.
8. **Reclassification of the Six-Page Budget (Area 3.8):** Reclassified the six-page conference layout from "VERIFIED" to **"PLANNING ESTIMATE / NOT YET COMPILED"**, clearly highlighting that two-column IEEE float placement, bibliography sizing, and equation wrapping require final compilation and inspection by the author in Overleaf.

---

## 2. Source Materials Inspected

The following primary files and source records were audited:
1. **Revision Guidance:** `docs/phase_reports/phase4_paper_revision_guidance_report.md` (Audited section-by-section).
2. **Historical Drafts:** `DLM_Research_Paper_v6.pdf` (8 pages), `DLM_Research_Paper_v5.pdf` (6 pages).
3. **Phase 3/3.1 Audit Reports:** `docs/phase_reports/phase3_1_final_results_audit_report.md`, `docs/phase_reports/phase3_full_matrix_evaluation_report.md`, `docs/phase_reports/phase3_experimental_freeze_record.md`.
4. **Underlying Data and Raw Results:** `results/phase3/phase3_evaluation_summary.json`, `results/phase3/anova_feature_ranking.json`, `results/metrics/summary_metrics_table.csv`, `configs/cnn/cnn_v1_plantvillage_metrics.json`.
5. **Implementation Codebase:** `src/deep_feature_selection/phase3_runner.py` (lines 405–480), `src/deep_feature_selection/anova_selector.py`, `src/deep_feature_selection/classifiers.py`, `src/deep_feature_selection/multimodal.py`, `scripts/generate_all_figures.py`.
6. **External Bibliographic Databases:** Cross-checked DOIs and metadata across IEEE Xplore, ACM Digital Library, Frontiers in Plant Science, and PubMed.

---

## 3. Eight-Area Correction Assessment

### 3.1 Statistical Language Throughout Recommendations
- **Identified Deficiency:** The Phase 4 report repeatedly utilized prohibited terminology such as "achieves statistical parity", "statistical equivalence was confirmed", and "proves that deep representations contain 50% redundancy" in `CHG-01`, `CHG-02`, `CHG-10`, and `CHG-11`. In classical hypothesis testing, a failure to reject the null hypothesis ($p = 0.2636 \ge 0.05$) indicates that the sample evidence is insufficient to detect a difference; it does **not** prove mathematical or statistical identity.
- **Required Terminology Implemented:**
  - *Permitted:* "No statistically significant difference was detected between the two configurations using the paired McNemar test ($p = 0.2636$)."
  - *Permitted:* "The reduced representation retained comparable observed performance."
  - *Permitted:* "The result suggests that the evaluated task can retain high classification performance with a reduced feature dimension."
- **PlantDoc Qualifications Preserved:**
  - Exp A at $k=384$: 31.51% to 34.25% (+2.74 pp, net gain of 6 images, McNemar $p = 0.1138$, exact binomial $p = 0.1094$, 95% Wilson CI [25.72%, 37.93%] vs [28.28%, 40.75%]).
  - Exp B at $k=384$: 33.79% to 34.25% (+0.46 pp, net gain of 1 image, McNemar $p = 1.0$).
  - Strictly framed as *empirical observations*, not statistically significant improvements.

### 3.2 PlantDoc Unsupported-Prediction Handling
- **Audit Findings:** In `src/deep_feature_selection/phase3_runner.py` (lines 435–448), PlantDoc samples are evaluated using:
  ```python
  sub_y = np.array([pdoc_samples[i]["mapped_class_index"] for i in indices], dtype=np.int64)
  preds = clf.predict(sub_x)
  correct = int(np.sum(preds == sub_y))
  acc = float(correct / len(indices))
  ```
  The ground-truth `sub_y` takes values strictly in the 25 mapped category indices ($[0, \dots, 24]$ within the 38-class space). The classifier produces predictions `preds` across all 38 PlantVillage classes ($[0, \dots, 37]$).
- **Handling of the 13 Unsupported Classes:** If `preds` selects one of the 13 unsupported PlantVillage classes (e.g., Blueberry, Orange, Soybean, or Squash classes not in PlantDoc), `preds == sub_y` evaluates to `False`. 
- **Correction Applied:** The text in `CHG-07` and `CHG-10` is updated to explicitly state:
  *"Predictions falling into any of the 13 unsupported PlantVillage output categories cannot match the 25 mapped ground-truth categories and are strictly counted as incorrect classifications. They are retained in the sample denominator ($N=219$ full mapped, $N=211$ conflict-excluded, $N=208$ deduplicated) rather than discarded or re-normalized, preventing artificial accuracy inflation."*

### 3.3 Historical 99.42% Attribution and Resolution
*(Detailed in Section 4 below).*

### 3.4 Unsupported Intrinsic-Rank Conclusion Removal
- **Identified Deficiency:** `CHG-10` asserted: *"identifying $k=64$ as the intrinsic rank boundary required to linearly separate 38 crop disease classes."*
- **Correction Applied:** Intrinsic rank cannot be deduced from a discrete drop in test accuracy across arbitrary power-of-two/multiples dimensions. Replaced throughout `CHG-10` with the approved phrasing:
  ```latex
  The decline in classification performance at smaller feature
  dimensions indicates that aggressive dimensionality reduction
  removes discriminative information that cannot be fully recovered
  by retraining the linear classification head.
  ```

### 3.5 Multimodal Architecture vs. Visual Feature Selection Separation
- **Identified Deficiency:** The Phase 4 recommendations (especially `CHG-12` Fig. 1 caption and introduction bullets) interwove the $k=256$ visual feature reduction into the complete multimodal pipeline without explicitly reminding readers that end-to-end multimodal classification accuracy was never evaluated on paired datasets.
- **Correction Applied:**
  - Preserved the multimodal architecture (BiLSTM + cross-attention + 26 pathological classes + rejection $\tau=0.65$) as the macro-framework.
  - Formulated ANOVA feature selection strictly as a deep visual representation study evaluating the visual classification head and projection layer.
  - Scoped Experiment C strictly as a *projection-level computational profiling analysis*.
  - Updated Figure 1 caption and Overleaf diagram guidance accordingly.

### 3.6 Full Matrix Numerical Verification
*(Detailed in Section 5 below).*

### 3.7 Bibliography Audit and Correction
*(Detailed in Section 8 below).*

### 3.8 Six-Page Feasibility Assessment Reclassification
*(Detailed in Section 9 below).*

---

## 4. Historical 99.42% Attribution Investigation and Conclusion

### 4.1 Forensic Investigation Questions and Answers

| Investigation Item | Finding / Evidence Source | Audit Verdict |
|---|---|---|
| **1. Which model produced 99.42%?** | In draft papers `DLM_Research_Paper_v5.pdf` and `v6.pdf` (Table I and Section IV-A), **99.42%** (with 0.992 Macro-F1) is explicitly attributed to the **Proposed Framework** (multimodal fusion). Standalone EfficientNet-B4 is listed as **97.8%** (0.971 Macro-F1). | Attributed historically to Multimodal Framework, NOT standalone CNN. |
| **2. Did standalone EfficientNet-B4 CNN v1 produce 99.42%?** | In the audited repository (`configs/cnn/cnn_v1_plantvillage_metrics.json`), standalone EfficientNet-B4 CNN v1 achieved **99.51% validation accuracy** and **0.9942 Macro Precision** ($N=8,146$). In Phase 3 validation benchmarking (`phase3_evaluation_summary.json`), Exp B at $k=512$ achieved **99.42% validation accuracy** (0.9924 Macro-F1). | Phase 4 conflated Macro Precision (0.9942) / Exp B validation with historical Table I. |
| **3. Which checkpoint supports 99.42%?** | No metrics JSON or evaluation log exists in the repository for the historical multimodal fusion checkpoint (`archive/increment_1_cnn/best_fusion.pth`). In contrast, `best_model.pth` supports CNN v1 (99.51% val acc, 99.42% val macro-precision). | Historical multimodal 99.42% lacks underlying raw evaluation metrics JSON. |
| **4. Which dataset split was used?** | The paper narrative calls it "PlantVillage" without specifying split; all repository historical comparisons (`phase4d_comparative_analysis.md`) were executed on the **validation partition** ($N=8,146$). | Validation partition ($N=8,146$). |
| **5. Was it validation or test accuracy?** | Validation accuracy. The locked test partition ($N=8,146$) was only established and evaluated in Phase 3 (achieving 99.44% Exp A, 99.30% Exp B). | Historical metrics are validation metrics. |
| **6. Original experimental protocol?** | Standalone VGG-16, ResNet-50, and EfficientNet-B4 compared against multimodal fusion (incorporating weather features). | Unpaired historical benchmarking without cryptographic freeze. |
| **7. Is it comparable to Phase 3?** | No. Phase 3 evaluated ANOVA feature selection on deep visual embeddings extracted from `best_model.pth` across frozen validation and test partitions. | Methodologically distinct. |
| **8. Can 99.42% legitimately remain in Table I?** | It CANNOT remain assigned to "EfficientNet-B4 CNN v1" as Phase 4 proposed. If retained, it must be labeled as the historical multimodal framework with an explicit qualification. | Requires Author Decision (Options A & B). |

### 4.2 Resolution Options for the Author

To resolve this discrepancy cleanly in Overleaf, the author is presented with two academically defensible paths:

#### Option A (Recommended): Pure Visual Backbone Baseline Table
Remove the unverified "Proposed Framework (Multimodal)" row from Table I entirely. Benchmark strictly the three visual CNN architectures on PlantVillage, reporting the audited metrics for EfficientNet-B4 CNN v1:
```latex
\begin{table}[t]
\caption{Visual Convolutional Backbone Comparison on PlantVillage}
\label{tab:baselines}
\centering
\footnotesize
\begin{tabular}{lcccc}
\hline
\textbf{Architecture} & \textbf{Input Res.} & \textbf{Params} & \textbf{Val. Acc.} & \textbf{Macro-F1} \\
\hline
VGG-16 & $224 \times 224$ & 138.4M & 93.40\% & 0.9210 \\
ResNet-50 & $224 \times 224$ & 25.6M & 95.70\% & 0.9480 \\
EfficientNet-B4 (CNN v1) & $380 \times 380$ & 18.6M & \textbf{99.51\%} & \textbf{0.9928} \\
\hline
\end{tabular}
\end{table}
```
*Justification:* Eliminates confusion between visual backbones and multimodal systems; reports the cryptographically audited CNN v1 validation metrics from `configs/cnn/cnn_v1_plantvillage_metrics.json` (99.51% accuracy, 0.9928 Macro-F1).

#### Option B: Historical Table with Qualified Multimodal Row
Retain the historical Table I structure, but accurately attribute 99.42% to the multimodal framework, reporting CNN v1 separately:
```latex
\begin{table}[t]
\caption{Baseline Architectural Comparison on PlantVillage}
\label{tab:baselines}
\centering
\footnotesize
\begin{tabular}{lcccc}
\hline
\textbf{Architecture} & \textbf{Modality} & \textbf{Params} & \textbf{Val. Acc.} & \textbf{Macro-F1} \\
\hline
VGG-16 & Visual & 138.4M & 93.40\% & 0.9210 \\
ResNet-50 & Visual & 25.6M & 95.70\% & 0.9480 \\
EfficientNet-B4 (CNN v1) & Visual & 18.6M & 99.51\% & 0.9928 \\
Proposed Framework$^\dagger$ & Multimodal & 19.8M & \textbf{99.42\%} & \textbf{0.9920} \\
\hline
\multicolumn{5}{l}{\footnotesize $^\dagger$Historical multimodal validation baseline; visual feature selection is} \\
\multicolumn{5}{l}{\footnotesize evaluated separately on the frozen visual representations in Section~\ref{sec:anova_results}.}
\end{tabular}
\end{table}
```

---

## 5. Complete Numerical Verification Table

Every numerical value reported across the abstract, introduction, results tables, and discussion has been audited against `results/phase3/phase3_evaluation_summary.json` and underlying checkpoint manifests:

| Change ID / Location | Metric / Parameter | Phase 4 Value | Verified Value | Evidence Source | Verdict / Corrective Action |
|---|---|---:|---:|---|---|
| **CHG-01 / Abstract** | Exp B $k=256$ Test Acc | 99.23% | 99.23% (8,083/8,146) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-01 / Abstract** | Baseline $k=512$ Test Acc | 99.30% | 99.30% (8,089/8,146) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-01 / Abstract** | McNemar Test Statistic | $\chi^2 = 1.25$ | $\chi^2 = 1.2500$ | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-01 / Abstract** | McNemar $p$-value | $p = 0.2636$ | $p = 0.2636$ | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-01 / Abstract** | Classifier Head Storage Cut | 50.0% | 49.95% (39,064 B / 78,128 B) | `phase3_evaluation_summary.json` | **VERIFIED** (Head scope required) |
| **CHG-01 / Abstract** | PlantDoc Gain at $k=384$ | +2.74 pp | +2.74 pp (69 $\to$ 75 corr.) | `phase3_evaluation_summary.json` | **VERIFIED** (Empirical gain only) |
| **CHG-01 / Abstract** | Projection Latency Drop | 45.7% (359.90 to 195.37) | 45.72% (359.90 $\to$ 195.37 $\mu$s) | `phase3_evaluation_summary.json` | **VERIFIED** (Measured at $k=256$) |
| **CHG-01 / Abstract** | Proj. Param Drop at $k=32$ | 93.6% (131,328 to 8,448) | 93.57% (131,328 $\to$ 8,448) | `phase3_evaluation_summary.json` | **DISCREPANCY RESOLVED:** Measured at $k=32$, not $k=256$. Conflation separated. |
| **CHG-04 / Sec III-A** | Train Subset Total | 9,299 | 9,299 | `plantvillage_train_subset.json` | **VERIFIED** ($36 \times 250 + 106 + 193$) |
| **CHG-04 / Sec III-A** | Potato Healthy Support | 106 | 106 | `plantvillage_train_subset.json` | **VERIFIED** |
| **CHG-04 / Sec III-A** | Apple Cedar Rust Support | 193 | 193 | `plantvillage_train_subset.json` | **VERIFIED** |
| **CHG-04 / Sec III-A** | PlantDoc Test Total | 236 | 236 | `plantdoc_test.json` | **VERIFIED** |
| **CHG-04 / Sec III-A** | PlantDoc Mapped Subset | 219 | 219 | `plantdoc_test.json` | **VERIFIED** (25 categories) |
| **CHG-04 / Sec III-A** | PlantDoc Conflict-Excluded | 211 | 211 | `plantdoc_test.json` | **VERIFIED** (Removes 8 duplicates) |
| **CHG-04 / Sec III-A** | PlantDoc Deduplicated | 208 | 208 | `plantdoc_test.json` | **VERIFIED** (Removes 11 duplicates) |
| **CHG-09 / Table I** | Standalone CNN v1 Val Acc | 99.42% | 99.51% | `cnn_v1_plantvillage_metrics.json` | **CORRECTED:** 99.42% was precision. Val acc is 99.51%. |
| **CHG-09 / Table I** | Standalone CNN v1 Macro-F1 | 0.9924 | 0.9928 | `cnn_v1_plantvillage_metrics.json` | **CORRECTED:** 0.9924 was Exp B val F1. CNN v1 is 0.9928. |
| **CHG-10 / Table II** | $k=512$ Exp A Test Acc | 99.44% | 99.435% (8,100 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.9914) |
| **CHG-10 / Table II** | $k=512$ Exp B Test Acc | 99.30% | 99.300% (8,089 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.9913) |
| **CHG-10 / Table II** | $k=384$ Exp A Test Acc | 99.34% | 99.337% (8,092 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.9906) |
| **CHG-10 / Table II** | $k=384$ Exp B Test Acc | 99.32% | 99.325% (8,091 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.9917) |
| **CHG-10 / Table II** | $k=256$ Exp A Test Acc | 98.20% | 98.195% (7,999 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.9759) |
| **CHG-10 / Table II** | $k=256$ Exp B Test Acc | 99.23% | 99.227% (8,083 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.9904) |
| **CHG-10 / Table II** | $k=192$ Exp A Test Acc | 94.23% | 94.230% (7,676 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.9335) |
| **CHG-10 / Table II** | $k=192$ Exp B Test Acc | 99.09% | 99.092% (8,072 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.9897) |
| **CHG-10 / Table II** | $k=128$ Exp A Test Acc | 84.88% | 84.876% (6,914 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.8166) |
| **CHG-10 / Table II** | $k=128$ Exp B Test Acc | 98.82% | 98.821% (8,050 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.9872) |
| **CHG-10 / Table II** | $k=64$ Exp A Test Acc | 63.05% | 63.050% (5,136 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.6283) |
| **CHG-10 / Table II** | $k=64$ Exp B Test Acc | 94.73% | 94.734% (7,717 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.9456) |
| **CHG-10 / Table II** | $k=32$ Exp A Test Acc | 36.47% | 36.472% (2,971 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.3986) |
| **CHG-10 / Table II** | $k=32$ Exp B Test Acc | 83.05% | 83.047% (6,765 / 8,146) | `phase3_evaluation_summary.json` | **VERIFIED** (Macro-F1: 0.8505) |
| **CHG-10 / Table II** | Head Storage Values | 77,976 B down to 5,016 B | Exp A: 77,976 B to 5,016 B (float32)<br>Exp B: 155,952 B to 10,032 B (float64) | `phase3_evaluation_summary.json` | **TABLE HEADING CORRECTED:** Unambiguously labeled as *Exp. A Classifier Head Storage*. |
| **CHG-10 / Table II** | PlantDoc Exp A Acc ($k=512$) | 31.51% | 31.507% (69 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp B Acc ($k=512$) | 33.79% | 33.790% (74 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp A Acc ($k=384$) | 34.25% | 34.247% (75 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp B Acc ($k=384$) | 34.25% | 34.247% (75 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp A Acc ($k=256$) | 31.05% | 31.050% (68 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp B Acc ($k=256$) | 31.96% | 31.963% (70 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp A Acc ($k=192$) | 27.85% | 27.854% (61 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp B Acc ($k=192$) | 33.33% | 33.333% (73 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp A Acc ($k=128$) | 21.46% | 21.461% (47 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp B Acc ($k=128$) | 31.51% | 31.507% (69 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp A Acc ($k=64$) | 16.89% | 16.895% (37 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp B Acc ($k=64$) | 26.94% | 26.941% (59 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp A Acc ($k=32$) | 10.96% | 10.959% (24 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Table II** | PlantDoc Exp B Acc ($k=32$) | 23.74% | 23.744% (52 / 219) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Sec IV-E.4**| Proj. Parameters ($k=512$) | 131,328 | 131,328 ($256 \times 512 + 256$) | `phase3_evaluation_summary.json` | **VERIFIED** |
| **CHG-10 / Sec IV-E.4**| Proj. Parameters ($k=256$) | 65,792 | 65,792 ($256 \times 256 + 256$) | `phase3_evaluation_summary.json` | **VERIFIED** (-49.90% reduction) |
| **CHG-10 / Sec IV-E.4**| Proj. Parameters ($k=32$) | 8,448 | 8,448 ($256 \times 32 + 256$) | `phase3_evaluation_summary.json` | **VERIFIED** (-93.57% reduction) |
| **CHG-10 / Sec IV-E.4**| Proj. Forward Latency ($k=512$)| 359.90 $\mu$s | 359.9016 $\mu$s/sample | `phase3_evaluation_summary.json` | **VERIFIED** (batch-64 forward pass) |
| **CHG-10 / Sec IV-E.4**| Proj. Forward Latency ($k=256$)| 195.37 $\mu$s | 195.3688 $\mu$s/sample | `phase3_evaluation_summary.json` | **VERIFIED** (-45.72% latency drop) |
| **CHG-10 / Sec IV-E.4**| Proj. Forward Latency ($k=32$) | 173.50 $\mu$s | 173.5000 $\mu$s/sample | `phase3_evaluation_summary.json` | **VERIFIED** (-51.79% latency drop) |

---

## 6. Corrected Overleaf-Ready LaTeX Replacements

Below are the complete, corrected LaTeX replacements for every affected Change ID. These blocks directly replace the corresponding sections in Overleaf.

### 6.1 Corrected Change ID: CHG-01 (Abstract)
- **Status:** CORRECTED
- **Corrections Applied:** Eliminated prohibited "statistical parity" wording; accurately scoped 50% storage savings to the classifier head; properly attributed 45.7% latency reduction to $k=256$ and 93.6% parameter reduction to $k=32$; framed PlantDoc transfer as an observational trend; removed "proves redundancy" phrasing.
```latex
\begin{abstract}
Automated crop disease diagnosis in uncontrolled agricultural environments is severely challenged by visual symptom variability, complex backgrounds, and domain shifts between laboratory datasets and field imagery. This paper presents a multimodal deep learning framework that integrates EfficientNet-B4 visual representations, temporal environmental context via a Bidirectional Long Short-Term Memory (BiLSTM) network, cross-attention feature fusion, and confidence-based prediction rejection. To examine whether deep visual embeddings can be compressed for edge deployment, we systematically investigate Analysis of Variance (ANOVA) feature selection on the 512-dimensional visual embedding space across seven dimensions ($k \in \{512, 384, 256, 192, 128, 64, 32\}$) under both unretrained weight slicing and retrained linear evaluation paradigms. Evaluated on the locked PlantVillage test partition ($N=8,146$), retaining the top 256 ANOVA features ($k=256$) retains comparable observed performance, achieving 99.23\% Top-1 test accuracy and a Macro-F1 of 0.9904. No statistically significant difference was detected compared to the full 512-dimensional baseline (99.30\%, paired McNemar $\chi^2 = 1.25, p = 0.2636$), while halving classifier head parameters and storage (19,494 to 9,766 parameters; 38.0~KB saved). On unconstrained real-world field images from PlantDoc ($N=219$ mapped), evaluating predictions against mapped categories (with unsupported classes counted as errors) reveals that moderate feature reduction ($k=384$) yields an empirical transfer gain of +2.74 percentage points (31.51\% to 34.25\%, McNemar $p = 0.1138$). Furthermore, cross-attention visual projection latency is reduced by 45.7\% at $k=256$ (359.90~$\mu$s to 195.37~$\mu$s), reaching a 93.6\% projection parameter reduction at $k=32$. These empirical results suggest that the evaluated visual diagnostic task can maintain high classification accuracy within a substantially reduced feature subspace.
\end{abstract}
```

### 6.2 Corrected Change ID: CHG-02 (Introduction & Stated Contributions)
- **Status:** CORRECTED
- **Corrections Applied:** Replaced "statistical parity" with permitted comparative phrasing; accurately separated $k=256$ latency from $k=32$ parameter reduction.
```latex
The primary contributions of this paper are summarized as follows:
\begin{itemize}
    \item We design a multimodal crop disease recognition framework coupling an EfficientNet-B4 visual backbone with a BiLSTM temporal weather network via multi-head cross-attention and confidence-based rejection ($\tau = 0.65$).
    \item We conduct a leakage-free empirical evaluation of filter-based ANOVA feature selection on deep 512-dimensional visual embeddings, comparing analytical weight slicing against retrained linear heads across seven feature dimensions ($k \in \{512, 384, 256, 192, 128, 64, 32\}$).
    \item We demonstrate that the top 256 ANOVA-selected features ($k=256$) achieve comparable performance to the uncompressed 512-d representation on locked PlantVillage test data (99.23\% vs. 99.30\%), with no statistically significant difference detected via paired McNemar testing ($p=0.2636$), halving classifier head parameters and weight memory.
    \item We evaluate out-of-distribution transfer on the real-world PlantDoc benchmark under strict zero-shot conditions, identifying an empirical gain of +2.74 percentage points at $k=384$ ($p=0.1138$), and document a 45.7\% latency reduction in the cross-attention projection stage at $k=256$.
\end{itemize}
```

### 6.3 Corrected Change ID: CHG-03 (Related Work / Literature Survey)
- **Status:** CORRECTED
- **Corrections Applied:** Replaced unverified citation `hollier1997` with established agricultural meteorological epidemiology literature; corrected `khaki2019` context (removed claim that it establishes BiLSTM disease forecasting); cited `li2017feature` for ANOVA filtering survey; pruned unverified citations.
```latex
\section{Related Work}
\label{sec:related_work}
Deep convolutional architectures have established state-of-the-art benchmarks in agricultural pathology, with models such as VGG-16, ResNet-50, and EfficientNet achieving $>95\%$ accuracy on laboratory benchmarks~\cite{mohanty2016}. EfficientNet-B4~\cite{tan2019} balances depth, width, and resolution via compound scaling, extracting rich morphological symptom representations. However, deep models trained under controlled conditions suffer significant accuracy degradation when transferred to unconstrained field imagery, such as the PlantDoc benchmark~\cite{singh2020plantdoc}, due to complex foliar backgrounds, variable illumination, and symptom co-occurrence~\cite{ramcharan2019}.

While visible symptoms represent current pathological state, disease proliferation is heavily governed by antecedent micrometeorological dynamics including temperature, relative humidity, and precipitation. Recurrent neural architectures, particularly Bidirectional LSTMs, effectively capture multi-day temporal environmental dependencies. Rather than naive vector concatenation, cross-attention mechanisms dynamically correlate visual symptom queries with environmental context keys, producing context-aware diagnostic representations.

Despite the expressive power of deep visual embeddings, high-dimensional representations often exhibit significant inter-feature correlation and dimensional redundancy. Filter-based feature selection using the univariate Analysis of Variance (ANOVA) $F$-statistic ranks features by their between-class to within-class variance ratio~\cite{li2017feature}. While widely studied in classical machine learning, evaluating whether univariate ANOVA selection preserves class-separating hyperplanes in deep post-backbone embeddings remains an open empirical question with profound implications for resource-constrained agricultural edge devices.
```

### 6.4 Corrected Change ID: CHG-06 (ANOVA Formulation & Dual Paradigms)
- **Status:** CORRECTED
- **Corrections Applied:** Mathematical formulation verified and retained; replaced citation `\cite{ding2005}` with `\cite{li2017feature}`.
```latex
\subsection{ANOVA Feature Selection and Evaluation Paradigms}
\label{sec:anova_method}
To systematically evaluate redundancy within the 512-dimensional embedding $v$, we implement univariate one-way Analysis of Variance (ANOVA)~\cite{li2017feature}. The selector is fitted strictly on the primary training partition ($N=9,299$), ensuring zero data leakage. For each embedding dimension $j \in \{0, \dots, 511\}$, the $F$-statistic measures the ratio of between-class variance to within-class variance across the $C=38$ categories:
\begin{equation}
    F_j = \frac{\mathrm{MSB}_j}{\mathrm{MSW}_j} = \frac{\sum_{c=1}^{C} N_c (\bar{v}_{c,j} - \bar{v}_j)^2 / (C - 1)}{\sum_{c=1}^{C} \sum_{i=1}^{N_c} (v_{c,i,j} - \bar{v}_{c,j})^2 / (N - C)}
    \label{eq:anova_f}
\end{equation}
where $N_c$ is class support, $\bar{v}_{c,j}$ is class mean, $\bar{v}_j$ is overall mean, and $v_{c,i,j}$ denotes the $j$-th feature of sample $i$ in class $c$. Features are sorted in descending order of $F_j$, resolving ties via ascending feature index:
\begin{equation}
    \mathcal{R} = \mathrm{argsort}\left(F_j, \text{descending}\right)
\end{equation}
Seven nested subsets $\mathcal{S}_k \subset \mathbb{R}^k$ are constructed for target dimensions $k \in \{512, 384, 256, 192, 128, 64, 32\}$, satisfying strict nestedness $\mathcal{S}_{32} \subset \mathcal{S}_{64} \subset \dots \subset \mathcal{S}_{512}$.

To evaluate the reduced subspaces without confounding variables, we implement two rigorous experimental paradigms:
\begin{enumerate}
    \item \textbf{Experiment A (Weight-Sliced Classifier):} Evaluates the intrinsic preservation of the frozen CNN classifier hyperplane without parameter retraining. For target dimension $k$, the weight matrix is analytically column-sliced:
    \begin{equation}
        W_{\mathrm{reduced}} = W_{\mathrm{cls}}[:, \mathcal{S}_k] \in \mathbb{R}^{38 \times k}, \quad b_{\mathrm{reduced}} = b_{\mathrm{cls}} \in \mathbb{R}^{38}
    \end{equation}
    \item \textbf{Experiment B (Retrained Linear Head):} Evaluates the linear separability of subspace $\mathcal{S}_k$ by training an optimal multinomial logistic regression head ($C=1.0$, L-BFGS, $\le 1000$ iterations) strictly on the reduced training representations $v[\mathcal{S}_k] \in \mathbb{R}^{9299 \times k}$.
\end{enumerate}
Validation Pareto optimization across Macro-F1, inference latency, and memory storage is conducted exclusively on the validation partition ($N=8,146$) prior to locked test evaluation.
```

### 6.5 Corrected Change ID: CHG-07 (PlantDoc Zero-Shot Protocol & Unsupported Handling)
- **Status:** CORRECTED
- **Corrections Applied:** Explicitly defines the treatment of the 13 unsupported PlantVillage output classes as incorrect predictions in the evaluation protocol.
```latex
\subsection{Zero-Shot Cross-Domain Field Evaluation Protocol}
\label{sec:zeroshot_protocol}
To evaluate model robustness under true operational distribution shifts without data contamination, we establish a strict zero-shot evaluation protocol on PlantDoc~\cite{singh2020plantdoc}. In strict compliance with experimental freeze standards, no PlantDoc field images or annotations were exposed during backbone training, ANOVA feature ranking, or linear head retraining.

Visual embeddings for PlantDoc images are extracted through the frozen EfficientNet-B4 backbone and reduced using the frozen ranking $\mathcal{S}_k$. Predictions are generated over the 38 PlantVillage output logits and evaluated against the 25 mapped PlantDoc ground-truth categories. Crucially, predictions falling into any of the 13 unsupported PlantVillage output categories cannot match the 25 mapped ground-truth categories and are strictly counted as incorrect classifications. They are retained in the sample denominator ($N=219$ full mapped, $N=211$ conflict-excluded, $N=208$ deduplicated) rather than discarded or re-normalized, preventing artificial accuracy inflation.
```

### 6.6 Corrected Change ID: CHG-08 (Multimodal Taxonomy & Projection Adaptation)
- **Status:** CORRECTED
- **Corrections Applied:** Clarified that the multimodal network is the parent framework; removed unverified citation `\cite{hollier1997}`.
```latex
\subsection{Multimodal Cross-Attention Fusion}
\label{sec:cross_attention}
The multimodal fusion module integrates the visual embedding $v \in \mathbb{R}^k$ with a 256-dimensional temporal weather context vector $h_{\mathrm{wx}} \in \mathbb{R}^{256}$ generated by a 2-layer BiLSTM processing 30-day sliding windows of micrometeorological data (Disease-Conducive Weather Scores, DCWS). 

To accommodate variable visual feature dimensions $k$, an adaptive linear projection maps visual features into the shared cross-attention dimension $d_{\mathrm{attn}} = 256$:
\begin{equation}
    Q_v = W_{\mathrm{cnn\_proj}} \cdot v[\mathcal{S}_k] + b_{\mathrm{cnn\_proj}}
    \label{eq:proj}
\end{equation}
where $W_{\mathrm{cnn\_proj}} \in \mathbb{R}^{256 \times k}$ and $b_{\mathrm{cnn\_proj}} \in \mathbb{R}^{256}$. Keys and values are projected from the weather context: $K_w = W_K h_{\mathrm{wx}}$, $V_w = W_V h_{\mathrm{wx}}$. Multi-head cross-attention ($h=8$ heads) dynamically weights environmental covariates conditioned on visual queries.

Crucially, the multimodal classification head predicts over $C_{\mathrm{multi}} = 26$ pathological disease categories, explicitly excluding the 12 healthy foliage categories from the 38-class PlantVillage taxonomy because environmental disease-conducive weather models apply strictly to active pathogens. Predictions are filtered through a confidence rejection threshold ($\tau = 0.65$), withholding diagnosis when maximum softmax probability satisfies $\max_c P(y=c|x) < \tau$.
```

### 6.7 Corrected Change ID: CHG-09 (Classification Performance & Baselines)
- **Status:** REQUIRES AUTHOR DECISION (Recommended Option A Provided Below)
- **Corrections Applied:** Table I updated to reflect verified validation metrics for CNN v1 (99.51% accuracy, 0.9928 Macro-F1). Disqualified 76.4% PlantDoc table removed.
```latex
\subsection{Baseline Classification Performance and Subspace Benchmarking}
\label{sec:baseline_results}
We first benchmark the full 512-dimensional EfficientNet-B4 CNN v1 against standard deep baselines on the PlantVillage validation partition ($N=8,146$), as summarized in Table~\ref{tab:baselines}. The verified standalone EfficientNet-B4 architecture achieves 99.51\% validation accuracy and 0.9928 Macro-F1, substantially outperforming VGG-16 (93.40\%) and ResNet-50 (95.70\%). On the locked PlantVillage test partition ($N=8,146$), the uncompressed baseline achieves 99.44\% under unretrained classification (Exp A) and 99.30\% under retrained linear evaluation (Exp B).

\begin{table}[t]
\caption{Visual Convolutional Backbone Comparison on PlantVillage}
\label{tab:baselines}
\centering
\footnotesize
\begin{tabular}{lcccc}
\hline
\textbf{Architecture} & \textbf{Input Res.} & \textbf{Params} & \textbf{Val. Acc.} & \textbf{Macro-F1} \\
\hline
VGG-16 & $224 \times 224$ & 138.4M & 93.40\% & 0.9210 \\
ResNet-50 & $224 \times 224$ & 25.6M & 95.70\% & 0.9480 \\
EfficientNet-B4 (CNN v1) & $380 \times 380$ & 18.6M & \textbf{99.51\%} & \textbf{0.9928} \\
\hline
\end{tabular}
\end{table}

When evaluated on the unconstrained PlantDoc benchmark ($N=219$ mapped) under strict zero-shot transfer conditions, performance drops to 31.51\% (Exp A) and 33.79\% (Exp B). This performance delta underscores the substantial domain shift between controlled laboratory backgrounds and complex natural field conditions.
```

### 6.8 Corrected Change ID: CHG-10 (Verified Feature Selection Results & Table I-B)
- **Status:** CORRECTED
- **Corrections Applied:**
  1. Replaced "statistical parity" with *"No statistically significant difference was detected via paired McNemar test ($p=0.2636$)"*.
  2. Table II heading unambiguously labeled: `\textbf{Classifier Head Storage (Exp. A)}`.
  3. Removed unsupported "intrinsic rank boundary at $k=64$" claim.
  4. Separated $k=256$ projection latency (-45.72%) from $k=32$ projection parameters (-93.57%).
  5. Clarified unsupported PlantDoc class handling.
```latex
\subsection{ANOVA Deep Feature Selection Evaluation}
\label{sec:anova_results}
Table~\ref{tab:anova_matrix} presents the comprehensive evaluation of ANOVA-based feature selection across all seven dimensions ($k \in \{512, 384, 256, 192, 128, 64, 32\}$) on both the locked PlantVillage test partition ($N=8,146$) and the zero-shot PlantDoc benchmark ($N=219$).

\begin{table*}[t]
\caption{Full Evaluation Matrix across Seven Feature Dimensions on Locked Test Partitions}
\label{tab:anova_matrix}
\centering
\footnotesize
\begin{tabular}{rccccccc}
\hline
\textbf{Dim.} & \textbf{Reduction} & \multicolumn{2}{c}{\textbf{Exp. A: Weight-Sliced Head}} & \multicolumn{2}{c}{\textbf{Exp. B: Retrained Linear Head}} & \textbf{Head Storage (Exp. A)} & \textbf{PlantDoc Zero-Shot} \\
\cline{3-4} \cline{5-6}
($k$) & (\%) & \textbf{PV Test Acc. (Correct)} & \textbf{Macro-F1} & \textbf{PV Test Acc. (Correct)} & \textbf{Macro-F1} & \textbf{(Bytes / Savings)} & \textbf{Exp. A / Exp. B Acc.} \\
\hline
512 & 0.0\% & 99.44\% (8,100 / 8,146) & 0.9914 & 99.30\% (8,089 / 8,146) & 0.9913 & 77,976 B (Baseline) & 31.51\% / 33.79\% \\
384 & 25.0\% & 99.34\% (8,092 / 8,146) & 0.9906 & 99.32\% (8,091 / 8,146) & 0.9917 & 58,520 B (-25.0\%) & \textbf{34.25\%} / \textbf{34.25\%} \\
\textbf{256} & \textbf{50.0\%} & \textbf{98.20\%} (7,999 / 8,146) & \textbf{0.9759} & \textbf{99.23\%} (8,083 / 8,146) & \textbf{0.9904} & \textbf{39,064 B (-50.0\%)} & 31.05\% / 31.96\% \\
192 & 62.5\% & 94.23\% (7,676 / 8,146) & 0.9335 & 99.09\% (8,072 / 8,146) & 0.9897 & 29,336 B (-62.5\%) & 27.85\% / 33.33\% \\
128 & 75.0\% & 84.88\% (6,914 / 8,146) & 0.8166 & 98.82\% (8,050 / 8,146) & 0.9872 & 19,608 B (-75.0\%) & 21.46\% / 31.51\% \\
64  & 87.5\% & 63.05\% (5,136 / 8,146) & 0.6283 & 94.73\% (7,717 / 8,146) & 0.9456 & 9,880 B (-87.5\%) & 16.89\% / 26.94\% \\
32  & 93.8\% & 36.47\% (2,971 / 8,146) & 0.3986 & 83.05\% (6,765 / 8,146) & 0.8505 & 5,016 B (-93.8\%) & 10.96\% / 23.74\% \\
\hline
\end{tabular}
\end{table*}

\subsubsection{Subspace Compression at $k=256$ and Hypothesis Testing}
On the locked PlantVillage test set, Experiment B reveals that retaining the top 256 ANOVA-ranked features preserves \textbf{99.23\% Top-1 accuracy} and a Macro-F1 of 0.9904, representing an observed difference of only 0.07 percentage points from the full 512-dimensional baseline (99.30\%). A paired McNemar test on the 8,146 individual test predictions reveals 8,076 concordant correct predictions, 50 concordant incorrect predictions, and 20 discordant pairs (7 correct only at $k=256$; 13 correct only at $k=512$). With Edwards continuity correction, $\chi^2 = 1.2500$ ($p = 0.2636$; exact binomial $p = 0.2632$). Because $p > 0.05$, no statistically significant difference was detected between the 256-dimensional subspace and the full 512-dimensional representation under the paired McNemar test.

Crucially, this 50\% dimensionality reduction halves classifier head parameters from 19,494 to 9,766, saving 38.0~KB of memory. We emphasize that this 50.0\% reduction applies specifically to the classifier head and embedding subspace; relative to the complete 18.6M-parameter EfficientNet-B4 model (74.7~MB checkpoint), the head reduction constitutes 0.052\% of total model parameters.

\subsubsection{Weight Slicing vs. Retrained Head Dynamics}
Comparing Experiment A and Experiment B illustrates how feature selection interacts with the learned representation. Direct unretrained weight slicing preserves 98.20\% accuracy at $k=256$ (-1.24 pp from baseline), confirming that the top 256 ANOVA dimensions align closely with the dominant separating hyperplanes of the original network. Below $k=192$, unretrained accuracy declines sharply (84.88\% at $k=128$, 63.05\% at $k=64$, 36.47\% at $k=32$), demonstrating that individual unretrained weights rely on cooperative cancellation across dimensions. In contrast, retraining the linear head recovers accuracy to 98.82\% at $k=128$ and 94.73\% at $k=64$ (+31.68 pp over sliced). The decline in classification performance at smaller feature dimensions indicates that aggressive dimensionality reduction removes discriminative information that cannot be fully recovered by retraining the linear classification head.

\subsubsection{Out-of-Distribution Field Generalization (PlantDoc)}
On the zero-shot PlantDoc benchmark ($N=219$), pruning the lowest-ranked 25\% of features ($k=384$) produces an empirical accuracy improvement of \textbf{+2.74 percentage points} for Experiment A (31.51\% to 34.25\%, net gain of 6 images) and \textbf{+0.46 percentage points} for Experiment B (33.79\% to 34.25\%, net gain of 1 image). Evaluated on the conflict-excluded ($N=211$) and deduplicated ($N=208$) subsets, accuracy similarly peaks at $k=384$ (35.07\% and 34.62\%, respectively). 

A paired McNemar test on the discordant predictions ($n_{01}=8, n_{10}=2$) yields $\chi^2 = 2.5000, p = 0.1138$ (exact binomial $p=0.1094$) with overlapping 95\% Wilson confidence intervals ([25.72\%, 37.93\%] vs. [28.28\%, 40.75\%]). Therefore, while this observed gain is consistent with dimensionality-reduction regularization, it cannot be claimed as statistically significant proof. Furthermore, whether the pruned features specifically correspond to laboratory background artifacts remains an unverified hypothesis requiring spatial attribution.

\subsubsection{Multimodal Projection Efficiency (Experiment C)}
Adapting the visual projection layer $W_{\mathrm{cnn\_proj}} \in \mathbb{R}^{256 \times k}$ scales parameters linearly with $k$. Reducing $k$ from 512 to 256 reduces projection parameters by 49.90\% (131,328 to 65,792), lowering forward projection latency for a batch of 64 from 359.90~$\mu$s/sample to 195.37~$\mu$s/sample (a 45.72\% latency reduction). At the most aggressive setting ($k=32$), projection parameters are reduced by 93.57\% (down to 8,448 parameters; 173.50~$\mu$s/sample latency), reducing total fusion module parameters by 10.58\% (1,160,986 to 1,038,106). Because end-to-end multimodal classification on paired field test sets was outside our visual feature audit scope, these gains are reported strictly as projection-level computational profiling benchmarks.
```

### 6.9 Corrected Change ID: CHG-11 (Conclusion & Future Work)
- **Status:** CORRECTED
- **Corrections Applied:** Replaced "confirmed statistical parity" with permitted null-hypothesis language.
```latex
\section{Conclusion and Future Work}
\label{sec:conclusion}
This study presented an empirical investigation of ANOVA-based deep feature selection within an EfficientNet-B4 multimodal crop disease identification framework. By evaluating seven feature dimensions under frozen weight-slicing and retrained linear head paradigms across 25,827 samples, we demonstrated that the visual classification head can be compressed by 50\% ($k=256$) while preserving 99.23\% Top-1 test accuracy and 0.9904 Macro-F1 across 38 crop disease classes. Paired McNemar analysis detected no statistically significant difference ($p = 0.2636$) between $k=256$ and the uncompressed 512-dimensional baseline, halving classifier head storage. Furthermore, moderate feature pruning ($k=384$) produced an observed empirical transfer gain of +2.74 percentage points on unseen PlantDoc field imagery, and reduced multimodal visual projection latency by 45.7\% at $k=256$.

Future work will focus on: (1) conducting spatial feature attribution and mask-guided Grad-CAM ablations to test whether low-ranking ANOVA dimensions correlate with non-foliar background pixels; (2) extending feature selection into an end-to-end multimodal evaluation on paired, synchronized agricultural field streams; and (3) deploying the optimized $k=256$ architecture onto embedded micro-edge accelerators for real-time mobile agricultural diagnosis.
```

### 6.10 Corrected Change ID: CHG-12 (Figure 1: Architecture Caption)
- **Status:** CORRECTED
- **Corrections Applied:** Revised caption to explicitly distinguish the multimodal macro-framework from the deep visual feature selection study.
```latex
\begin{figure*}[t]
\centering
\includegraphics[width=0.92\textwidth]{architecture.png}
\caption{Overall architecture of the proposed multimodal crop disease recognition framework. EfficientNet-B4 extracts a 512-dimensional visual embedding $v$ from $380 \times 380$ input images, which is systematically investigated under ANOVA feature selection (Section~\ref{sec:anova_results}). A 2-layer BiLSTM models 30-day temporal weather sequences (DCWS). Multi-head cross-attention dynamically fuses visual queries with environmental context keys, predicting over 26 pathological disease classes with confidence-based prediction rejection ($\tau = 0.65$). Visual feature selection is evaluated experimentally on the visual classification head and projection stage.}
\label{fig:architecture}
\end{figure*}
```

---

## 7. Revised Figure and Architecture Recommendations

### 7.1 Figure 1 (Multimodal Framework Diagram)
- **Current Observation:** The diagram in `v6` illustrates the end-to-end flow from image and weather inputs to cross-attention and 26-class output.
- **Guidance for Author in Overleaf:**
  - Do **NOT** modify or regenerate the image asset itself.
  - Apply the corrected caption from `CHG-12`.
  - In Section III-F and Section IV-E.4, emphasize to readers that the ANOVA selector operates on the visual embedding $v$ and that its downstream effects in this study are evaluated on: (1) the visual classification heads (Exp A and Exp B), and (2) the visual projection layer $W_{\mathrm{cnn\_proj}}$ (Exp C). End-to-end multimodal fine-tuning under feature reduction remains future work.

### 7.2 Figure 3 (Dual-Panel Accuracy & Pareto Plot)
- **Status:** RETAINED (`CHG-13`).
- **Action:** Replace `classification_accuracy_comparison.png` with `results/phase3/figures/fig1_accuracy_vs_dimension.png`. This asset displays both the PlantVillage test accuracy curve across all seven dimensions and the Pareto trade-off.

### 7.3 Figure 4 (Grad-CAM Visualisation)
- **Status:** RETAINED (`CHG-14`).
- **Action:** Constrain width to `\includegraphics[width=0.88\columnwidth]{Potato leaf early blight (3).jpg}` and anchor to the top of Page 6 (`\begin{figure}[t]`). This prevents float spillover onto Page 7/8.

---

## 8. Complete Bibliography Audit

### 8.1 Detailed Bibliography Audit Table

| Citation Key | Status in `v6` | Existence & Metadata Verification | Claim Supported in Paper | Relevance / Audit Verdict | Recommended Action |
|---|---|---|---|---|---|
| `mohanty2016` | [2] Full | Verified: Front. Plant Sci. 7:1419, 2016. DOI: `10.3389/fpls.2016.01419` | PlantVillage laboratory dataset benchmark (>95% acc). | **VERIFIED & RELEVANT** | RETAIN. |
| `singh2020plantdoc` | [3] Full | Verified: ACM IKDD CoDS/COMAD 2020, pp. 249–253. DOI: `10.1145/3371158.3371196` | PlantDoc field dataset and real-world domain shift. | **VERIFIED & RELEVANT** | RETAIN. |
| `tan2019` | [4] Full | Verified: ICML 2019, PMLR 97:6105–6114. | EfficientNet compound scaling architecture. | **VERIFIED & RELEVANT** | RETAIN. |
| `garg2021` | [5] Corrupted DOI | Paper exists in ICCCNT 2021; Phase 4 DOI (`10.1109/ICCCNT51525.2021.9579720`) belongs to an unrelated dehazing paper! | Deep learning transfer learning on crop diseases. | **CORRUPTED DOI** | RETAIN without bad DOI, or replace with verified baseline. |
| `ramcharan2019` | [6] Full | Verified: Front. Plant Sci. 10:272, 2019. DOI: `10.3389/fpls.2019.00272` | Mobile/field domain shift and environmental degradation. | **VERIFIED & RELEVANT** | RETAIN. |
| `ganin2016` | [7] Full | Verified: JMLR 17(59):1–35, 2016. | Domain adversarial training. | **OBSOLETE (CNN v2 disqualified)** | REMOVE (saves 2 lines). |
| `hollier1997` | [8] Corrupted | **PHANTOM / UNVERIFIED:** *Plant Disease* 81:1099 is a Focus article by T. Kommedahl. No paper by Hollier & Schepers under this title exists in that volume. | Weather-based disease forecasting. | **UNVERIFIED CITATION** | REMOVE or replace with verified agro-weather reference. |
| `khaki2019` | [9] Full | Verified: Front. Plant Sci. 10:621, 2019. DOI: `10.3389/fpls.2019.00621` | Crop yield prediction using ResNet-like feed-forward DNN. | **MISATTRIBUTED CLAIM:** Does not model BiLSTM or disease forecasting. | RETAIN only for general agro-deep-learning, or replace with LSTM reference. |
| `tajbakhsh2016` | [10] Full | Verified: IEEE TMI 35(5):1299–1312, 2016. | Fine-tuning vs scratch training in medical CNNs. | **OBSOLETE** | REMOVE (saves 2 lines). |
| `nasa2018` | [11] Partial | NASA POWER agroclimatology database. | Source of meteorological temperature and RH data. | **VALID DATA SOURCE** | RETAIN with complete metadata. |
| `li2017feature` | [21] Wrong Venue | Verified in **ACM Comput. Surv. (CSUR)**, 50(6):94, 2017. DOI: `10.1145/3136625`. (Not IEEE TKDE!). | Comprehensive survey of filter-based feature selection. | **VERIFIED (Venue corrected)** | RETAIN with corrected BibTeX. |
| `ding2005` | Proposed | Verified: J. Bioinform. Comput. Biol. 3(2):185–205, 2005. | Minimum redundancy feature selection (mRMR). | **MISPLACED FOR ANOVA:** mRMR is not standard one-way ANOVA. | REPLACE citation with `li2017feature`. |
| `[12]–[20]` in `v6` | Corrupted | Listed without authors (title and URL only); [17] and [19] share duplicate DOIs. | Informal general background. | **DEFECTIVE OVERLEAF DRAFTING** | REMOVE all 9 entries. |

### 8.2 Corrected, Compilable BibTeX Records

Replace corrupted or defective BibTeX records in Overleaf with these verified entries:

```bibtex
@article{mohanty2016,
  author = {S. P. Mohanty and D. P. Hughes and M. Salath{\'e}},
  title = {Using Deep Learning for Image-Based Plant Disease Detection},
  journal = {Frontiers in Plant Science},
  volume = {7},
  pages = {1419},
  year = {2016},
  doi = {10.3389/fpls.2016.01419}
}

@inproceedings{singh2020plantdoc,
  author = {D. Singh and N. Jain and P. Jain and P. Kayal and S. Kumawat and N. Batra},
  title = {{PlantDoc}: A Dataset for Visual Plant Disease Detection},
  booktitle = {Proc. 7th ACM IKDD CoDS and 25th COMAD},
  pages = {249--253},
  year = {2020},
  doi = {10.1145/3371158.3371196}
}

@inproceedings{tan2019,
  author = {M. Tan and Q. V. Le},
  title = {{EfficientNet}: Rethinking Model Scaling for Convolutional Neural Networks},
  booktitle = {Proc. International Conference on Machine Learning (ICML)},
  pages = {6105--6114},
  year = {2019}
}

@article{ramcharan2019,
  author = {A. Ramcharan and P. McCloskey and K. Baranowski and N. Mbilinyi and L. Mrisho and M. Ndalahwa and J. Legg and D. P. Hughes},
  title = {A Mobile-Based Deep Learning Model for Cassava Disease Diagnosis},
  journal = {Frontiers in Plant Science},
  volume = {10},
  pages = {272},
  year = {2019},
  doi = {10.3389/fpls.2019.00272}
}

@article{li2017feature,
  author = {J. Li and K. Cheng and S. Wang and F. Morstatter and R. P. Trevino and J. Tang and H. Liu},
  title = {Feature Selection: A Data Perspective},
  journal = {ACM Computing Surveys},
  volume = {50},
  number = {6},
  pages = {94:1--94:45},
  year = {2017},
  doi = {10.1145/3136625}
}

@misc{nasapower2018,
  author = {P. W. Stackhouse and others},
  title = {{NASA POWER}: Prediction of Worldwide Energy Resources Agroclimatology Methodology},
  howpublished = {NASA Langley Research Center},
  year = {2018},
  url = {https://power.larc.nasa.gov}
}
```

---

## 9. Revised Six-Page Feasibility Assessment

### Assessment Classification: PLANNING ESTIMATE / NOT YET COMPILED

> [!WARNING]
> The six-page assessment in the Phase 4 report was formulated based on line counts and manual page budgeting. **No compiled LaTeX output or revised PDF exists as evidence that the revised text fits within exactly 6.0 pages.**

### Factors Subject to Overleaf LaTeX Engine Behavior:
1. **Two-Column Float Scheduling:** In IEEEtran, `table*` and `figure*` environments (such as Table~\ref{tab:anova_matrix} and Figure~\ref{fig:architecture}) float to the top of the next page following their definition. If placed late on Page 3 or 4, they may unexpectedly float to Page 5, pushing text into Page 7.
2. **Bibliography Font Size and Formatting:** Depending on whether `\small` or `\footnotesize` is applied to `\bibliography`, references [1]–[12] may take between 25 and 45 lines.
3. **Subfigure and Math Line Wraps:** Equations (\ref{eq:anova_f}) and (\ref{eq:embedding}) must not break column boundaries.

### Estimated Line Savings (Planning Guide Only):
- **Literature Survey Condensation (`CHG-03`):** Estimated savings: ~55 lines.
- **Elimination of Disqualified PlantDoc Fine-Tuning (`CHG-07`):** Estimated savings: ~25 lines.
- **Grad-CAM Sizing (`CHG-14`, width=0.88\columnwidth):** Anchored to Page 6; eliminates Page 8 overflow.
- **Pruning Malformed References (`CHG-15`):** Reduces 21 references (many with orphaned URL lines) to 10–12 clean entries, saving ~35 lines.

**Action Required by Author:** Apply the corrected text in Overleaf, compile the PDF, inspect page breaks, and adjust float positions (`[t]`) to guarantee that the final document terminates on Page 6.

---

## 10. Updated Change ID Status Matrix (CHG-01 through CHG-15)

| Change ID | Title / Section | Phase 4 Status | Phase 4.1 Audited Status | Core Problem Identified & Correction |
|---|---|---|---|---|
| **CHG-01** | Abstract (Page 1) | Proposed | **CORRECTED** | Removed "statistical parity" and "proves redundancy"; properly scoped 50% head compression and projection latency vs parameter metrics. |
| **CHG-02** | Introduction Contributions (Page 2) | Proposed | **CORRECTED** | Replaced "statistical parity" with permitted null-hypothesis phrasing; separated projection latency and parameter settings. |
| **CHG-03** | Related Work Condensation (Pages 2–3) | Proposed | **CORRECTED** | Removed unverified `hollier1997` and misattributed `khaki2019`; cited `li2017feature` for ANOVA filtering. |
| **CHG-04** | Dataset Composition (Page 3) | Proposed | **RETAINED** | Partition arithmetic ($N=9,299 = 36 \times 250 + 106 + 193$) is completely verified and accurate. |
| **CHG-05** | Visual Feature Extraction (Page 4) | Proposed | **RETAINED** | Layer dimensions ($1792 \to 512$) and activation equations are verified against `best_model.pth`. |
| **CHG-06** | ANOVA Formulation & Dual Paradigms (Pages 4–5)| Proposed | **CORRECTED** | Replaced misplaced `\cite{ding2005}` with `\cite{li2017feature}`. |
| **CHG-07** | PlantDoc Zero-Shot Protocol (Page 5) | Proposed | **CORRECTED** | Explicitly documented that predictions into the 13 unsupported PlantVillage classes are counted as incorrect predictions ($N=219$). |
| **CHG-08** | Multimodal Taxonomy & Projection (Page 5) | Proposed | **CORRECTED** | Clarified multimodal macro-framework; removed unverified citation `hollier1997`. |
| **CHG-09** | Baselines & PlantVillage Benchmarks (Page 6) | Proposed | **REQUIRES AUTHOR DECISION** | Corrected standalone CNN v1 validation metrics (99.51% acc, 0.9928 Macro-F1). Provided Option A and Option B for historical multimodal row. |
| **CHG-10** | ANOVA Full Matrix & Analysis (Pages 6–7) | Proposed | **CORRECTED** | Replaced "statistical parity"; labeled storage column as *Exp. A Head Storage*; removed unsupported intrinsic-rank claim; scoped projection metrics. |
| **CHG-11** | Conclusion & Future Scope (Page 7) | Proposed | **CORRECTED** | Replaced "confirmed statistical parity" with permitted comparative phrasing. |
| **CHG-12** | Figure 1 Architecture Caption (Page 4) | Proposed | **CORRECTED** | Revised caption to separate multimodal architecture from visual feature-selection evaluation. |
| **CHG-13** | Figure 3 Dual-Panel Plot (Page 6) | Proposed | **RETAINED** | Dual-panel trajectory plot is verified and ready for Overleaf upload. |
| **CHG-14** | Figure 4 Grad-CAM Repositioning (Page 6) | Proposed | **RETAINED** | Constraining width to `0.88\columnwidth` pulls float to Page 6. |
| **CHG-15** | References & BibTeX Cleanup (Page 7/8) | Proposed | **CORRECTED** | Fixed ACM CSUR venue for `li2017feature`; removed corrupted DOI for `garg2021`; removed phantom and obsolete citations. |

---

## 11. Prioritized Manual Overleaf Correction Checklist

Follow these steps sequentially in Overleaf:

1. **Step 1: Update Abstract (`CHG-01`)**
   - Replace the `\begin{abstract}...\end{abstract}` block with the text from Section 6.1.
2. **Step 2: Update Introduction Contributions (`CHG-02`)**
   - Replace the four bullet points in Section I with Section 6.2.
3. **Step 3: Condense Related Work (`CHG-03`)**
   - Replace Section II with the three unified paragraphs in Section 6.3.
4. **Step 4: Confirm Dataset Arithmetic (`CHG-04`)**
   - Keep the partition description from Phase 4 (already verified).
5. **Step 5: Insert ANOVA Math & Paradigms (`CHG-06`)**
   - Insert Section III-C from Section 6.4 (citing `li2017feature`).
6. **Step 6: Insert Zero-Shot Protocol with Unsupported Class Semantics (`CHG-07`)**
   - Insert Section III-D from Section 6.5.
7. **Step 7: Update Multimodal Architecture Description (`CHG-08`)**
   - Update Section III-F with the text from Section 6.6.
8. **Step 8: Select Baseline Table Strategy (`CHG-09`)**
   - Insert Table~\ref{tab:baselines} using Option A (Section 6.7) or Option B (Section 4.2).
9. **Step 9: Insert Full ANOVA Matrix & Corrected Analysis (`CHG-10`)**
   - Insert full-width Table~\ref{tab:anova_matrix} and the four subsections from Section 6.8.
10. **Step 10: Replace Conclusion (`CHG-11`)**
    - Replace Section V with Section 6.9.
11. **Step 11: Update Figure 1 Caption (`CHG-12`)**
    - Update Figure 1 caption using Section 6.10.
12. **Step 12: Upload and Replace Figure 3 (`CHG-13`)**
    - Upload `results/phase3/figures/fig1_accuracy_vs_dimension.png` to Overleaf.
13. **Step 13: Reposition Grad-CAM Figure 4 (`CHG-14`)**
    - Set `\includegraphics[width=0.88\columnwidth]{Potato leaf early blight (3).jpg}` and anchor to top of Page 6.
14. **Step 14: Replace BibTeX File (`CHG-15`)**
    - Replace malformed bib entries with the clean BibTeX records from Section 8.2.
15. **Step 15: Compile and Verify Page Budget (`Section 9`)**
    - Click **Recompile** in Overleaf. Inspect the resulting PDF to verify that the text terminates neatly on Page 6 with zero overflow onto Page 7.

---

## 12. Remaining Uncertainties and Evidence Gaps

The following minor items cannot be resolved without direct input from the author:
1. **Selection between Option A and Option B for Table I (`CHG-09`):** Whether the author prefers a clean visual-only baseline comparison (Option A) or retaining the historical multimodal row with an explicit qualification footnote (Option B).
2. **Replacement for Weather Epidemiology Citation (`CHG-08`):** Since `hollier1997` was verified as phantom, the author should decide whether to cite a specific local extension bulletin or rely on general agrometeorological disease forecasting literature.
3. **Final LaTeX Engine Float Decisions:** Float layout behavior in Overleaf is dependent on exact column breaks and requires compilation verification by the author.

---

## 13. Final Scope and Integrity Verification

Before concluding, the integrity of the workspace was formally verified:
1. **Original Phase 4 Report Unchanged:** `docs/phase_reports/phase4_paper_revision_guidance_report.md` was inspected and remains unmodified.
2. **Original Paper PDFs Unchanged:** `DLM_Research_Paper_v6.pdf` and `DLM_Research_Paper_v5.pdf` are untouched.
3. **No Code, Checkpoints, Manifests, or TeX Files Modified:** No `.tex`, `.bib`, image, dataset, or source file in the repository was altered.
4. **No New Experiments or Evaluations Run:** All audits utilized pre-existing frozen evidence and records.
5. **Zero Git Commits or Pushes:** The working tree remains unchanged relative to tracked files.
6. **Exactly One New Report Produced:** `docs/phase_reports/phase4_1_correction_and_evidence_verification_report.md`.

---
*End of Phase 4.1 Correction and Evidence Verification Report.*

