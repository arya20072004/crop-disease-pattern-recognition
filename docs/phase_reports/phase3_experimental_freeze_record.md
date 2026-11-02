# Phase 3 Experimental Freeze Gate Record

**Execution Timestamp:** 2026-11-01T12:21:55Z  
**Status:** **LOCKED & FROZEN PRIOR TO TEST EVALUATION**  
**Authorized Reference Checkpoint:** `models/weights/best_model.pth`  
**Checkpoint SHA-256:** `e48a4d2f0562282d930bc0583017a45ad243083b38e102aa5eacfd401b6fcabb`  

---

## 1. Frozen Experimental Configuration

| Component | Frozen Specification |
|---|---|
| **Reference Architecture** | EfficientNet-B4 CNN v1 (`CropDiseaseClassifier`) |
| **Feature Extraction Layer** | `Linear(1792, 512) -> ReLU` |
| **Output Embedding Space** | $\mathbb{R}^{512}$ |
| **Preprocessing Parameters** | Resolution: $380 \times 380$, RGB, `cv2.INTER_AREA`, ImageNet Norm |
| **Primary Training Partition** | `data/manifests/plantvillage_train_subset.json` ($N=9,299$) |
| **Validation Partition** | `data/manifests/plantvillage_val.json` ($N=8,146$) |
| **Frozen PlantVillage Test Partition** | `data/manifests/plantvillage_test.json` ($N=8,146$) |
| **Frozen PlantDoc Test Partition** | `data/manifests/plantdoc_test.json` ($N=236$, 219 mapped) |
| **Feature Selector** | One-Way ANOVA $F$-statistic (Train-Only Fit) |
| **Tie-Breaking Rule** | 64-bit float score descending, feature index ascending |
| **Evaluated Dimensions ($K$)** | `[512, 384, 256, 192, 128, 64, 32]` |
| **Experiment A Head** | Weight-Sliced ($W_{:, I_k} \in \mathbb{R}^{38 \times k}$), Zero Retraining |
| **Experiment B Head** | Retrained Linear Head (Multinomial Logistic Regression, $C=1.0$, L-BFGS) |
| **Experiment C Architecture** | `AdaptiveCrossAttentionFusion` ($W_{\mathrm{cnn\_proj}} \in \mathbb{R}^{256 \times k}$, 26 Disease Classes) |
| **Confidence Rejection Threshold** | $\tau = 0.65$ |

---

## 2. Frozen ANOVA Feature Subsets

Master Ranking SHA-256: `c677dbab0d408f73050b99358f7b867d393f9542ed93ed2fa4ffde57fbdc2d68`

| Target Dimension ($k$) | Retained (%) | Reduction (%) | Indices SHA-256 Checksum |
|---|---|---|---|
| **$k=512$** | 100.0% | 0.0% | `5738153ec97595b1c1e4dc027f7b7fb4534f19ed2ce9f9ee712e6d34a384cde7` |
| **$k=384$** | 75.0% | 25.0% | `b9e19d45a71f07facdd1902886a606f1981d68463ddcc37269024ff553bf2fb1` |
| **$k=256$** | 50.0% | 50.0% | `4b920692c27e8d76c6bf15248991f366f6830321e210861f19b42a6353b3e84b` |
| **$k=192$** | 37.5% | 62.5% | `4d0a55fcc972eaf420fc7a0cfc49d7c46cab81c2b1538000bfa7a72c8c680481` |
| **$k=128$** | 25.0% | 75.0% | `859251cb42cad16379dae10929aebeedb37f9211acf55fee0d4c35a78ee2199b` |
| **$k=64$** | 12.5% | 87.5% | `c7341e32733bad09d621340587b48cf3d9a043ac9dcf243cf9bc49f505282e14` |
| **$k=32$** | 6.2% | 93.8% | `57c9887060eb598e9252ac7e2c687a4aaa41739fc6ba55af4b0d923873f48f25` |

---

## 3. Validation Pareto Frontiers & Model Selection

### 3.1 Experiment A (Weight-Sliced) Validation Pareto Frontier
- **$k=512$**: Acc = 99.45%, Macro-F1 = 0.9922, Latency = 5.34 $\mu$s, Weights = 77,976 bytes
- **$k=384$**: Acc = 99.40%, Macro-F1 = 0.9917, Latency = 4.76 $\mu$s, Weights = 58,520 bytes
- **$k=256$**: Acc = 98.24%, Macro-F1 = 0.9754, Latency = 2.22 $\mu$s, Weights = 39,064 bytes
- **$k=192$**: Acc = 94.67%, Macro-F1 = 0.9395, Latency = 1.43 $\mu$s, Weights = 29,336 bytes
- **$k=128$**: Acc = 85.21%, Macro-F1 = 0.8212, Latency = 1.06 $\mu$s, Weights = 19,608 bytes
- **$k=64$**: Acc = 63.28%, Macro-F1 = 0.6287, Latency = 0.72 $\mu$s, Weights = 9,880 bytes
- **$k=32$**: Acc = 36.67%, Macro-F1 = 0.4058, Latency = 0.65 $\mu$s, Weights = 5,016 bytes

### 3.2 Experiment B (Retrained Head) Validation Pareto Frontier
- **$k=384$**: Acc = 99.42%, Macro-F1 = 0.9925, Latency = 11.01 $\mu$s, Weights = 117,040 bytes
- **$k=256$**: Acc = 99.36%, Macro-F1 = 0.9918, Latency = 8.34 $\mu$s, Weights = 78,128 bytes
- **$k=192$**: Acc = 99.23%, Macro-F1 = 0.9906, Latency = 6.38 $\mu$s, Weights = 58,672 bytes
- **$k=128$**: Acc = 98.75%, Macro-F1 = 0.9859, Latency = 4.24 $\mu$s, Weights = 39,216 bytes
- **$k=64$**: Acc = 94.27%, Macro-F1 = 0.9417, Latency = 3.81 $\mu$s, Weights = 19,760 bytes
- **$k=32$**: Acc = 83.39%, Macro-F1 = 0.8539, Latency = 4.14 $\mu$s, Weights = 10,032 bytes

### 3.3 Selected Final Locked Configurations for Test Evaluation
- **Experiment A Selected Pareto Dimension:** **$k = 256$** (Acc = 98.24%, Macro-F1 = 0.9754)
- **Experiment B Selected Pareto Dimension:** **$k = 256$** (Acc = 99.36%, Macro-F1 = 0.9918)
- **Uncompressed Reference Baseline:** **$k = 512$** (Acc = 99.45%, Macro-F1 = 0.9922)

**CERTIFICATION:** This configuration is locked. Test evaluations will be executed strictly against these frozen parameters without refitting or alteration.
