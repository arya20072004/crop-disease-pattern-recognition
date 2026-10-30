# Dataset Documentation & Provenance

## 1. PlantVillage Benchmark (In-Domain Laboratory Data)
- **Source:** Hughes & Salathé (2015), Penn State / EPFL.
- **Samples:** 54,305 high-resolution leaf photographs.
- **Classes:** 38 crop disease categories across 14 plant species.
- **Partitioning:**
  - Training Partition: 38,013 images (70%).
  - Validation Partition: 8,146 images (15%).
  - Held-Out Test Partition: 8,146 images (15%).
- **Class Distribution & Imbalance:**
  - Most frequent class: `Soybean___healthy` (409 validation images).
  - Least frequent class: `Apple___Cedar_apple_rust` (41 validation images).
  - Imbalance Ratio: $10.0\times$.

---

## 2. PlantDoc Benchmark (Real-World Out-of-Domain Data)
- **Source:** Singh et al. (2020), IIIT Delhi.
- **Samples:** 236 images in the official test partition.
- **Class Mapping Audit (Stage 4B Reconciliation):**
  - Exactly **25 categories** mapped cleanly to official PlantVillage classes.
  - Exactly **2 categories** (17 images: 8 `Bell_pepper leaf`, 9 `Bell_pepper leaf spot`) are excluded as unmapped.
- **Duplication & Label Conflict Audit:**
  - Binary image matching (MD5 hash analysis) identified **11 duplicate pairs** between PlantDoc test and PlantVillage training sets.
  - 3 pairs exhibited consistent diagnoses.
  - 8 pairs exhibited contradictory medical diagnoses (e.g. `Tomato Early blight` vs `Tomato Late blight`).
- **Audit Variants:**
  - *Full Mapped:* $N = 219$ (25 classes).
  - *Conflict-Excluded:* $N = 211$ (excludes the 8 contradictory duplicates).
  - *Fully Deduplicated:* $N = 208$ (excludes all 11 duplicate pairs).
