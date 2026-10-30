# Dataset Protocols, Manifests, and Access Guide

This directory documents the dataset partitions, class mappings, and manifest structures utilized throughout the Pattern Recognition CCA 1 case study.

In compliance with repository hygiene and copyright redistribution policies, raw image files (totaling >2 GB) are **not tracked in this Git repository**. Instead, full dataset manifests containing deterministic image identifiers, class mappings, split assignments, and cryptographic verification flags are provided in [`data/manifests/`](manifests/).

---

## 1. Datasets Overview

### 1.1 PlantVillage (In-Domain Laboratory Benchmark)
- **Origin:** Hughes, D. P., & Salathé, M. (2015). *"An open access repository of images on plant health to enable the development of mobile disease diagnostics."* arXiv:1511.08060.
- **Characteristics:** Detached leaves mounted against homogeneous solid grey or black paper backgrounds under uniform laboratory studio illumination.
- **Total Dataset Size:** 54,305 images across 38 distinct crop-disease categories (14 crop species, 26 diseases, 12 healthy controls).
- **Partitioning Protocol:**
  - **Full Training Split:** 38,013 images used to train the deep CNN baseline (EfficientNet-B4 CNN v1).
  - **Standardized Classical Training Subset:** Exactly **9,299 samples** used to fit the Linear SVM, Random Forest, and train-only `StandardScaler`.
  - **Held-Out Validation Split:** Exactly **8,146 samples** evaluated identically across all three models.
- **Manifest:** [`data/manifests/plantvillage_val.json`](manifests/plantvillage_val.json) ($N = 8,146$ samples).

### 1.2 PlantDoc (Real-World Out-of-Domain Field Benchmark)
- **Origin:** Singh, D., Jain, N., Jain, A., Kayal, P., Kumawat, S., & Batra, N. (2020). *"PlantDoc: A Dataset for Collaborative Computer Vision Applications in Agriculture."* ACM CoDS-COMAD 2020.
- **Characteristics:** Uncontrolled agricultural field photography containing natural sunlight, complex soil/weed backgrounds, camera noise, scale variation, and multiple overlapping leaves.
- **Test Set Size:** 236 images total on disk.
- **Class Reconciliation & Mapping Audit:**
  - **Mapped Categories:** Exactly **25 classes** (219 images) successfully mapped to official PlantVillage classes.
  - **Excluded Categories:** Exactly **2 classes** (17 images: 8 `Bell_pepper leaf`, 9 `Bell_pepper leaf spot`) excluded as unmapped.
  - **Train-Test Duplication Audit:** 11 duplicate pairs identified between PlantDoc test and PlantVillage training sets (3 consistent, 8 conflicting).
- **Three Documented Evaluation Variants:**
  1. *Full Mapped Variant ($N = 219$):* All 219 mapped field images.
  2. *Conflict-Excluded Variant ($N = 211$):* Excludes the 8 contradictory-label duplicate images.
  3. *Fully Deduplicated Variant ($N = 208$):* Excludes all 11 train-test duplicate images.
- **Manifest:** [`data/manifests/plantdoc_test.json`](manifests/plantdoc_test.json) ($N = 236$ samples, including mapping and duplicate flags).

---

## 2. Manifest Schema

Every sample entry in `plantvillage_val.json` and `plantdoc_test.json` follows a strict JSON schema:

```json
{
  "image_id": "8c11697b-df51-49c6-b240-c6e5b6a529e4___CREC_HLB 7268.JPG",
  "relative_path": "plantvillage/Orange___Haunglongbing_(Citrus_greening)/...",
  "class_name": "Orange___Haunglongbing_(Citrus_greening)",
  "class_index": 15,
  "is_mapped": true,
  "is_duplicate_with_train": false,
  "has_label_conflict": false
}
```

---

## 3. How to Obtain the Raw Datasets

To execute full model evaluation from raw image files:
1. **PlantVillage Dataset:** Download the color image collection from [PlantVillage Crowdsourced Repository](https://github.com/spMohanty/PlantVillage-Dataset).
2. **PlantDoc Dataset:** Download from [PlantDoc Official Repository](https://github.com/pratikkayal/PlantDoc-Dataset).
3. Place raw images in `data/raw/plantvillage/` and `data/raw/plantdoc/` respectively, matching the relative paths in the manifests.
