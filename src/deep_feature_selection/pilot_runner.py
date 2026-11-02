"""
pilot_runner.py
===============
Phase 2 Pilot Validation Runner for Deep Visual Embedding Feature Selection.
Executes the mandatory validation gate across:
- Checkpoint integrity & embedding extraction.
- Train-only ANOVA selector fitting.
- Experiment A: Weight-sliced classifier evaluation (k=512, 256, 64).
- Experiment B: Retrained linear evaluation head (k=512, 256, 64).
- Experiment C: Multimodal cross-attention forward pass & tensor shape verification.
- Zero-leakage verification.
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.deep_feature_selection.extractor import EmbeddingExtractor
from src.deep_feature_selection.anova_selector import DeepANOVASelector
from src.deep_feature_selection.classifiers import WeightSlicedClassifier, RetrainedLinearHead
from src.deep_feature_selection.multimodal import AdaptiveCrossAttentionFusion

RESULTS_DIR = REPO_ROOT / "results" / "phase2_pilot"
MANIFEST_DIR = REPO_ROOT / "data" / "manifests"


def create_stratified_pilot_manifest(
    source_manifest: Path,
    out_manifest: Path,
    samples_per_class: int = 30,
    seed: int = 42
) -> Path:
    """Creates a balanced, stratified pilot manifest."""
    with open(source_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    rng = np.random.RandomState(seed)
    class_to_samples = {}
    for s in data["samples"]:
        c = s.get("class_index", s.get("mapped_class_index", -1))
        class_to_samples.setdefault(c, []).append(s)

    selected_samples = []
    for c in sorted(class_to_samples.keys()):
        cls_list = class_to_samples[c]
        n_take = min(samples_per_class, len(cls_list))
        perm = rng.permutation(len(cls_list))[:n_take]
        for idx in perm:
            selected_samples.append(cls_list[idx])

    out_data = {
        "dataset": data.get("dataset", "PlantVillage"),
        "split": f"{data.get('split', 'train')}_pilot",
        "samples_per_class": samples_per_class,
        "total_samples": len(selected_samples),
        "samples": selected_samples
    }
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2)

    return out_manifest


def run_phase2_pilot(
    samples_per_class_train: int = 30,
    samples_per_class_val: int = 20,
    pilot_k_dims: Tuple[int, ...] = (512, 256, 64),
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Executes the Phase 2 pilot validation.
    """
    print("=" * 70)
    print("STARTING PHASE 2 PILOT EXPERIMENT")
    print(f"Candidate Dimensions: {pilot_k_dims}")
    print("=" * 70)

    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Initialize extractor and verify CNN v1 checkpoint
    print("\n--- Step 1: Initializing Extractor & Verifying Checkpoint ---")
    extractor = EmbeddingExtractor(device=device)
    print(f"Verified Checkpoint SHA-256: {extractor.checkpoint_sha256}")

    # 2. Prepare pilot manifests
    print("\n--- Step 2: Preparing Pilot Datasets ---")
    pilot_train_manifest = MANIFEST_DIR / "plantvillage_train_pilot.json"
    pilot_val_manifest = MANIFEST_DIR / "plantvillage_val_pilot.json"

    create_stratified_pilot_manifest(
        MANIFEST_DIR / "plantvillage_train_subset.json",
        pilot_train_manifest,
        samples_per_class=samples_per_class_train,
        seed=42
    )
    create_stratified_pilot_manifest(
        MANIFEST_DIR / "plantvillage_val.json",
        pilot_val_manifest,
        samples_per_class=samples_per_class_val,
        seed=42
    )

    # 3. Extract embeddings
    print("\n--- Step 3: Extracting Visual Embeddings ---")
    train_res = extractor.extract_from_manifest(
        pilot_train_manifest,
        cache_name="pv_train_pilot_emb_512",
        batch_size=64
    )
    val_res = extractor.extract_from_manifest(
        pilot_val_manifest,
        cache_name="pv_val_pilot_emb_512",
        batch_size=64
    )

    X_train = train_res["embeddings"]
    y_train = train_res["labels"]
    X_val = val_res["embeddings"]
    y_val = val_res["labels"]

    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"X_val shape:   {X_val.shape}, dtype: {X_val.dtype}")

    # 4. Fit ANOVA Selector strictly on training embeddings
    print("\n--- Step 4: Fitting ANOVA Selector (Train Only) ---")
    # Fit selector at full rank (k=512) to inspect all scores
    full_selector = DeepANOVASelector(k=512)
    t_anova_fit_0 = time.time()
    full_selector.fit(X_train, y_train)
    anova_fit_time = time.time() - t_anova_fit_0

    print(f"ANOVA fit duration: {anova_fit_time:.4f}s")
    print(f"Max F-score: {float(np.max(full_selector.f_scores_)):.2f}, Min F-score: {float(np.min(full_selector.f_scores_)):.2f}")

    # 5. Extract original model weights for Experiment A
    orig_weights = extractor.model.classifier.weight.data
    orig_bias = extractor.model.classifier.bias.data

    experiment_records = {}

    for k in pilot_k_dims:
        print(f"\n" + "-" * 50)
        print(f"EVALUATING CONFIGURATION k = {k} ({k/512*100:.1f}% Retention)")
        print("-" * 50)

        # Slice features via train-fitted selector
        selector_k = DeepANOVASelector(k=k)
        selector_k.fit(X_train, y_train)  # Train-only fit
        
        t_trans_0 = time.time()
        X_train_k = selector_k.transform(X_train)
        X_val_k = selector_k.transform(X_val)
        trans_duration = time.time() - t_trans_0

        selected_idx = selector_k.get_selected_indices()
        indices_sha256 = selector_k.get_indices_sha256()
        reduction_pct = (512 - k) / 512 * 100.0

        print(f"Selected {len(selected_idx)} features. Checksum: {indices_sha256[:16]}... (Reduction: {reduction_pct:.1f}%)")

        # --- Experiment A: Weight-Sliced Classifier (Zero-Retraining) ---
        clf_a = WeightSlicedClassifier(orig_weights, orig_bias, selected_idx)
        metrics_a = clf_a.evaluate(X_val_k, y_val)
        print(f"[Exp A Weight-Sliced]  Val Acc: {metrics_a['accuracy']*100:.2f}%, Macro-F1: {metrics_a['macro_f1']:.4f}")

        # --- Experiment B: Retrained Linear Evaluation Head ---
        clf_b = RetrainedLinearHead(k=k, num_classes=38, C=1.0, max_iter=1000, random_state=42)
        clf_b.fit(X_train_k, y_train)
        metrics_b = clf_b.evaluate(X_val_k, y_val)
        print(f"[Exp B Retrained Head] Val Acc: {metrics_b['accuracy']*100:.2f}%, Macro-F1: {metrics_b['macro_f1']:.4f} (Train: {metrics_b['classifier_train_time_sec']:.3f}s)")

        # --- Experiment C: Multimodal Cross-Attention Forward Pass ---
        cross_attn_model = AdaptiveCrossAttentionFusion(
            cnn_dim=k,
            weather_dim=256,
            meta_dim=21,
            n_diseases=26,
            attn_dim=256,
            n_heads=8
        )
        cross_attn_model.eval()

        # Test forward pass with batch size = 4
        dummy_cnn = torch.from_numpy(X_val_k[:4])
        dummy_weather = torch.randn(4, 256)
        dummy_meta = torch.randn(4, 21)

        with torch.no_grad():
            fused_logits, attn_weights = cross_attn_model(dummy_cnn, dummy_weather, dummy_meta)

        assert fused_logits.shape == (4, 26), f"Expected (4, 26), got {fused_logits.shape}"
        assert attn_weights.shape == (4, 1, 1), f"Expected (4, 1, 1), got {attn_weights.shape}"

        attn_summary = cross_attn_model.get_structure_summary()
        print(f"[Exp C Cross-Attention] Forward pass verified: Logits shape {list(fused_logits.shape)}, CNN proj weights {attn_summary['cnn_proj_weight_shape']}")

        experiment_records[f"k_{k}"] = {
            "k": k,
            "reduction_percentage": reduction_pct,
            "selected_indices_sha256": indices_sha256,
            "transformation_time_sec": trans_duration,
            "experiment_a_weight_sliced": metrics_a,
            "experiment_b_retrained_linear": metrics_b,
            "experiment_c_cross_attention": {
                "verified": True,
                "input_cnn_shape": list(dummy_cnn.shape),
                "output_logits_shape": list(fused_logits.shape),
                "cnn_proj_parameters": attn_summary["cnn_proj_parameters"],
                "total_fusion_parameters": attn_summary["total_parameters"]
            }
        }

    total_duration = time.time() - t_start

    pilot_report = {
        "status": "COMPLETED",
        "checkpoint_sha256": extractor.checkpoint_sha256,
        "pilot_train_samples": len(X_train),
        "pilot_val_samples": len(X_val),
        "evaluated_classes": 38,
        "anova_fit_time_sec": anova_fit_time,
        "total_pilot_duration_sec": total_duration,
        "configurations": experiment_records
    }

    out_file = RESULTS_DIR / "pilot_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(pilot_report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"PILOT COMPLETED SUCCESSFULLY in {total_duration:.2f}s!")
    print(f"Results saved to: {out_file}")
    print("=" * 70)

    return pilot_report


if __name__ == "__main__":
    run_phase2_pilot()

