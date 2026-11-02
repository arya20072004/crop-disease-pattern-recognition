"""
extractor.py
============
Deterministic, reproducible EfficientNet-B4 visual embedding extraction pipeline.
Extracts 512-dimensional output from the verified CNN v1 checkpoint:
Dropout(0.4) -> Linear(1792, 512) -> ReLU.
"""

import sys
import os
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cnn.cnn_adapter import CropDiseaseClassifier, preprocess_image, CONFIG_CLASS_NAMES

CHECKPOINT_V1_EXPECTED_SHA256 = "e48a4d2f0562282d930bc0583017a45ad243083b38e102aa5eacfd401b6fcabb"
DEFAULT_CHECKPOINT_PATH = REPO_ROOT / "models" / "weights" / "best_model.pth"
FALLBACK_CHECKPOINT_PATH = REPO_ROOT.parent / "crop-disease-detection" / "archive" / "increment_1_cnn" / "best_model.pth"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"


def compute_file_sha256(filepath: Union[str, Path]) -> str:
    """Computes SHA-256 digest of a file in chunks."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class EmbeddingExtractor:
    """
    Extracts and caches 512-dimensional visual embeddings using the verified CNN v1 model.
    """
    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.checkpoint_path = self._resolve_checkpoint(checkpoint_path)
        self.checkpoint_sha256 = self._verify_checkpoint(self.checkpoint_path)
        
        # Load class names
        with open(CONFIG_CLASS_NAMES, "r", encoding="utf-8") as f:
            self.class_names = json.load(f)
        self.num_classes = len(self.class_names)
        
        # Initialize architecture
        self.model = CropDiseaseClassifier(num_classes=self.num_classes)
        state_dict = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False

    def _resolve_checkpoint(self, path: Optional[Union[str, Path]]) -> Path:
        if path is not None and Path(path).exists():
            return Path(path)
        if DEFAULT_CHECKPOINT_PATH.exists():
            return DEFAULT_CHECKPOINT_PATH
        if FALLBACK_CHECKPOINT_PATH.exists():
            return FALLBACK_CHECKPOINT_PATH
        raise FileNotFoundError(
            f"CNN v1 checkpoint not found at {DEFAULT_CHECKPOINT_PATH} or {FALLBACK_CHECKPOINT_PATH}"
        )

    def _verify_checkpoint(self, path: Path) -> str:
        sha = compute_file_sha256(path)
        if sha != CHECKPOINT_V1_EXPECTED_SHA256:
            raise ValueError(
                f"Checkpoint integrity violation! Expected {CHECKPOINT_V1_EXPECTED_SHA256}, got {sha}"
            )
        return sha

    def extract_from_manifest(
        self,
        manifest_path: Union[str, Path],
        cache_name: Optional[str] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        batch_size: int = 64,
        max_samples: Optional[int] = None,
        force_extract: bool = False
    ) -> Dict[str, Any]:
        """
        Extracts 512-d embeddings for all samples in a dataset manifest.
        Uses cached .npz if available and valid.
        """
        m_path = Path(manifest_path)
        if not m_path.exists():
            raise FileNotFoundError(f"Manifest not found: {m_path}")

        c_dir = Path(cache_dir) if cache_dir else DEFAULT_CACHE_DIR
        c_dir.mkdir(parents=True, exist_ok=True)

        name = cache_name or f"{m_path.stem}_emb_512"
        if max_samples:
            name += f"_sample{max_samples}"

        npz_path = c_dir / f"{name}.npz"
        meta_path = c_dir / f"{name}_meta.json"

        # Check existing cache
        if not force_extract and npz_path.exists() and meta_path.exists():
            print(f"[CACHE HIT] Loading embeddings from {npz_path.name}")
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            data = np.load(npz_path, allow_pickle=False)
            return {
                "embeddings": data["embeddings"],
                "labels": data["labels"],
                "sample_ids": [str(x) for x in data["sample_ids"]],
                "metadata": meta,
                "cached": True,
                "npz_path": str(npz_path)
            }

        print(f"[EXTRACTING] Processing manifest {m_path.name}...")
        with open(m_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        manifest_sha = compute_file_sha256(m_path)
        samples = manifest_data["samples"]
        if max_samples:
            samples = samples[:max_samples]

        total = len(samples)
        sample_ids = []
        labels = []
        img_paths = []

        for s in samples:
            sample_ids.append(s.get("filename", s.get("relative_path", "")))
            if "class_index" in s:
                labels.append(int(s["class_index"]))
            elif "mapped_class_index" in s:
                labels.append(int(s["mapped_class_index"]))
            else:
                labels.append(-1)
            
            # Resolve relative path locally or via fallback
            p = REPO_ROOT / s["relative_path"]
            if not p.exists():
                p = REPO_ROOT.parent / "crop-disease-detection" / s["relative_path"]
            if not p.exists():
                raise FileNotFoundError(f"Image not found on disk: {s['relative_path']}")
            img_paths.append(p)

        # Shards directory for resilient sequential batch caching
        shards_dir = c_dir / "_shards" / name
        shards_dir.mkdir(parents=True, exist_ok=True)

        # Batch extraction loop with sharded caching
        t0 = time.time()
        num_batches = (total + batch_size - 1) // batch_size
        all_embeddings = []

        with ThreadPoolExecutor(max_workers=4) as executor:
            for b_idx in range(num_batches):
                start_i = b_idx * batch_size
                end_i = min(start_i + batch_size, total)
                shard_path = shards_dir / f"batch_{b_idx:05d}.npz"

                if shard_path.exists():
                    shard_data = np.load(shard_path, allow_pickle=False)
                    emb_batch = shard_data["embeddings"]
                else:
                    batch_paths = img_paths[start_i:end_i]
                    # Parallel image loading & preprocessing across 4 worker threads
                    batch_tensors = list(executor.map(preprocess_image, batch_paths))
                    x_batch = torch.from_numpy(np.stack(batch_tensors)).to(self.device)

                    with torch.no_grad():
                        emb_batch = self.model.extract_embedding(x_batch).cpu().numpy()

                    # Save shard checkpoint
                    np.savez(
                        shard_path,
                        embeddings=emb_batch,
                        labels=np.array(labels[start_i:end_i], dtype=np.int64),
                        sample_ids=np.array(sample_ids[start_i:end_i], dtype=str)
                    )

                all_embeddings.append(emb_batch)

                if (b_idx + 1) % 10 == 0 or (b_idx + 1) == num_batches:
                    curr_elapsed = time.time() - t0
                    print(f"[{m_path.stem}] Batch {b_idx + 1}/{num_batches} ({end_i}/{total}) - {end_i/curr_elapsed:.1f} img/s", flush=True)

        elapsed = time.time() - t0
        embeddings_arr = np.concatenate(all_embeddings, axis=0).astype(np.float32)
        labels_arr = np.array(labels, dtype=np.int64)
        sample_ids_arr = np.array(sample_ids, dtype=str)

        assert embeddings_arr.shape == (total, 512), f"Expected ({total}, 512), got {embeddings_arr.shape}"
        assert len(labels_arr) == total

        # Comprehensive Phase 3 Metadata Run Envelope
        meta = {
            "model_name": "EfficientNet-B4_CNN_v1",
            "checkpoint_sha256": self.checkpoint_sha256,
            "manifest_name": m_path.name,
            "manifest_sha256": manifest_sha,
            "num_samples": total,
            "embedding_dim": 512,
            "batch_size": batch_size,
            "num_batches": num_batches,
            "preprocessing": {
                "input_resolution": [380, 380],
                "color_format": "RGB",
                "interpolation": "INTER_AREA",
                "normalization_mean": [0.485, 0.456, 0.406],
                "normalization_std": [0.229, 0.224, 0.225]
            },
            "class_names_path": str(CONFIG_CLASS_NAMES.relative_to(REPO_ROOT)),
            "class_names_sha256": compute_file_sha256(CONFIG_CLASS_NAMES),
            "extraction_duration_sec": elapsed,
            "throughput_img_per_sec": float(total / elapsed) if elapsed > 0 else 0.0,
            "device": str(self.device),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "COMPLETED",
            "sha256_npz": None
        }

        # Save consolidated file to disk
        np.savez_compressed(
            npz_path,
            embeddings=embeddings_arr,
            labels=labels_arr,
            sample_ids=sample_ids_arr
        )
        meta["sha256_npz"] = compute_file_sha256(npz_path)

        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        # Cleanup temporary batch shards after successful consolidation
        for shard_file in shards_dir.glob("batch_*.npz"):
            try:
                shard_file.unlink()
            except Exception:
                pass
        try:
            shards_dir.rmdir()
        except Exception:
            pass

        print(f"[EXTRACTED] Saved {total} samples (512-d) to {npz_path.name} in {elapsed:.2f}s ({total/elapsed:.1f} img/s)", flush=True)

        return {
            "embeddings": embeddings_arr,
            "labels": labels_arr,
            "sample_ids": sample_ids,
            "metadata": meta,
            "cached": False,
            "npz_path": str(npz_path)
        }

