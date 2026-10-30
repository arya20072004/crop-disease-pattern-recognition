"""
feature_extraction.py
=====================
CCA 1 Pattern Recognition Shared Handcrafted Feature Extraction Pipeline.

Extracts:
1. HSV Colour Histograms (32 bins per channel -> 96-dimensional descriptor).
2. Histogram of Oriented Gradients (HOG) (9 bins, 8x8 cell, 2x2 block -> 8,100-d descriptor for 128x128).
3. Combined concatenated feature vector (8,196-d).
4. Deterministic preprocessing, feature caching, and
   strict train-only StandardScaler fitting to guarantee zero test leakage.
"""

import sys
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, Optional, Union, Dict, Any, List

import cv2
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ImageExtractionError(Exception):
    """Base exception for image loading and feature extraction failures."""
    pass


class ImageCorruptedError(ImageExtractionError):
    """Raised when an image file cannot be decoded or is corrupted."""
    pass


class ImageInvalidShapeError(ImageExtractionError):
    """Raised when an image has invalid dimensions or zero area."""
    pass


@dataclass
class FeatureConfig:
    """
    Configuration contract for handcrafted feature extraction.
    Default parameters conform to CCA 1 specification.
    """
    image_size: Tuple[int, int] = (128, 128)  # (width, height)
    hsv_bins: int = 32                         # bins per channel for HSV (total 96)
    hog_cell_size: Tuple[int, int] = (8, 8)    # pixels per cell
    hog_block_size: Tuple[int, int] = (16, 16) # pixels per block (2x2 cells)
    hog_block_stride: Tuple[int, int] = (8, 8) # block stride (50% overlap)
    hog_bins: int = 9                          # orientation bins
    apply_segmentation: bool = False           # optional leaf segmentation flag

    @property
    def color_dim(self) -> int:
        return self.hsv_bins * 3

    @property
    def hog_dim(self) -> int:
        w, h = self.image_size
        bw, bh = self.hog_block_size
        sw, sh = self.hog_block_stride
        cw, ch = self.hog_cell_size
        nb = self.hog_bins
        # Number of blocks
        nx = (w - bw) // sw + 1
        ny = (h - bh) // sh + 1
        cells_per_block = (bw // cw) * (bh // ch)
        return nx * ny * cells_per_block * nb

    @property
    def total_dim(self) -> int:
        return self.color_dim + self.hog_dim


def load_and_validate_image(
    image_input: Union[str, Path, np.ndarray, Image.Image]
) -> np.ndarray:
    """
    Loads and validates an image, returning an RGB uint8 numpy array.
    """
    if isinstance(image_input, (str, Path)):
        p = Path(image_input)
        if not p.exists():
            raise FileNotFoundError(f"Image file does not exist: {p}")
        try:
            with Image.open(p) as pil_img:
                pil_img = pil_img.convert("RGB")
                img_rgb = np.array(pil_img, dtype=np.uint8)
        except Exception as e:
            raise ImageCorruptedError(f"Failed to read/decode image at {p}: {e}")
    elif isinstance(image_input, Image.Image):
        try:
            img_rgb = np.array(image_input.convert("RGB"), dtype=np.uint8)
        except Exception as e:
            raise ImageCorruptedError(f"Failed to convert PIL Image: {e}")
    elif isinstance(image_input, np.ndarray):
        if image_input.dtype != np.uint8:
            if np.issubdtype(image_input.dtype, np.floating) and image_input.max() <= 1.0:
                img_rgb = (image_input * 255).astype(np.uint8)
            else:
                img_rgb = image_input.astype(np.uint8)
        else:
            img_rgb = image_input.copy()
    else:
        raise TypeError(f"Unsupported image_input type: {type(image_input)}")

    # Shape validation
    if img_rgb.ndim == 2:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
    elif img_rgb.ndim == 3 and img_rgb.shape[2] == 4:
        img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2RGB)
    elif img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ImageInvalidShapeError(f"Expected 3-channel RGB image, got shape {img_rgb.shape}")

    h, w, c = img_rgb.shape
    if h == 0 or w == 0:
        raise ImageInvalidShapeError(f"Image has zero dimension: shape={img_rgb.shape}")

    return img_rgb


def extract_hsv_histogram(
    img_rgb: np.ndarray,
    bins: int = 32
) -> np.ndarray:
    """
    Extracts normalized HSV color histogram with `bins` per channel.
    Output dimension: bins * 3 = 96 (default).
    Each channel histogram is L1-normalized (sums to 1.0).
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    hist_h = cv2.calcHist([hsv], [0], None, [bins], [0, 180]).flatten()
    hist_s = cv2.calcHist([hsv], [1], None, [bins], [0, 256]).flatten()
    hist_v = cv2.calcHist([hsv], [2], None, [bins], [0, 256]).flatten()

    sum_h = np.sum(hist_h)
    sum_s = np.sum(hist_s)
    sum_v = np.sum(hist_v)

    hist_h = hist_h / (sum_h + 1e-7)
    hist_s = hist_s / (sum_s + 1e-7)
    hist_v = hist_v / (sum_v + 1e-7)

    return np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)


def extract_hog_descriptor(
    img_rgb: np.ndarray,
    config: FeatureConfig
) -> np.ndarray:
    """
    Extracts Histogram of Oriented Gradients (HOG) descriptor using cv2.HOGDescriptor.
    Parameters match config: 9 orientation bins, 8x8 cell size, 16x16 block size (2x2 cells),
    8x8 stride.
    """
    target_w, target_h = config.image_size
    resized = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

    win_size = (target_w, target_h)
    block_size = config.hog_block_size
    block_stride = config.hog_block_stride
    cell_size = config.hog_cell_size
    nbins = config.hog_bins

    hog = cv2.HOGDescriptor(
        win_size,
        block_size,
        block_stride,
        cell_size,
        nbins
    )
    descriptor = hog.compute(gray)
    if descriptor is None:
        raise ImageExtractionError("HOG computation returned None.")
    return descriptor.flatten().astype(np.float32)


def extract_handcrafted_features(
    image_input: Union[str, Path, np.ndarray, Image.Image],
    config: Optional[FeatureConfig] = None
) -> np.ndarray:
    """
    Complete end-to-end extraction pipeline for a single image:
    1. Validation & RGB loading.
    2. HSV Colour Histogram extraction (96-d).
    3. HOG descriptor extraction (8,100-d for 128x128).
    4. Feature concatenation (8,196-d total).
    """
    if config is None:
        config = FeatureConfig()

    img_rgb = load_and_validate_image(image_input)

    color_feats = extract_hsv_histogram(img_rgb, bins=config.hsv_bins)
    hog_feats = extract_hog_descriptor(img_rgb, config=config)

    combined = np.concatenate([color_feats, hog_feats]).astype(np.float32)
    expected_dim = config.total_dim
    if combined.shape[0] != expected_dim:
        raise ValueError(f"Feature dimension mismatch: expected {expected_dim}, got {combined.shape[0]}")

    return combined


class FeatureScaler:
    """
    Strict train-only feature standardizer preventing data leakage.
    Wraps sklearn.preprocessing.StandardScaler with state enforcement.
    """
    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train: np.ndarray) -> "FeatureScaler":
        if len(X_train) == 0:
            raise ValueError("Cannot fit scaler on empty array.")
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError(
                "FeatureScaler must be fitted on training features before transforming validation/test data!"
            )
        return self.scaler.transform(X).astype(np.float32)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        self.fit(X_train)
        return self.transform(X_train)


def save_feature_cache(
    cache_path: Union[str, Path],
    features: np.ndarray,
    labels: np.ndarray,
    sample_paths: List[str],
    config: FeatureConfig,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """Persists extracted features and labels into compressed .npz format."""
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    meta_dict = metadata or {}
    meta_dict["config"] = {
        "image_size": list(config.image_size),
        "hsv_bins": config.hsv_bins,
        "hog_cell_size": list(config.hog_cell_size),
        "hog_block_size": list(config.hog_block_size),
        "hog_block_stride": list(config.hog_block_stride),
        "hog_bins": config.hog_bins,
        "total_dim": config.total_dim
    }

    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        sample_paths=np.array(sample_paths, dtype=object),
        metadata=np.array([meta_dict], dtype=object)
    )


def load_feature_cache(cache_path: Union[str, Path]) -> Dict[str, Any]:
    """Loads features from a compressed .npz cache file."""
    path = Path(cache_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature cache file not found: {path}")

    with np.load(path, allow_pickle=True) as data:
        return {
            "features": data["features"],
            "labels": data["labels"],
            "sample_paths": data["sample_paths"].tolist(),
            "metadata": data["metadata"][0] if "metadata" in data else {}
        }


def extract_manifest_features(
    manifest_path: Union[str, Path],
    config: Optional[FeatureConfig] = None,
    cache_path: Optional[Union[str, Path]] = None,
    force_recompute: bool = False
) -> Dict[str, Any]:
    """Extracts features for all samples listed in a JSON manifest."""
    if config is None:
        config = FeatureConfig()

    if cache_path is not None and not force_recompute:
        c_path = Path(cache_path)
        if c_path.exists():
            return load_feature_cache(c_path)

    m_path = Path(manifest_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Manifest not found: {m_path}")

    with open(m_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data.get("samples", [])
    features_list = []
    labels_list = []
    valid_paths = []

    for s in samples:
        rel_p = s.get("relative_path")
        abs_p = REPO_ROOT / rel_p
        lbl = s.get("class_index", s.get("mapped_class_index", -1))
        try:
            feat = extract_handcrafted_features(abs_p, config=config)
            features_list.append(feat)
            labels_list.append(lbl)
            valid_paths.append(rel_p)
        except Exception as e:
            print(f"Warning: Failed to extract features for {rel_p}: {e}")

    X = np.stack(features_list).astype(np.float32)
    y = np.array(labels_list, dtype=np.int64)

    res = {
        "features": X,
        "labels": y,
        "sample_paths": valid_paths,
        "metadata": {"manifest": str(m_path.name), "num_samples": len(valid_paths)}
    }

    if cache_path is not None:
        save_feature_cache(
            cache_path=cache_path,
            features=X,
            labels=y,
            sample_paths=valid_paths,
            config=config,
            metadata=res["metadata"]
        )

    return res


if __name__ == "__main__":
    cfg = FeatureConfig()
    print(f"FeatureConfig: color_dim={cfg.color_dim}, hog_dim={cfg.hog_dim}, total_dim={cfg.total_dim}")
    dummy = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    feats = extract_handcrafted_features(dummy, cfg)
    print(f"Extracted feature shape: {feats.shape}")
    assert feats.shape[0] == cfg.total_dim
    print("Self-test PASSED.")
