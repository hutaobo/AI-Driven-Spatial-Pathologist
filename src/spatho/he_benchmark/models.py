"""Segmenter adapters for the H&E tumor-region benchmark.

Heavy foundation models are optional. The harness prefers precomputed masks
when a prediction directory is provided, so GPU inference can happen elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
import os

import numpy as np

from .catalog import MODEL_CATALOG
from .datasets import load_array
from .metrics import as_binary_mask


class Segmenter(Protocol):
    model_id: str

    def available(self) -> tuple[bool, str]:
        ...

    def predict(self, image: np.ndarray, *, case: dict[str, Any]) -> np.ndarray:
        ...


class StainThresholdSegmenter:
    """Weak H&E heuristic: purple, non-white tissue as tumor-like foreground."""

    model_id = "stain_threshold"

    def available(self) -> tuple[bool, str]:
        return True, "built-in heuristic"

    def predict(self, image: np.ndarray, *, case: dict[str, Any]) -> np.ndarray:
        rgb = _as_rgb(image)
        r = rgb[..., 0].astype(np.float32)
        g = rgb[..., 1].astype(np.float32)
        b = rgb[..., 2].astype(np.float32)
        tissue = (r + g + b) < 700
        purple = (b > r) & (r > g) & tissue
        dark = (r + g + b) < 520
        return np.logical_and(tissue, np.logical_or(purple, dark))


class ExternalMaskSegmenter:
    model_id: str

    def __init__(self, model_id: str, pred_dir: Path) -> None:
        self.model_id = model_id
        self.pred_dir = Path(pred_dir).expanduser().resolve()

    def available(self) -> tuple[bool, str]:
        if not self.pred_dir.exists():
            return False, f"prediction directory missing: {self.pred_dir}"
        return True, f"precomputed masks in {self.pred_dir}"

    def predict(self, image: np.ndarray, *, case: dict[str, Any]) -> np.ndarray:
        stem = Path(str(case.get("image"))).stem
        candidates = [
            self.pred_dir / f"{case.get('case_id')}.npy",
            self.pred_dir / f"{case.get('case_id')}.png",
            self.pred_dir / f"{stem}.npy",
            self.pred_dir / f"{stem}.png",
        ]
        for path in candidates:
            if path.exists():
                pred = load_array(path)
                return as_binary_mask(pred)
        raise FileNotFoundError(
            f"No precomputed mask for {case.get('case_id')} under {self.pred_dir}"
        )


class CheckpointSegmenter:
    """Placeholder for local GPU checkpoints referenced by environment variables."""

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self.env_name = f"SPATHO_{model_id.upper()}_CKPT"

    def available(self) -> tuple[bool, str]:
        checkpoint = os.environ.get(self.env_name)
        if not checkpoint:
            return False, f"set {self.env_name} or pass prediction_dirs.{self.model_id}"
        path = Path(checkpoint)
        if not path.exists():
            return False, f"checkpoint missing: {path}"
        return False, (
            f"checkpoint listed at {path}, but in-process inference for {self.model_id} "
            "is not bundled; dump masks and set prediction_dirs"
        )

    def predict(self, image: np.ndarray, *, case: dict[str, Any]) -> np.ndarray:
        raise RuntimeError(
            f"{self.model_id} in-process inference is not bundled. "
            "Run the model on GPU, write per-case masks, and point prediction_dirs at them."
        )


class PlipFullTileSegmenter:
    model_id = "plip_fulltile"

    def available(self) -> tuple[bool, str]:
        try:
            from pathology_ai_service.core import PLIPZeroShotContourClassifier
        except Exception as exc:
            return False, f"PLIP classifier unavailable: {exc}"
        return True, "pathology_ai_service.PLIPZeroShotContourClassifier"

    def predict(self, image: np.ndarray, *, case: dict[str, Any]) -> np.ndarray:
        from pathology_ai_service.core import PLIPZeroShotContourClassifier

        classifier = PLIPZeroShotContourClassifier()
        rgb = _as_rgb(image)
        result = classifier.classify_array(rgb) if hasattr(classifier, "classify_array") else None
        if result is None:
            raise RuntimeError("PLIP classifier does not expose classify_array; skip this model.")
        label = str(result.get("top_label") or result.get("label") or "").lower()
        tumorish = any(token in label for token in ("tumor", "carcinoma", "neoplastic", "invasive", "malignant"))
        return np.full(rgb.shape[:2], tumorish, dtype=bool)


def _as_rgb(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC image, got {arr.shape}")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        finite = arr[np.isfinite(arr)]
        high = float(finite.max()) if finite.size else 1.0
        scale = 255.0 if high <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return arr


def build_segmenters(
    model_ids: list[str],
    *,
    prediction_dirs: dict[str, str] | None = None,
) -> dict[str, Segmenter]:
    prediction_dirs = prediction_dirs or {}
    segmenters: dict[str, Segmenter] = {}
    for model_id in model_ids:
        if model_id not in MODEL_CATALOG and not str(model_id).startswith("external:"):
            raise KeyError(f"Unknown model id {model_id}. Known: {sorted(MODEL_CATALOG)}")
        pred_dir = prediction_dirs.get(model_id)
        if pred_dir:
            segmenters[model_id] = ExternalMaskSegmenter(model_id, Path(pred_dir))
            continue
        if model_id == "stain_threshold":
            segmenters[model_id] = StainThresholdSegmenter()
        elif model_id == "plip_fulltile":
            segmenters[model_id] = PlipFullTileSegmenter()
        elif str(model_id).startswith("external:"):
            raise ValueError(f"{model_id} requires prediction_dirs[{model_id!r}]")
        else:
            segmenters[model_id] = CheckpointSegmenter(model_id)
    return segmenters
