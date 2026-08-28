"""Pixel-level overlap and boundary metrics for binary tumor masks."""

from __future__ import annotations

from typing import Any

import numpy as np


def as_binary_mask(array: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 3:
        if arr.shape[-1] in {3, 4}:
            arr = np.max(arr[..., :3], axis=-1)
        else:
            arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected a 2D mask, got shape {arr.shape}.")
    if arr.dtype == np.bool_:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        return arr >= float(threshold)
    unique = np.unique(arr)
    if unique.size <= 2 and set(unique.tolist()) <= {0, 1, 255}:
        return arr > 0
    return arr >= float(threshold)


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def overlap_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    pred_m = as_binary_mask(pred)
    gt_m = as_binary_mask(gt)
    if pred_m.shape != gt_m.shape:
        raise ValueError(f"Mask shape mismatch: pred {pred_m.shape} vs gt {gt_m.shape}.")
    tp = float(np.logical_and(pred_m, gt_m).sum())
    fp = float(np.logical_and(pred_m, np.logical_not(gt_m)).sum())
    fn = float(np.logical_and(np.logical_not(pred_m), gt_m).sum())
    tn = float(np.logical_and(np.logical_not(pred_m), np.logical_not(gt_m)).sum())
    return {
        "dice": _safe_div(2 * tp, 2 * tp + fp + fn),
        "iou": _safe_div(tp, tp + fp + fn),
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "accuracy": _safe_div(tp + tn, tp + tn + fp + fn),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    mask_b = np.asarray(mask, dtype=bool)
    if mask_b.size == 0 or not mask_b.any():
        return mask_b
    padded = np.pad(mask_b, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask_b & np.logical_not(interior)


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_m = as_binary_mask(pred)
    gt_m = as_binary_mask(gt)
    pred_s = _surface(pred_m)
    gt_s = _surface(gt_m)
    if not pred_s.any() or not gt_s.any():
        return float("nan")
    try:
        from scipy.ndimage import distance_transform_edt
    except Exception:
        return _hd95_numpy(pred_s, gt_s)
    dt_gt = distance_transform_edt(np.logical_not(gt_s))
    dt_pred = distance_transform_edt(np.logical_not(pred_s))
    d1 = dt_gt[pred_s]
    d2 = dt_pred[gt_s]
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))


def _hd95_numpy(pred_s: np.ndarray, gt_s: np.ndarray) -> float:
    pred_pts = np.argwhere(pred_s).astype(float)
    gt_pts = np.argwhere(gt_s).astype(float)
    if pred_pts.shape[0] > 4000 or gt_pts.shape[0] > 4000:
        pred_pts = pred_pts[:: max(1, pred_pts.shape[0] // 4000)]
        gt_pts = gt_pts[:: max(1, gt_pts.shape[0] // 4000)]
    d_pred = np.sqrt(((pred_pts[:, None, :] - gt_pts[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    d_gt = np.sqrt(((gt_pts[:, None, :] - pred_pts[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    return float(max(np.percentile(d_pred, 95), np.percentile(d_gt, 95)))


def case_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    metrics = overlap_metrics(pred, gt)
    metrics["hd95"] = hd95(pred, gt)
    return metrics


def summarize_metrics(rows: list[dict[str, Any]], *, keys: tuple[str, ...] = ("dice", "iou", "precision", "recall", "accuracy", "hd95")) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_cases": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row.get(key, np.nan))]
        if not values:
            summary[f"{key}_mean"] = None
            summary[f"{key}_std"] = None
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values, ddof=0))
    return summary
