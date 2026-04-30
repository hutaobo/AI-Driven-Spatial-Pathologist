from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib import error, request
import json
import re

import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon, shape

from histoseg.spatial_pathologist.artifact_loader import build_case_bundle, load_base_pipeline_config
from histoseg.spatial_pathologist.config import SpatialPathologistConfig
from histoseg.spatial_pathologist.report import write_html_report

from .schema import WorkflowConfig


_IMAGE_CORE_ATTRS = {
    "axes",
    "dtype",
    "multiscale_levels",
    "level_shapes",
    "source_path",
    "transform_kind",
    "transform_direction",
    "transform_input_space",
    "transform_output_space",
    "transform_output_unit",
    "xenium_physical_unit",
    "image_to_xenium_affine",
    "alignment_csv_path",
    "pixel_size_um",
    "xenium_pixel_size_um",
    "keypoints_validation",
}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def _post_json(url: str, payload: dict[str, Any], *, timeout: float = 300.0) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        url=url,
        data=body,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {details}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Expected JSON object from {url}, got {type(parsed).__name__}")
    return parsed


def _slug(value: str, *, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return (slug or "contour")[:max_len]


def resolve_contour_geojson(cfg: WorkflowConfig, base_cfg: dict[str, Any] | None = None) -> Path:
    if cfg.he_contour_geojson is not None:
        path = Path(cfg.he_contour_geojson).expanduser().resolve()
        if path.exists():
            return path
        raise FileNotFoundError(f"Configured he_contour_geojson does not exist: {path}")

    resolved_base = base_cfg or load_base_pipeline_config(cfg.base_pipeline_config)
    dataset_root = Path(resolved_base["dataset_root"]).expanduser().resolve()
    candidates = [
        dataset_root / "xenium_explorer_annotations.generated.geojson",
        dataset_root / "xenium_explorer_annotations.s1_s5.generated.geojson",
        dataset_root / "xenium_explorer_annotations.geojson",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No contour GeoJSON found. Expected one of: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def _load_geojson_properties(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    props_by_id: dict[str, dict[str, Any]] = {}
    for index, feature in enumerate(payload.get("features", []), start=1):
        props = dict(feature.get("properties", {}) or {})
        contour_id = str(props.get("name") or props.get("polygon_id") or props.get("id") or f"contour_{index}")
        props_by_id[contour_id] = props
    return props_by_id


def _iter_geojson_feature_batches(
    path: Path,
    *,
    output_dir: Path,
    batch_size: int = 32,
) -> tuple[int, list[Path]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError(f"Contour GeoJSON has no features: {path}")

    template = {key: value for key, value in payload.items() if key != "features"}
    template.setdefault("type", "FeatureCollection")
    batches_dir = output_dir / "contour_geojson_batches"
    batches_dir.mkdir(parents=True, exist_ok=True)
    batch_paths: list[Path] = []
    for start in range(0, len(features), int(batch_size)):
        stop = min(start + int(batch_size), len(features))
        batch_payload = {**template, "features": features[start:stop]}
        batch_path = batches_dir / f"batch_{len(batch_paths) + 1:04d}.geojson"
        batch_path.write_text(
            json.dumps(batch_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        batch_paths.append(batch_path)
    return len(features), batch_paths


def _to_rgb(array: np.ndarray, axes: str) -> np.ndarray:
    arr = np.asarray(array)
    normalized_axes = str(axes).lower()
    if arr.ndim == 3 and "c" in normalized_axes and normalized_axes.index("c") == 0:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        arr = np.squeeze(arr)
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr], axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Cannot convert patch array with shape {arr.shape} to RGB.")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] > 3:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        finite = arr[np.isfinite(arr)]
        if finite.size:
            low = float(np.percentile(finite, 1))
            high = float(np.percentile(finite, 99))
        else:
            low, high = 0.0, 1.0
        if high <= low:
            high = low + 1.0
        arr = np.clip((arr.astype(float) - low) / (high - low), 0.0, 1.0) * 255.0
    return arr.astype(np.uint8, copy=False)


def _save_patch(image: Any, output_path: Path, *, max_side_px: int) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = _to_rgb(np.asarray(image.levels[0]), image.axes)
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - PDC workflow installs pillow
        raise RuntimeError("Saving H&E contour patches requires Pillow.") from exc
    pil = Image.fromarray(rgb)
    original_size = pil.size
    if max(pil.size) > int(max_side_px):
        pil.thumbnail((int(max_side_px), int(max_side_px)), Image.Resampling.LANCZOS)
    pil.save(output_path)
    nonzero = float(np.count_nonzero(rgb.sum(axis=-1) > 3) / max(rgb.shape[0] * rgb.shape[1], 1))
    return {
        "path": str(output_path),
        "original_width": int(original_size[0]),
        "original_height": int(original_size[1]),
        "saved_width": int(pil.size[0]),
        "saved_height": int(pil.size[1]),
        "nonzero_fraction": nonzero,
    }


def _crop_image_level(
    level: Any,
    *,
    axes: str,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    slices = [slice(None)] * len(level.shape)
    slices[axes.index("x")] = slice(x0, x1)
    slices[axes.index("y")] = slice(y0, y1)
    try:
        return np.asarray(level[tuple(slices)]).copy()
    except TypeError:
        if hasattr(level, "open_zarr_source"):
            store, source = level.open_zarr_source()
            try:
                return np.asarray(source[tuple(slices)]).copy()
            finally:
                if hasattr(store, "close"):
                    store.close()
        if hasattr(level, "asarray"):
            return np.asarray(level.asarray()[tuple(slices)]).copy()
        raise


def _polygon_mask_for_bbox(
    *,
    image_geometry: Polygon | MultiPolygon,
    bbox: tuple[int, int, int, int],
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    width = int(x1 - x0)
    height = int(y1 - y0)
    local_geometry = affinity.translate(image_geometry, xoff=-float(x0), yoff=-float(y0))
    yy, xx = np.mgrid[0:height, 0:width]
    sample_points = np.column_stack(
        [
            xx.reshape(-1).astype(float) + 0.5,
            yy.reshape(-1).astype(float) + 0.5,
        ]
    )

    mask = np.zeros(sample_points.shape[0], dtype=bool)
    polygons = (
        list(local_geometry.geoms)
        if isinstance(local_geometry, MultiPolygon)
        else [local_geometry]
    )
    for polygon in polygons:
        polygon_mask = _polygon_contains_points(polygon, sample_points)
        mask |= polygon_mask
    return mask.reshape(height, width)


def _polygon_contains_points(polygon: Polygon, sample_points: np.ndarray) -> np.ndarray:
    exterior_path = MplPath(np.asarray(polygon.exterior.coords, dtype=float))
    contained = exterior_path.contains_points(sample_points, radius=1e-9)
    for interior in polygon.interiors:
        hole_path = MplPath(np.asarray(interior.coords, dtype=float))
        contained &= ~hole_path.contains_points(sample_points)
    return contained


def _apply_polygon_mask(
    patch_array: np.ndarray,
    *,
    mask: np.ndarray,
    axes: str,
) -> np.ndarray:
    y_index = axes.index("y")
    x_index = axes.index("x")
    broadcast_shape = [1] * patch_array.ndim
    broadcast_shape[y_index] = mask.shape[0]
    broadcast_shape[x_index] = mask.shape[1]
    broadcast_mask = mask.reshape(broadcast_shape)
    fill_value = np.zeros((), dtype=patch_array.dtype)
    return np.where(broadcast_mask, patch_array, fill_value)


def _json_ready_attr(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_ready_attr(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready_attr(item) for item in value]
    return value


def _read_he_image_from_sdata_store(store_path: Path) -> Any | None:
    """Load only ``images/he`` from a pyXenium SData zarr store.

    ``pyXenium.io.read_sdata`` materializes points and tables, which is too heavy
    for full Xenium transcript stores. The H&E pyramid itself can remain lazy.
    """

    try:
        import zarr
        from pyXenium.io.sdata_model import XeniumImage
    except Exception:
        return None

    if not store_path.exists():
        return None
    try:
        root = zarr.open_group(str(store_path), mode="r")
        if root.attrs.get("format") != "pyxenium.sdata":
            return None
        if "images" not in root or "he" not in root["images"]:
            return None
        group = root["images"]["he"]
        level_names = sorted(
            (str(level_key) for level_key in group.keys()),
            key=lambda value: int(value) if value.isdigit() else value,
        )
        if not level_names:
            return None
        attrs = group.attrs
        metadata = {
            str(key): _json_ready_attr(attrs[key])
            for key in attrs.keys()
            if str(key) not in _IMAGE_CORE_ATTRS
        }
        for key in (
            "transform_direction",
            "transform_input_space",
            "transform_output_space",
            "transform_output_unit",
            "xenium_physical_unit",
        ):
            if key in attrs:
                metadata[key] = _json_ready_attr(attrs[key])
        pixel_size_um = attrs.get("pixel_size_um", attrs.get("xenium_pixel_size_um", None))
        return XeniumImage(
            levels=[group[level_name] for level_name in level_names],
            axes=str(attrs["axes"]),
            dtype=str(attrs["dtype"]),
            source_path=str(attrs.get("source_path", store_path)),
            transform_kind=str(attrs.get("transform_kind", "affine")),
            image_to_xenium_affine=_json_ready_attr(attrs.get("image_to_xenium_affine", None)),
            alignment_csv_path=attrs.get("alignment_csv_path", None),
            pixel_size_um=pixel_size_um,
            keypoints_validation=_json_ready_attr(attrs.get("keypoints_validation", None)),
            metadata=metadata,
        )
    except Exception:
        return None


def _load_he_image_for_dataset(dataset_root: Path) -> tuple[Any, str]:
    sdata_store = dataset_root / "spatialdata.zarr"
    he_from_store = _read_he_image_from_sdata_store(sdata_store)
    if he_from_store is not None:
        return he_from_store, "spatialdata.zarr/images/he"
    try:
        from pyXenium.io.xenium_artifacts import read_he_image
    except Exception as exc:
        raise RuntimeError("H&E contour foundation mode requires pyXenium image readers.") from exc
    he_image = read_he_image(str(dataset_root))
    if he_image is None:
        raise RuntimeError(
            "H&E contour foundation mode requires aligned H&E image data. "
            f"Could not find {sdata_store}/images/he or Xenium *_he_image.ome.tif artifacts."
        )
    return he_image, "xenium_he_artifact"


def _geometry_xenium_pixel_to_image_xy(geometry: Polygon | MultiPolygon, he_image: Any) -> Polygon | MultiPolygon:
    if he_image.image_to_xenium_affine is None:
        raise ValueError("H&E image is missing image_to_xenium_affine metadata.")
    inverse = np.linalg.inv(np.asarray(he_image.image_to_xenium_affine, dtype=float))

    def transform_xy(x: Any, y: Any, z: Any | None = None) -> tuple[Any, Any] | tuple[Any, Any, Any]:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)
        flat = np.column_stack([x_array.reshape(-1), y_array.reshape(-1), np.ones(x_array.size)])
        transformed = flat @ inverse.T
        out_x = transformed[:, 0].reshape(x_array.shape)
        out_y = transformed[:, 1].reshape(y_array.shape)
        if z is None:
            return out_x, out_y
        return out_x, out_y, z

    from shapely.ops import transform

    return transform(transform_xy, geometry)


def _select_pyramid_level(he_image: Any, *, bbox_level0: tuple[int, int, int, int], max_side_px: int) -> tuple[int, float, float]:
    x0, y0, x1, y1 = bbox_level0
    max_side = max(int(x1 - x0), int(y1 - y0), 1)
    target_scale = max(float(max_side) / max(float(max_side_px), 1.0), 1.0)
    shapes = he_image.multiscale_shapes()
    x_index = he_image.axes.index("x")
    y_index = he_image.axes.index("y")
    level0_shape = shapes[0]
    selected = len(shapes) - 1
    selected_scale_x = float(level0_shape[x_index]) / float(shapes[selected][x_index])
    selected_scale_y = float(level0_shape[y_index]) / float(shapes[selected][y_index])
    for level_index, shape_at_level in enumerate(shapes):
        scale_x = float(level0_shape[x_index]) / float(shape_at_level[x_index])
        scale_y = float(level0_shape[y_index]) / float(shape_at_level[y_index])
        if max(scale_x, scale_y) >= target_scale:
            selected = level_index
            selected_scale_x = scale_x
            selected_scale_y = scale_y
            break
    return selected, selected_scale_x, selected_scale_y


def _extract_feature_patch(
    *,
    feature: dict[str, Any],
    he_image: Any,
    output_path: Path,
    max_side_px: int,
) -> dict[str, Any]:
    geometry = shape(feature.get("geometry"))
    if not isinstance(geometry, (Polygon, MultiPolygon)):
        raise ValueError(f"Contour geometry must be Polygon or MultiPolygon, got {geometry.geom_type!r}.")
    image_geometry_level0 = _geometry_xenium_pixel_to_image_xy(geometry, he_image)
    min_x, min_y, max_x, max_y = image_geometry_level0.bounds
    bbox_level0 = (
        int(np.floor(min_x)),
        int(np.floor(min_y)),
        int(np.ceil(max_x)),
        int(np.ceil(max_y)),
    )
    level_index, scale_x, scale_y = _select_pyramid_level(
        he_image,
        bbox_level0=bbox_level0,
        max_side_px=max_side_px,
    )
    image_geometry = affinity.scale(
        image_geometry_level0,
        xfact=1.0 / scale_x,
        yfact=1.0 / scale_y,
        origin=(0.0, 0.0),
    )
    level = he_image.levels[level_index]
    shape_at_level = tuple(int(value) for value in getattr(level, "shape", np.shape(level)))
    x_index = he_image.axes.index("x")
    y_index = he_image.axes.index("y")
    image_width = shape_at_level[x_index]
    image_height = shape_at_level[y_index]
    min_x, min_y, max_x, max_y = image_geometry.bounds
    bbox = (
        max(int(np.floor(min_x)), 0),
        max(int(np.floor(min_y)), 0),
        min(int(np.ceil(max_x)), image_width),
        min(int(np.ceil(max_y)), image_height),
    )
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Contour bbox does not intersect H&E image at level {level_index}: {bbox}")
    patch_array = _crop_image_level(level, axes=he_image.axes, bbox=bbox)
    mask = _polygon_mask_for_bbox(image_geometry=image_geometry, bbox=bbox)
    masked_patch = _apply_polygon_mask(patch_array, mask=mask, axes=he_image.axes)
    patch_meta = _save_patch(
        SimpleNamespace(levels=[masked_patch], axes=he_image.axes),
        output_path,
        max_side_px=max_side_px,
    )
    patch_meta.update(
        {
            "pyramid_level": int(level_index),
            "level_downsample_x": float(scale_x),
            "level_downsample_y": float(scale_y),
            "bbox_level_xy": [int(value) for value in bbox],
            "bbox_level0_xy": [int(value) for value in bbox_level0],
        }
    )
    return patch_meta


def _extract_he_contour_patches(
    *,
    cfg: WorkflowConfig,
    base_cfg: dict[str, Any],
    contour_geojson: Path,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], Path]:
    dataset_root = Path(base_cfg["dataset_root"]).expanduser().resolve()
    he_image, sdata_source = _load_he_image_for_dataset(dataset_root)
    payload = json.loads(contour_geojson.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not isinstance(features, list) or not features:
        raise ValueError(f"Contour GeoJSON has no features: {contour_geojson}")
    patches_dir = output_dir / "contour_patches"
    records: list[dict[str, Any]] = []
    for index, feature in enumerate(features, start=1):
        props = dict(feature.get("properties", {}) or {})
        contour_id = str(props.get("name") or props.get("polygon_id") or props.get("id") or f"contour_{index}")
        if index == 1 or index % 25 == 0 or index == len(features):
            print(f"[he-foundation] extracting contour patch {index}/{len(features)}", flush=True)
        patch_path = patches_dir / f"{index:05d}_{_slug(contour_id)}.png"
        patch_meta = _extract_feature_patch(
            feature=feature,
            he_image=he_image,
            output_path=patch_path,
            max_side_px=cfg.he_foundation_max_patch_side_px,
        )
        structure_id = props.get("structure_id")
        records.append(
            {
                "contour_id": contour_id,
                "image_path": str(patch_path),
                "structure_id": int(structure_id) if structure_id is not None else None,
                "structure_name": props.get("assigned_structure"),
                "component_index": props.get("component_index"),
                "polygon_index": props.get("polygon_index"),
                "sdata_source": sdata_source,
                "patch": patch_meta,
            }
        )
    manifest_path = _write_json(output_dir / "he_contour_patches_manifest.json", records)
    return records, manifest_path


def _classify_contours(
    *,
    cfg: WorkflowConfig,
    records: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    classifications: list[dict[str, Any]] = []
    warnings: list[str] = []
    classify_url = cfg.pathology_ai_api_base_url.rstrip("/") + "/v1/he/contours/classify"
    chunk_size = 24
    for start in range(0, len(records), chunk_size):
        chunk = records[start : start + chunk_size]
        payload = {
            "case_name": cfg.case_name,
            "model_id": cfg.he_foundation_model_id,
            "prompt_set": cfg.he_foundation_prompt_set,
            "top_k": cfg.he_foundation_top_k,
            "contours": [
                {
                    "contour_id": item["contour_id"],
                    "image_path": item["image_path"],
                    "structure_id": item.get("structure_id"),
                    "structure_name": item.get("structure_name"),
                }
                for item in chunk
            ],
        }
        response = _post_json(classify_url, payload, timeout=900.0)
        classifications.extend(response.get("classifications", []))
        warnings.extend(str(item) for item in response.get("warnings", []))

    classification_json = _write_json(output_dir / "he_contour_classification.json", classifications)
    rows = []
    for item in classifications:
        top = (item.get("top_classes") or [{}])[0]
        rows.append(
            {
                "contour_id": item.get("contour_id"),
                "image_path": item.get("image_path"),
                "structure_id": item.get("structure_id"),
                "structure_name": item.get("structure_name"),
                "top_label_id": top.get("label_id"),
                "top_label": top.get("label"),
                "top_score": top.get("score"),
                "top_classes_json": json.dumps(item.get("top_classes", []), ensure_ascii=False),
                "patch_quality_json": json.dumps(item.get("patch_quality", {}), ensure_ascii=False),
                "error": item.get("error"),
            }
        )
    classification_csv = output_dir / "he_contour_classification.csv"
    pd.DataFrame(rows).to_csv(classification_csv, index=False)
    return {
        "classifications": classifications,
        "classification_json": str(classification_json),
        "classification_csv": str(classification_csv),
        "warnings": warnings,
    }


def summarize_contours_by_structure(classifications: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int | None, str], dict[str, Any]] = {}
    for item in classifications:
        key = (item.get("structure_id"), str(item.get("structure_name") or "unassigned"))
        bucket = grouped.setdefault(
            key,
            {
                "structure_id": item.get("structure_id"),
                "structure_name": item.get("structure_name") or "unassigned",
                "n_contours": 0,
                "n_classified": 0,
                "label_scores": {},
                "example_contours": [],
            },
        )
        bucket["n_contours"] += 1
        top_classes = item.get("top_classes") or []
        if not top_classes:
            continue
        bucket["n_classified"] += 1
        if len(bucket["example_contours"]) < 6:
            bucket["example_contours"].append(item.get("contour_id"))
        for cls in top_classes:
            label_id = str(cls.get("label_id"))
            entry = bucket["label_scores"].setdefault(
                label_id,
                {
                    "label_id": label_id,
                    "label": cls.get("label"),
                    "scores": [],
                    "contour_count": 0,
                },
            )
            entry["scores"].append(float(cls.get("score", 0.0)))
            entry["contour_count"] += 1

    summaries: list[dict[str, Any]] = []
    for bucket in grouped.values():
        labels = []
        for entry in bucket.pop("label_scores").values():
            scores = entry.pop("scores")
            entry["mean_score"] = float(np.mean(scores)) if scores else 0.0
            entry["max_score"] = float(np.max(scores)) if scores else 0.0
            labels.append(entry)
        labels.sort(key=lambda row: (row["mean_score"], row["max_score"]), reverse=True)
        bucket["top_visual_labels"] = labels[:5]
        if labels:
            bucket["top_label_id"] = labels[0]["label_id"]
            bucket["top_label"] = labels[0]["label"]
            bucket["top_mean_score"] = labels[0]["mean_score"]
            bucket["top_max_score"] = labels[0]["max_score"]
        else:
            bucket["top_label_id"] = None
            bucket["top_label"] = None
            bucket["top_mean_score"] = 0.0
            bucket["top_max_score"] = 0.0
        summaries.append(bucket)
    summaries.sort(key=lambda row: (row["structure_id"] is None, row["structure_id"] or 0, row["structure_name"]))
    return summaries


def _write_structure_summary(output_dir: Path, summaries: list[dict[str, Any]]) -> dict[str, str]:
    summary_json = _write_json(output_dir / "he_contour_to_structure_summary.json", summaries)
    rows = []
    for item in summaries:
        rows.append(
            {
                "structure_id": item.get("structure_id"),
                "structure_name": item.get("structure_name"),
                "n_contours": item.get("n_contours"),
                "n_classified": item.get("n_classified"),
                "top_label_id": item.get("top_label_id"),
                "top_label": item.get("top_label"),
                "top_mean_score": item.get("top_mean_score"),
                "top_max_score": item.get("top_max_score"),
                "top_visual_labels_json": json.dumps(item.get("top_visual_labels", []), ensure_ascii=False),
            }
        )
    summary_csv = output_dir / "he_contour_to_structure_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    return {"structure_summary_json": str(summary_json), "structure_summary_csv": str(summary_csv)}


def _call_structure_naming(
    *,
    cfg: WorkflowConfig,
    structure: dict[str, Any],
    current_review: dict[str, Any],
    he_summary: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "case_name": cfg.case_name,
        "study_context": cfg.study_context,
        "annotation_taxonomy": cfg.annotation_taxonomy,
        "structure": structure,
        "current_review": current_review,
        "he_visual_summary": he_summary,
        "multimodal_evidence": {
            "top_clusters": structure.get("top_clusters", []),
            "harmonized_composition": structure.get("harmonized_composition", {}),
            "raw_empirical_composition": structure.get("raw_empirical_composition", {}),
            "top_candidates": structure.get("top_candidates", []),
        },
        "override_policy": {
            "he_visual_override_enabled": cfg.he_visual_override_enabled,
            "he_visual_override_min_llm_confidence": cfg.he_visual_override_min_llm_confidence,
            "he_visual_override_min_foundation_score": cfg.he_visual_override_min_foundation_score,
        },
    }
    return _post_json(
        cfg.pathology_ai_api_base_url.rstrip("/") + "/v1/annotations/structure-multimodal",
        payload,
        timeout=300.0,
    )


def _apply_multimodal_names(
    *,
    cfg: WorkflowConfig,
    case_bundle: dict[str, Any],
    pathology_outputs: dict[str, Any],
    structure_summaries: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, str]:
    structure_by_id = {int(item["structure_id"]): item for item in case_bundle["structures"]}
    summary_by_id = {
        int(item["structure_id"]): item
        for item in structure_summaries
        if item.get("structure_id") is not None
    }
    structure_reviews_path = Path(pathology_outputs["structure_reviews_json"]).resolve()
    cluster_reviews_path = Path(pathology_outputs["cluster_reviews_json"]).resolve()
    case_summary_path = Path(pathology_outputs["case_summary_json"]).resolve()
    structure_reviews = json.loads(structure_reviews_path.read_text(encoding="utf-8"))
    cluster_reviews = json.loads(cluster_reviews_path.read_text(encoding="utf-8"))
    case_summary = json.loads(case_summary_path.read_text(encoding="utf-8"))

    naming_rows: list[dict[str, Any]] = []
    accepted = 0
    for review in structure_reviews:
        structure_id = int(review.get("structure_id"))
        structure = structure_by_id.get(structure_id, {"structure_id": structure_id})
        he_summary = summary_by_id.get(structure_id)
        if he_summary is None:
            naming_rows.append(
                {
                    "structure_id": structure_id,
                    "pre_visual_name": review.get("title") or review.get("assigned_label"),
                    "final_name": review.get("title") or review.get("assigned_label"),
                    "visual_override": False,
                    "accepted": False,
                    "confidence": review.get("confidence", 0.0),
                    "error": "no_he_visual_summary",
                }
            )
            continue
        try:
            naming = _call_structure_naming(
                cfg=cfg,
                structure=structure,
                current_review=review,
                he_summary=he_summary,
            )
        except Exception as exc:
            naming = {
                "structure_id": structure_id,
                "pre_visual_name": review.get("title") or review.get("assigned_label"),
                "final_name": review.get("title") or review.get("assigned_label"),
                "visual_override": False,
                "confidence": review.get("confidence", 0.0),
                "reasoning_summary": "",
                "visual_evidence": [],
                "molecular_evidence": [],
                "conflicts": [str(exc)],
                "recommended_checks": [],
                "error": str(exc),
            }
        top_score = float(he_summary.get("top_max_score") or he_summary.get("top_mean_score") or 0.0)
        accepted_override = (
            bool(cfg.he_visual_override_enabled)
            and bool(naming.get("visual_override"))
            and float(naming.get("confidence", 0.0)) >= cfg.he_visual_override_min_llm_confidence
            and top_score >= cfg.he_visual_override_min_foundation_score
        )
        original_title = str(review.get("title") or review.get("assigned_label") or "")
        final_name = str(naming.get("final_name") or original_title)
        review["he_foundation_visual_summary"] = he_summary
        review["he_multimodal_naming"] = {
            **naming,
            "accepted_visual_override": accepted_override,
            "foundation_top_score": top_score,
        }
        visual_lines = [str(item) for item in naming.get("visual_evidence", [])]
        molecular_lines = [str(item) for item in naming.get("molecular_evidence", [])]
        if accepted_override:
            accepted += 1
            review["title"] = final_name
            review["assigned_label"] = final_name
            review["confidence"] = float(naming.get("confidence", review.get("confidence", 0.0)))
        if naming.get("reasoning_summary"):
            review["summary"] = (
                f"{naming['reasoning_summary']}\n\n"
                f"Previous molecular-only summary: {review.get('summary', '')}"
            )
        review["key_evidence"] = (visual_lines + molecular_lines + list(review.get("key_evidence", [])))[:10]
        review["recommended_checks"] = (
            [str(item) for item in naming.get("recommended_checks", [])]
            + list(review.get("recommended_checks", []))
        )[:8]
        naming_rows.append(
            {
                "structure_id": structure_id,
                "pre_visual_name": naming.get("pre_visual_name") or original_title,
                "final_name": final_name,
                "visual_override": bool(naming.get("visual_override")),
                "accepted": accepted_override,
                "confidence": naming.get("confidence"),
                "foundation_top_label": he_summary.get("top_label"),
                "foundation_top_score": top_score,
                "reasoning_summary": naming.get("reasoning_summary"),
                "error": naming.get("error"),
            }
        )

    if accepted:
        case_summary["key_findings"] = (
            [f"H&E contour foundation model accepted visual overrides for {accepted} structure names."]
            + list(case_summary.get("key_findings", []))
        )[:8]
    else:
        case_summary["key_findings"] = (
            ["H&E contour foundation model reviewed structure names; no visual override met acceptance thresholds."]
            + list(case_summary.get("key_findings", []))
        )[:8]

    structure_reviews_path.write_text(json.dumps(structure_reviews, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    pd.DataFrame(structure_reviews).to_csv(structure_reviews_path.with_suffix(".csv"), index=False)
    case_summary_path.write_text(json.dumps(case_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    names_json = _write_json(output_dir / "structure_multimodal_names.json", naming_rows)
    names_csv = output_dir / "structure_multimodal_names.csv"
    pd.DataFrame(naming_rows).to_csv(names_csv, index=False)

    report_path = write_html_report(
        output_dir=Path(pathology_outputs["output_dir"]).resolve(),
        case_bundle=case_bundle,
        cluster_reviews=cluster_reviews,
        structure_reviews=structure_reviews,
        case_summary=case_summary,
    )
    return {
        "structure_multimodal_names_json": str(names_json),
        "structure_multimodal_names_csv": str(names_csv),
        "updated_structure_reviews_json": str(structure_reviews_path),
        "updated_structure_reviews_csv": str(structure_reviews_path.with_suffix(".csv")),
        "updated_case_summary_json": str(case_summary_path),
        "updated_report_html": str(report_path),
        "accepted_visual_overrides": str(accepted),
    }


def apply_he_contour_foundation(
    cfg: WorkflowConfig,
    workflow_result: dict[str, str],
) -> dict[str, str]:
    workflow_summary_path = Path(workflow_result["workflow_summary_json"]).resolve()
    workflow_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))
    output_root = Path(workflow_summary["output_root"]).resolve()
    he_output_dir = output_root / "he_foundation"
    he_output_dir.mkdir(parents=True, exist_ok=True)

    runtime_base_config = Path(workflow_summary["runtime_base_pipeline_config"]).resolve()
    base_cfg = load_base_pipeline_config(runtime_base_config)
    contour_geojson = resolve_contour_geojson(cfg, base_cfg)
    patch_records, patch_manifest = _extract_he_contour_patches(
        cfg=cfg,
        base_cfg=base_cfg,
        contour_geojson=contour_geojson,
        output_dir=he_output_dir,
    )
    classification = _classify_contours(cfg=cfg, records=patch_records, output_dir=he_output_dir)
    structure_summaries = summarize_contours_by_structure(classification["classifications"])
    structure_summary_paths = _write_structure_summary(he_output_dir, structure_summaries)

    spatial_cfg = SpatialPathologistConfig(
        case_name=cfg.case_name,
        study_context=cfg.study_context,
        base_pipeline_config=runtime_base_config,
        output_dir=Path(workflow_summary["pathology_outputs"]["output_dir"]).resolve(),
        pathology_review_backend=cfg.pathology_review_backend,
        pathology_ai_api_base_url=cfg.pathology_ai_api_base_url,
        pathology_ai_top_k=cfg.pathology_ai_top_k,
        pathology_ai_answer_language=cfg.pathology_ai_answer_language,
        pathology_ai_document_ids=tuple(cfg.pathology_ai_document_ids),
        openai_enabled=cfg.openai_enabled,
        openai_api_key_env=cfg.openai_api_key_env,
        openai_model=cfg.openai_model,
        openai_reasoning_effort=cfg.openai_reasoning_effort,
        openai_store=cfg.openai_store,
        force_recompute_pipeline=False,
        low_confidence_threshold=cfg.low_confidence_threshold,
        ambiguity_margin_threshold=cfg.ambiguity_margin_threshold,
        top_clusters_per_structure=cfg.top_clusters_per_structure,
    )
    case_bundle = build_case_bundle(spatial_cfg)
    naming_outputs = _apply_multimodal_names(
        cfg=cfg,
        case_bundle=case_bundle,
        pathology_outputs=workflow_summary["pathology_outputs"],
        structure_summaries=structure_summaries,
        output_dir=he_output_dir,
    )
    metadata = {
        "enabled": True,
        "contour_geojson": str(contour_geojson),
        "contour_key": cfg.he_contour_key,
        "model_id": cfg.he_foundation_model_id,
        "prompt_set": cfg.he_foundation_prompt_set,
        "top_k": cfg.he_foundation_top_k,
        "max_patch_side_px": cfg.he_foundation_max_patch_side_px,
        "patch_count": len(patch_records),
        "classification_count": len(classification["classifications"]),
        "classification_warnings": classification["warnings"],
        "visual_override_enabled": cfg.he_visual_override_enabled,
        "visual_override_min_llm_confidence": cfg.he_visual_override_min_llm_confidence,
        "visual_override_min_foundation_score": cfg.he_visual_override_min_foundation_score,
    }
    metadata_path = _write_json(he_output_dir / "he_foundation_metadata.json", metadata)

    he_outputs = {
        "he_foundation_dir": str(he_output_dir),
        "patch_manifest_json": str(patch_manifest),
        "classification_json": classification["classification_json"],
        "classification_csv": classification["classification_csv"],
        **structure_summary_paths,
        **naming_outputs,
        "metadata_json": str(metadata_path),
    }
    workflow_summary["he_foundation_outputs"] = he_outputs
    workflow_summary["pathology_outputs"]["report_html"] = naming_outputs["updated_report_html"]
    workflow_summary_path.write_text(
        json.dumps(workflow_summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        **workflow_result,
        "he_contour_classification_csv": classification["classification_csv"],
        "he_contour_to_structure_summary_csv": structure_summary_paths["structure_summary_csv"],
        "structure_multimodal_names_csv": naming_outputs["structure_multimodal_names_csv"],
        "workflow_summary_json": str(workflow_summary_path),
    }
