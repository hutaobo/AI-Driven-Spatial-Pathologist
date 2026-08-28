"""Load paired H&E images and tumor masks, including GeoJSON rasterization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator
import json
import re

import numpy as np

from .catalog import DEFAULT_TUMOR_TOKENS


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".npy"}
MASK_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def _slug(value: str, *, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")
    return (slug or "case")[:max_len]


def load_array(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path))
    try:
        from PIL import Image
    except Exception:
        Image = None
    if Image is not None:
        with Image.open(path) as handle:
            return np.asarray(handle.convert("RGB") if handle.mode != "L" else handle)
    try:
        import matplotlib.image as mpimg
    except Exception as exc:
        raise RuntimeError(f"Reading {path} requires Pillow or matplotlib.") from exc
    array = np.asarray(mpimg.imread(path))
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8 and np.issubdtype(array.dtype, np.floating):
        array = np.clip(array * (255.0 if array.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
    return array


def save_array(path: Path, array: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array)
    if path.suffix.lower() == ".npy":
        np.save(path, arr)
        return path
    if arr.dtype != np.uint8:
        if arr.dtype == np.bool_ or set(np.unique(arr).tolist()) <= {0, 1}:
            arr = (arr.astype(np.uint8) * 255)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
    except Exception as exc:
        if path.suffix.lower() != ".npy":
            npy_path = path.with_suffix(".npy")
            np.save(npy_path, arr)
            return npy_path
        raise RuntimeError(f"Saving {path} requires Pillow.") from exc
    if arr.ndim == 2:
        Image.fromarray(arr, mode="L").save(path)
    else:
        Image.fromarray(arr[..., :3]).save(path)
    return path


def _feature_label(properties: dict[str, Any]) -> str:
    classification = properties.get("classification")
    if isinstance(classification, dict):
        name = classification.get("name")
        if name:
            return str(name)
    for key in ("name", "label", "class", "tissue", "region"):
        value = properties.get(key)
        if value:
            return str(value)
    return ""


def is_tumor_label(label: str, *, tokens: tuple[str, ...] = DEFAULT_TUMOR_TOKENS) -> bool:
    lowered = str(label).lower()
    return any(token in lowered for token in tokens)


def rasterize_geojson(
    geojson_path: Path,
    *,
    height: int,
    width: int,
    tumor_tokens: tuple[str, ...] = DEFAULT_TUMOR_TOKENS,
) -> np.ndarray:
    payload = json.loads(Path(geojson_path).read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not isinstance(features, list):
        raise ValueError(f"GeoJSON has no features list: {geojson_path}")
    ys, xs = np.mgrid[0:height, 0:width]
    points = np.stack([xs.ravel(), ys.ravel()], axis=1)
    mask = np.zeros(points.shape[0], dtype=bool)
    try:
        from matplotlib.path import Path as MplPath
        from shapely.geometry import shape
    except Exception as exc:
        raise RuntimeError("GeoJSON rasterization requires matplotlib and shapely.") from exc
    matched = 0
    for feature in features:
        if not isinstance(feature, dict):
            continue
        label = _feature_label(dict(feature.get("properties") or {}))
        if label and not is_tumor_label(label, tokens=tumor_tokens):
            continue
        geometry = feature.get("geometry")
        if not geometry:
            continue
        geom = shape(geometry)
        geoms = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
        for polygon in geoms:
            if polygon.is_empty or polygon.geom_type != "Polygon":
                continue
            path = MplPath(np.asarray(polygon.exterior.coords, dtype=float))
            contained = path.contains_points(points)
            for interior in polygon.interiors:
                hole = MplPath(np.asarray(interior.coords, dtype=float))
                contained &= np.logical_not(hole.contains_points(points))
            mask |= contained
            matched += 1
    if matched == 0:
        raise ValueError(f"No tumor polygons found in {geojson_path}.")
    return mask.reshape(height, width)


def write_cases_jsonl(path: Path, cases: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(case, ensure_ascii=False, default=str) for case in cases]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def read_cases_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    cases: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{path}:{line_no} is not a JSON object.")
        cases.append(payload)
    return cases


def _index_by_stem(directory: Path, suffixes: set[str]) -> dict[str, Path]:
    indexed: dict[str, Path] = {}
    if not directory.exists():
        return indexed
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in suffixes:
            indexed[path.stem] = path
    return indexed


def ingest_paired_directories(
    *,
    dataset_id: str,
    images_dir: Path,
    masks_dir: Path | None,
    output_dir: Path,
    kind: str = "private",
    organ: str | None = None,
    pixel_size_um: float | None = None,
    split: str = "test",
) -> dict[str, Any]:
    images_dir = Path(images_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    image_index = _index_by_stem(images_dir, IMAGE_SUFFIXES)
    mask_index = _index_by_stem(Path(masks_dir).expanduser().resolve(), MASK_SUFFIXES) if masks_dir else {}
    if not image_index:
        raise FileNotFoundError(f"No images found in {images_dir}")
    cases: list[dict[str, Any]] = []
    missing_masks = 0
    for stem, image_path in image_index.items():
        mask_path = mask_index.get(stem)
        if mask_path is None:
            missing_masks += 1
        cases.append(
            {
                "case_id": f"{dataset_id}__{_slug(stem)}",
                "dataset_id": dataset_id,
                "kind": kind,
                "split": split,
                "organ": organ,
                "pixel_size_um": pixel_size_um,
                "image": str(image_path),
                "mask": str(mask_path) if mask_path else None,
                "qualitative_only": mask_path is None,
            }
        )
    cases_path = write_cases_jsonl(output_dir / "cases.jsonl", cases)
    metadata = {
        "dataset_id": dataset_id,
        "kind": kind,
        "n_cases": len(cases),
        "n_with_masks": len(cases) - missing_masks,
        "n_qualitative_only": missing_masks,
        "cases_path": str(cases_path),
        "images_dir": str(images_dir),
        "masks_dir": str(Path(masks_dir).expanduser().resolve()) if masks_dir else None,
    }
    _write_json(output_dir / "dataset_metadata.json", metadata)
    return metadata


def ingest_image_geojson(
    *,
    dataset_id: str,
    image_path: Path,
    geojson_path: Path,
    output_dir: Path,
    kind: str = "private",
    organ: str | None = None,
    pixel_size_um: float | None = None,
    split: str = "test",
    tumor_tokens: tuple[str, ...] = DEFAULT_TUMOR_TOKENS,
) -> dict[str, Any]:
    image_path = Path(image_path).expanduser().resolve()
    geojson_path = Path(geojson_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    image = load_array(image_path)
    height, width = int(image.shape[0]), int(image.shape[1])
    mask = rasterize_geojson(geojson_path, height=height, width=width, tumor_tokens=tumor_tokens)
    image_out = save_array(output_dir / "images" / f"{image_path.stem}.npy", image)
    mask_out = save_array(output_dir / "masks" / f"{image_path.stem}.npy", mask.astype(np.uint8))
    case = {
        "case_id": f"{dataset_id}__{_slug(image_path.stem)}",
        "dataset_id": dataset_id,
        "kind": kind,
        "split": split,
        "organ": organ,
        "pixel_size_um": pixel_size_um,
        "image": str(image_out),
        "mask": str(mask_out),
        "qualitative_only": False,
        "source_image": str(image_path),
        "source_geojson": str(geojson_path),
        "n_tumor_pixels": int(mask.sum()),
    }
    cases_path = write_cases_jsonl(output_dir / "cases.jsonl", [case])
    metadata = {
        "dataset_id": dataset_id,
        "kind": kind,
        "n_cases": 1,
        "n_with_masks": 1,
        "n_qualitative_only": 0,
        "cases_path": str(cases_path),
        "image": str(image_out),
        "mask": str(mask_out),
    }
    _write_json(output_dir / "dataset_metadata.json", metadata)
    return metadata


def iter_cases(cases: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for case in cases:
        image = load_array(Path(case["image"]))
        mask = None
        if case.get("mask"):
            mask = load_array(Path(case["mask"]))
        yield {**case, "image_array": image, "mask_array": mask}


def write_synthetic_fixture(output_dir: Path, *, dataset_id: str = "private_he") -> dict[str, Any]:
    """Create a tiny purple-blob tumor image plus an exact mask.

    Used by tests and `spatho he-benchmark init --with-synthetic-fixture`.
    """
    output_dir = Path(output_dir).expanduser().resolve()
    height, width = 64, 64
    image = np.full((height, width, 3), 230, dtype=np.uint8)
    image[:, :] = (220, 180, 200)
    yy, xx = np.ogrid[:height, :width]
    blob = (yy - 32) ** 2 + (xx - 28) ** 2 <= 14**2
    image[blob] = (110, 30, 150)
    artifact = (yy - 10) ** 2 + (xx - 10) ** 2 <= 6**2
    image[artifact] = (40, 40, 40)
    whitespace = np.zeros((height, width), dtype=bool)
    whitespace[:, 55:] = True
    image[whitespace] = (245, 245, 245)
    mask = blob & np.logical_not(whitespace)
    images_dir = output_dir / "images"
    masks_dir = output_dir / "masks"
    image_path = save_array(images_dir / "synthetic_tumor.npy", image)
    mask_path = save_array(masks_dir / "synthetic_tumor.npy", mask.astype(np.uint8))
    case = {
        "case_id": f"{dataset_id}__synthetic_tumor",
        "dataset_id": dataset_id,
        "kind": "private",
        "split": "test",
        "organ": "breast",
        "pixel_size_um": 0.5,
        "image": str(image_path),
        "mask": str(mask_path),
        "qualitative_only": False,
    }
    cases_path = write_cases_jsonl(output_dir / "cases.jsonl", [case])
    metadata = {
        "dataset_id": dataset_id,
        "kind": "private",
        "synthetic": True,
        "n_cases": 1,
        "n_with_masks": 1,
        "cases_path": str(cases_path),
    }
    _write_json(output_dir / "dataset_metadata.json", metadata)
    return metadata
