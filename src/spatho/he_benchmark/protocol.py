"""Run and initialize the H&E tumor-region benchmark protocol."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal
import json

from pydantic import BaseModel, ConfigDict, Field

from .catalog import MODEL_CATALOG, catalog_payload
from .datasets import (
    ingest_image_geojson,
    ingest_paired_directories,
    iter_cases,
    read_cases_jsonl,
    save_array,
    write_synthetic_fixture,
)
from .metrics import case_metrics, summarize_metrics
from .models import build_segmenters
from .report import make_overlay_rgb, write_benchmark_report


class DatasetRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1)
    cases_path: Path
    kind: Literal["public", "private"] = "private"
    enabled: bool = True


class HeBenchmarkProtocol(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(default="he_tumor_region_v1", min_length=1)
    tile_size_px: int = Field(default=1024, ge=64, le=4096)
    models: list[str] = Field(default_factory=lambda: ["stain_threshold", "dino_nested_unet", "uni2_upernet", "segtme_uni2"])
    datasets: list[DatasetRef] = Field(default_factory=list)
    prediction_dirs: dict[str, Path] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=lambda: ["dice", "iou", "precision", "recall", "accuracy", "hd95"])
    allow_partial: bool = True
    max_overlay_cases: int = Field(default=12, ge=0, le=64)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def default_protocol(*, root: Path) -> HeBenchmarkProtocol:
    datasets = [
        DatasetRef(dataset_id="private_he", cases_path=Path("datasets/private_he/cases.jsonl"), kind="private"),
    ]
    if (root / "datasets" / "public_camelyon16" / "cases.jsonl").exists():
        datasets.append(
            DatasetRef(
                dataset_id="public_camelyon16",
                cases_path=Path("datasets/public_camelyon16/cases.jsonl"),
                kind="public",
            )
        )
    return HeBenchmarkProtocol(datasets=datasets)


def init_benchmark(
    output_dir: Path | str,
    *,
    with_synthetic_fixture: bool = False,
    private_images_dir: Path | str | None = None,
    private_masks_dir: Path | str | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "datasets" / "private_he"
    public_dir = output_dir / "datasets" / "public_camelyon16"
    public_dir.mkdir(parents=True, exist_ok=True)
    ingest_meta = None
    if private_images_dir is not None:
        ingest_meta = ingest_paired_directories(
            dataset_id="private_he",
            images_dir=Path(private_images_dir),
            masks_dir=Path(private_masks_dir) if private_masks_dir else None,
            output_dir=private_dir,
        )
    elif with_synthetic_fixture:
        ingest_meta = write_synthetic_fixture(private_dir, dataset_id="private_he")
    else:
        private_dir.mkdir(parents=True, exist_ok=True)
        readme = private_dir / "README.md"
        readme.write_text(
            "Put paired H&E tiles in `images/` and tumor masks in `masks/` with the same stem,\n"
            "then run `spatho he-benchmark ingest --dataset-id private_he --images ... --masks ...`.\n",
            encoding="utf-8",
        )
    protocol = default_protocol(root=output_dir)
    protocol_path = _write_json(output_dir / "protocol.json", json.loads(protocol.model_dump_json()))
    catalog_path = _write_json(output_dir / "catalog.json", catalog_payload())
    return {
        "benchmark_dir": str(output_dir),
        "protocol_json": str(protocol_path),
        "catalog_json": str(catalog_path),
        "private_dataset": ingest_meta,
        "public_dataset_dir": str(public_dir),
    }


def load_protocol(path: Path) -> HeBenchmarkProtocol:
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocol = HeBenchmarkProtocol.model_validate(payload)
    datasets = []
    for dataset in protocol.datasets:
        cases_path = dataset.cases_path
        if not cases_path.is_absolute():
            cases_path = (path.parent / cases_path).resolve()
        datasets.append(dataset.model_copy(update={"cases_path": cases_path}))
    prediction_dirs = {}
    for key, value in protocol.prediction_dirs.items():
        pred_path = Path(value)
        if not pred_path.is_absolute():
            pred_path = (path.parent / pred_path).resolve()
        prediction_dirs[key] = pred_path
    return protocol.model_copy(update={"datasets": datasets, "prediction_dirs": prediction_dirs})


def doctor_benchmark(protocol_path: Path) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []
    protocol_file = Path(protocol_path)
    if not protocol_file.exists():
        return {
            "protocol_exists": False,
            "ready_to_run": False,
            "issues": [f"protocol missing: {protocol_file}"],
            "warnings": [],
        }
    protocol = load_protocol(protocol_file)
    dataset_reports = []
    n_scored = 0
    for dataset in protocol.datasets:
        exists = dataset.cases_path.exists()
        n_cases = 0
        n_masks = 0
        if exists:
            cases = read_cases_jsonl(dataset.cases_path)
            n_cases = len(cases)
            n_masks = sum(1 for case in cases if case.get("mask"))
            n_scored += n_masks
        else:
            if dataset.enabled:
                issues.append(f"cases.jsonl missing for {dataset.dataset_id}: {dataset.cases_path}")
        dataset_reports.append(
            {
                "dataset_id": dataset.dataset_id,
                "kind": dataset.kind,
                "enabled": dataset.enabled,
                "cases_exist": exists,
                "n_cases": n_cases,
                "n_with_masks": n_masks,
            }
        )
        if exists and n_masks == 0:
            warnings.append(f"{dataset.dataset_id} has no masks; ranking will be qualitative only.")
    pred_dirs = {key: str(value) for key, value in protocol.prediction_dirs.items()}
    segmenters = build_segmenters(protocol.models, prediction_dirs=pred_dirs)
    model_reports = []
    runnable = 0
    for model_id, segmenter in segmenters.items():
        ok, reason = segmenter.available()
        spec = MODEL_CATALOG.get(model_id, {})
        model_reports.append(
            {
                "model_id": model_id,
                "available": ok,
                "required": bool(spec.get("required")),
                "reason": reason,
                "track": spec.get("track"),
            }
        )
        if ok:
            runnable += 1
        elif spec.get("required") and not protocol.allow_partial:
            issues.append(f"required model unavailable: {model_id} ({reason})")
        elif spec.get("required"):
            warnings.append(f"required model unavailable, will skip: {model_id} ({reason})")
    ready = not issues and runnable >= 1
    return {
        "protocol_exists": True,
        "protocol_name": protocol.name,
        "ready_to_run": ready,
        "issues": issues,
        "warnings": warnings,
        "n_runnable_models": runnable,
        "n_scored_cases": n_scored,
        "datasets": dataset_reports,
        "models": model_reports,
    }


def run_benchmark(protocol_path: Path, *, output_dir: Path | None = None) -> dict[str, Any]:
    protocol_file = Path(protocol_path).expanduser().resolve()
    protocol = load_protocol(protocol_file)
    output_dir = Path(output_dir or (protocol_file.parent / "runs" / protocol.name)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_dirs = {key: str(value) for key, value in protocol.prediction_dirs.items()}
    segmenters = build_segmenters(protocol.models, prediction_dirs=pred_dirs)
    runnable: dict[str, Any] = {}
    skipped_unique: list[dict[str, str]] = []
    for model_id, segmenter in segmenters.items():
        ok, reason = segmenter.available()
        if ok:
            runnable[model_id] = segmenter
        else:
            skipped_unique.append({"model_id": model_id, "reason": reason})

    case_rows: list[dict[str, Any]] = []
    overlays: list[dict[str, Any]] = []
    predictions_by_case: dict[str, dict[str, Any]] = {}
    overlay_count = 0

    for dataset in protocol.datasets:
        if not dataset.enabled:
            continue
        if not dataset.cases_path.exists():
            if protocol.allow_partial:
                skipped_unique.append({"model_id": "*", "reason": f"missing dataset {dataset.dataset_id}"})
                continue
            raise FileNotFoundError(dataset.cases_path)
        for loaded in iter_cases(read_cases_jsonl(dataset.cases_path)):
            image = loaded["image_array"]
            gt = loaded["mask_array"]
            case_id = str(loaded["case_id"])
            predictions_by_case.setdefault(case_id, {})
            wrote_overlay = False
            for model_id, segmenter in runnable.items():
                pred = segmenter.predict(image, case=loaded)
                predictions_by_case[case_id][model_id] = pred
                row: dict[str, Any] = {
                    "case_id": case_id,
                    "dataset_id": dataset.dataset_id,
                    "kind": dataset.kind,
                    "model_id": model_id,
                    "qualitative_only": bool(gt is None or loaded.get("qualitative_only")),
                    "image": loaded.get("image"),
                }
                if gt is not None:
                    row.update(case_metrics(pred, gt))
                case_rows.append(row)
                if overlay_count < int(protocol.max_overlay_cases):
                    overlay = make_overlay_rgb(image, pred, gt)
                    overlay_path = save_array(
                        output_dir / "overlays" / f"{case_id}__{model_id}.npy",
                        overlay,
                    )
                    overlays.append(
                        {
                            "case_id": case_id,
                            "model_id": model_id,
                            "path": str(overlay_path),
                        }
                    )
                    wrote_overlay = True
            if wrote_overlay:
                overlay_count += 1

    leaderboard = _build_leaderboard(case_rows, protocol.models)
    agreement = _inter_model_agreement(predictions_by_case, protocol.models)
    report = write_benchmark_report(
        output_dir=output_dir,
        protocol=json.loads(protocol.model_dump_json()),
        case_rows=case_rows,
        leaderboard=leaderboard,
        agreement=agreement,
        skipped=skipped_unique,
        overlays=overlays,
    )
    summary = {
        "protocol_json": str(protocol_file),
        "output_dir": str(output_dir),
        "n_case_rows": len(case_rows),
        "n_runnable_models": len({row["model_id"] for row in case_rows}),
        "skipped_models": skipped_unique,
        **report,
    }
    _write_json(output_dir / "run_summary.json", summary)
    return summary


def _build_leaderboard(case_rows: list[dict[str, Any]], model_ids: list[str]) -> list[dict[str, Any]]:
    board: list[dict[str, Any]] = []
    for kind in ("private", "public"):
        for model_id in model_ids:
            rows = [
                row
                for row in case_rows
                if row["model_id"] == model_id and row.get("kind") == kind and not row.get("qualitative_only")
            ]
            if not rows:
                continue
            summary = summarize_metrics(rows)
            spec = MODEL_CATALOG.get(model_id, {})
            board.append(
                {
                    "kind": kind,
                    "model_id": model_id,
                    "track": spec.get("track"),
                    "role": spec.get("role"),
                    **summary,
                }
            )
    board.sort(
        key=lambda row: (
            0 if row["kind"] == "private" else 1,
            -(row.get("dice_mean") if row.get("dice_mean") is not None else -1.0),
            row.get("hd95_mean") if row.get("hd95_mean") is not None else 1e9,
            row["model_id"],
        )
    )
    return board


def _inter_model_agreement(
    predictions_by_case: dict[str, dict[str, Any]],
    model_ids: list[str],
) -> list[dict[str, Any]]:
    import numpy as np

    from .metrics import overlap_metrics

    pairs: dict[tuple[str, str], list[float]] = {}
    for preds in predictions_by_case.values():
        present = [model_id for model_id in model_ids if model_id in preds]
        for i, left in enumerate(present):
            for right in present[i + 1 :]:
                dice = overlap_metrics(preds[left], preds[right]).get("dice")
                if dice is None:
                    continue
                pairs.setdefault((left, right), []).append(float(dice))
    rows = []
    for (left, right), values in sorted(pairs.items()):
        if not values:
            continue
        rows.append(
            {
                "model_a": left,
                "model_b": right,
                "n_cases": len(values),
                "dice_mean": float(np.mean(values)),
            }
        )
    return rows


def ingest_dataset(
    *,
    dataset_id: str,
    output_dir: Path | str,
    images_dir: Path | str | None = None,
    masks_dir: Path | str | None = None,
    image_path: Path | str | None = None,
    geojson_path: Path | str | None = None,
    kind: str = "private",
    organ: str | None = None,
    pixel_size_um: float | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    if image_path is not None and geojson_path is not None:
        return ingest_image_geojson(
            dataset_id=dataset_id,
            image_path=Path(image_path),
            geojson_path=Path(geojson_path),
            output_dir=output_dir,
            kind=kind,
            organ=organ,
            pixel_size_um=pixel_size_um,
        )
    if images_dir is None:
        raise ValueError("Provide --images or --image plus --geojson.")
    return ingest_paired_directories(
        dataset_id=dataset_id,
        images_dir=Path(images_dir),
        masks_dir=Path(masks_dir) if masks_dir else None,
        output_dir=output_dir,
        kind=kind,
        organ=organ,
        pixel_size_um=pixel_size_um,
    )
