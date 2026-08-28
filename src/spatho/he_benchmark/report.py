"""Write JSON/CSV/Markdown artifacts for an H&E tumor-region benchmark run."""

from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json

import numpy as np

from .metrics import as_binary_mask


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(title for _, title in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_fmt(row.get(key)) for key, _ in columns) + " |")
    if not body:
        body.append("| " + " | ".join("—" for _ in columns) + " |")
    return "\n".join([header, sep, *body])


def make_overlay_rgb(image: np.ndarray, pred: np.ndarray, gt: np.ndarray | None = None) -> np.ndarray:
    rgb = np.asarray(image)
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        finite = rgb[np.isfinite(rgb)]
        high = float(finite.max()) if finite.size else 1.0
        scale = 255.0 if high <= 1.0 else 1.0
        rgb = np.clip(rgb * scale, 0, 255).astype(np.uint8)
    overlay = rgb.copy().astype(np.float32)
    pred_m = as_binary_mask(pred)
    green = np.array([40.0, 200.0, 40.0])
    red = np.array([220.0, 40.0, 40.0])
    blue = np.array([40.0, 80.0, 220.0])
    if gt is None:
        overlay[pred_m] = 0.45 * overlay[pred_m] + 0.55 * red
    else:
        gt_m = as_binary_mask(gt)
        tp = np.logical_and(pred_m, gt_m)
        fp = np.logical_and(pred_m, np.logical_not(gt_m))
        fn = np.logical_and(np.logical_not(pred_m), gt_m)
        overlay[tp] = 0.45 * overlay[tp] + 0.55 * green
        overlay[fp] = 0.45 * overlay[fp] + 0.55 * red
        overlay[fn] = 0.45 * overlay[fn] + 0.55 * blue
    return np.clip(overlay, 0, 255).astype(np.uint8)


def write_benchmark_report(
    *,
    output_dir: Path,
    protocol: dict[str, Any],
    case_rows: list[dict[str, Any]],
    leaderboard: list[dict[str, Any]],
    agreement: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    overlays: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir = Path(output_dir)
    leaderboard_json = _write_json(output_dir / "leaderboard.json", leaderboard)
    cases_json = _write_json(output_dir / "case_metrics.json", case_rows)
    cases_csv = _write_csv(output_dir / "case_metrics.csv", case_rows)
    agreement_json = _write_json(output_dir / "model_agreement.json", agreement)
    skipped_json = _write_json(output_dir / "skipped_models.json", skipped)
    protocol_json = _write_json(output_dir / "protocol.snapshot.json", protocol)

    private_board = [row for row in leaderboard if row.get("kind") == "private"]
    public_board = [row for row in leaderboard if row.get("kind") == "public"]
    columns = [
        ("kind", "Set"),
        ("model_id", "Model"),
        ("track", "Track"),
        ("n_cases", "N"),
        ("dice_mean", "Dice"),
        ("dice_std", "Dice std"),
        ("iou_mean", "IoU"),
        ("precision_mean", "Prec"),
        ("recall_mean", "Rec"),
        ("hd95_mean", "HD95"),
    ]
    lines = [
        f"# H&E tumor-region benchmark: {protocol.get('name', 'he_tumor_region')}",
        "",
        "Private slides decide which model to keep. Public sets only check that the implementation is sane.",
        "",
        "## Private ranking",
        "",
        _markdown_table(private_board, columns),
        "",
        "## Public ranking",
        "",
        _markdown_table(public_board, columns),
        "",
        "## Inter-model agreement (no ground truth required)",
        "",
        _markdown_table(
            agreement,
            [("model_a", "A"), ("model_b", "B"), ("n_cases", "N"), ("dice_mean", "Dice")],
        ),
        "",
        "## Skipped models",
        "",
    ]
    if skipped:
        for item in skipped:
            lines.append(f"- `{item.get('model_id')}`: {item.get('reason')}")
    else:
        lines.append("- none")
    lines.extend(["", "## Case rows", "", f"- `{cases_csv.name}`", ""])
    report_md = output_dir / "leaderboard.md"
    report_md.write_text("\n".join(lines), encoding="utf-8")
    return {
        "leaderboard_json": str(leaderboard_json),
        "leaderboard_md": str(report_md),
        "case_metrics_json": str(cases_json),
        "case_metrics_csv": str(cases_csv),
        "model_agreement_json": str(agreement_json),
        "skipped_models_json": str(skipped_json),
        "protocol_snapshot_json": str(protocol_json),
        "n_overlays": str(len(overlays)),
    }
