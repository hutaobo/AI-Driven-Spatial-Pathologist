from __future__ import annotations

import json
import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Iterable


def bootstrap_runtime_env() -> None:
    """Point caches to writable paths before importing HistoSeg/matplotlib."""
    os.environ.setdefault("HOME", "/tmp")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("GRADIO_TEMP_DIR", "/tmp/gradio")

    for key in ("HOME", "XDG_CACHE_HOME", "MPLCONFIGDIR", "GRADIO_TEMP_DIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


bootstrap_runtime_env()

import gradio as gr

try:
    from histoseg import Pattern1IsolineConfig, run_pattern1_isoline

    HISTOSEG_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - startup fallback only
    Pattern1IsolineConfig = None  # type: ignore[assignment]
    run_pattern1_isoline = None  # type: ignore[assignment]
    HISTOSEG_IMPORT_ERROR = str(exc)


APP_NAME = "AI Driven Spatial Pathologist"
APP_DESCRIPTION = (
    "Upload a Xenium bundle or the required HistoSeg input files, "
    "run Pattern1 isoline analysis, and download the generated contours."
)
DEFAULT_PATTERN1 = "10,23,19,27,14,20,25,26"
PREFERRED_WORK_DIR = Path(os.environ.get("APP_DATA_DIR", "./project-vol")).resolve()
FALLBACK_WORK_DIR = Path("/tmp/project-vol")


def resolve_work_dir() -> Path:
    for candidate in (PREFERRED_WORK_DIR, FALLBACK_WORK_DIR):
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return candidate
        except OSError:
            continue
    raise PermissionError(
        f"Could not find a writable work directory. Tried: {PREFERRED_WORK_DIR} and {FALLBACK_WORK_DIR}"
    )


DEFAULT_WORK_DIR = resolve_work_dir()
RUNS_DIR = DEFAULT_WORK_DIR / "runs"


def ensure_workdirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)


def parse_pattern1_clusters(raw: str) -> list[int | str]:
    values: list[int | str] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        if token.lstrip("-").isdigit():
            values.append(int(token))
        else:
            values.append(token)
    if not values:
        raise ValueError("Pattern1 clusters cannot be empty.")
    return values


def stage_uploaded_file(uploaded: object | None, target_dir: Path, explicit_name: str | None = None) -> Path | None:
    if uploaded is None:
        return None
    source = Path(str(uploaded))
    if not source.exists():
        raise FileNotFoundError(f"Uploaded file not found: {source}")
    filename = explicit_name or source.name
    destination = target_dir / filename
    shutil.copy2(source, destination)
    return destination


def extract_zip_bundle(bundle_zip: object | None, target_dir: Path) -> Path | None:
    if bundle_zip is None:
        return None
    archive_path = stage_uploaded_file(bundle_zip, target_dir)
    assert archive_path is not None
    extract_dir = target_dir / "bundle"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(extract_dir)
    return extract_dir


def _pick_first(paths: Iterable[Path]) -> Path | None:
    for candidate in paths:
        return candidate
    return None


def find_bundle_file(bundle_dir: Path, exact_relative_path: str, filename: str) -> Path | None:
    exact_path = bundle_dir / exact_relative_path
    if exact_path.exists():
        return exact_path

    lowercase_filename = filename.lower()
    matches = (
        path
        for path in bundle_dir.rglob("*")
        if path.is_file() and path.name.lower() == lowercase_filename
    )
    return _pick_first(matches)


def resolve_inputs(
    *,
    bundle_dir: Path | None,
    cells_upload: object | None,
    clusters_upload: object | None,
    tissue_upload: object | None,
    target_dir: Path,
) -> tuple[Path, Path, Path | None]:
    cells_path = stage_uploaded_file(cells_upload, target_dir)
    clusters_path = stage_uploaded_file(clusters_upload, target_dir)
    tissue_path = stage_uploaded_file(tissue_upload, target_dir)

    if bundle_dir is not None:
        if cells_path is None:
            cells_path = find_bundle_file(bundle_dir, "cells.parquet", "cells.parquet")
        if clusters_path is None:
            clusters_path = find_bundle_file(
                bundle_dir,
                "analysis/clustering/gene_expression_graphclust/clusters.csv",
                "clusters.csv",
            )
        if tissue_path is None:
            tissue_path = find_bundle_file(bundle_dir, "tissue_boundary.csv", "tissue_boundary.csv")

    if cells_path is None:
        raise ValueError("Missing cells.parquet. Upload it directly or include it in the Xenium zip.")
    if clusters_path is None:
        raise ValueError(
            "Missing clusters.csv. Upload it directly or include "
            "analysis/clustering/gene_expression_graphclust/clusters.csv in the zip."
        )

    return cells_path, clusters_path, tissue_path


def build_run_dir() -> Path:
    ensure_workdirs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / f"run-{stamp}"
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = RUNS_DIR / f"run-{stamp}-{suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def zip_outputs(output_dir: Path) -> Path:
    archive_base = output_dir / "histoseg_outputs"
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)
    return Path(archive_path)


def format_summary(result: object, *, used_tissue_boundary: bool, work_dir: Path) -> dict[str, object]:
    payload = {
        "work_dir": str(work_dir),
        "out_dir": str(result.out_dir),
        "id_col_used": result.id_col_used,
        "x_col": result.x_col,
        "y_col": result.y_col,
        "n_target_cells": result.n_target_cells,
        "n_bg0_points": result.n_bg0_points,
        "n_contours": len(result.contours),
        "label_scheme": result.label_scheme,
        "used_tissue_boundary": used_tissue_boundary,
    }
    if result.segmentation_confidence_score is not None:
        payload["segmentation_confidence_score"] = result.segmentation_confidence_score
    if result.segmentation_confidence_stats is not None:
        payload["segmentation_confidence_stats"] = result.segmentation_confidence_stats
    return payload


def run_analysis(
    bundle_zip: object | None,
    cells_parquet: object | None,
    clusters_csv: object | None,
    tissue_boundary_csv: object | None,
    pattern1_clusters: str,
    grid_n: int,
    knn_k: int,
    smooth_sigma: float,
    min_cells_inside: int,
    label_scheme: str,
    use_synth_bg: bool,
    compute_confidence_score: bool,
    bbox_expand_um: float,
    syn_bg_density: float,
) -> tuple[str, str | None, dict[str, object], list[str]]:
    if HISTOSEG_IMPORT_ERROR is not None:
        raise gr.Error(
            "HistoSeg could not be imported inside the app container. "
            f"Import error: {HISTOSEG_IMPORT_ERROR}"
        )

    run_dir = build_run_dir()
    upload_dir = run_dir / "inputs"
    output_dir = run_dir / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = extract_zip_bundle(bundle_zip, upload_dir)
    cells_path, clusters_path, tissue_path = resolve_inputs(
        bundle_dir=bundle_dir,
        cells_upload=cells_parquet,
        clusters_upload=clusters_csv,
        tissue_upload=tissue_boundary_csv,
        target_dir=upload_dir,
    )

    parsed_clusters = parse_pattern1_clusters(pattern1_clusters)

    effective_use_synth_bg = bool(use_synth_bg and tissue_path is not None)
    messages = []
    if use_synth_bg and tissue_path is None:
        messages.append(
            "No tissue_boundary.csv was provided, so synthetic background was disabled automatically."
        )

    cfg = Pattern1IsolineConfig(
        clusters_csv=clusters_path,
        cells_parquet=cells_path,
        tissue_boundary_csv=tissue_path,
        out_dir=output_dir,
        pattern1_clusters=parsed_clusters,
        grid_n=int(grid_n),
        knn_k=int(knn_k),
        smooth_sigma=float(smooth_sigma),
        min_cells_inside=int(min_cells_inside),
        label_scheme=label_scheme,
        use_synth_bg=effective_use_synth_bg,
        compute_confidence_score=bool(compute_confidence_score),
        bbox_expand_um=float(bbox_expand_um),
        syn_bg_density=float(syn_bg_density),
    )

    result = run_pattern1_isoline(cfg)
    archive_path = zip_outputs(output_dir)

    output_files = [str(archive_path)]
    if result.preview_png is not None:
        output_files.append(str(result.preview_png))
    if result.params_json is not None:
        output_files.append(str(result.params_json))
    output_files.extend(str(path) for path in sorted(output_dir.glob("pattern1_isoline_*.npy")))

    summary = format_summary(
        result,
        used_tissue_boundary=tissue_path is not None,
        work_dir=run_dir,
    )

    status_lines = [
        f"{APP_NAME} finished successfully.",
        f"Contours generated: {len(result.contours)}",
        f"Run directory: {run_dir}",
    ]
    status_lines.extend(messages)

    preview_path = str(result.preview_png) if result.preview_png is not None else None
    return "\n".join(status_lines), preview_path, summary, output_files


CUSTOM_CSS = """
.hero {
  background: linear-gradient(135deg, #f4efe6 0%, #efe3cf 48%, #e1efe8 100%);
  border: 1px solid #d8c8ab;
  border-radius: 20px;
  padding: 24px;
  margin-bottom: 18px;
}
.hero h1 {
  font-size: 2.2rem;
  margin: 0 0 8px 0;
  color: #203028;
}
.hero p {
  margin: 0;
  color: #38463d;
  font-size: 1.02rem;
}
.callout {
  border-left: 4px solid #9f5c3f;
  background: #fbf6f1;
  padding: 14px 16px;
  border-radius: 10px;
}
"""


with gr.Blocks(
    title=APP_NAME,
    css=CUSTOM_CSS,
    theme=gr.themes.Soft(
        primary_hue="emerald",
        secondary_hue="stone",
        neutral_hue="slate",
    ),
) as demo:
    ensure_workdirs()

    gr.HTML(
        """
        <div class="hero">
          <h1>AI Driven Spatial Pathologist</h1>
          <p>
            A SciLifeLab Serve-ready wrapper around HistoSeg for Xenium-driven Pattern1 isoline analysis.
            Upload a Xenium zip bundle or the required files directly, tune the parameters, and download the contours.
          </p>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="callout">
        Required analysis inputs: <code>cells.parquet</code> and <code>clusters.csv</code>.
        Recommended: <code>tissue_boundary.csv</code> so synthetic background can be used.
        A zipped Xenium output folder works if it contains those files, especially
        <code>analysis/clustering/gene_expression_graphclust/clusters.csv</code>.
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            bundle_zip = gr.File(
                label="Xenium zip bundle",
                file_types=[".zip"],
                type="filepath",
            )
            cells_parquet = gr.File(
                label="cells.parquet",
                file_types=[".parquet"],
                type="filepath",
            )
            clusters_csv = gr.File(
                label="clusters.csv",
                file_types=[".csv"],
                type="filepath",
            )
            tissue_boundary_csv = gr.File(
                label="tissue_boundary.csv",
                file_types=[".csv"],
                type="filepath",
            )
            pattern1_clusters = gr.Textbox(
                label="Pattern1 clusters",
                value=DEFAULT_PATTERN1,
                info="Comma-separated cluster labels or integers.",
            )
            label_scheme = gr.Dropdown(
                label="Label scheme",
                choices=["p1_is_one", "p1_is_zero"],
                value="p1_is_one",
            )

            with gr.Accordion("Advanced parameters", open=False):
                grid_n = gr.Slider(label="grid_n", minimum=200, maximum=1600, step=50, value=1200)
                knn_k = gr.Slider(label="knn_k", minimum=5, maximum=100, step=1, value=30)
                smooth_sigma = gr.Slider(label="smooth_sigma", minimum=0.5, maximum=12.0, step=0.5, value=5.0)
                min_cells_inside = gr.Slider(label="min_cells_inside", minimum=1, maximum=200, step=1, value=10)
                bbox_expand_um = gr.Slider(label="bbox_expand_um", minimum=0, maximum=500, step=10, value=100)
                syn_bg_density = gr.Slider(label="syn_bg_density", minimum=0.001, maximum=0.05, step=0.001, value=0.01)
                use_synth_bg = gr.Checkbox(label="Use synthetic background", value=True)
                compute_confidence_score = gr.Checkbox(label="Compute segmentation confidence score", value=False)

            run_button = gr.Button("Run HistoSeg analysis", variant="primary")

        with gr.Column(scale=1):
            status_text = gr.Textbox(label="Status", lines=5)
            preview_image = gr.Image(label="Preview", type="filepath")
            summary_json = gr.JSON(label="Run summary")
            output_files = gr.File(label="Download outputs", file_count="multiple")

    run_button.click(
        fn=run_analysis,
        inputs=[
            bundle_zip,
            cells_parquet,
            clusters_csv,
            tissue_boundary_csv,
            pattern1_clusters,
            grid_n,
            knn_k,
            smooth_sigma,
            min_cells_inside,
            label_scheme,
            use_synth_bg,
            compute_confidence_score,
            bbox_expand_um,
            syn_bg_density,
        ],
        outputs=[status_text, preview_image, summary_json, output_files],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_api=False,
    )
