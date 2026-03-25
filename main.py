from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


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
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import KNeighborsRegressor

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover - optional runtime helper
    pq = None

try:
    from histoseg import Pattern1IsolineConfig
    from histoseg.contours.pattern1_isoline import (
        Pattern1IsolineResult,
        _normalize_cluster_label,
        _validate_label_scheme,
        align_clusters_with_cells,
        extract_contour_paths,
        filter_loops_by_cell_count,
        generate_synthetic_bg_in_bbox,
        load_tissue_boundary_csv,
        make_mesh_from_xy,
        sample_background_from_other_cells_plus_synth,
        tissue_mask_from_xy,
    )
    from histoseg.sfplot.Searcher_Findee_Score import (
        compute_cophenetic_from_distance_matrix,
        compute_searcher_findee_distance_matrix_from_df,
        plot_cophenetic_heatmap,
    )

    HISTOSEG_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - startup fallback only
    Pattern1IsolineConfig = None  # type: ignore[assignment]
    Pattern1IsolineResult = None  # type: ignore[assignment]
    _normalize_cluster_label = None  # type: ignore[assignment]
    _validate_label_scheme = None  # type: ignore[assignment]
    align_clusters_with_cells = None  # type: ignore[assignment]
    extract_contour_paths = None  # type: ignore[assignment]
    filter_loops_by_cell_count = None  # type: ignore[assignment]
    generate_synthetic_bg_in_bbox = None  # type: ignore[assignment]
    load_tissue_boundary_csv = None  # type: ignore[assignment]
    make_mesh_from_xy = None  # type: ignore[assignment]
    sample_background_from_other_cells_plus_synth = None  # type: ignore[assignment]
    tissue_mask_from_xy = None  # type: ignore[assignment]
    compute_cophenetic_from_distance_matrix = None  # type: ignore[assignment]
    compute_searcher_findee_distance_matrix_from_df = None  # type: ignore[assignment]
    plot_cophenetic_heatmap = None  # type: ignore[assignment]
    HISTOSEG_IMPORT_ERROR = str(exc)


APP_NAME = "AI Driven Spatial Pathologist"
APP_DESCRIPTION = (
    "Upload cell coordinates and cluster assignments, run target-cluster contour analysis, "
    "and download the contour preview plus the cophenetic heatmap."
)
DEFAULT_PATTERN1 = "10,23,19,27,14,20,25,26"
LABEL_SCHEME_OPTIONS = {
    "Selected clusters score high (recommended)": "p1_is_one",
    "Selected clusters score low (invert scoring)": "p1_is_zero",
}
DEFAULT_LABEL_SCHEME = "Selected clusters score high (recommended)"
PREFERRED_WORK_DIR = Path(os.environ.get("APP_DATA_DIR", "./project-vol")).resolve()
FALLBACK_WORK_DIR = Path("/tmp/project-vol")


@dataclass(frozen=True)
class RuntimeProfile:
    grid_n: int
    bg_max_points: int
    syn_bg_density: float
    syn_bg_min: int
    syn_bg_max: int
    scale_label: str
    notes: tuple[str, ...]


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


def log_event(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def to_internal_label_scheme(label_scheme: str) -> str:
    if label_scheme in LABEL_SCHEME_OPTIONS:
        return LABEL_SCHEME_OPTIONS[label_scheme]
    return _validate_label_scheme(label_scheme)


def describe_label_scheme(label_scheme: str) -> str:
    internal = to_internal_label_scheme(label_scheme)
    if internal == "p1_is_one":
        return "Selected clusters score high"
    return "Selected clusters score low (inverted)"


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
        raise ValueError("Clusters to outline cannot be empty.")
    return values


def safe_count_parquet_rows(parquet_path: Path) -> int | None:
    if pq is None:
        return None
    try:
        return int(pq.ParquetFile(parquet_path).metadata.num_rows)
    except Exception:
        return None


def safe_count_csv_rows(csv_path: Path) -> int | None:
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            count = sum(1 for _ in handle) - 1
        return max(count, 0)
    except Exception:
        return None


def choose_runtime_profile(
    *,
    requested_grid_n: int,
    requested_syn_bg_density: float,
    use_synth_bg: bool,
    estimated_rows: int | None,
) -> RuntimeProfile:
    effective_grid_n = int(requested_grid_n)
    bg_max_points = 60000
    syn_bg_density = float(requested_syn_bg_density)
    syn_bg_min = 20000
    syn_bg_max = 120000
    notes: list[str] = []

    ref_rows = estimated_rows or 0
    if ref_rows >= 80000:
        scale_label = "large"
        effective_grid_n = min(effective_grid_n, 450)
        bg_max_points = 12000
        syn_bg_density = min(syn_bg_density, 0.0015)
        syn_bg_min = 4000
        syn_bg_max = 12000
    elif ref_rows >= 40000:
        scale_label = "medium-large"
        effective_grid_n = min(effective_grid_n, 550)
        bg_max_points = 18000
        syn_bg_density = min(syn_bg_density, 0.0025)
        syn_bg_min = 5000
        syn_bg_max = 18000
    elif ref_rows >= 20000:
        scale_label = "medium"
        effective_grid_n = min(effective_grid_n, 650)
        bg_max_points = 25000
        syn_bg_density = min(syn_bg_density, 0.0035)
        syn_bg_min = 8000
        syn_bg_max = 25000
    elif ref_rows >= 10000:
        scale_label = "small-medium"
        effective_grid_n = min(effective_grid_n, 800)
        bg_max_points = 35000
        syn_bg_density = min(syn_bg_density, 0.005)
        syn_bg_min = 12000
        syn_bg_max = 35000
    else:
        scale_label = "small"

    if effective_grid_n != int(requested_grid_n):
        notes.append(
            f"Auto-reduced grid_n from {requested_grid_n} to {effective_grid_n} for Serve runtime stability."
        )
    if use_synth_bg and syn_bg_density != float(requested_syn_bg_density):
        notes.append(
            f"Auto-reduced synthetic background density from {requested_syn_bg_density:.4f} to {syn_bg_density:.4f}."
        )

    return RuntimeProfile(
        grid_n=effective_grid_n,
        bg_max_points=bg_max_points,
        syn_bg_density=syn_bg_density,
        syn_bg_min=syn_bg_min,
        syn_bg_max=syn_bg_max,
        scale_label=scale_label,
        notes=tuple(notes),
    )


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


def resolve_inputs(
    *,
    cells_upload: object | None,
    clusters_upload: object | None,
    tissue_upload: object | None,
    target_dir: Path,
) -> tuple[Path, Path, Path | None]:
    cells_path = stage_uploaded_file(cells_upload, target_dir)
    clusters_path = stage_uploaded_file(clusters_upload, target_dir)
    tissue_path = stage_uploaded_file(tissue_upload, target_dir)

    if cells_path is None:
        raise ValueError("Missing cells.parquet. Please upload the cell coordinate file.")
    if clusters_path is None:
        raise ValueError("Missing clusters.csv. Please upload the cluster assignment file.")

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


def cleanup_old_runs(max_keep: int = 2) -> list[str]:
    ensure_workdirs()
    runs = sorted(
        [path for path in RUNS_DIR.glob("run-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: list[str] = []
    for stale in runs[max_keep:]:
        try:
            shutil.rmtree(stale)
            removed.append(stale.name)
        except OSError:
            continue
    return removed


def directory_size_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def zip_outputs(output_dir: Path) -> tuple[Path | None, str | None]:
    archive_base = output_dir / "histoseg_outputs"
    archive_path = Path(f"{archive_base}.zip")
    output_bytes = directory_size_bytes(output_dir)
    free_bytes = shutil.disk_usage(output_dir).free

    # Creating a zip duplicates the output payload temporarily, so we keep a safety margin.
    required_free = max(output_bytes * 2, 256 * 1024 * 1024)
    if free_bytes < required_free:
        return None, (
            "Skipped zip archive because disk space is low on the Serve instance. "
            "The raw output files are still available below."
        )

    try:
        archive_path_str = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)
        return Path(archive_path_str), None
    except OSError as exc:
        try:
            archive_path.unlink(missing_ok=True)
        except OSError:
            pass
        if getattr(exc, "errno", None) == 28:
            return None, (
                "Skipped zip archive because the Serve instance ran out of disk space. "
                "The raw output files are still available below."
            )
        raise


def compute_cophenetic_outputs(
    *,
    merged_df: object,
    pattern1_clusters: list[int | str],
    x_col: str,
    y_col: str,
    output_dir: Path,
    linkage_method: str,
    show_corr: bool,
) -> tuple[float, dict[str, object], Path]:
    distance_matrix = compute_searcher_findee_distance_matrix_from_df(
        merged_df,
        x_col=x_col,
        y_col=y_col,
        z_col=None,
        celltype_col="cluster",
    )
    if getattr(distance_matrix, "shape", (0, 0))[0] < 2 or getattr(distance_matrix, "shape", (0, 0))[1] < 2:
        raise ValueError("Need at least two cluster groups to build the cophenetic heatmap.")

    row_coph, _col_coph = compute_cophenetic_from_distance_matrix(
        distance_matrix,
        method=linkage_method,
        show_corr=show_corr,
    )

    labels = [_normalize_cluster_label(label) for label in row_coph.index]
    row_coph = row_coph.copy()
    row_coph.index = labels
    row_coph.columns = labels

    selected_clusters: list[str] = []
    for item in pattern1_clusters:
        normalized = _normalize_cluster_label(item)
        if normalized and normalized in row_coph.index and normalized not in selected_clusters:
            selected_clusters.append(normalized)

    if not selected_clusters:
        raise ValueError("None of the selected clusters are present in the cophenetic matrix.")

    other_clusters = [label for label in row_coph.index if label not in set(selected_clusters)]
    if not other_clusters:
        raise ValueError("Need both selected and non-selected clusters to compute the cophenetic score.")

    blue_band_matrix = row_coph.loc[selected_clusters, other_clusters]
    band = blue_band_matrix.to_numpy().ravel()
    band = band[~np.isnan(band)]
    if band.size == 0:
        raise ValueError("The cophenetic comparison block has no finite values.")

    stats: dict[str, object] = {
        "n_pairs": int(band.size),
        "min": float(np.min(band)),
        "p05": float(np.quantile(band, 0.05)),
        "median": float(np.median(band)),
        "mean": float(np.mean(band)),
        "p95": float(np.quantile(band, 0.95)),
        "max": float(np.max(band)),
    }

    heatmap_image = plot_cophenetic_heatmap(
        row_coph,
        matrix_name="row_coph",
        sample="selected clusters",
        return_image=True,
        dpi=300,
    )
    heatmap_path = output_dir / "cophenetic_heatmap_row_coph.png"
    heatmap_image.save(heatmap_path)

    return float(stats["mean"]), stats, heatmap_path


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
        "label_scheme": describe_label_scheme(result.label_scheme),
        "label_scheme_internal": to_internal_label_scheme(result.label_scheme),
        "used_tissue_boundary": used_tissue_boundary,
    }
    if result.segmentation_confidence_score is not None:
        payload["segmentation_confidence_score"] = result.segmentation_confidence_score
    if result.segmentation_confidence_stats is not None:
        payload["segmentation_confidence_stats"] = result.segmentation_confidence_stats
    return payload


def emit_status(
    *,
    phase: str,
    run_dir: Path,
    lines: list[str],
    summary: dict[str, object],
    preview_path: str | None = None,
    cophenetic_heatmap_path: str | None = None,
    output_files: list[str] | None = None,
) -> tuple[str, str | None, str | None, dict[str, object], list[str]]:
    status_lines = [f"Phase: {phase}", f"Run directory: {run_dir}"]
    status_lines.extend(lines)
    return "\n".join(status_lines), preview_path, cophenetic_heatmap_path, summary, output_files or []


def run_analysis(
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
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if HISTOSEG_IMPORT_ERROR is not None:
        raise gr.Error(
            "HistoSeg could not be imported inside the app container. "
            f"Import error: {HISTOSEG_IMPORT_ERROR}"
        )
    removed_runs = cleanup_old_runs(max_keep=2)
    start_time = time.perf_counter()
    run_dir = build_run_dir()
    upload_dir = run_dir / "inputs"
    output_dir = run_dir / "outputs"
    upload_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"work_dir": str(run_dir)}

    try:
        progress(0.03, desc="Staging uploaded files")
        log_event("Staging uploaded files")
        yield emit_status(
            phase="staging-inputs",
            run_dir=run_dir,
            lines=["Copying uploaded files into the app workspace."] + (
                [f"Cleaned old run directories: {', '.join(removed_runs)}"] if removed_runs else []
            ),
            summary=summary,
        )

        cells_path, clusters_path, tissue_path = resolve_inputs(
            cells_upload=cells_parquet,
            clusters_upload=clusters_csv,
            tissue_upload=tissue_boundary_csv,
            target_dir=upload_dir,
        )

        parsed_clusters = parse_pattern1_clusters(pattern1_clusters)
        internal_label_scheme = to_internal_label_scheme(label_scheme)
        estimated_cells_rows = safe_count_parquet_rows(cells_path)
        estimated_cluster_rows = safe_count_csv_rows(clusters_path)
        estimated_rows = max(x for x in [estimated_cells_rows, estimated_cluster_rows] if x is not None) if any(
            x is not None for x in [estimated_cells_rows, estimated_cluster_rows]
        ) else None

        effective_use_synth_bg = bool(use_synth_bg and tissue_path is not None)
        if use_synth_bg and tissue_path is None:
            log_event("No tissue_boundary.csv uploaded; synthetic background disabled automatically")

        profile = choose_runtime_profile(
            requested_grid_n=int(grid_n),
            requested_syn_bg_density=float(syn_bg_density),
            use_synth_bg=effective_use_synth_bg,
            estimated_rows=estimated_rows,
        )
        summary.update(
            {
                "estimated_cells_rows": estimated_cells_rows,
                "estimated_cluster_rows": estimated_cluster_rows,
                "dataset_scale": profile.scale_label,
                "requested_grid_n": int(grid_n),
                "effective_grid_n": profile.grid_n,
                "requested_syn_bg_density": float(syn_bg_density),
                "effective_syn_bg_density": profile.syn_bg_density,
                "effective_bg_max_points": profile.bg_max_points,
                "used_tissue_boundary": tissue_path is not None,
                "selected_clusters": [str(item) for item in parsed_clusters],
                "label_scheme": describe_label_scheme(internal_label_scheme),
                "label_scheme_internal": internal_label_scheme,
            }
        )

        preflight_lines = [
            f"Estimated cells.parquet rows: {estimated_cells_rows if estimated_cells_rows is not None else 'unknown'}",
            f"Estimated clusters.csv rows: {estimated_cluster_rows if estimated_cluster_rows is not None else 'unknown'}",
            f"Dataset scale profile: {profile.scale_label}",
            f"Effective grid_n: {profile.grid_n}",
            f"Clusters to outline: {', '.join(str(item) for item in parsed_clusters)}",
            f"Scoring direction: {describe_label_scheme(internal_label_scheme)}",
        ]
        if not effective_use_synth_bg:
            preflight_lines.append("Synthetic background is disabled for this run.")
        preflight_lines.extend(profile.notes)

        progress(0.12, desc="Inputs ready")
        log_event(
            f"Inputs ready | cells_rows={estimated_cells_rows} | cluster_rows={estimated_cluster_rows} | "
            f"grid_n={profile.grid_n} | synth_bg={effective_use_synth_bg}"
        )
        yield emit_status(
            phase="preflight",
            run_dir=run_dir,
            lines=preflight_lines,
            summary=summary,
        )

        cfg = Pattern1IsolineConfig(
            clusters_csv=clusters_path,
            cells_parquet=cells_path,
            tissue_boundary_csv=tissue_path,
            out_dir=output_dir,
            pattern1_clusters=parsed_clusters,
            grid_n=profile.grid_n,
            knn_k=int(knn_k),
            smooth_sigma=float(smooth_sigma),
            min_cells_inside=int(min_cells_inside),
            label_scheme=internal_label_scheme,
            use_synth_bg=effective_use_synth_bg,
            compute_confidence_score=bool(compute_confidence_score),
            bbox_expand_um=float(bbox_expand_um),
            syn_bg_density=profile.syn_bg_density,
            bg_max_points=profile.bg_max_points,
            syn_bg_min=profile.syn_bg_min,
            syn_bg_max=profile.syn_bg_max,
        )

        progress(0.2, desc="Aligning clusters and cell coordinates")
        log_event("Aligning clusters with cells.parquet")
        yield emit_status(
            phase="aligning-cells",
            run_dir=run_dir,
            lines=["Matching GraphClust barcodes with cell coordinates."],
            summary=summary,
        )

        merged, id_col_used, x_col, y_col = align_clusters_with_cells(
            cfg.clusters_csv,
            cfg.cells_parquet,
            barcode_col=cfg.barcode_col,
            cluster_col=cfg.cluster_col,
        )
        merged = merged.copy()
        merged["cluster"] = merged["cluster"].map(_normalize_cluster_label)
        merged = merged.loc[merged["cluster"] != ""].copy()

        p1 = set(_normalize_cluster_label(x) for x in cfg.pattern1_clusters)
        p1 = {x for x in p1 if x != ""}
        if len(p1) == 0:
            raise ValueError("The selected clusters could not be matched after normalization.")

        merged["_is_p1"] = merged["cluster"].isin(p1)
        p1_df = merged.loc[merged["_is_p1"], [id_col_used, x_col, y_col]].copy()
        if len(p1_df) < 10:
            raise RuntimeError(f"Too few cells were found in the selected clusters after merging: {len(p1_df)}")

        target_ids = set(p1_df[id_col_used].astype(str))
        target_xy = p1_df[[x_col, y_col]].to_numpy(float)
        summary.update(
            {
                "id_col_used": id_col_used,
                "x_col": x_col,
                "y_col": y_col,
                "merged_rows": int(len(merged)),
                "n_target_cells": int(len(target_xy)),
            }
        )

        progress(0.34, desc="Sampling background points")
        log_event(f"Sampling background points | merged_rows={len(merged)} | target_cells={len(target_xy)}")
        yield emit_status(
            phase="sampling-background",
            run_dir=run_dir,
            lines=[
                f"Merged rows: {len(merged)}",
                f"Selected-cluster cells: {len(target_xy)}",
                "Sampling real-cell background and optional synthetic background.",
            ],
            summary=summary,
        )

        syn_bg_xy: np.ndarray | None = None
        if cfg.use_synth_bg:
            if cfg.tissue_boundary_csv is None:
                raise ValueError("use_synth_bg=True but tissue_boundary_csv is missing.")
            boundary_xy = load_tissue_boundary_csv(cfg.tissue_boundary_csv)
            syn_bg_xy = generate_synthetic_bg_in_bbox(
                boundary_xy,
                expand_um=cfg.bbox_expand_um,
                density=cfg.syn_bg_density,
                min_n=cfg.syn_bg_min,
                max_n=cfg.syn_bg_max,
                seed=cfg.random_state,
            )
            summary["n_synthetic_bg_points"] = int(len(syn_bg_xy))

        bg0_xy = sample_background_from_other_cells_plus_synth(
            cells_df=merged.rename(columns={id_col_used: "tmp_id"}),
            synthetic_bg_xy=syn_bg_xy,
            target_ids=set(str(x) for x in target_ids),
            target_xy=target_xy,
            cell_id_col="tmp_id",
            x_col=x_col,
            y_col=y_col,
            d_min=cfg.bg_d_min,
            d_max=cfg.bg_d_max,
            max_points=cfg.bg_max_points,
            seed=cfg.random_state,
            margin_um=cfg.margin_um,
        )
        if len(bg0_xy) == 0:
            raise RuntimeError("No bg0 points sampled. Try relaxing bg_d_min/bg_d_max, or disabling synth bg.")
        summary["n_bg0_points"] = int(len(bg0_xy))

        progress(0.48, desc="Training spatial KNN model")
        log_event(f"Training KNN | bg0={len(bg0_xy)} | target={len(target_xy)} | k={cfg.knn_k}")
        yield emit_status(
            phase="training-knn",
            run_dir=run_dir,
            lines=[
                f"Background points kept: {len(bg0_xy)}",
                f"Training KNN with k={cfg.knn_k}.",
            ],
            summary=summary,
        )

        X_train = np.vstack([bg0_xy, target_xy])
        if _validate_label_scheme(cfg.label_scheme) == "p1_is_one":
            y_train = np.hstack([np.zeros(len(bg0_xy)), np.ones(len(target_xy))])
        else:
            y_train = np.hstack([np.ones(len(bg0_xy)), np.zeros(len(target_xy))])

        reg = KNeighborsRegressor(n_neighbors=cfg.knn_k, weights="distance")
        reg.fit(X_train, y_train)

        progress(0.62, desc="Predicting on spatial mesh")
        log_event(f"Predicting on mesh | grid_n={cfg.grid_n} | grid_points={cfg.grid_n * cfg.grid_n}")
        yield emit_status(
            phase="predicting-mesh",
            run_dir=run_dir,
            lines=[
                f"Predicting on a {cfg.grid_n} x {cfg.grid_n} mesh.",
                "This is usually the slowest step for larger Xenium inputs.",
            ],
            summary=summary,
        )

        xx, yy, grid = make_mesh_from_xy(
            target_xy,
            grid_n=cfg.grid_n,
            pad_fraction=cfg.pad_fraction,
            margin_um=cfg.margin_um,
        )
        prob = reg.predict(grid).reshape(xx.shape)
        prob_smooth = gaussian_filter(prob, sigma=cfg.smooth_sigma)

        progress(0.76, desc="Extracting contours")
        log_event("Applying tissue mask and extracting isolines")
        yield emit_status(
            phase="extracting-contours",
            run_dir=run_dir,
            lines=["Applying tissue mask, smoothing, and contour extraction."],
            summary=summary,
        )

        all_xy = merged[[x_col, y_col]].to_numpy(float)
        tissue_mask = tissue_mask_from_xy(all_xy, xx, yy, max_dist_threshold=cfg.max_dist_threshold)
        prob_smooth_masked = prob_smooth.copy()
        prob_smooth_masked[~tissue_mask] = np.nan

        verts_list = extract_contour_paths(xx, yy, prob_smooth_masked, level=cfg.isoline_level)
        verts_list = filter_loops_by_cell_count(verts_list, target_xy, min_cells_inside=cfg.min_cells_inside)
        if len(verts_list) == 0:
            raise RuntimeError(
                "No isoline found.\n"
                "Try reducing min_cells_inside, increasing smooth_sigma, increasing knn_k, or lowering grid_n."
            )

        conf_score: float | None = None
        conf_stats: dict[str, object] | None = None
        cophenetic_heatmap_path: Path | None = None
        cophenetic_note: str | None = None
        if cfg.compute_confidence_score:
            progress(0.86, desc="Computing cophenetic heatmap")
            log_event("Computing cophenetic heatmap and confidence score")
            yield emit_status(
                phase="cophenetic-analysis",
                run_dir=run_dir,
                lines=["Computing the cophenetic heatmap and summary confidence score."],
                summary=summary,
            )
            try:
                conf_score, conf_stats, cophenetic_heatmap_path = compute_cophenetic_outputs(
                    merged_df=merged,
                    pattern1_clusters=list(cfg.pattern1_clusters),
                    x_col=x_col,
                    y_col=y_col,
                    output_dir=output_dir,
                    linkage_method=cfg.confidence_linkage_method,
                    show_corr=cfg.confidence_show_corr,
                )
            except Exception as exc:
                cophenetic_note = f"Skipped cophenetic heatmap: {exc}"
                log_event(cophenetic_note)
                summary["cophenetic_note"] = cophenetic_note

        progress(0.93, desc="Saving outputs")
        log_event(f"Saving outputs | contours={len(verts_list)}")
        yield emit_status(
            phase="saving-outputs",
            run_dir=run_dir,
            lines=[
                f"Contours extracted: {len(verts_list)}",
                "Writing the contour preview, cophenetic heatmap, and downloadable outputs.",
            ],
            summary=summary,
            cophenetic_heatmap_path=str(cophenetic_heatmap_path) if cophenetic_heatmap_path is not None else None,
        )

        params_path: Path | None = None
        params = asdict(cfg)
        params.update(
            dict(
                id_col_used=id_col_used,
                x_col=x_col,
                y_col=y_col,
                n_target_cells=int(len(target_xy)),
                n_bg0=int(len(bg0_xy)),
                n_contours=int(len(verts_list)),
                label_scheme=describe_label_scheme(cfg.label_scheme),
                label_scheme_internal=_validate_label_scheme(cfg.label_scheme),
                segmentation_confidence_score=conf_score,
                segmentation_confidence_stats=conf_stats,
                cophenetic_heatmap_path=str(cophenetic_heatmap_path) if cophenetic_heatmap_path is not None else None,
            )
        )
        if cfg.save_params_json:
            params_path = output_dir / "params.json"
            with params_path.open("w", encoding="utf-8") as handle:
                json.dump(params, handle, indent=2, ensure_ascii=False, default=str)

        if cfg.save_contours_npy:
            for i, vertices in enumerate(verts_list):
                np.save(output_dir / f"pattern1_isoline_{cfg.isoline_level:g}_{i}.npy", vertices)

        preview_path: Path | None = None
        if cfg.save_preview_png:
            plt.figure(figsize=(10, 10))
            plt.scatter(bg0_xy[:, 0], bg0_xy[:, 1], s=1, alpha=0.05, label="background points")
            plt.scatter(target_xy[:, 0], target_xy[:, 1], s=3, alpha=0.85, label="selected-cluster cells")
            for vertices in verts_list:
                plt.plot(vertices[:, 0], vertices[:, 1], linewidth=2)
            plt.gca().set_aspect("equal")
            title = (
                f"Target-cluster contours | isoline={cfg.isoline_level:g} | "
                f"contours={len(verts_list)} | scoring={describe_label_scheme(cfg.label_scheme)}"
            )
            if conf_score is not None:
                title += f" | cophenetic confidence={conf_score:.4f}"
            plt.title(title)
            plt.legend(frameon=False)
            plt.tight_layout()
            preview_path = output_dir / f"pattern1_isoline_{cfg.isoline_level:g}.png"
            plt.savefig(preview_path, dpi=200)
            plt.close()

        archive_path, archive_note = zip_outputs(output_dir)
        result = Pattern1IsolineResult(
            out_dir=output_dir,
            id_col_used=id_col_used,
            x_col=x_col,
            y_col=y_col,
            n_target_cells=int(len(target_xy)),
            n_bg0_points=int(len(bg0_xy)),
            contours=list(verts_list),
            label_scheme=_validate_label_scheme(cfg.label_scheme),
            segmentation_confidence_score=conf_score,
            segmentation_confidence_stats=conf_stats,
            params_json=params_path,
            preview_png=preview_path,
        )

        output_files: list[str] = []
        if archive_path is not None:
            output_files.append(str(archive_path))
        if preview_path is not None:
            output_files.append(str(preview_path))
        if cophenetic_heatmap_path is not None:
            output_files.append(str(cophenetic_heatmap_path))
        if params_path is not None:
            output_files.append(str(params_path))
        output_files.extend(str(path) for path in sorted(output_dir.glob("pattern1_isoline_*.npy")))

        summary.update(format_summary(result, used_tissue_boundary=tissue_path is not None, work_dir=run_dir))
        summary["effective_runtime_seconds"] = round(time.perf_counter() - start_time, 2)
        summary["profile_notes"] = list(profile.notes)
        summary["cophenetic_heatmap_generated"] = cophenetic_heatmap_path is not None
        summary["cophenetic_heatmap_path"] = (
            str(cophenetic_heatmap_path) if cophenetic_heatmap_path is not None else None
        )

        log_event(
            f"Run finished successfully | contours={len(result.contours)} | "
            f"elapsed_s={summary['effective_runtime_seconds']}"
        )
        progress(1.0, desc="Finished")
        yield emit_status(
            phase="finished",
            run_dir=run_dir,
            lines=[
                f"{APP_NAME} finished successfully.",
                f"Contours generated: {len(result.contours)}",
                f"Elapsed time: {summary['effective_runtime_seconds']} seconds",
            ]
            + ([f"Cophenetic confidence score: {conf_score:.4f}"] if conf_score is not None else [])
            + ([cophenetic_note] if cophenetic_note else [])
            + list(profile.notes)
            + ([archive_note] if archive_note else []),
            summary=summary,
            preview_path=str(preview_path) if preview_path is not None else None,
            cophenetic_heatmap_path=str(cophenetic_heatmap_path) if cophenetic_heatmap_path is not None else None,
            output_files=output_files,
        )
    except Exception as exc:
        log_event(f"Run failed: {exc}")
        print(traceback.format_exc(), flush=True)
        raise gr.Error(str(exc))


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
            A SciLifeLab Serve-ready wrapper around HistoSeg for target-cluster contour analysis.
            Upload the required analysis files, tune the parameters, and download the contour preview plus the cophenetic heatmap.
          </p>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="callout">
        Required analysis inputs: <code>cells.parquet</code> and <code>clusters.csv</code>.
        Recommended: <code>tissue_boundary.csv</code> so synthetic background can be used.
        Pick the cluster IDs you want to outline as the target region, then run the app to generate contours and the cophenetic heatmap.
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            cells_parquet = gr.File(
                label="Cell coordinates (cells.parquet)",
                file_types=[".parquet"],
                type="filepath",
            )
            clusters_csv = gr.File(
                label="Cluster assignments (clusters.csv)",
                file_types=[".csv"],
                type="filepath",
            )
            tissue_boundary_csv = gr.File(
                label="Tissue boundary (optional: tissue_boundary.csv)",
                file_types=[".csv"],
                type="filepath",
            )
            pattern1_clusters = gr.Textbox(
                label="Clusters to outline",
                value=DEFAULT_PATTERN1,
                info="Enter the cluster IDs or labels that should be treated as the target region. Separate them with commas.",
            )
            label_scheme = gr.Dropdown(
                label="Scoring direction",
                choices=list(LABEL_SCHEME_OPTIONS.keys()),
                value=DEFAULT_LABEL_SCHEME,
                info="Recommended: selected clusters score high and background scores low.",
            )

            with gr.Accordion("Advanced parameters", open=False):
                grid_n = gr.Slider(label="Mesh resolution", minimum=200, maximum=1600, step=50, value=650)
                knn_k = gr.Slider(label="KNN neighbors", minimum=5, maximum=100, step=1, value=30)
                smooth_sigma = gr.Slider(label="Smoothing strength", minimum=0.5, maximum=12.0, step=0.5, value=5.0)
                min_cells_inside = gr.Slider(label="Minimum cells inside each contour", minimum=1, maximum=200, step=1, value=10)
                bbox_expand_um = gr.Slider(label="Boundary expansion (um)", minimum=0, maximum=500, step=10, value=100)
                syn_bg_density = gr.Slider(label="Synthetic background density", minimum=0.001, maximum=0.05, step=0.001, value=0.003)
                use_synth_bg = gr.Checkbox(label="Use synthetic background", value=True)
                compute_confidence_score = gr.Checkbox(
                    label="Generate cophenetic heatmap and confidence score",
                    value=True,
                )

            run_button = gr.Button("Run HistoSeg analysis", variant="primary")

        with gr.Column(scale=1):
            status_text = gr.Textbox(label="Status", lines=8)
            preview_image = gr.Image(label="Contour preview", type="filepath")
            cophenetic_heatmap_image = gr.Image(label="Cophenetic heatmap", type="filepath")
            summary_json = gr.JSON(label="Run summary")
            output_files = gr.File(label="Download outputs", file_count="multiple")

    run_button.click(
        fn=run_analysis,
        inputs=[
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
        outputs=[status_text, preview_image, cophenetic_heatmap_image, summary_json, output_files],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        show_api=False,
    )
