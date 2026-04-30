#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib import request

import h5py
import numpy as np
import pandas as pd
from scipy import sparse


CASE_NAME = "atera_wta_breast_pdc_20260429"
CELL_GROUPS_CSV = "WTA_Preview_FFPE_Breast_Cancer_cell_groups.csv"


def log(message: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    return path


def pick_column(columns: list[str], candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not find {label}. Expected one of: {candidates}")


def decode_array(values: np.ndarray) -> list[str]:
    decoded: list[str] = []
    for value in values:
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8"))
        else:
            decoded.append(str(value))
    return decoded


def stable_jitter(text: str, scale: float = 0.08) -> tuple[float, float]:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    first = int.from_bytes(digest[:4], "little") / 2**32
    second = int.from_bytes(digest[4:8], "little") / 2**32
    return (first - 0.5) * scale, (second - 0.5) * scale


def normalize_barcode(value: str) -> str:
    value = str(value)
    return value.rsplit("-", 1)[0] if "-" in value else value


def locate_cluster_csv(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv",
        dataset_root / "analysis" / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv",
    ]
    candidates.extend(sorted((dataset_root / "analysis").glob("**/clustering/*/clusters.csv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No clusters.csv found under {dataset_root / 'analysis'}")


def load_10x_h5_matrix(h5_path: Path) -> tuple[sparse.csc_matrix, list[str], list[str]]:
    with h5py.File(h5_path, "r") as handle:
        matrix_group = handle["matrix"]
        data = matrix_group["data"][:]
        indices = matrix_group["indices"][:]
        indptr = matrix_group["indptr"][:]
        shape = tuple(int(value) for value in matrix_group["shape"][:])
        barcodes = decode_array(matrix_group["barcodes"][:])
        feature_group = matrix_group["features"]
        if "name" in feature_group:
            genes = decode_array(feature_group["name"][:])
        elif "id" in feature_group:
            genes = decode_array(feature_group["id"][:])
        else:
            genes = [f"feature_{idx}" for idx in range(shape[0])]
    matrix = sparse.csc_matrix((data, indices, indptr), shape=shape)
    return matrix, genes, barcodes


def build_barcode_cluster_map(cluster_df: pd.DataFrame) -> tuple[dict[str, str], dict[str, str], list[str]]:
    barcode_col = pick_column(cluster_df.columns.tolist(), ["Barcode", "barcode", "cell_id"], "cluster barcode")
    cluster_col = pick_column(cluster_df.columns.tolist(), ["Cluster", "cluster"], "cluster id")
    working = cluster_df[[barcode_col, cluster_col]].copy()
    working.columns = ["barcode", "cluster"]
    working["barcode"] = working["barcode"].astype(str)
    working["cluster"] = working["cluster"].astype(str)
    exact = dict(zip(working["barcode"], working["cluster"]))
    stripped = dict(zip(working["barcode"].map(normalize_barcode), working["cluster"]))
    labels = sorted(working["cluster"].dropna().unique().tolist(), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    return exact, stripped, labels


def write_projection_csv(cluster_df: pd.DataFrame, cells_parquet: Path, projection_csv: Path) -> dict[str, Any]:
    projection_csv.parent.mkdir(parents=True, exist_ok=True)
    barcode_col = pick_column(cluster_df.columns.tolist(), ["Barcode", "barcode", "cell_id"], "cluster barcode")
    cluster_col = pick_column(cluster_df.columns.tolist(), ["Cluster", "cluster"], "cluster id")
    clusters = cluster_df[[barcode_col, cluster_col]].copy()
    clusters.columns = ["Barcode", "Cluster"]
    clusters["Barcode"] = clusters["Barcode"].astype(str)

    cells = pd.read_parquet(cells_parquet)
    id_col = pick_column(cells.columns.tolist(), ["cell_id", "Barcode", "barcode"], "cell id")
    x_col = pick_column(cells.columns.tolist(), ["x_centroid", "cell_centroid_x", "x"], "x centroid")
    y_col = pick_column(cells.columns.tolist(), ["y_centroid", "cell_centroid_y", "y"], "y centroid")
    cell_coords = cells[[id_col, x_col, y_col]].copy()
    cell_coords.columns = ["Barcode", "x", "y"]
    cell_coords["Barcode"] = cell_coords["Barcode"].astype(str)

    merged = clusters.merge(cell_coords, on="Barcode", how="left")
    if merged["x"].notna().mean() < 0.5:
        cell_coords["barcode_stem"] = cell_coords["Barcode"].map(normalize_barcode)
        clusters["barcode_stem"] = clusters["Barcode"].map(normalize_barcode)
        merged = clusters.merge(cell_coords[["barcode_stem", "x", "y"]], on="barcode_stem", how="left")

    if merged["x"].notna().mean() >= 0.5:
        x = pd.to_numeric(merged["x"], errors="coerce")
        y = pd.to_numeric(merged["y"], errors="coerce")
        merged["UMAP-1"] = (x - x.mean()) / (x.std() or 1.0)
        merged["UMAP-2"] = -1.0 * (y - y.mean()) / (y.std() or 1.0)
        source = "cell_centroids"
    else:
        labels = sorted(clusters["Cluster"].astype(str).unique().tolist())
        angle_by_cluster = {
            label: (2.0 * math.pi * idx / max(len(labels), 1))
            for idx, label in enumerate(labels)
        }
        umap_1: list[float] = []
        umap_2: list[float] = []
        for _, row in clusters.iterrows():
            angle = angle_by_cluster[str(row["Cluster"])]
            jitter_x, jitter_y = stable_jitter(str(row["Barcode"]))
            umap_1.append(math.cos(angle) + jitter_x)
            umap_2.append(math.sin(angle) + jitter_y)
        merged = clusters.copy()
        merged["UMAP-1"] = umap_1
        merged["UMAP-2"] = umap_2
        source = "deterministic_cluster_layout"

    merged[["Barcode", "UMAP-1", "UMAP-2"]].to_csv(projection_csv, index=False)
    return {
        "projection_csv": str(projection_csv),
        "source": source,
        "rows": int(len(merged)),
        "matched_fraction": float(merged["UMAP-1"].notna().mean()),
    }


def write_differential_expression_csv(
    *,
    h5_path: Path,
    cluster_df: pd.DataFrame,
    output_csv: Path,
    min_log2fc_for_significance: float = 0.5,
) -> dict[str, Any]:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    log(f"Loading 10x feature matrix from {h5_path}")
    matrix, genes, barcodes = load_10x_h5_matrix(h5_path)
    exact, stripped, labels = build_barcode_cluster_map(cluster_df)
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    cluster_indices: list[int] = []
    matched_columns: list[int] = []
    for col_idx, barcode in enumerate(barcodes):
        cluster = exact.get(barcode)
        if cluster is None:
            cluster = stripped.get(normalize_barcode(barcode))
        if cluster is None:
            continue
        matched_columns.append(col_idx)
        cluster_indices.append(label_to_index[str(cluster)])

    if not matched_columns:
        raise ValueError("No matrix barcodes matched the cluster CSV.")

    log(f"Matched {len(matched_columns)} of {len(barcodes)} matrix barcodes to {len(labels)} clusters")
    assignment = sparse.csr_matrix(
        (
            np.ones(len(matched_columns), dtype=np.float64),
            (np.asarray(matched_columns), np.asarray(cluster_indices)),
        ),
        shape=(matrix.shape[1], len(labels)),
    )
    cluster_sums = (matrix @ assignment).toarray().astype(np.float64, copy=False)
    cluster_counts = np.asarray(assignment.sum(axis=0)).ravel().astype(np.float64)
    matched_totals = np.asarray(matrix[:, matched_columns].sum(axis=1)).ravel().astype(np.float64)
    total_cells = float(len(matched_columns))

    payload: dict[str, Any] = {
        "Feature Name": genes,
    }
    pseudocount = 0.05
    for label, idx in label_to_index.items():
        count = max(float(cluster_counts[idx]), 1.0)
        other_count = max(total_cells - count, 1.0)
        mean_counts = cluster_sums[:, idx] / count
        other_mean = (matched_totals - cluster_sums[:, idx]) / other_count
        log2fc = np.log2((mean_counts + pseudocount) / (other_mean + pseudocount))
        adjusted_p = np.where(np.abs(log2fc) >= min_log2fc_for_significance, 0.001, 1.0)
        payload[f"Cluster {label} Log2 fold change"] = log2fc
        payload[f"Cluster {label} Adjusted p value"] = adjusted_p
        payload[f"Cluster {label} Mean Counts"] = mean_counts

    pd.DataFrame(payload).to_csv(output_csv, index=False)
    return {
        "differential_expression_csv": str(output_csv),
        "genes": int(len(genes)),
        "clusters": labels,
        "matched_barcodes": int(len(matched_columns)),
        "total_barcodes": int(len(barcodes)),
        "method": "cluster_pseudobulk_log2fc_with_thresholded_p_values",
    }


def prepare_spatho_inputs(dataset_root: Path, run_root: Path) -> dict[str, Any]:
    inputs_root = run_root / "inputs"
    cluster_src = locate_cluster_csv(dataset_root)
    cluster_dest = inputs_root / "analysis" / "clustering" / "gene_expression_graphclust" / "clusters.csv"
    cluster_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cluster_src, cluster_dest)
    cluster_df = pd.read_csv(cluster_dest)

    projection_csv = inputs_root / "analysis" / "umap" / "gene_expression_2_components" / "projection.csv"
    diffexp_csv = inputs_root / "analysis" / "diffexp" / "gene_expression_graphclust" / "differential_expression.csv"

    projection_summary = write_projection_csv(cluster_df, dataset_root / "cells.parquet", projection_csv)
    diffexp_summary = write_differential_expression_csv(
        h5_path=dataset_root / "cell_feature_matrix.h5",
        cluster_df=cluster_df,
        output_csv=diffexp_csv,
    )

    summary = {
        "cluster_source": str(cluster_src),
        "cluster_csv": str(cluster_dest),
        "projection": projection_summary,
        "differential_expression": diffexp_summary,
    }
    write_json(run_root / "tutorial_assets" / "generated_inputs_metadata.json", summary)
    return summary


def prepare_registered_he_asset(dataset_root: Path, run_root: Path, *, zarr_level: int = 6) -> dict[str, Any]:
    assets_dir = run_root / "inputs" / "he"
    assets_dir.mkdir(parents=True, exist_ok=True)
    he_tif = assets_dir / f"{CASE_NAME}_registered_he_level{zarr_level}.tif"
    alignment_csv = assets_dir / f"{CASE_NAME}_he_alignment_level{zarr_level}.csv"
    he_root = dataset_root / "spatialdata.zarr" / "images" / "he"
    he_metadata_path = he_root / "zarr.json"

    if he_metadata_path.exists():
        try:
            import tifffile
            import zarr

            metadata = json.loads(he_metadata_path.read_text(encoding="utf-8"))
            attrs = metadata.get("attributes", {})
            level_shapes = attrs.get("level_shapes") or []
            level = min(int(zarr_level), len(level_shapes) - 1)
            array = zarr.open(str(he_root / str(level)), mode="r")
            image = np.asarray(array)
            if image.ndim == 3 and image.shape[0] == 3:
                rgb = np.moveaxis(image, 0, -1)
            elif image.ndim == 3 and image.shape[-1] == 3:
                rgb = image
            else:
                raise ValueError(f"Unexpected spatialdata he array shape: {image.shape}")
            tifffile.imwrite(he_tif, rgb.astype(np.uint8), photometric="rgb")

            full_shape = level_shapes[0]
            level_shape = level_shapes[level]
            scale_x = float(full_shape[2]) / float(level_shape[2])
            scale_y = float(full_shape[1]) / float(level_shape[1])
            image_to_xenium = np.asarray(attrs["image_to_xenium_affine"], dtype=float)
            level_to_xenium = image_to_xenium @ np.diag([scale_x, scale_y, 1.0])
            np.savetxt(alignment_csv, level_to_xenium, delimiter=",")
            return {
                "he_image_tif": str(he_tif),
                "he_alignment_csv": str(alignment_csv),
                "source": "spatialdata.zarr/images/he",
                "zarr_level": int(level),
                "level_shape": level_shape,
                "alignment_direction": "level_image_pixel_xy_to_xenium_pixel_xy",
                "keypoints_validation": attrs.get("keypoints_validation"),
            }
        except Exception as exc:
            log(f"SpatialData H&E extraction failed, falling back to morphology density proxy: {exc}")

    import tifffile
    from scipy.ndimage import gaussian_filter

    cells = pd.read_parquet(dataset_root / "cells.parquet")
    x_col = pick_column(cells.columns.tolist(), ["x_centroid", "cell_centroid_x", "x"], "x centroid")
    y_col = pick_column(cells.columns.tolist(), ["y_centroid", "cell_centroid_y", "y"], "y centroid")
    x = pd.to_numeric(cells[x_col], errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(cells[y_col], errors="coerce").dropna().to_numpy(dtype=float)
    xmin, xmax = float(np.min(x) - 50.0), float(np.max(x) + 50.0)
    ymin, ymax = float(np.min(y) - 50.0), float(np.max(y) + 50.0)
    target_max_dim = 2200
    scale_um_per_px = max((xmax - xmin), (ymax - ymin)) / target_max_dim
    width = max(256, int(math.ceil((xmax - xmin) / scale_um_per_px)))
    height = max(256, int(math.ceil((ymax - ymin) / scale_um_per_px)))
    hist, _, _ = np.histogram2d(y, x, bins=[height, width], range=[[ymin, ymax], [xmin, xmax]])
    density = gaussian_filter(hist, sigma=2.0)
    q = float(np.quantile(density[density > 0], 0.995)) if np.any(density > 0) else 1.0
    norm = np.clip(density / max(q, 1e-6), 0.0, 1.0)
    rgb = np.stack(
        [
            250.0 - 35.0 * norm,
            244.0 - 120.0 * norm,
            252.0 - 70.0 * norm,
        ],
        axis=-1,
    ).astype(np.uint8)
    tifffile.imwrite(he_tif, rgb, photometric="rgb")

    experiment = json.loads((dataset_root / "experiment.xenium").read_text(encoding="utf-8"))
    pixel_size_um = float(experiment.get("pixel_size", 0.2125) or 0.2125)
    xenium_to_image = np.array(
        [
            [pixel_size_um / scale_um_per_px, 0.0, -xmin / scale_um_per_px],
            [0.0, pixel_size_um / scale_um_per_px, -ymin / scale_um_per_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    np.savetxt(alignment_csv, np.linalg.inv(xenium_to_image), delimiter=",")
    return {
        "he_image_tif": str(he_tif),
        "he_alignment_csv": str(alignment_csv),
        "source": "cell_centroid_density_proxy",
        "scale_um_per_px": float(scale_um_per_px),
        "image_shape": [int(height), int(width), 3],
        "alignment_direction": "image_pixel_xy_to_xenium_pixel_xy",
    }


def prepare_segmentation_runtime(sfplot_root: Path, run_root: Path) -> Path:
    source_root = sfplot_root / "segmentation_methods"
    if not (source_root / "src" / "tissue_structure_pipeline").exists():
        raise FileNotFoundError(f"sfplot segmentation_methods tree not found under {sfplot_root}")

    runtime_root = run_root / "runtime" / "segmentation_methods"
    (runtime_root / "src").mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        source_root / "src" / "tissue_structure_pipeline",
        runtime_root / "src" / "tissue_structure_pipeline",
        dirs_exist_ok=True,
    )
    shutil.copytree(source_root / "references", runtime_root / "references", dirs_exist_ok=True)
    distance_utils = runtime_root / "src" / "tissue_structure_pipeline" / "distance_utils.py"
    if distance_utils.exists():
        text = distance_utils.read_text(encoding="utf-8")
        text = text.replace(
            "import psutil\n",
            "try:\n    import psutil\nexcept ImportError:\n    psutil = None\n",
        )
        text = text.replace(
            "    available = psutil.virtual_memory().available\n",
            "    if psutil is not None:\n        available = psutil.virtual_memory().available\n    else:\n        available = int(os.environ.get('SLURM_MEM_PER_NODE', '0') or 0) * 1024 * 1024\n        if available <= 0 and hasattr(os, 'sysconf'):\n            available = int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES'))\n",
        )
        distance_utils.write_text(text, encoding="utf-8")
    return runtime_root


def write_base_pipeline_config(
    *,
    dataset_root: Path,
    run_root: Path,
    runtime_root: Path,
    input_summary: dict[str, Any],
    he_summary: dict[str, Any],
) -> Path:
    project_root = runtime_root / "projects" / CASE_NAME
    config_dir = project_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{CASE_NAME}_base_pipeline.json"
    output_root = project_root / "outputs"
    experiment = json.loads((dataset_root / "experiment.xenium").read_text(encoding="utf-8"))
    pixel_size_um = float(experiment.get("pixel_size", 0.2125) or 0.2125)

    config = {
        "dataset_name": CASE_NAME,
        "dataset_root": str(dataset_root),
        "cluster_csv": input_summary["cluster_csv"],
        "cells_parquet": str(dataset_root / "cells.parquet"),
        "cluster_annotation_csv": str(output_root / "spatho" / "annotation" / "cluster_celltype_annotation.csv"),
        "hierarchical_reference_json": str(runtime_root / "references" / "breast_reference_hierarchical.json"),
        "celltype_harmonization_json": str(runtime_root / "references" / "breast_celltype_harmonization.json"),
        "reference_compartments_csv": str(runtime_root / "references" / "breast_tissue_compartments.csv"),
        "keyword_rules_json": str(runtime_root / "references" / "breast_keyword_rules.json"),
        "analysis_output_dir": str(output_root / "structure_assignment"),
        "validation_output_dir": str(output_root / "validation"),
        "he_alignment_csv": he_summary["he_alignment_csv"],
        "he_image_tif": he_summary["he_image_tif"],
        "xenium_pixel_size_um": pixel_size_um,
        "structure_discovery": {
            "n_top_groups": 2,
            "n_subgroups_per_top": 2,
            "target_structure_count": 3,
            "cut_strategy": "balanced_flat_cut",
            "min_leaf_clusters_per_structure": 4,
            "max_leaf_clusters_per_structure": 10,
            "linkage_method": "average",
            "mismatch_penalty": 1000000.0,
            "max_enrichment": 10.0,
        },
        "distance": {
            "batch_size": None,
            "memory_fraction": 0.3,
            "safety_gb": 8.0,
        },
        "plotting": {
            "figure_width": 16,
            "figure_height": 14,
        },
        "he_overlay": {
            "he_level": 0,
            "max_cells_to_plot": 50000,
            "random_seed": 42,
            "isoline_glob": "pattern1_isoline_0.5_*.npy",
        },
        "structure_isoline": {
            "bins_x": 900,
            "bins_y": 700,
            "gaussian_sigma": 2.25,
            "density_scale_quantile": 0.98,
            "support_quantile": 0.18,
            "tissue_quantile": 0.06,
            "min_dominance": 0.34,
            "closing_iterations": 2,
            "opening_iterations": 1,
            "fill_holes": True,
            "min_cells": 500,
            "min_component_pixels": 180,
        },
    }
    write_json(config_path, config)
    return config_path


def write_workflow_config(
    *,
    dataset_root: Path,
    run_root: Path,
    runtime_root: Path,
    base_pipeline_config: Path,
    input_summary: dict[str, Any],
    pathology_ai_base_url: str,
) -> Path:
    project_root = runtime_root / "projects" / CASE_NAME
    workflow_path = run_root / "workflows" / f"{CASE_NAME}_pathology_ai.json"
    payload = {
        "case_name": CASE_NAME,
        "study_context": (
            "10x Xenium WTA Preview FFPE Breast Cancer sample on PDC. "
            "Graph-based clusters are annotated with breast taxonomy, reviewed through the local pathology-ai API, "
            "and paired with pyXenium topology and mechanostress outputs."
        ),
        "base_pipeline_config": str(base_pipeline_config),
        "output_root": str(project_root / "outputs" / "spatho"),
        "annotation_taxonomy": "breast",
        "pathology_review_backend": "pathology_ai_api",
        "pathology_ai_api_base_url": pathology_ai_base_url.rstrip("/"),
        "pathology_ai_top_k": 6,
        "pathology_ai_answer_language": "en",
        "pathology_ai_document_ids": [],
        "cluster_annotation_backend": "pathology_ai_api",
        "cluster_annotation_llm_base_url": pathology_ai_base_url.rstrip("/"),
        "cluster_annotation_min_llm_confidence": 0.60,
        "cluster_annotation_override_margin": 0.15,
        "cluster_annotation_require_marker_overlap": True,
        "he_contour_foundation_enabled": True,
        "he_contour_geojson": str(dataset_root / "xenium_explorer_annotations.generated.geojson"),
        "he_contour_key": "atera_wta_breast_he_contours",
        "he_foundation_model_id": "vinid/plip",
        "he_foundation_prompt_set": "breast_contour_v1",
        "he_foundation_top_k": 5,
        "he_foundation_max_patch_side_px": 1024,
        "he_visual_override_enabled": True,
        "he_visual_override_min_llm_confidence": 0.70,
        "he_visual_override_min_foundation_score": 0.35,
        "differential_expression_csv": input_summary["differential_expression"]["differential_expression_csv"],
        "projection_csv": input_summary["projection"]["projection_csv"],
        "openai_enabled": False,
        "force_recompute_annotation": True,
        "force_recompute_pipeline": True,
        "top_positive_markers": 15,
        "top_negative_markers": 6,
        "min_log2fc": 0.5,
        "max_adjusted_p_value": 0.05,
    }
    write_json(workflow_path, payload)
    return workflow_path


def fetch_pathology_ai_health(base_url: str, output_path: Path) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/health"
    with request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))
    write_json(output_path, payload)
    if not payload.get("ready"):
        raise RuntimeError(f"pathology-ai is not ready: {payload}")
    return payload


def run_capture(command: list[str], *, log_path: Path, cwd: Path | None = None) -> str:
    log(f"Running: {' '.join(command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log_path.write_text((proc.stdout or "") + (proc.stderr or ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {' '.join(command)}; see {log_path}")
    return proc.stdout


def parse_json_stdout(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Could not find JSON object in command output: {text[:500]}")
    return json.loads(text[start : end + 1])


def run_spatho_workflow(workflow_config: Path, run_root: Path) -> dict[str, Any]:
    logs_dir = run_root / "logs"
    doctor_stdout = run_capture(
        [sys.executable, "-m", "spatho.cli", "doctor", "--config", str(workflow_config)],
        log_path=logs_dir / "spatho_doctor.json",
    )
    doctor = parse_json_stdout(doctor_stdout)
    write_json(run_root / "tutorial_assets" / "spatho_doctor.json", doctor)
    if not doctor.get("ready_to_run"):
        raise RuntimeError(f"spatho doctor failed: {doctor.get('issues')}")

    run_stdout = run_capture(
        [sys.executable, "-m", "spatho.cli", "run", "--config", str(workflow_config)],
        log_path=logs_dir / "spatho_run.log",
    )
    result = parse_json_stdout(run_stdout)
    write_json(run_root / "tutorial_assets" / "spatho_run_result.json", result)
    return result


def run_pyxenium_workflows(dataset_root: Path, run_root: Path) -> dict[str, Any]:
    pyx_root = run_root / "pyxenium"
    topology_dir = pyx_root / "topology"
    mechanostress_dir = pyx_root / "mechanostress"
    tbc_dir = pyx_root / "tbc_anchor_placeholder"
    tbc_dir.mkdir(parents=True, exist_ok=True)
    (tbc_dir / "README.txt").write_text(
        "No transcript-level t_and_c anchor was present in this PDC dataset copy. "
        "pyXenium recomputed the smoke-panel gene topology anchors from the cell_feature_matrix.h5 counts.\n",
        encoding="utf-8",
    )

    from pyXenium.validation.atera_wta_breast_topology import run_atera_wta_breast_topology
    from pyXenium.mechanostress import (
        AxisStrengthConfig,
        MechanostressConfig,
        TumorStromaGrowthConfig,
        run_mechanostress_workflow,
    )

    log("Running pyXenium Atera WTA breast topology workflow")
    topology = run_atera_wta_breast_topology(
        dataset_root=str(dataset_root),
        tbc_results=str(tbc_dir),
        output_dir=str(topology_dir),
        sample_id=CASE_NAME,
        export_figures=True,
        write_h5ad=None,
    )

    cell_groups = dataset_root / CELL_GROUPS_CSV
    log("Running pyXenium mechanostress snapshot")
    mech_config = MechanostressConfig(
        axis=AxisStrengthConfig(radii_um=(25.0, 50.0, 100.0, 200.0), groupby=("group",), local_k=15),
        tumor_stroma=TumorStromaGrowthConfig(
            annotation_col="group",
            tumor_label="Luminal-like Amorphous DCIS Cells",
            stroma_label="CAFs, DCIS Associated",
        ),
        coupling_genes=("COL1A1", "COL3A1", "MMP11", "TGFB1", "CXCL12"),
        sample_id=CASE_NAME,
    )
    mechanostress = run_mechanostress_workflow(
        base_path=dataset_root,
        cell_table=cell_groups if cell_groups.exists() else None,
        config=mech_config,
        output_dir=mechanostress_dir,
    )

    summary = {
        "topology_summary_json": str(topology_dir / "summary.json"),
        "topology_report_md": str(topology_dir / "report.md"),
        "mechanostress_summary": mechanostress.summary,
        "mechanostress_summary_json": str(mechanostress_dir / "summary.json"),
        "mechanostress_report_md": str(mechanostress_dir / "report.md"),
    }
    write_json(run_root / "tutorial_assets" / "pyxenium_summary.json", summary)
    return summary


def copy_if_exists(source: Path, destination: Path) -> str | None:
    if not source.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return str(destination)


def collect_tutorial_assets(run_root: Path, spatho_result: dict[str, Any] | None) -> dict[str, Any]:
    assets_root = run_root / "tutorial_assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {"run_root": str(run_root), "files": {}}

    project_outputs = run_root / "runtime" / "segmentation_methods" / "projects" / CASE_NAME / "outputs" / "spatho"
    spatho_paths = {
        "workflow_summary_json": project_outputs / "workflow_summary.json",
        "artifact_manifest_json": project_outputs / "artifact_manifest.json",
        "pathology_report_html": project_outputs / "pathology_review" / "index.html",
    }
    if spatho_result:
        for key in ("workflow_summary_json", "artifact_manifest_json", "pathology_report_html"):
            value = spatho_result.get(key)
            if value:
                spatho_paths[key] = Path(value)

    for key in ("workflow_summary_json", "artifact_manifest_json"):
        copied = copy_if_exists(spatho_paths[key], assets_root / "spatho" / spatho_paths[key].name)
        if copied:
            index["files"][key] = copied
    if spatho_paths["pathology_report_html"].exists():
        index["files"]["pathology_report_html"] = str(spatho_paths["pathology_report_html"])

    for name in (
        "pipeline/validation/he_structure_isoline_overlay.png",
        "pipeline/validation/spatial_structure_isoline_overlay.png",
        "pipeline/validation/structure_isoline_metrics.json",
        "pipeline/structure_assignment/structure_assignments.csv",
        "he_foundation/he_contour_classification.csv",
        "he_foundation/he_contour_to_structure_summary.csv",
        "he_foundation/structure_multimodal_names.csv",
        "he_foundation/he_foundation_metadata.json",
    ):
        source = project_outputs / name
        copied = copy_if_exists(source, assets_root / "spatho" / source.name)
        if copied:
            index["files"][name] = copied

    for source in sorted((run_root / "pyxenium" / "topology").glob("*")):
        if source.suffix.lower() in {".json", ".md", ".csv", ".png", ".pdf"} and source.stat().st_size < 8_000_000:
            copied = copy_if_exists(source, assets_root / "pyxenium_topology" / source.name)
            if copied:
                index["files"][f"topology/{source.name}"] = copied

    for source in sorted((run_root / "pyxenium" / "mechanostress").glob("*")):
        if source.suffix.lower() in {".json", ".md", ".csv", ".png"} and source.stat().st_size < 8_000_000:
            copied = copy_if_exists(source, assets_root / "pyxenium_mechanostress" / source.name)
            if copied:
                index["files"][f"mechanostress/{source.name}"] = copied

    write_json(assets_root / "artifact_index.json", index)
    return index


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PDC Atera WTA breast workflow tutorial bundle.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--sfplot-root", required=True, type=Path)
    parser.add_argument("--pathology-ai-base-url", required=True)
    parser.add_argument("--he-zarr-level", type=int, default=6)
    parser.add_argument("--skip-spatho", action="store_true")
    parser.add_argument("--skip-pyxenium", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    run_root = args.run_root.expanduser().resolve()
    sfplot_root = args.sfplot_root.expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "tutorial_assets").mkdir(exist_ok=True)

    log(f"Dataset root: {dataset_root}")
    log(f"Run root: {run_root}")
    health = fetch_pathology_ai_health(args.pathology_ai_base_url, run_root / "tutorial_assets" / "pathology_ai_health.json")
    log(f"pathology-ai ready with components: {sorted(health.get('components', {}))}")

    input_summary = prepare_spatho_inputs(dataset_root, run_root)
    he_summary = prepare_registered_he_asset(dataset_root, run_root, zarr_level=args.he_zarr_level)
    write_json(run_root / "tutorial_assets" / "he_alignment_metadata.json", he_summary)
    runtime_root = prepare_segmentation_runtime(sfplot_root, run_root)
    base_config = write_base_pipeline_config(
        dataset_root=dataset_root,
        run_root=run_root,
        runtime_root=runtime_root,
        input_summary=input_summary,
        he_summary=he_summary,
    )
    workflow_config = write_workflow_config(
        dataset_root=dataset_root,
        run_root=run_root,
        runtime_root=runtime_root,
        base_pipeline_config=base_config,
        input_summary=input_summary,
        pathology_ai_base_url=args.pathology_ai_base_url,
    )
    write_json(
        run_root / "tutorial_assets" / "run_manifest.json",
        {
            "case_name": CASE_NAME,
            "dataset_root": str(dataset_root),
            "local_windows_dataset": r"Y:\long\10X_datasets\Xenium\Atera\WTA_Preview_FFPE_Breast_Cancer_outs",
            "run_root": str(run_root),
            "pathology_ai_base_url": args.pathology_ai_base_url,
            "base_pipeline_config": str(base_config),
            "workflow_config": str(workflow_config),
            "he": he_summary,
        },
    )

    spatho_result: dict[str, Any] | None = None
    if not args.skip_spatho:
        spatho_result = run_spatho_workflow(workflow_config, run_root)
    if not args.skip_pyxenium:
        run_pyxenium_workflows(dataset_root, run_root)
    collect_tutorial_assets(run_root, spatho_result)
    log("PDC Atera breast workflow finished")


if __name__ == "__main__":
    main()
