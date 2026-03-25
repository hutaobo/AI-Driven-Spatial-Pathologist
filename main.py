from __future__ import annotations

import json
import os
import shutil
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import dendrogram, fcluster, leaves_list, linkage, to_tree
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import squareform
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
    "A dendrogram-guided Xenium analysis workspace that turns related clusters into interpretable "
    "spatial structures before running the final HistoSeg contour analysis."
)
DEFAULT_PATTERN1 = "10,23,19,27,14,20,25,26"
LABEL_SCHEME_OPTIONS = {
    "Treat the selected structures as the signal of interest (recommended)": "p1_is_one",
    "Treat the selected structures as background (invert the score)": "p1_is_zero",
}
DEFAULT_LABEL_SCHEME = "Treat the selected structures as the signal of interest (recommended)"
GROUP_SELECTION_EMPTY = "No structures selected yet. Click one or more colored badges on the dendrogram, or type cluster IDs manually."
GROUP_PALETTE = [
    "#6EF0D4",
    "#78B9FF",
    "#FFB870",
    "#C8A2FF",
    "#FF8DA1",
    "#90F184",
    "#FFD76C",
    "#80E1FF",
    "#F4A6FF",
    "#FFA07A",
    "#B8F0DE",
    "#A7BFFF",
]
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
SELECTIONS_DIR = DEFAULT_WORK_DIR / "structure-selections"


def ensure_workdirs() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    SELECTIONS_DIR.mkdir(parents=True, exist_ok=True)


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
        return "Selected structures are treated as the signal of interest"
    return "Selected structures are treated as background"


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


def parse_optional_clusters(raw: str) -> list[int | str]:
    if raw is None:
        return []
    if not str(raw).strip():
        return []
    return parse_pattern1_clusters(str(raw))


def stringify_clusters(clusters: list[int | str]) -> str:
    return ",".join(str(item) for item in clusters)


def summarize_clusters(clusters: list[str], max_items: int = 8) -> str:
    if len(clusters) <= max_items:
        return ", ".join(clusters)
    head = ", ".join(clusters[:max_items])
    return f"{head}, ... (+{len(clusters) - max_items} more)"


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


def build_selection_dir() -> Path:
    ensure_workdirs()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    selection_dir = SELECTIONS_DIR / f"selection-{stamp}"
    suffix = 1
    while selection_dir.exists():
        suffix += 1
        selection_dir = SELECTIONS_DIR / f"selection-{stamp}-{suffix}"
    selection_dir.mkdir(parents=True, exist_ok=False)
    return selection_dir


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


def cleanup_old_selections(max_keep: int = 2) -> list[str]:
    ensure_workdirs()
    selections = sorted(
        [path for path in SELECTIONS_DIR.glob("selection-*") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: list[str] = []
    for stale in selections[max_keep:]:
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


def prepare_merged_clusters(cells_path: Path, clusters_path: Path) -> tuple[pd.DataFrame, str, str, str]:
    merged, id_col_used, x_col, y_col = align_clusters_with_cells(
        clusters_path,
        cells_path,
        barcode_col="Barcode",
        cluster_col="Cluster",
    )
    merged = merged.copy()
    merged["cluster"] = merged["cluster"].map(_normalize_cluster_label)
    merged = merged.loc[merged["cluster"] != ""].copy()
    return merged, id_col_used, x_col, y_col


def normalize_row_cophenetic(row_coph: pd.DataFrame) -> pd.DataFrame:
    labels = [_normalize_cluster_label(label) for label in row_coph.index]
    normalized = row_coph.copy()
    normalized.index = labels
    normalized.columns = labels
    return normalized


def remap_flat_clusters_by_leaf_order(cluster_ids: list[str], linkage_matrix, flat_labels) -> dict[str, int]:
    if linkage_matrix is None:
        return {cluster_ids[0]: 1}

    raw_map = {str(cluster_id): int(raw_label) for cluster_id, raw_label in zip(cluster_ids, flat_labels)}
    ordered_cluster_ids = [cluster_ids[index] for index in leaves_list(linkage_matrix)]

    raw_order: list[int] = []
    seen_raw: set[int] = set()
    for cluster_id in ordered_cluster_ids:
        raw_label = raw_map[str(cluster_id)]
        if raw_label in seen_raw:
            continue
        seen_raw.add(raw_label)
        raw_order.append(raw_label)

    remap = {raw_label: index + 1 for index, raw_label in enumerate(raw_order)}
    return {str(cluster_id): int(remap[raw_map[str(cluster_id)]]) for cluster_id in cluster_ids}


def group_color(group_id: int) -> str:
    return GROUP_PALETTE[(int(group_id) - 1) % len(GROUP_PALETTE)]


def build_structure_choice_label(group_id: int, clusters: list[str]) -> str:
    return f"Structure {group_id} | {len(clusters)} cluster IDs"


def build_group_table(
    group_state: dict[str, object] | None,
    selected_groups: list[str] | None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if not group_state:
        return pd.DataFrame(rows, columns=["Selected", "Structure", "Cluster count", "Cluster IDs"])

    selected_set = set(selected_groups or [])
    for record in group_state.get("group_records", []):
        choice_label = str(record["choice_label"])
        rows.append(
            {
                "Selected": "Yes" if choice_label in selected_set else "",
                "Structure": str(record["group_name"]),
                "Cluster count": int(record["cluster_count"]),
                "Cluster IDs": ", ".join(str(item) for item in record["clusters"]),
            }
        )
    return pd.DataFrame(rows, columns=["Selected", "Structure", "Cluster count", "Cluster IDs"])


def build_structure_group_state(
    row_coph: pd.DataFrame,
    *,
    n_groups: int,
    linkage_method: str = "average",
) -> dict[str, object]:
    cluster_ids = [str(value) for value in row_coph.index]
    if not cluster_ids:
        raise ValueError("No clusters were available for dendrogram building.")

    cluster_to_leaf_index = {cluster_id: index for index, cluster_id in enumerate(cluster_ids)}
    if len(cluster_ids) == 1:
        ordered_clusters = cluster_ids
        group_to_clusters = {1: ordered_clusters}
        linkage_matrix = None
        leaf_positions = {0: 5.0}
        node_leaf_map = {0: [0]}
    else:
        condensed = squareform(row_coph.values, checks=False)
        linkage_matrix = linkage(condensed, method=linkage_method)
        n_groups = max(1, min(int(n_groups), len(cluster_ids)))
        flat_labels = fcluster(linkage_matrix, t=n_groups, criterion="maxclust")
        cluster_to_group = remap_flat_clusters_by_leaf_order(cluster_ids, linkage_matrix, flat_labels)
        leaf_order = [int(index) for index in leaves_list(linkage_matrix)]
        ordered_clusters = [cluster_ids[index] for index in leaf_order]
        group_to_clusters: dict[int, list[str]] = {}
        for cluster_id in ordered_clusters:
            group_id = int(cluster_to_group[cluster_id])
            group_to_clusters.setdefault(group_id, []).append(cluster_id)
        leaf_positions = {leaf_id: 5.0 + 10.0 * order_index for order_index, leaf_id in enumerate(leaf_order)}

        root_node, node_list = to_tree(linkage_matrix, rd=True)
        node_leaf_map: dict[int, list[int]] = {}

        def collect_leaf_ids(node) -> list[int]:
            if node.is_leaf():
                leaves = [int(node.id)]
            else:
                leaves = collect_leaf_ids(node.left) + collect_leaf_ids(node.right)
            node_leaf_map[int(node.id)] = leaves
            return leaves

        collect_leaf_ids(root_node)
        _ = node_list  # Keeps the rd=True unpacking explicit for readability.

    leaf_set_to_node: dict[frozenset[int], dict[str, float]] = {}
    if linkage_matrix is None:
        leaf_set_to_node[frozenset({0})] = {"node_id": 0.0, "dist": 0.0}
    else:
        root_node, node_list = to_tree(linkage_matrix, rd=True)
        for node in node_list:
            leaves = node_leaf_map.get(int(node.id), [])
            leaf_set_to_node[frozenset(int(value) for value in leaves)] = {
                "node_id": float(node.id),
                "dist": float(node.dist),
            }

    ordered_cluster_to_position = {cluster_id: index for index, cluster_id in enumerate(ordered_clusters)}

    choices: list[str] = []
    choice_to_clusters: dict[str, list[str]] = {}
    table_rows: list[dict[str, object]] = []
    group_records: list[dict[str, Any]] = []
    for group_id, clusters in sorted(group_to_clusters.items()):
        choice_label = build_structure_choice_label(group_id, clusters)
        leaf_ids = [int(cluster_to_leaf_index[cluster_id]) for cluster_id in clusters]
        x_points = [float(leaf_positions[leaf_id]) for leaf_id in leaf_ids]
        leaf_span = sorted(int(ordered_cluster_to_position[cluster_id]) for cluster_id in clusters)
        node_summary = leaf_set_to_node.get(frozenset(leaf_ids), {"node_id": float(group_id), "dist": 0.0})
        span_left = min(x_points) - 5.0
        span_right = max(x_points) + 5.0
        y_data = float(node_summary["dist"])
        marker_y = y_data
        color = group_color(group_id)

        choices.append(choice_label)
        choice_to_clusters[choice_label] = list(clusters)
        table_rows.append(
            {
                "Selected": "",
                "Structure": f"Structure {group_id}",
                "Cluster count": len(clusters),
                "Cluster IDs": ", ".join(clusters),
            }
        )
        group_records.append(
            {
                "group_id": int(group_id),
                "group_name": f"Structure {group_id}",
                "choice_label": choice_label,
                "clusters": list(clusters),
                "cluster_count": int(len(clusters)),
                "color": color,
                "leaf_start": int(min(leaf_span)),
                "leaf_end": int(max(leaf_span)),
                "span_left": float(span_left),
                "span_right": float(span_right),
                "x_data": float(np.mean(x_points)),
                "y_data": float(y_data),
                "marker_y": float(marker_y),
            }
        )

    return {
        "n_groups": len(group_to_clusters),
        "ordered_clusters": ordered_clusters,
        "choices": choices,
        "choice_to_clusters": choice_to_clusters,
        "table_rows": table_rows,
        "group_records": group_records,
        "row_coph_labels": cluster_ids,
        "row_coph_values": row_coph.to_numpy().tolist(),
        "linkage_matrix": linkage_matrix.tolist() if linkage_matrix is not None else None,
        "selected_groups": [],
    }


def collect_clusters_from_groups(selected_groups: list[str], group_state: dict[str, object] | None) -> list[str]:
    if not group_state:
        return []
    choice_to_clusters = group_state.get("choice_to_clusters", {})
    ordered_clusters = group_state.get("ordered_clusters", [])
    selected_set = set(selected_groups or [])
    cluster_order: list[str] = []
    for cluster_id in ordered_clusters:
        for choice, clusters in choice_to_clusters.items():
            if choice in selected_set and cluster_id in clusters and cluster_id not in cluster_order:
                cluster_order.append(cluster_id)
    return cluster_order


def update_clusters_to_outline_from_groups(
    selected_groups: list[str] | None,
    group_state: dict[str, object] | None,
) -> tuple[str, str]:
    clusters = collect_clusters_from_groups(selected_groups or [], group_state)
    if not clusters:
        return "", GROUP_SELECTION_EMPTY

    cluster_text = stringify_clusters(clusters)
    summary = (
        f"Selected {len(selected_groups or [])} structure(s) -> "
        f"{len(clusters)} cluster ID(s): {summarize_clusters(clusters)}"
    )
    return cluster_text, summary


def normalize_selected_groups(
    selected_groups: list[str] | None,
    group_state: dict[str, object] | None,
) -> list[str]:
    if not group_state:
        return []
    selected_set = set(selected_groups or [])
    return [choice for choice in group_state.get("choices", []) if choice in selected_set]


def render_structure_selector_image(
    group_state: dict[str, object],
    selected_groups: list[str] | None,
) -> tuple[Path, dict[str, object]]:
    output_dir = Path(str(group_state["selector_output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_groups = normalize_selected_groups(selected_groups, group_state)
    selected_set = set(selected_groups)
    group_records = [dict(record) for record in group_state.get("group_records", [])]
    row_coph = pd.DataFrame(
        np.asarray(group_state["row_coph_values"], dtype=float),
        index=list(group_state["row_coph_labels"]),
        columns=list(group_state["row_coph_labels"]),
    )
    ordered_clusters = list(group_state["ordered_clusters"])
    ordered_matrix = row_coph.loc[ordered_clusters, ordered_clusters]
    linkage_payload = group_state.get("linkage_matrix")
    linkage_matrix = np.asarray(linkage_payload, dtype=float) if linkage_payload is not None else None

    selector_key = "none"
    if selected_set:
        selected_ids = [str(record["group_id"]) for record in group_records if record["choice_label"] in selected_set]
        selector_key = "_".join(selected_ids)
    selector_path = output_dir / f"interactive_structure_selector_{selector_key}.png"

    cluster_to_record: dict[str, dict[str, Any]] = {}
    for record in group_records:
        for cluster_id in record["clusters"]:
            cluster_to_record[str(cluster_id)] = record

    fig = plt.figure(figsize=(13.5, 8.7), facecolor="#07111D")
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.55], hspace=0.08)
    ax_dendro = fig.add_subplot(outer[0])
    heat_grid = outer[1].subgridspec(2, 2, height_ratios=[0.075, 1.0], width_ratios=[0.075, 1.0], hspace=0.03, wspace=0.03)
    ax_corner = fig.add_subplot(heat_grid[0, 0])
    ax_top_band = fig.add_subplot(heat_grid[0, 1])
    ax_left_band = fig.add_subplot(heat_grid[1, 0])
    ax_heat = fig.add_subplot(heat_grid[1, 1])

    for axis in (ax_dendro, ax_top_band, ax_left_band, ax_heat):
        axis.set_facecolor("#0C1726")
    ax_corner.set_facecolor("#07111D")
    ax_corner.axis("off")

    if linkage_matrix is not None:
        dendrogram(
            linkage_matrix,
            no_labels=True,
            color_threshold=0,
            above_threshold_color="#6B8198",
            link_color_func=lambda _node_id: "#6B8198",
            ax=ax_dendro,
        )
        max_dist = float(np.max(linkage_matrix[:, 2])) if len(linkage_matrix) else 1.0
    else:
        ax_dendro.plot([5.0, 5.0], [0.0, 1.0], color="#6B8198", linewidth=2.5)
        max_dist = 1.0

    marker_offset = max(max_dist * 0.08, 0.24)
    marker_positions: dict[str, dict[str, float]] = {}
    marker_size = 310

    for record in group_records:
        is_selected = record["choice_label"] in selected_set
        color = str(record["color"])
        x_data = float(record["x_data"])
        y_data = float(record["y_data"])
        marker_y = y_data + marker_offset
        record["marker_y"] = marker_y

        ax_dendro.axvspan(
            float(record["span_left"]),
            float(record["span_right"]),
            color=color,
            alpha=0.22 if is_selected else 0.06,
            zorder=0,
        )
        ax_dendro.plot(
            [x_data, x_data],
            [y_data, marker_y - marker_offset * 0.18],
            color=color,
            linewidth=2.1 if is_selected else 1.3,
            alpha=0.95,
            zorder=4,
        )
        ax_dendro.scatter(
            [x_data],
            [marker_y],
            s=marker_size + (95 if is_selected else 0),
            color=color,
            edgecolors="#F7FBFF" if is_selected else "#132236",
            linewidths=2.0,
            zorder=6,
        )
        ax_dendro.text(
            x_data,
            marker_y,
            f"S{record['group_id']}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#07111D",
            zorder=7,
        )
        ax_dendro.text(
            x_data,
            marker_y + marker_offset * 0.72,
            f"{record['cluster_count']} IDs",
            ha="center",
            va="bottom",
            fontsize=8.3,
            color="#D9E8F8" if is_selected else "#9CB0C7",
            zorder=7,
        )

    ax_dendro.text(
        0.015,
        0.96,
        "Click a colored structure badge to add or remove that branch from the final contour run.",
        transform=ax_dendro.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="#E9F2FD",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#101C2B", edgecolor="#1C3550", alpha=0.97),
    )
    ax_dendro.set_title("Interactive structure selector", loc="left", fontsize=15, color="#F5F9FF", pad=12)
    ax_dendro.set_ylabel("Cophenetic distance", color="#A8BCD3")
    ax_dendro.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_dendro.tick_params(axis="y", colors="#90A6BF")
    for spine in ax_dendro.spines.values():
        spine.set_color("#20354A")
    ax_dendro.set_ylim(-marker_offset * 0.4, max_dist + marker_offset * 2.0)

    band_rgba = np.array([[mcolors.to_rgba(cluster_to_record[cluster]["color"]) for cluster in ordered_clusters]])
    ax_top_band.imshow(band_rgba, aspect="auto")
    ax_left_band.imshow(np.transpose(band_rgba, (1, 0, 2)), aspect="auto")
    for axis in (ax_top_band, ax_left_band):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#20354A")

    heat = ax_heat.imshow(ordered_matrix.to_numpy(float), cmap="magma", interpolation="nearest", aspect="auto")
    for record in group_records:
        start = int(record["leaf_start"])
        end = int(record["leaf_end"])
        size = end - start + 1
        ax_heat.add_patch(
            Rectangle(
                (start - 0.5, start - 0.5),
                size,
                size,
                fill=False,
                edgecolor=str(record["color"]),
                linewidth=2.4 if record["choice_label"] in selected_set else 0.9,
                alpha=0.95 if record["choice_label"] in selected_set else 0.28,
            )
        )

    if len(ordered_clusters) <= 18:
        tick_positions = range(len(ordered_clusters))
        tick_labels = ordered_clusters
    else:
        step = max(1, len(ordered_clusters) // 12)
        tick_positions = list(range(0, len(ordered_clusters), step))
        tick_labels = [ordered_clusters[index] for index in tick_positions]

    ax_heat.set_xticks(list(tick_positions))
    ax_heat.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8, color="#A9BDD4")
    ax_heat.set_yticks(list(tick_positions))
    ax_heat.set_yticklabels(tick_labels, fontsize=8, color="#A9BDD4")
    ax_heat.set_title("Cophenetic heatmap ordered by the dendrogram", loc="left", fontsize=13, color="#F5F9FF", pad=8)
    for spine in ax_heat.spines.values():
        spine.set_color("#20354A")
    cbar = fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.02)
    cbar.outline.set_edgecolor("#20354A")
    cbar.ax.tick_params(colors="#A8BCD3")

    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    for record in group_records:
        x_disp, y_disp = ax_dendro.transData.transform((float(record["x_data"]), float(record["marker_y"])))
        marker_positions[str(record["choice_label"])] = {
            "x": float(x_disp),
            "y": float(height - y_disp),
            "x_norm": float(x_disp / width),
            "y_norm": float((height - y_disp) / height),
            "radius": 32.0,
            "radius_norm": float(32.0 / max(width, height)),
        }

    fig.savefig(selector_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)

    next_state = dict(group_state)
    next_state["selected_groups"] = list(selected_groups)
    next_state["marker_positions"] = marker_positions
    next_state["selector_path"] = str(selector_path)
    return selector_path, next_state


def resolve_clicked_structure(
    click_index: object,
    group_state: dict[str, object] | None,
) -> str | None:
    if not group_state:
        return None

    marker_positions = group_state.get("marker_positions", {})
    if not marker_positions:
        return None

    candidate_points: list[tuple[float, float]] = []
    if isinstance(click_index, dict):
        if "x" in click_index and "y" in click_index:
            candidate_points.append((float(click_index["x"]), float(click_index["y"])))
    elif isinstance(click_index, (list, tuple)) and len(click_index) >= 2:
        a = float(click_index[0])
        b = float(click_index[1])
        candidate_points.append((a, b))
        if abs(a - b) > 1:
            candidate_points.append((b, a))
    else:
        return None

    best_choice: str | None = None
    best_distance = float("inf")
    best_threshold = float("inf")
    for choice_label, marker in marker_positions.items():
        for x_click, y_click in candidate_points:
            if max(abs(x_click), abs(y_click)) <= 1.5:
                distance = float(np.hypot(x_click - marker["x_norm"], y_click - marker["y_norm"]))
                threshold = float(marker["radius_norm"]) * 1.6
            else:
                distance = float(np.hypot(x_click - marker["x"], y_click - marker["y"]))
                threshold = float(marker["radius"]) * 1.6

            if distance < best_distance:
                best_choice = str(choice_label)
                best_distance = distance
                best_threshold = threshold

    if best_choice is not None and best_distance <= best_threshold:
        return best_choice
    return None


def refresh_structure_selection(
    selected_groups: list[str] | None,
    group_state: dict[str, object] | None,
    note: str | None = None,
) -> tuple[str | None, pd.DataFrame, dict[str, object], str, str, dict[str, object]]:
    if not group_state:
        empty_table = build_group_table({}, [])
        return None, empty_table, gr.update(choices=[], value=[]), "", note or GROUP_SELECTION_EMPTY, {}

    normalized_groups = normalize_selected_groups(selected_groups, group_state)
    selector_path, next_state = render_structure_selector_image(group_state, normalized_groups)
    cluster_text, summary = update_clusters_to_outline_from_groups(normalized_groups, next_state)
    if note:
        summary = f"{summary}\n{note}"

    return (
        str(selector_path),
        build_group_table(next_state, normalized_groups),
        gr.update(choices=next_state["choices"], value=normalized_groups),
        cluster_text,
        summary,
        next_state,
    )


def toggle_structure_group_from_selector(
    group_state: dict[str, object] | None,
    evt: gr.SelectData,
) -> tuple[str | None, pd.DataFrame, dict[str, object], str, str, dict[str, object]]:
    if not group_state:
        empty_table = build_group_table({}, [])
        return None, empty_table, gr.update(choices=[], value=[]), "", GROUP_SELECTION_EMPTY, {}

    current_groups = normalize_selected_groups(group_state.get("selected_groups", []), group_state)
    clicked_choice = resolve_clicked_structure(getattr(evt, "index", None), group_state)
    if clicked_choice is None:
        return refresh_structure_selection(
            current_groups,
            group_state,
            note="Click directly on one of the colored badges labelled S1, S2, S3, ... to toggle a structure.",
        )

    next_groups = list(current_groups)
    if clicked_choice in next_groups:
        next_groups = [choice for choice in next_groups if choice != clicked_choice]
    else:
        next_groups.append(clicked_choice)

    return refresh_structure_selection(next_groups, group_state)


def clear_structure_selection(
    group_state: dict[str, object] | None,
) -> tuple[str | None, pd.DataFrame, dict[str, object], str, str, dict[str, object]]:
    return refresh_structure_selection([], group_state, note="Selection cleared. Choose one or more structures to continue.")


def build_structure_groups(
    cells_parquet: object | None,
    clusters_csv: object | None,
    n_structure_groups: int,
    progress: gr.Progress = gr.Progress(track_tqdm=False),
):
    if HISTOSEG_IMPORT_ERROR is not None:
        raise gr.Error(
            "HistoSeg could not be imported inside the app container. "
            f"Import error: {HISTOSEG_IMPORT_ERROR}"
        )

    removed_previews = cleanup_old_selections(max_keep=2)
    selection_dir = build_selection_dir()
    input_dir = selection_dir / "inputs"
    output_dir = selection_dir / "outputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress(0.1, desc="Staging files for dendrogram")
    cells_path, clusters_path, _unused_tissue_path = resolve_inputs(
        cells_upload=cells_parquet,
        clusters_upload=clusters_csv,
        tissue_upload=None,
        target_dir=input_dir,
    )

    progress(0.35, desc="Aligning cells and clusters")
    merged, id_col_used, x_col, y_col = prepare_merged_clusters(cells_path, clusters_path)
    if merged["cluster"].nunique() < 2:
        raise gr.Error("Need at least two clusters to build a dendrogram.")

    progress(0.6, desc="Computing cophenetic dendrogram")
    distance_matrix = compute_searcher_findee_distance_matrix_from_df(
        merged,
        x_col=x_col,
        y_col=y_col,
        z_col=None,
        celltype_col="cluster",
    )
    row_coph, _col_coph = compute_cophenetic_from_distance_matrix(
        distance_matrix,
        method="average",
        show_corr=False,
    )
    row_coph = normalize_row_cophenetic(row_coph)

    heatmap_image = plot_cophenetic_heatmap(
        row_coph,
        matrix_name="row_coph",
        sample="all clusters",
        return_image=True,
        dpi=300,
    )
    heatmap_path = output_dir / "cophenetic_heatmap_row_coph.png"
    heatmap_image.save(heatmap_path)

    progress(0.85, desc="Cutting dendrogram into structure groups")
    group_state = build_structure_group_state(row_coph, n_groups=int(n_structure_groups), linkage_method="average")
    group_state["selector_output_dir"] = str(output_dir)
    selector_path, group_state = render_structure_selector_image(group_state, [])
    group_df = build_group_table(group_state, [])
    status_lines = [
        f"Built a cophenetic dendrogram for {merged['cluster'].nunique()} clusters.",
        f"Cut the dendrogram into {group_state['n_groups']} candidate structure(s).",
        "Click one or more colored badges on the interactive dendrogram to choose the structures that should enter the final contour run.",
    ]
    if removed_previews:
        status_lines.append(f"Cleaned old dendrogram sessions: {', '.join(removed_previews)}")
    status_lines.append(f"Merged cells available for grouping: {len(merged)}")
    status_lines.append(f"Coordinate columns: {x_col}, {y_col}")
    status_lines.append(f"Cell identifier column: {id_col_used}")

    progress(1.0, desc="Structure groups ready")
    return (
        "\n".join(status_lines),
        str(selector_path),
        str(heatmap_path),
        group_df,
        gr.update(choices=group_state["choices"], value=[]),
        "",
        GROUP_SELECTION_EMPTY,
        group_state,
    )


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
            f"Cluster IDs entering contour analysis: {', '.join(str(item) for item in parsed_clusters)}",
            f"Scoring mode: {describe_label_scheme(internal_label_scheme)}",
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
            plt.scatter(target_xy[:, 0], target_xy[:, 1], s=3, alpha=0.85, label="selected-structure cells")
            for vertices in verts_list:
                plt.plot(vertices[:, 0], vertices[:, 1], linewidth=2)
            plt.gca().set_aspect("equal")
            title = (
                f"Selected-structure contours | isoline={cfg.isoline_level:g} | "
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
:root {
  --app-bg: #07111d;
  --app-bg-soft: #0b1523;
  --panel-bg: rgba(12, 23, 38, 0.96);
  --panel-bg-strong: rgba(15, 29, 46, 0.98);
  --panel-border: #1f3850;
  --panel-border-strong: #2a5270;
  --text-main: #f4f8ff;
  --text-muted: #9db1c9;
  --accent: #6ef0d4;
  --accent-warm: #ffbd73;
  --accent-cool: #7ab8ff;
  --shadow-strong: 0 24px 70px rgba(0, 0, 0, 0.38);
}

body {
  background:
    radial-gradient(circle at top left, rgba(110, 240, 212, 0.10), transparent 28%),
    radial-gradient(circle at top right, rgba(122, 184, 255, 0.12), transparent 32%),
    linear-gradient(180deg, #050d16 0%, #08111d 100%) !important;
}

.gradio-container {
  max-width: 1480px !important;
  color: var(--text-main);
  background: transparent !important;
  font-family: "Aptos", "Bahnschrift", "Segoe UI Variable", "Segoe UI", sans-serif;
}

.gradio-container .gr-box,
.gradio-container .block,
.gradio-container .form,
.gradio-container .panel,
.gradio-container .gr-accordion,
.gradio-container .gr-dataframe,
.gradio-container .gradio-file {
  background: var(--panel-bg) !important;
  border: 1px solid var(--panel-border) !important;
  box-shadow: none !important;
}

.gradio-container .label-wrap,
.gradio-container .label-wrap span,
.gradio-container label,
.gradio-container .prose,
.gradio-container .prose p,
.gradio-container .prose li,
.gradio-container .gr-markdown,
.gradio-container .gr-markdown p,
.gradio-container .gr-markdown li,
.gradio-container .gr-html,
.gradio-container .gr-html p,
.gradio-container .gr-html li,
.gradio-container .gr-json,
.gradio-container .gr-dataframe,
.gradio-container .gr-textbox,
.gradio-container .gr-form label {
  color: var(--text-main) !important;
}

.gradio-container .gr-markdown strong,
.gradio-container .gr-html strong,
.gradio-container .gr-markdown code,
.gradio-container .gr-html code {
  color: #f9fcff !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select {
  background: #091321 !important;
  color: var(--text-main) !important;
  border: 1px solid #284059 !important;
}

.gradio-container table {
  background: #0c1726 !important;
  color: var(--text-main) !important;
}

.gradio-container th {
  background: #101d2d !important;
  color: var(--text-main) !important;
}

.gradio-container td {
  background: #0c1726 !important;
  color: #e6eef8 !important;
}

.gradio-container .gr-button {
  border-radius: 14px !important;
  font-weight: 700 !important;
  letter-spacing: 0.01em;
}

.gradio-container .gr-button-primary {
  background: linear-gradient(135deg, #6ef0d4 0%, #3ec7ff 100%) !important;
  color: #04111b !important;
  border: none !important;
  box-shadow: 0 14px 30px rgba(62, 199, 255, 0.25);
}

.gradio-container .gr-button-secondary {
  background: #132335 !important;
  color: var(--text-main) !important;
  border: 1px solid #284761 !important;
}

.hero-shell {
  background:
    linear-gradient(135deg, rgba(16, 27, 42, 0.98) 0%, rgba(12, 20, 31, 0.96) 48%, rgba(14, 29, 44, 0.98) 100%);
  border: 1px solid rgba(110, 240, 212, 0.18);
  border-radius: 28px;
  padding: 30px 32px;
  margin-bottom: 18px;
  box-shadow: var(--shadow-strong);
}

.hero-kicker {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(110, 240, 212, 0.10);
  border: 1px solid rgba(110, 240, 212, 0.18);
  color: var(--accent);
  font-size: 0.88rem;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  margin-bottom: 16px;
}

.hero-shell h1 {
  margin: 0;
  font-size: 2.9rem;
  line-height: 1.04;
  letter-spacing: -0.03em;
  color: #f6fbff;
}

.hero-shell p {
  margin: 14px 0 0 0;
  max-width: 980px;
  color: #d5e3f4;
  font-size: 1.08rem;
  line-height: 1.7;
}

.hero-metrics {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 20px;
}

.hero-metrics span {
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid rgba(255, 255, 255, 0.08);
  color: #eaf3ff;
  font-size: 0.92rem;
}

.guide-shell {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}

.guide-card {
  background: var(--panel-bg-strong);
  border: 1px solid var(--panel-border);
  border-radius: 22px;
  padding: 18px;
  min-height: 172px;
}

.guide-card .guide-step {
  color: var(--accent-warm);
  font-size: 0.86rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  margin-bottom: 10px;
}

.guide-card h3 {
  margin: 0 0 10px 0;
  color: #f6fbff;
  font-size: 1.08rem;
}

.guide-card p {
  margin: 0;
  color: #c8d7ea;
  line-height: 1.62;
  font-size: 0.96rem;
}

.app-note {
  margin-bottom: 18px;
  padding: 16px 18px;
  border-radius: 20px;
  background: rgba(13, 23, 38, 0.96);
  border: 1px solid rgba(255, 189, 115, 0.18);
  color: #eff6ff;
  line-height: 1.7;
}

.app-note strong {
  color: var(--accent-warm);
}

.micro-guide {
  padding: 14px 16px;
  border-radius: 16px;
  background: rgba(14, 24, 38, 0.96);
  border: 1px solid rgba(122, 184, 255, 0.18);
  color: #d9e7f6;
  line-height: 1.62;
  margin-bottom: 14px;
}

#left-rail,
#right-rail {
  gap: 14px;
}

#structure-selector,
#cophenetic-preview,
#contour-preview {
  border-radius: 20px !important;
  overflow: hidden;
}

#selection-summary textarea,
#dendrogram-status textarea,
#run-status textarea {
  min-height: 112px !important;
}

@media (max-width: 1120px) {
  .guide-shell {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 700px) {
  .hero-shell {
    padding: 24px 20px;
  }
  .hero-shell h1 {
    font-size: 2.2rem;
  }
  .guide-shell {
    grid-template-columns: 1fr;
  }
}
"""


with gr.Blocks(
    title=APP_NAME,
    css=CUSTOM_CSS,
    theme=gr.themes.Base(primary_hue="cyan", secondary_hue="blue", neutral_hue="slate"),
) as demo:
    ensure_workdirs()

    gr.HTML(
        """
        <div class="hero-shell">
          <div class="hero-kicker">SciLifeLab Serve app | Xenium spatial pathology</div>
          <h1>AI Driven Spatial Pathologist</h1>
          <p>
            Turn Xenium cluster maps into interpretable spatial structures before drawing the final contour map.
            This app first builds a cophenetic dendrogram from the uploaded cluster assignments, then lets you pick one or more structure branches, and finally sends the merged cluster IDs into HistoSeg for contour generation.
          </p>
          <div class="hero-metrics">
            <span>Dendrogram-guided structure picking</span>
            <span>Multi-structure contour analysis</span>
            <span>Cophenetic heatmap + contour preview</span>
          </div>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="guide-shell">
          <div class="guide-card">
            <div class="guide-step">Step 1</div>
            <h3>Upload the Xenium-derived tables</h3>
            <p>Provide <code>cells.parquet</code> and <code>clusters.csv</code>. Add <code>tissue_boundary.csv</code> if you want synthetic background support during contour generation.</p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 2</div>
            <h3>Build the structure dendrogram</h3>
            <p>The app groups cluster IDs by spatial similarity and cuts the dendrogram into candidate structures. Each colored badge marks one branch after the current cut.</p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 3</div>
            <h3>Select one or more structures</h3>
            <p>Click the dendrogram badges or use the checklist fallback. The selected structures are merged automatically into the cluster-ID textbox used by the final HistoSeg run.</p>
          </div>
          <div class="guide-card">
            <div class="guide-step">Step 4</div>
            <h3>Run the final contour analysis</h3>
            <p>You will receive a contour preview, the raw cophenetic heatmap from HistoSeg, parameter metadata, and downloadable contour files for downstream inspection.</p>
          </div>
        </div>
        <div class="app-note">
          <strong>What this app is for.</strong> Use it when you want to convert related Xenium clusters into larger spatial structures that can be interpreted like tissue compartments.
          The dendrogram is not just a picture: it is the structure-picking step that decides which branches enter the final contour analysis.
        </div>
        """
    )

    group_state = gr.State(value={})

    with gr.Row():
        with gr.Column(scale=1, elem_id="left-rail"):
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
            n_structure_groups = gr.Slider(
                label="Cut the dendrogram into this many candidate structures",
                minimum=2,
                maximum=12,
                step=1,
                value=4,
            )
            build_groups_button = gr.Button("1. Build dendrogram and candidate structures", variant="secondary")
            gr.HTML(
                """
                <div class="micro-guide">
                  Build the dendrogram first. Then click one or more colored badges on the interactive structure selector.
                  The checkbox list below stays available as a fallback and stays synchronized with the image selection.
                </div>
                """
            )
            structure_group_selector = gr.CheckboxGroup(
                label="Structures to include in the final contour run",
                choices=[],
                value=[],
                info="You can click the dendrogram badges directly, or use this checklist if you prefer a textual fallback.",
            )
            clear_selection_button = gr.Button("Clear selected structures", variant="secondary")
            pattern1_clusters = gr.Textbox(
                label="Cluster IDs included in the final contour",
                value="",
                info="This field is auto-filled from the selected structures, but you can still edit it manually if you already know the exact cluster IDs.",
            )
            selection_summary = gr.Textbox(
                label="Current structure selection",
                value=GROUP_SELECTION_EMPTY,
                lines=3,
                elem_id="selection-summary",
            )
            label_scheme = gr.Dropdown(
                label="How should the selected structures be scored?",
                choices=list(LABEL_SCHEME_OPTIONS.keys()),
                value=DEFAULT_LABEL_SCHEME,
                info="Recommended: treat the selected structures as the signal of interest and let background score low.",
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
                    label="Include the cophenetic heatmap and confidence score in the final outputs",
                    value=True,
                )

            run_button = gr.Button("2. Run final HistoSeg contour analysis", variant="primary")

        with gr.Column(scale=1, elem_id="right-rail"):
            structure_status = gr.Textbox(label="Step 1 status", lines=6, elem_id="dendrogram-status")
            structure_selector_image = gr.Image(
                label="Interactive structure selector",
                type="filepath",
                interactive=True,
                sources=[],
                elem_id="structure-selector",
            )
            structure_group_table = gr.Dataframe(
                label="Candidate structures and their cluster IDs",
                headers=["Selected", "Structure", "Cluster count", "Cluster IDs"],
                datatype=["str", "str", "number", "str"],
                interactive=False,
                wrap=True,
            )
            cophenetic_heatmap_image = gr.Image(
                label="Raw cophenetic heatmap from HistoSeg",
                type="filepath",
                elem_id="cophenetic-preview",
            )
            status_text = gr.Textbox(label="Step 2 status", lines=8, elem_id="run-status")
            preview_image = gr.Image(label="Final contour preview", type="filepath", elem_id="contour-preview")
            summary_json = gr.JSON(label="Run summary")
            output_files = gr.File(label="Download outputs", file_count="multiple")

    build_groups_button.click(
        fn=build_structure_groups,
        inputs=[cells_parquet, clusters_csv, n_structure_groups],
        outputs=[
            structure_status,
            structure_selector_image,
            cophenetic_heatmap_image,
            structure_group_table,
            structure_group_selector,
            pattern1_clusters,
            selection_summary,
            group_state,
        ],
    )

    structure_group_selector.change(
        fn=refresh_structure_selection,
        inputs=[structure_group_selector, group_state],
        outputs=[
            structure_selector_image,
            structure_group_table,
            structure_group_selector,
            pattern1_clusters,
            selection_summary,
            group_state,
        ],
    )

    clear_selection_button.click(
        fn=clear_structure_selection,
        inputs=[group_state],
        outputs=[
            structure_selector_image,
            structure_group_table,
            structure_group_selector,
            pattern1_clusters,
            selection_summary,
            group_state,
        ],
    )

    structure_selector_image.select(
        fn=toggle_structure_group_from_selector,
        inputs=[group_state],
        outputs=[
            structure_selector_image,
            structure_group_table,
            structure_group_selector,
            pattern1_clusters,
            selection_summary,
            group_state,
        ],
        show_progress="hidden",
    )

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
