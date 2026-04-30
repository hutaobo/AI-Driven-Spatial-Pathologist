from __future__ import annotations

from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any
import csv
import json
import math
import re

from .schema import WorkflowConfig


PATHWAY_GENE_SETS: dict[str, set[str]] = {
    "epithelial_tumor": {"EPCAM", "KRT8", "KRT18", "KRT19", "KRT7", "MUC1", "ERBB2", "ESR1", "PGR"},
    "proliferation": {"MKI67", "TOP2A", "PCNA", "UBE2C", "BIRC5", "CCNB1"},
    "immune_inflammation": {"PTPRC", "CD3D", "CD3E", "MS4A1", "CD79A", "NKG7", "GNLY"},
    "macrophage_myeloid": {"CD68", "CD163", "LYZ", "LST1", "CSF1R", "AIF1"},
    "fibroblast_stroma": {"COL1A1", "COL1A2", "COL3A1", "DCN", "LUM", "CXCL12", "ACTA2"},
    "endothelial_angiogenesis": {"PECAM1", "VWF", "KDR", "FLT1", "ENG", "EMCN"},
    "myoepithelial_basal": {"KRT5", "KRT14", "ACTA2", "TP63", "MYL9", "TAGLN"},
    "interferon": {"ISG15", "IFIT1", "IFIT3", "MX1", "OAS1", "STAT1"},
    "hypoxia_stress": {"VEGFA", "CA9", "ENO1", "LDHA", "ALDOA", "NDRG1"},
}


def foundation_evidence_requested(cfg: WorkflowConfig) -> bool:
    return bool(cfg.rna_foundation_enabled or cfg.pathway_activity_enabled or cfg.niche_fusion_enabled)


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    return path


def _write_csv(path: Path, rows: list[dict[str, Any]], *, fieldnames: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(fieldnames or [])
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in headers})
    return path


def _read_rows(path: Path) -> list[dict[str, str]]:
    delimiter = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        return [dict(row) for row in reader]


def _load_json(path: str | Path | None, default: Any) -> Any:
    if path is None:
        return default
    resolved = Path(path)
    if not resolved.exists():
        return default
    return json.loads(resolved.read_text(encoding="utf-8"))


def _pick_column(keys: list[str], candidates: list[str]) -> str | None:
    normalized = {key.lower().strip().replace(" ", "_").replace("-", "_"): key for key in keys}
    for candidate in candidates:
        found = normalized.get(candidate.lower().strip().replace(" ", "_").replace("-", "_"))
        if found is not None:
            return found
    return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return str(int(float(text)))
    except ValueError:
        return text


def _parse_json_cell(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _entropy(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0 or len(counts) <= 1:
        return 0.0
    raw = 0.0
    for count in counts.values():
        probability = count / total
        if probability > 0:
            raw -= probability * math.log(probability)
    return raw / math.log(len(counts))


def _foundation_paths(foundation_dir: Path) -> dict[str, Path]:
    return {
        "rna_foundation_cluster_summary_csv": foundation_dir / "rna_foundation_cluster_summary.csv",
        "rna_foundation_cluster_summary_json": foundation_dir / "rna_foundation_cluster_summary.json",
        "rna_foundation_structure_summary_csv": foundation_dir / "rna_foundation_structure_summary.csv",
        "rna_foundation_structure_summary_json": foundation_dir / "rna_foundation_structure_summary.json",
        "pathway_activity_cluster_summary_csv": foundation_dir / "pathway_activity_cluster_summary.csv",
        "pathway_activity_cluster_summary_json": foundation_dir / "pathway_activity_cluster_summary.json",
        "pathway_activity_structure_summary_csv": foundation_dir / "pathway_activity_structure_summary.csv",
        "pathway_activity_structure_summary_json": foundation_dir / "pathway_activity_structure_summary.json",
        "he_morphology_feature_summary_csv": foundation_dir / "he_morphology_feature_summary.csv",
        "he_morphology_feature_summary_json": foundation_dir / "he_morphology_feature_summary.json",
        "niche_fusion_summary_csv": foundation_dir / "niche_fusion_summary.csv",
        "niche_fusion_summary_json": foundation_dir / "niche_fusion_summary.json",
        "metadata_json": foundation_dir / "foundation_evidence_metadata.json",
    }


def _load_structure_by_cluster(pathology_outputs: dict[str, Any]) -> dict[str, str]:
    reviews = _load_json(pathology_outputs.get("structure_reviews_json"), [])
    best: dict[str, tuple[str, float]] = {}
    if not isinstance(reviews, list):
        return {}
    for review in reviews:
        if not isinstance(review, dict):
            continue
        structure_id = _id(review.get("structure_id"))
        if structure_id is None:
            continue
        for item in review.get("top_clusters", []) or []:
            if not isinstance(item, dict):
                continue
            raw_cluster_id = None
            for key in ("cluster_id", "cluster", "id", "label"):
                if item.get(key) not in (None, ""):
                    raw_cluster_id = item.get(key)
                    break
            cluster_id = _id(raw_cluster_id)
            if cluster_id is None:
                continue
            weight = max(
                _safe_float(item.get("cell_count")),
                _safe_float(item.get("count")),
                _safe_float(item.get("fraction")),
                _safe_float(item.get("fraction_of_structure")),
            )
            if cluster_id not in best or weight > best[cluster_id][1]:
                best[cluster_id] = (structure_id, weight)
    return {cluster_id: structure_id for cluster_id, (structure_id, _) in best.items()}


def _normalize_cluster_summary_rows(rows: list[dict[str, str]], source_path: Path) -> list[dict[str, Any]]:
    if not rows:
        return []
    keys = list(rows[0].keys())
    cluster_col = _pick_column(keys, ["cluster_id", "cluster", "group", "group_id"])
    label_col = _pick_column(keys, ["top_reference_label", "reference_label", "predicted_label", "cell_type", "label"])
    confidence_col = _pick_column(keys, ["mean_reference_confidence", "reference_confidence", "confidence", "score", "probability"])
    fraction_col = _pick_column(keys, ["top_reference_fraction", "reference_fraction", "fraction", "proportion"])
    count_col = _pick_column(keys, ["n_cells", "cell_count", "count", "n"])
    normalized: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        cluster_id = _id(row.get(cluster_col)) if cluster_col else str(index)
        label = str(row.get(label_col, "") or "").strip() if label_col else ""
        normalized.append(
            {
                **row,
                "cluster_id": cluster_id,
                "n_cells": _safe_int(row.get(count_col), 1) if count_col else 1,
                "top_reference_label": label or "unknown",
                "top_reference_fraction": _safe_float(row.get(fraction_col), 1.0) if fraction_col else 1.0,
                "mean_reference_confidence": _safe_float(row.get(confidence_col), 0.0) if confidence_col else 0.0,
                "ambiguous_fraction": _safe_float(row.get("ambiguous_fraction"), 0.0),
                "label_distribution_json": row.get("label_distribution_json") or json.dumps({label or "unknown": 1}),
                "source": str(source_path),
            }
        )
    return normalized


def _aggregate_cell_mapping_to_clusters(rows: list[dict[str, str]], source_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    if not rows:
        return [], ["RNA foundation cell mapping was empty."]
    keys = list(rows[0].keys())
    cluster_col = _pick_column(keys, ["cluster_id", "cluster", "group", "group_id", "leiden", "louvain"])
    label_col = _pick_column(keys, ["predicted_label", "reference_label", "cell_type", "label", "celltype"])
    confidence_col = _pick_column(keys, ["confidence", "score", "probability", "max_probability"])
    if cluster_col is None:
        return [], ["RNA foundation cell mapping is missing a cluster_id/cluster column."]
    if label_col is None:
        return [], ["RNA foundation cell mapping is missing a predicted/reference label column."]

    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "labels": Counter(), "confidences": [], "ambiguous": 0}
    )
    for row in rows:
        cluster_id = _id(row.get(cluster_col))
        if cluster_id is None:
            continue
        label = str(row.get(label_col) or "unknown").strip() or "unknown"
        normalized_label = label.lower()
        grouped[cluster_id]["n"] += 1
        grouped[cluster_id]["labels"][label] += 1
        if normalized_label in {"unknown", "ambiguous", "unassigned", "other"}:
            grouped[cluster_id]["ambiguous"] += 1
        if confidence_col is not None:
            grouped[cluster_id]["confidences"].append(_safe_float(row.get(confidence_col), 0.0))

    cluster_rows: list[dict[str, Any]] = []
    for cluster_id, payload in sorted(grouped.items(), key=lambda item: item[0]):
        n_cells = int(payload["n"])
        top_label, top_count = payload["labels"].most_common(1)[0]
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "n_cells": n_cells,
                "top_reference_label": top_label,
                "top_reference_fraction": top_count / max(n_cells, 1),
                "mean_reference_confidence": _mean(payload["confidences"]),
                "ambiguous_fraction": payload["ambiguous"] / max(n_cells, 1),
                "label_distribution_json": json.dumps(dict(payload["labels"]), ensure_ascii=False, sort_keys=True),
                "source": str(source_path),
            }
        )
    return cluster_rows, warnings


def _aggregate_clusters_to_structures(
    cluster_rows: list[dict[str, Any]],
    structure_by_cluster: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"n": 0, "labels": Counter(), "weighted_confidence": 0.0, "weighted_ambiguous": 0.0, "clusters": []}
    )
    for row in cluster_rows:
        cluster_id = _id(row.get("cluster_id"))
        if cluster_id is None:
            continue
        structure_id = structure_by_cluster.get(cluster_id)
        if structure_id is None:
            continue
        n_cells = max(_safe_int(row.get("n_cells"), 1), 1)
        label = str(row.get("top_reference_label") or "unknown")
        grouped[structure_id]["n"] += n_cells
        grouped[structure_id]["labels"][label] += int(round(n_cells * _safe_float(row.get("top_reference_fraction"), 1.0)))
        grouped[structure_id]["weighted_confidence"] += n_cells * _safe_float(row.get("mean_reference_confidence"), 0.0)
        grouped[structure_id]["weighted_ambiguous"] += n_cells * _safe_float(row.get("ambiguous_fraction"), 0.0)
        grouped[structure_id]["clusters"].append(cluster_id)

    structure_rows: list[dict[str, Any]] = []
    for structure_id, payload in sorted(grouped.items(), key=lambda item: item[0]):
        total = max(int(payload["n"]), 1)
        top_label, top_count = payload["labels"].most_common(1)[0] if payload["labels"] else ("unknown", 0)
        structure_rows.append(
            {
                "structure_id": structure_id,
                "n_cells": total,
                "top_reference_label": top_label,
                "top_reference_fraction": top_count / total,
                "mean_reference_confidence": payload["weighted_confidence"] / total,
                "ambiguous_fraction": payload["weighted_ambiguous"] / total,
                "clusters_json": json.dumps(payload["clusters"], ensure_ascii=False),
                "label_distribution_json": json.dumps(dict(payload["labels"]), ensure_ascii=False, sort_keys=True),
            }
        )
    return structure_rows


def _build_rna_foundation_outputs(
    cfg: WorkflowConfig,
    paths: dict[str, Path],
    structure_by_cluster: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    cluster_rows: list[dict[str, Any]] = []
    if cfg.rna_foundation_enabled:
        if cfg.rna_foundation_cluster_summary_path is not None and cfg.rna_foundation_cluster_summary_path.exists():
            cluster_rows = _normalize_cluster_summary_rows(
                _read_rows(cfg.rna_foundation_cluster_summary_path),
                cfg.rna_foundation_cluster_summary_path,
            )
        elif cfg.rna_foundation_cell_mapping_path is not None and cfg.rna_foundation_cell_mapping_path.exists():
            cluster_rows, warnings = _aggregate_cell_mapping_to_clusters(
                _read_rows(cfg.rna_foundation_cell_mapping_path),
                cfg.rna_foundation_cell_mapping_path,
            )
        else:
            warnings.append(
                "RNA foundation evidence is enabled, but no existing cell mapping or cluster summary path was provided."
            )
    structure_rows = _aggregate_clusters_to_structures(cluster_rows, structure_by_cluster)
    _write_csv(paths["rna_foundation_cluster_summary_csv"], cluster_rows)
    _write_json(paths["rna_foundation_cluster_summary_json"], cluster_rows)
    _write_csv(paths["rna_foundation_structure_summary_csv"], structure_rows)
    _write_json(paths["rna_foundation_structure_summary_json"], structure_rows)
    return cluster_rows, structure_rows, warnings


def _score_diffexp_pathways(diffexp_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _read_rows(diffexp_path)
    warnings: list[str] = []
    if not rows:
        return [], ["Differential-expression table was empty."]
    keys = list(rows[0].keys())
    cluster_col = _pick_column(keys, ["cluster_id", "cluster", "group", "group_id"])
    gene_col = _pick_column(keys, ["gene", "genes", "feature", "feature_name", "symbol"])
    score_col = _pick_column(
        keys,
        [
            "log2fc",
            "log2_fold_change",
            "log_fold_change",
            "logfoldchanges",
            "avg_log2fc",
            "score",
            "statistic",
        ],
    )
    if gene_col is not None and (cluster_col is None or score_col is None):
        wide_score_cols: list[tuple[str, str]] = []
        for key in keys:
            match = re.match(r"Cluster\s+(.+?)\s+Log2\s+fold\s+change$", key, flags=re.IGNORECASE)
            if match:
                wide_score_cols.append((_id(match.group(1)) or match.group(1), key))
        if wide_score_cols:
            grouped_wide: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
            for row in rows:
                gene = str(row.get(gene_col) or "").strip().upper()
                if not gene:
                    continue
                for pathway, genes in PATHWAY_GENE_SETS.items():
                    if gene not in genes:
                        continue
                    for cluster_id, column in wide_score_cols:
                        grouped_wide[(cluster_id, pathway)].append((gene, _safe_float(row.get(column), 0.0)))
            scored_wide: list[dict[str, Any]] = []
            for (cluster_id, pathway), hits in sorted(grouped_wide.items(), key=lambda item: item[0]):
                scores = [score for _, score in hits]
                scored_wide.append(
                    {
                        "cluster_id": cluster_id,
                        "pathway": pathway,
                        "score": _mean(scores),
                        "n_genes": len({gene for gene, _ in hits}),
                        "matched_genes": ";".join(sorted({gene for gene, _ in hits})),
                        "source": str(diffexp_path),
                    }
                )
            if not scored_wide:
                warnings.append("No built-in pathway marker genes were found in the wide differential-expression table.")
            return scored_wide, warnings
    if cluster_col is None or gene_col is None or score_col is None:
        return [], ["Differential-expression table must contain cluster, gene, and score/log2FC columns."]
    grouped: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    for row in rows:
        cluster_id = _id(row.get(cluster_col))
        if cluster_id is None:
            continue
        gene = str(row.get(gene_col) or "").strip().upper()
        if not gene:
            continue
        score = _safe_float(row.get(score_col), 0.0)
        for pathway, genes in PATHWAY_GENE_SETS.items():
            if gene in genes:
                grouped[(cluster_id, pathway)].append((gene, score))

    scored: list[dict[str, Any]] = []
    for (cluster_id, pathway), hits in sorted(grouped.items(), key=lambda item: item[0]):
        scores = [score for _, score in hits]
        scored.append(
            {
                "cluster_id": cluster_id,
                "pathway": pathway,
                "score": _mean(scores),
                "n_genes": len({gene for gene, _ in hits}),
                "matched_genes": ";".join(sorted({gene for gene, _ in hits})),
                "source": str(diffexp_path),
            }
        )
    if not scored:
        warnings.append("No built-in pathway marker genes were found in the differential-expression table.")
    return scored, warnings


def _aggregate_pathways_to_structures(
    cluster_rows: list[dict[str, Any]],
    structure_by_cluster: dict[str, str],
    direct_structure_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    grouped_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in direct_structure_rows or []:
        structure_id = _id(row.get("structure_id"))
        pathway = str(row.get("pathway") or "").strip()
        if structure_id and pathway:
            grouped_scores[structure_id][pathway].append(_safe_float(row.get("score"), 0.0))
    for row in cluster_rows:
        cluster_id = _id(row.get("cluster_id"))
        pathway = str(row.get("pathway") or "").strip()
        structure_id = structure_by_cluster.get(cluster_id or "")
        if structure_id and pathway:
            grouped_scores[structure_id][pathway].append(_safe_float(row.get("score"), 0.0))

    structure_rows: list[dict[str, Any]] = []
    for structure_id, pathway_scores in sorted(grouped_scores.items(), key=lambda item: item[0]):
        averaged = {pathway: _mean(scores) for pathway, scores in pathway_scores.items()}
        activated = sorted(
            ((pathway, score) for pathway, score in averaged.items() if score > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        suppressed = sorted(
            ((pathway, score) for pathway, score in averaged.items() if score < 0),
            key=lambda item: item[1],
        )[:5]
        scores = list(averaged.values())
        structure_rows.append(
            {
                "structure_id": structure_id,
                "top_activated_pathways": ";".join(f"{name}:{score:.3f}" for name, score in activated),
                "top_suppressed_pathways": ";".join(f"{name}:{score:.3f}" for name, score in suppressed),
                "pathway_separation_score": (max(scores) - min(scores)) if scores else 0.0,
                "pathway_scores_json": json.dumps(averaged, ensure_ascii=False, sort_keys=True),
            }
        )
    return structure_rows


def _build_pathway_outputs(
    cfg: WorkflowConfig,
    paths: dict[str, Path],
    structure_by_cluster: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    warnings: list[str] = []
    cluster_rows: list[dict[str, Any]] = []
    direct_structure_rows: list[dict[str, Any]] = []
    if cfg.pathway_activity_enabled:
        if cfg.pathway_activity_csv is not None and cfg.pathway_activity_csv.exists():
            rows = _read_rows(cfg.pathway_activity_csv)
            if rows:
                keys = list(rows[0].keys())
                structure_col = _pick_column(keys, ["structure_id", "structure", "region_id"])
                cluster_col = _pick_column(keys, ["cluster_id", "cluster", "group", "group_id"])
                pathway_col = _pick_column(keys, ["pathway", "gene_set", "signature", "term"])
                score_col = _pick_column(keys, ["score", "activity", "zscore", "z_score", "mean_score"])
                if pathway_col is None or score_col is None:
                    warnings.append("Pathway activity CSV is missing pathway and score columns.")
                else:
                    for row in rows:
                        pathway = str(row.get(pathway_col) or "").strip()
                        score = _safe_float(row.get(score_col), 0.0)
                        if structure_col is not None and row.get(structure_col):
                            direct_structure_rows.append(
                                {"structure_id": _id(row.get(structure_col)), "pathway": pathway, "score": score, "source": str(cfg.pathway_activity_csv)}
                            )
                        elif cluster_col is not None and row.get(cluster_col):
                            cluster_rows.append(
                                {"cluster_id": _id(row.get(cluster_col)), "pathway": pathway, "score": score, "source": str(cfg.pathway_activity_csv)}
                            )
            else:
                warnings.append("Pathway activity CSV was empty.")
        elif cfg.differential_expression_csv is not None and cfg.differential_expression_csv.exists():
            cluster_rows, warnings = _score_diffexp_pathways(cfg.differential_expression_csv)
        else:
            warnings.append("Pathway activity is enabled, but neither pathway_activity_csv nor differential_expression_csv exists.")
    structure_rows = _aggregate_pathways_to_structures(cluster_rows, structure_by_cluster, direct_structure_rows)
    _write_csv(paths["pathway_activity_cluster_summary_csv"], cluster_rows)
    _write_json(paths["pathway_activity_cluster_summary_json"], cluster_rows)
    _write_csv(paths["pathway_activity_structure_summary_csv"], structure_rows)
    _write_json(paths["pathway_activity_structure_summary_json"], structure_rows)
    return cluster_rows, structure_rows, warnings


def _signal_bucket(label: str) -> str | None:
    normalized = label.lower()
    if any(token in normalized for token in ("artifact", "blur", "empty", "fold", "low_quality", "background")):
        return "artifact_signal"
    if any(token in normalized for token in ("tumor", "carcinoma", "invasive", "dcis", "epithel")):
        return "tumor_signal"
    if any(token in normalized for token in ("immune", "inflamm", "macrophage", "lymphocyte", "plasma")):
        return "inflammation_signal"
    if any(token in normalized for token in ("stroma", "fibro", "collagen", "vascular", "endothel")):
        return "stroma_signal"
    return None


def _he_class_scores(row: dict[str, str]) -> dict[str, float]:
    signals = {
        "artifact_signal": 0.0,
        "tumor_signal": 0.0,
        "inflammation_signal": 0.0,
        "stroma_signal": 0.0,
    }
    top_label = str(row.get("top_label") or row.get("top_label_id") or row.get("foundation_top_label") or "")
    top_score = _safe_float(row.get("top_score") or row.get("foundation_top_score"), 0.0)
    bucket = _signal_bucket(top_label)
    if bucket:
        signals[bucket] = max(signals[bucket], top_score)
    classes = _parse_json_cell(row.get("top_classes_json") or row.get("top_classes"), [])
    if isinstance(classes, list):
        for item in classes:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label") or item.get("label_id") or item.get("id") or "")
            bucket = _signal_bucket(label)
            if bucket:
                signals[bucket] = max(signals[bucket], _safe_float(item.get("score"), 0.0))
    return signals


def _build_he_morphology_outputs(workflow_summary: dict[str, Any], paths: dict[str, Path]) -> tuple[list[dict[str, Any]], list[str]]:
    he_outputs = workflow_summary.get("he_foundation_outputs", {})
    classification_path = he_outputs.get("classification_csv")
    warnings: list[str] = []
    rows: list[dict[str, str]] = []
    if classification_path and Path(classification_path).exists():
        rows = _read_rows(Path(classification_path))
    elif he_outputs:
        warnings.append("H&E foundation outputs exist, but classification_csv was not found.")
    else:
        warnings.append("H&E morphology summary skipped because no H&E foundation outputs were present.")

    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "scores": [],
            "labels": Counter(),
            "artifact_signal": [],
            "tumor_signal": [],
            "inflammation_signal": [],
            "stroma_signal": [],
        }
    )
    for row in rows:
        structure_id = _id(row.get("structure_id"))
        if structure_id is None:
            continue
        label = str(row.get("top_label") or row.get("top_label_id") or "unknown")
        score = _safe_float(row.get("top_score"), 0.0)
        grouped[structure_id]["scores"].append(score)
        grouped[structure_id]["labels"][label] += 1
        for key, value in _he_class_scores(row).items():
            grouped[structure_id][key].append(value)

    summary_rows: list[dict[str, Any]] = []
    for structure_id, payload in sorted(grouped.items(), key=lambda item: item[0]):
        label, count = payload["labels"].most_common(1)[0] if payload["labels"] else ("unknown", 0)
        total = max(sum(payload["labels"].values()), 1)
        signal_means = {
            key: _mean(payload[key])
            for key in ("artifact_signal", "tumor_signal", "inflammation_signal", "stroma_signal")
        }
        dominant_signal = max(signal_means.items(), key=lambda item: item[1])[0].replace("_signal", "")
        summary_rows.append(
            {
                "structure_id": structure_id,
                "n_contours": total,
                "dominant_visual_label": label,
                "dominant_visual_fraction": count / total,
                "mean_foundation_score": _mean(payload["scores"]),
                **signal_means,
                "dominant_signal": dominant_signal,
                "visual_entropy": _entropy(payload["labels"]),
                "label_distribution_json": json.dumps(dict(payload["labels"]), ensure_ascii=False, sort_keys=True),
            }
        )
    _write_csv(paths["he_morphology_feature_summary_csv"], summary_rows)
    _write_json(paths["he_morphology_feature_summary_json"], summary_rows)
    return summary_rows, warnings


def _row_by_structure(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        structure_id = _id(row.get("structure_id"))
        if structure_id is not None:
            out[structure_id] = row
    return out


def _top_pathways(row: dict[str, Any]) -> list[str]:
    raw = str(row.get("top_activated_pathways") or "")
    return [item.split(":", 1)[0] for item in raw.split(";") if item.strip()][:3]


def _niche_consistency(
    *,
    rna: dict[str, Any] | None,
    pathway: dict[str, Any] | None,
    he: dict[str, Any] | None,
) -> str:
    notes: list[str] = []
    rna_label = str((rna or {}).get("top_reference_label") or "").lower()
    he_signal = str((he or {}).get("dominant_signal") or "").lower()
    pathways = " ".join(_top_pathways(pathway or {})).lower()
    if rna_label and he_signal:
        if "tumor" in rna_label and he_signal == "tumor":
            notes.append("RNA reference and H&E morphology both support tumor/epithelial identity")
        elif any(token in rna_label for token in ("fibro", "stroma")) and he_signal == "stroma":
            notes.append("RNA reference and H&E morphology both support stromal identity")
        elif he_signal == "inflammation":
            notes.append("H&E adds inflammatory context that may complement RNA/cell-type evidence")
        elif he_signal == "artifact":
            notes.append("H&E morphology raises a tissue-quality caveat for this structure")
    if pathways:
        notes.append(f"pathway activity adds molecular program context: {', '.join(_top_pathways(pathway or {}))}")
    return "; ".join(notes) if notes else "No strong cross-modal concordance or conflict detected"


def _build_niche_fusion_outputs(
    workflow_summary: dict[str, Any],
    paths: dict[str, Path],
    *,
    rna_structure_rows: list[dict[str, Any]],
    pathway_structure_rows: list[dict[str, Any]],
    he_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    structure_reviews = _load_json(workflow_summary.get("pathology_outputs", {}).get("structure_reviews_json"), [])
    review_by_id = _row_by_structure(structure_reviews if isinstance(structure_reviews, list) else [])
    rna_by_id = _row_by_structure(rna_structure_rows)
    pathway_by_id = _row_by_structure(pathway_structure_rows)
    he_by_id = _row_by_structure(he_rows)
    structure_ids = sorted(set(review_by_id) | set(rna_by_id) | set(pathway_by_id) | set(he_by_id), key=lambda value: (len(value), value))

    rows: list[dict[str, Any]] = []
    for structure_id in structure_ids:
        review = review_by_id.get(structure_id, {})
        rna = rna_by_id.get(structure_id)
        pathway = pathway_by_id.get(structure_id)
        he = he_by_id.get(structure_id)
        modalities = [name for name, value in (("rna_foundation", rna), ("pathway", pathway), ("he_morphology", he)) if value]
        rows.append(
            {
                "structure_id": structure_id,
                "niche_id": f"N{structure_id}",
                "current_name": review.get("title") or review.get("assigned_label") or "",
                "rna_top_reference_label": (rna or {}).get("top_reference_label", ""),
                "rna_reference_confidence": (rna or {}).get("mean_reference_confidence", ""),
                "top_activated_pathways": (pathway or {}).get("top_activated_pathways", ""),
                "pathway_separation_score": (pathway or {}).get("pathway_separation_score", ""),
                "he_dominant_signal": (he or {}).get("dominant_signal", ""),
                "he_visual_entropy": (he or {}).get("visual_entropy", ""),
                "evidence_modality_count": len(modalities),
                "evidence_modalities_json": json.dumps(modalities, ensure_ascii=False),
                "consistency_summary": _niche_consistency(rna=rna, pathway=pathway, he=he),
                "evidence_json": json.dumps(
                    {"rna_foundation": rna, "pathway_activity": pathway, "he_morphology": he},
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                ),
            }
        )
    _write_csv(paths["niche_fusion_summary_csv"], rows)
    _write_json(paths["niche_fusion_summary_json"], rows)
    return rows


def _foundation_by_structure(
    *,
    rna_structure_rows: list[dict[str, Any]],
    pathway_structure_rows: list[dict[str, Any]],
    he_rows: list[dict[str, Any]],
    niche_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rna_by_id = _row_by_structure(rna_structure_rows)
    pathway_by_id = _row_by_structure(pathway_structure_rows)
    he_by_id = _row_by_structure(he_rows)
    niche_by_id = _row_by_structure(niche_rows)
    ids = set(rna_by_id) | set(pathway_by_id) | set(he_by_id) | set(niche_by_id)
    return {
        structure_id: {
            "rna_foundation": rna_by_id.get(structure_id),
            "pathway_activity": pathway_by_id.get(structure_id),
            "he_morphology": he_by_id.get(structure_id),
            "niche_fusion": niche_by_id.get(structure_id),
        }
        for structure_id in ids
    }


def _foundation_lines(evidence: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    rna = evidence.get("rna_foundation") or {}
    pathway = evidence.get("pathway_activity") or {}
    he = evidence.get("he_morphology") or {}
    niche = evidence.get("niche_fusion") or {}
    if rna:
        lines.append(
            "RNA foundation/reference mapping: "
            f"{rna.get('top_reference_label', 'unknown')} "
            f"(confidence {float(rna.get('mean_reference_confidence') or 0.0):.2f}, "
            f"ambiguous {float(rna.get('ambiguous_fraction') or 0.0):.2f})."
        )
    if pathway:
        lines.append(f"Pathway activity: {pathway.get('top_activated_pathways') or 'no dominant activated pathway'}.")
    if he:
        lines.append(
            "H&E morphology foundation signal: "
            f"{he.get('dominant_signal', 'unknown')} via {he.get('dominant_visual_label', 'unknown')} "
            f"(entropy {float(he.get('visual_entropy') or 0.0):.2f})."
        )
    if niche:
        lines.append(f"Cross-modal consistency: {niche.get('consistency_summary')}.")
    return lines


def _update_pathology_reviews(
    workflow_summary: dict[str, Any],
    foundation_map: dict[str, dict[str, Any]],
) -> None:
    pathology_outputs = workflow_summary.get("pathology_outputs", {})
    structure_reviews_path = Path(pathology_outputs.get("structure_reviews_json", ""))
    case_summary_path = Path(pathology_outputs.get("case_summary_json", ""))
    if not structure_reviews_path.exists():
        return
    reviews = _load_json(structure_reviews_path, [])
    if not isinstance(reviews, list):
        return
    for review in reviews:
        if not isinstance(review, dict):
            continue
        structure_id = _id(review.get("structure_id"))
        if structure_id is None or structure_id not in foundation_map:
            continue
        evidence = foundation_map[structure_id]
        review["foundation_evidence"] = evidence
        review["key_evidence"] = (_foundation_lines(evidence) + list(review.get("key_evidence", [])))[:14]
    structure_reviews_path.write_text(json.dumps(reviews, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_csv(structure_reviews_path.with_suffix(".csv"), reviews)

    if case_summary_path.exists():
        case_summary = _load_json(case_summary_path, {})
        if isinstance(case_summary, dict):
            message = (
                "Foundation evidence layer aggregated RNA reference mapping, pathway activity, "
                "H&E morphology signals, and lightweight niche fusion before final review."
            )
            case_summary["key_findings"] = ([message] + list(case_summary.get("key_findings", [])))[:10]
            case_summary_path.write_text(json.dumps(case_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_foundation_section(niche_rows: list[dict[str, Any]], outputs: dict[str, str]) -> str:
    rows_html = []
    for row in niche_rows[:12]:
        rows_html.append(
            "<tr>"
            f"<td>{escape(str(row.get('structure_id', '')))}</td>"
            f"<td>{escape(str(row.get('current_name', '')))}</td>"
            f"<td>{escape(str(row.get('rna_top_reference_label', '')))}</td>"
            f"<td>{escape(str(row.get('top_activated_pathways', '')))}</td>"
            f"<td>{escape(str(row.get('he_dominant_signal', '')))}</td>"
            f"<td>{escape(str(row.get('consistency_summary', '')))}</td>"
            "</tr>"
        )
    if not rows_html:
        rows_html.append("<tr><td colspan=\"6\">No structure-level foundation evidence was available.</td></tr>")
    links = [
        ("RNA foundation cluster summary", outputs.get("rna_foundation_cluster_summary_csv")),
        ("Pathway activity summary", outputs.get("pathway_activity_structure_summary_csv")),
        ("H&E morphology feature summary", outputs.get("he_morphology_feature_summary_csv")),
        ("Niche fusion summary", outputs.get("niche_fusion_summary_csv")),
    ]
    link_html = " ".join(
        f"<a href=\"{escape(Path(path).name)}\">{escape(label)}</a>"
        for label, path in links
        if path
    )
    return (
        "<!-- spatho-foundation-evidence:start -->\n"
        "<section>\n"
        "<h2>Foundation Evidence</h2>\n"
        "<p>This section separates marker heuristic evidence, scGPT-like RNA reference mapping, "
        "pathway activity, PLIP H&E morphology evidence, and final LLM adjudication. "
        "The v1 fusion layer is frozen-feature and audit-oriented; it does not train a new "
        "SpatialFusion-style joint embedding model.</p>\n"
        f"<p>{link_html}</p>\n"
        "<table><thead><tr><th>Structure</th><th>Current name</th><th>RNA reference</th>"
        "<th>Pathway</th><th>H&E signal</th><th>Consistency</th></tr></thead><tbody>\n"
        + "\n".join(rows_html)
        + "\n</tbody></table>\n"
        "</section>\n"
        "<!-- spatho-foundation-evidence:end -->"
    )


def _insert_foundation_report_section(report_path: str | Path | None, niche_rows: list[dict[str, Any]], outputs: dict[str, str]) -> None:
    if not report_path:
        return
    path = Path(report_path)
    if not path.exists():
        return
    html = path.read_text(encoding="utf-8")
    start = "<!-- spatho-foundation-evidence:start -->"
    end = "<!-- spatho-foundation-evidence:end -->"
    if start in html and end in html:
        before, rest = html.split(start, 1)
        _, after = rest.split(end, 1)
        html = before + after
    section = _render_foundation_section(niche_rows, outputs)
    if "</main>" in html:
        html = html.replace("</main>", section + "\n</main>", 1)
    else:
        html = html + "\n" + section + "\n"
    path.write_text(html, encoding="utf-8")


def apply_foundation_evidence(
    cfg: WorkflowConfig,
    workflow_result: dict[str, str],
) -> dict[str, str]:
    workflow_summary_path = Path(workflow_result["workflow_summary_json"]).resolve()
    workflow_summary = json.loads(workflow_summary_path.read_text(encoding="utf-8"))
    output_root = Path(workflow_summary["output_root"]).resolve()
    foundation_dir = output_root / "foundation"
    foundation_dir.mkdir(parents=True, exist_ok=True)
    paths = _foundation_paths(foundation_dir)
    structure_by_cluster = _load_structure_by_cluster(workflow_summary.get("pathology_outputs", {}))

    rna_cluster_rows, rna_structure_rows, rna_warnings = _build_rna_foundation_outputs(cfg, paths, structure_by_cluster)
    pathway_cluster_rows, pathway_structure_rows, pathway_warnings = _build_pathway_outputs(cfg, paths, structure_by_cluster)
    he_rows, he_warnings = _build_he_morphology_outputs(workflow_summary, paths)
    if cfg.niche_fusion_enabled:
        niche_rows = _build_niche_fusion_outputs(
            workflow_summary,
            paths,
            rna_structure_rows=rna_structure_rows,
            pathway_structure_rows=pathway_structure_rows,
            he_rows=he_rows,
        )
    else:
        niche_rows = []
        _write_csv(paths["niche_fusion_summary_csv"], niche_rows)
        _write_json(paths["niche_fusion_summary_json"], niche_rows)
    foundation_map = _foundation_by_structure(
        rna_structure_rows=rna_structure_rows,
        pathway_structure_rows=pathway_structure_rows,
        he_rows=he_rows,
        niche_rows=niche_rows,
    )
    _update_pathology_reviews(workflow_summary, foundation_map)

    outputs = {
        "enabled": True,
        "foundation_dir": str(foundation_dir),
        **{key: str(path) for key, path in paths.items()},
    }
    metadata = {
        "enabled": True,
        "rna_foundation_enabled": cfg.rna_foundation_enabled,
        "rna_foundation_backend": cfg.rna_foundation_backend,
        "rna_foundation_cell_mapping_path": str(cfg.rna_foundation_cell_mapping_path) if cfg.rna_foundation_cell_mapping_path else None,
        "rna_foundation_cluster_summary_path": str(cfg.rna_foundation_cluster_summary_path) if cfg.rna_foundation_cluster_summary_path else None,
        "pathway_activity_enabled": cfg.pathway_activity_enabled,
        "pathway_activity_csv": str(cfg.pathway_activity_csv) if cfg.pathway_activity_csv else None,
        "niche_fusion_enabled": cfg.niche_fusion_enabled,
        "niche_fusion_backend": cfg.niche_fusion_backend,
        "he_contour_foundation_enabled": cfg.he_contour_foundation_enabled,
        "cluster_count_with_rna_foundation": len(rna_cluster_rows),
        "structure_count_with_rna_foundation": len(rna_structure_rows),
        "cluster_count_with_pathway_activity": len(pathway_cluster_rows),
        "structure_count_with_pathway_activity": len(pathway_structure_rows),
        "structure_count_with_he_morphology": len(he_rows),
        "structure_count_with_niche_fusion": len(niche_rows),
        "warnings": rna_warnings + pathway_warnings + he_warnings,
    }
    _write_json(paths["metadata_json"], metadata)
    workflow_summary["foundation_outputs"] = outputs
    workflow_summary_path.write_text(json.dumps(workflow_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _insert_foundation_report_section(
        workflow_summary.get("pathology_outputs", {}).get("report_html"),
        niche_rows,
        outputs,
    )
    return {
        **workflow_result,
        "foundation_dir": str(foundation_dir),
        "niche_fusion_summary_csv": str(paths["niche_fusion_summary_csv"]),
        "workflow_summary_json": str(workflow_summary_path),
    }
