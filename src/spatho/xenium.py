from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence
import csv
import json


DEFAULT_XENIUM_PIXEL_SIZE_UM = 0.2125
DATASET_MODALITY_XENIUM_RNA_PROTEIN = "xenium_rna_protein"
CANONICAL_SPACE_PHYSICAL_UM = "physical_um"
EXPORT_SPACE_XENIUM_EXPLORER_PIXEL = "xenium_explorer_pixel"
VALID_SEGMENTATION_SOURCES = (
    "ranger_protein_assisted",
    "ranger_default",
    "third_party_import",
)
DEFAULT_PROTEIN_PREFIXES = (
    "PROTEIN_",
    "PROTEIN:",
    "ADT_",
    "ADT:",
    "AB_",
    "AB:",
)


def _coerce_positive_float(value: float | int | str, *, label: str) -> float:
    numeric = float(value)
    if numeric <= 0:
        raise ValueError(f"{label} must be > 0, got {value!r}")
    return numeric


def validate_segmentation_source(value: str) -> str:
    normalized = str(value).strip()
    if normalized not in VALID_SEGMENTATION_SOURCES:
        supported = ", ".join(VALID_SEGMENTATION_SOURCES)
        raise ValueError(f"segmentation_source must be one of: {supported}")
    return normalized


def resolve_xenium_pixel_size_um(
    *,
    metadata_pixel_size_um: float | int | str | None = None,
    fallback_pixel_size_um: float | int | str = DEFAULT_XENIUM_PIXEL_SIZE_UM,
) -> tuple[float, str]:
    if metadata_pixel_size_um is not None:
        return _coerce_positive_float(metadata_pixel_size_um, label="metadata_pixel_size_um"), "metadata"
    return _coerce_positive_float(fallback_pixel_size_um, label="fallback_pixel_size_um"), "fallback"


def um_to_xenium_explorer_pixels(
    coords: Sequence[Sequence[float]],
    *,
    pixel_size_um: float = DEFAULT_XENIUM_PIXEL_SIZE_UM,
) -> list[tuple[float, float]]:
    scale = 1.0 / _coerce_positive_float(pixel_size_um, label="pixel_size_um")
    return [(float(x) * scale, float(y) * scale) for x, y in coords]


def xenium_explorer_pixels_to_um(
    coords: Sequence[Sequence[float]],
    *,
    pixel_size_um: float = DEFAULT_XENIUM_PIXEL_SIZE_UM,
) -> list[tuple[float, float]]:
    scale = _coerce_positive_float(pixel_size_um, label="pixel_size_um")
    return [(float(x) * scale, float(y) * scale) for x, y in coords]


def infer_feature_modalities(
    feature_names: Sequence[str],
    *,
    protein_feature_names: Iterable[str] | None = None,
    protein_prefixes: Sequence[str] = DEFAULT_PROTEIN_PREFIXES,
) -> list[str]:
    explicit = {str(item).upper() for item in (protein_feature_names or [])}
    prefixes = tuple(str(prefix).upper() for prefix in protein_prefixes)
    modalities: list[str] = []
    for raw_name in feature_names:
        name = str(raw_name)
        upper_name = name.upper()
        if upper_name in explicit or upper_name.startswith(prefixes):
            modalities.append("protein")
        else:
            modalities.append("rna")
    return modalities


def build_feature_records(
    feature_names: Sequence[str],
    *,
    protein_feature_names: Iterable[str] | None = None,
    protein_prefixes: Sequence[str] = DEFAULT_PROTEIN_PREFIXES,
) -> list[dict[str, str]]:
    modalities = infer_feature_modalities(
        feature_names,
        protein_feature_names=protein_feature_names,
        protein_prefixes=protein_prefixes,
    )
    return [
        {
            "feature_name": str(name),
            "feature_modality": modality,
        }
        for name, modality in zip(feature_names, modalities, strict=True)
    ]


@dataclass(frozen=True)
class XeniumAlignmentFixtureCase:
    case_id: str
    description: str
    source_space: str
    target_space: str
    segmentation_source: str
    transform_chain: tuple[dict[str, Any], ...]
    input_coordinates: tuple[tuple[str, float, float], ...]
    expected_output_coordinates: tuple[tuple[str, float, float], ...]
    axes: tuple[str, str] = ("x", "y")
    units: tuple[str, str] = ("um", "um")
    notes: tuple[str, ...] = ()

    def transform_payload(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "description": self.description,
            "source_space": self.source_space,
            "target_space": self.target_space,
            "axes": list(self.axes),
            "units": {
                "source": self.units[0],
                "target": self.units[1],
            },
            "segmentation_source": self.segmentation_source,
            "transform_chain": list(self.transform_chain),
        }


def _write_coordinate_csv(rows: Sequence[tuple[str, float, float]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "x", "y"])
        for identifier, x_value, y_value in rows:
            writer.writerow([identifier, f"{float(x_value):.6f}", f"{float(y_value):.6f}"])


def write_xenium_alignment_fixture_case(
    case: XeniumAlignmentFixtureCase,
    output_dir: str | Path,
) -> dict[str, str]:
    case_dir = Path(output_dir).resolve() / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    transform_path = case_dir / "transform.json"
    transform_path.write_text(
        json.dumps(case.transform_payload(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    input_path = case_dir / "input.csv"
    _write_coordinate_csv(case.input_coordinates, input_path)

    expected_path = case_dir / "expected_output.csv"
    _write_coordinate_csv(case.expected_output_coordinates, expected_path)

    readme_path = case_dir / "README.md"
    note_lines = "\n".join(f"- {note}" for note in case.notes) if case.notes else "- No extra notes."
    readme_path.write_text(
        "\n".join(
            [
                f"# {case.case_id}",
                "",
                case.description,
                "",
                f"- Source space: `{case.source_space}`",
                f"- Target space: `{case.target_space}`",
                f"- Segmentation source: `{case.segmentation_source}`",
                f"- Axes: `{case.axes[0]}`, `{case.axes[1]}`",
                f"- Units: `{case.units[0]}` -> `{case.units[1]}`",
                "",
                "## Notes",
                note_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )

    return {
        "case_dir": str(case_dir),
        "transform_json": str(transform_path),
        "input_csv": str(input_path),
        "expected_output_csv": str(expected_path),
        "readme_md": str(readme_path),
    }


def build_xenium_rna_protein_alignment_note(
    *,
    pixel_size_um: float,
    segmentation_source: str,
) -> str:
    return "\n".join(
        [
            "# Xenium RNA+Protein Alignment Note",
            "",
            "This note treats Xenium Gene + Protein data as a same-cell, multimodal spatial dataset.",
            "",
            "## Core spaces",
            "",
            f"- Canonical analysis space: `{CANONICAL_SPACE_PHYSICAL_UM}`",
            f"- Export space: `{EXPORT_SPACE_XENIUM_EXPLORER_PIXEL}`",
            f"- Pixel size (um): `{pixel_size_um:.6f}`",
            f"- Segmentation source: `{segmentation_source}`",
            "",
            "## Why this matters",
            "",
            "- RNA transcript points, protein images, H&E or morphology images, segmentation labels, and exported polygons must stay traceable across spaces.",
            "- A single `AnnData.obsm['spatial']` centroid is not enough to express RNA+protein alignment for the same tissue section.",
            "- Polygon export to Xenium Explorer should be treated as an explicit `um -> pixel` transform, not an implicit plotting detail.",
            "- The polygon becomes the smallest pathology-facing analysis unit where RNA, protein, and H&E evidence can be fused.",
            "",
            "## Recommended modeling",
            "",
            f"- Keep the same-cell table in `{CANONICAL_SPACE_PHYSICAL_UM}`.",
            "- Preserve `feature_modality` so protein channels are not flattened into gene-like features.",
            "- Preserve protein images, H&E or morphology images, autofluorescence, and segmentation labels as first-class spatial objects.",
            "- Use exported polygons as the bridge between cell-level molecular signals and pathology-AI review on image patches.",
            "",
            "## Alignment checks",
            "",
            "- Transcript points should map back to the segmentation used for quantification.",
            "- Protein image signal should remain spatially consistent with centroids and exported polygons after transforms.",
            "- H&E or morphology tiles sampled for pathology AI should stay locked to the same polygon geometry.",
            "- `um -> pixel -> um` round-trips should stay within tolerance.",
            "",
        ]
    )


def _default_alignment_cases(
    *,
    pixel_size_um: float,
    segmentation_source: str,
) -> list[XeniumAlignmentFixtureCase]:
    centroid_um = (12.75, 8.50)
    transcript_um = (12.5375, 8.925)
    protein_roi_um = (11.90, 8.075)
    polygon_um = [
        (12.1125, 8.0750),
        (13.3875, 8.0750),
        (13.3875, 9.1375),
        (12.1125, 9.1375),
        (12.1125, 8.0750),
    ]
    polygon_px = um_to_xenium_explorer_pixels(polygon_um, pixel_size_um=pixel_size_um)

    return [
        XeniumAlignmentFixtureCase(
            case_id="identity_points",
            description="Leave canonical physical coordinates unchanged for internal alignment checks.",
            source_space=CANONICAL_SPACE_PHYSICAL_UM,
            target_space=CANONICAL_SPACE_PHYSICAL_UM,
            segmentation_source=segmentation_source,
            transform_chain=({"type": "identity"},),
            input_coordinates=(
                ("cell_centroid", centroid_um[0], centroid_um[1]),
                ("transcript_point", transcript_um[0], transcript_um[1]),
            ),
            expected_output_coordinates=(
                ("cell_centroid", centroid_um[0], centroid_um[1]),
                ("transcript_point", transcript_um[0], transcript_um[1]),
            ),
            notes=(
                "Use this case to verify that RNA and protein objects already expressed in micron space stay stable.",
            ),
        ),
        XeniumAlignmentFixtureCase(
            case_id="scale_um_to_pixel",
            description="Convert same-cell micron coordinates into Xenium Explorer pixel space.",
            source_space=CANONICAL_SPACE_PHYSICAL_UM,
            target_space=EXPORT_SPACE_XENIUM_EXPLORER_PIXEL,
            segmentation_source=segmentation_source,
            transform_chain=(
                {
                    "type": "scale",
                    "scale_x": round(1.0 / pixel_size_um, 6),
                    "scale_y": round(1.0 / pixel_size_um, 6),
                },
            ),
            input_coordinates=(
                ("cell_centroid", centroid_um[0], centroid_um[1]),
                ("protein_roi_anchor", protein_roi_um[0], protein_roi_um[1]),
            ),
            expected_output_coordinates=tuple(
                (identifier, x_value, y_value)
                for identifier, (x_value, y_value) in (
                    ("cell_centroid", um_to_xenium_explorer_pixels([centroid_um], pixel_size_um=pixel_size_um)[0]),
                    ("protein_roi_anchor", um_to_xenium_explorer_pixels([protein_roi_um], pixel_size_um=pixel_size_um)[0]),
                )
            ),
            units=("um", "pixel"),
            notes=(
                "This is the core Xenium export transform used by polygon and ROI overlays.",
                "The same export path also anchors H&E or morphology image patches to the polygon seen by pathology AI.",
            ),
        ),
        XeniumAlignmentFixtureCase(
            case_id="translation_origin_shift",
            description="Apply an origin shift after data are lifted into a shared physical coordinate system.",
            source_space=CANONICAL_SPACE_PHYSICAL_UM,
            target_space=CANONICAL_SPACE_PHYSICAL_UM,
            segmentation_source=segmentation_source,
            transform_chain=(
                {
                    "type": "translation",
                    "translation_x": 2.5,
                    "translation_y": -1.25,
                },
            ),
            input_coordinates=(
                ("cell_centroid", centroid_um[0], centroid_um[1]),
                ("protein_roi_anchor", protein_roi_um[0], protein_roi_um[1]),
            ),
            expected_output_coordinates=(
                ("cell_centroid", centroid_um[0] + 2.5, centroid_um[1] - 1.25),
                ("protein_roi_anchor", protein_roi_um[0] + 2.5, protein_roi_um[1] - 1.25),
            ),
            notes=(
                "Use this case when a protein image or imported segmentation has a known origin offset.",
            ),
        ),
        XeniumAlignmentFixtureCase(
            case_id="axis_order_xy_yx",
            description="Make axis-order handling explicit when moving between x/y coordinates and image row/column semantics.",
            source_space=CANONICAL_SPACE_PHYSICAL_UM,
            target_space=CANONICAL_SPACE_PHYSICAL_UM,
            segmentation_source=segmentation_source,
            transform_chain=({"type": "swap_axes", "axes": ["x", "y"]},),
            input_coordinates=(
                ("transcript_point", transcript_um[0], transcript_um[1]),
                ("protein_roi_anchor", protein_roi_um[0], protein_roi_um[1]),
            ),
            expected_output_coordinates=(
                ("transcript_point", transcript_um[1], transcript_um[0]),
                ("protein_roi_anchor", protein_roi_um[1], protein_roi_um[0]),
            ),
            notes=(
                "This catches x/y vs row/column confusion before polygon export drifts in the viewer.",
            ),
        ),
        XeniumAlignmentFixtureCase(
            case_id="compose_scale_then_translate",
            description="Compose micron-to-pixel scaling with a viewer-space translation for polygon overlays.",
            source_space=CANONICAL_SPACE_PHYSICAL_UM,
            target_space=EXPORT_SPACE_XENIUM_EXPLORER_PIXEL,
            segmentation_source=segmentation_source,
            transform_chain=(
                {
                    "type": "scale",
                    "scale_x": round(1.0 / pixel_size_um, 6),
                    "scale_y": round(1.0 / pixel_size_um, 6),
                },
                {
                    "type": "translation",
                    "translation_x": 10.0,
                    "translation_y": -4.0,
                },
            ),
            input_coordinates=tuple(
                (f"polygon_v{i}", x_value, y_value)
                for i, (x_value, y_value) in enumerate(polygon_um[:4], start=1)
            ),
            expected_output_coordinates=tuple(
                (f"polygon_v{i}", x_value + 10.0, y_value - 4.0)
                for i, (x_value, y_value) in enumerate(polygon_px[:4], start=1)
            ),
            units=("um", "pixel"),
            notes=(
                "This mirrors the real export pattern used for structure polygons after canonical-space processing.",
                "Use the same transformed polygon to crop the corresponding H&E region for pathology-AI review.",
            ),
        ),
    ]


def write_xenium_rna_protein_alignment_bundle(
    output_dir: str | Path,
    *,
    metadata_pixel_size_um: float | int | str | None = None,
    fallback_pixel_size_um: float | int | str = DEFAULT_XENIUM_PIXEL_SIZE_UM,
    segmentation_source: str = "ranger_default",
) -> dict[str, str]:
    bundle_dir = Path(output_dir).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    segmentation_source = validate_segmentation_source(segmentation_source)
    pixel_size_um, pixel_size_source = resolve_xenium_pixel_size_um(
        metadata_pixel_size_um=metadata_pixel_size_um,
        fallback_pixel_size_um=fallback_pixel_size_um,
    )

    note_path = bundle_dir / "xenium_rna_protein_alignment_note.md"
    note_path.write_text(
        build_xenium_rna_protein_alignment_note(
            pixel_size_um=pixel_size_um,
            segmentation_source=segmentation_source,
        ),
        encoding="utf-8",
    )

    cases_dir = bundle_dir / "fixtures"
    cases_dir.mkdir(parents=True, exist_ok=True)
    case_outputs = [
        write_xenium_alignment_fixture_case(case, cases_dir)
        for case in _default_alignment_cases(
            pixel_size_um=pixel_size_um,
            segmentation_source=segmentation_source,
        )
    ]

    manifest_path = bundle_dir / "xenium_rna_protein_fixture_manifest.json"
    manifest_payload = {
        "dataset_modality": DATASET_MODALITY_XENIUM_RNA_PROTEIN,
        "canonical_space": CANONICAL_SPACE_PHYSICAL_UM,
        "export_space": EXPORT_SPACE_XENIUM_EXPLORER_PIXEL,
        "pixel_size_um": pixel_size_um,
        "pixel_size_source": pixel_size_source,
        "segmentation_source": segmentation_source,
        "cases": case_outputs,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "output_dir": str(bundle_dir),
        "alignment_note_md": str(note_path),
        "fixture_manifest_json": str(manifest_path),
        "fixtures_dir": str(cases_dir),
    }
