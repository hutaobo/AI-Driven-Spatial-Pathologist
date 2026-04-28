from __future__ import annotations

import csv
import json
from pathlib import Path

from spatho.api import write_xenium_alignment_fixtures
from spatho.xenium import (
    DEFAULT_XENIUM_PIXEL_SIZE_UM,
    build_feature_records,
    infer_feature_modalities,
    um_to_xenium_explorer_pixels,
    xenium_explorer_pixels_to_um,
)


def test_feature_modality_inference_preserves_rna_and_protein() -> None:
    feature_names = ["EPCAM", "PROTEIN_CD3", "ADT_PD1", "COL1A1"]
    modalities = infer_feature_modalities(feature_names)

    assert modalities == ["rna", "protein", "protein", "rna"]

    records = build_feature_records(feature_names)
    assert records[1]["feature_name"] == "PROTEIN_CD3"
    assert records[1]["feature_modality"] == "protein"


def test_um_pixel_round_trip_is_stable() -> None:
    coords_um = [(12.75, 8.5), (3.1875, 1.0625)]
    coords_px = um_to_xenium_explorer_pixels(coords_um, pixel_size_um=DEFAULT_XENIUM_PIXEL_SIZE_UM)
    round_trip = xenium_explorer_pixels_to_um(coords_px, pixel_size_um=DEFAULT_XENIUM_PIXEL_SIZE_UM)

    for expected, observed in zip(coords_um, round_trip, strict=True):
        assert observed[0] == expected[0]
        assert observed[1] == expected[1]


def test_write_xenium_alignment_fixtures_writes_note_and_cases(tmp_path) -> None:
    result = write_xenium_alignment_fixtures(
        tmp_path,
        metadata_pixel_size_um=0.25,
        segmentation_source="ranger_protein_assisted",
    )

    note_path = Path(result["alignment_note_md"])
    manifest_path = Path(result["fixture_manifest_json"])
    fixtures_dir = Path(result["fixtures_dir"])

    assert note_path.exists()
    assert manifest_path.exists()
    assert fixtures_dir.exists()

    note_text = note_path.read_text(encoding="utf-8")
    assert "RNA+protein" in note_text
    assert "H&E" in note_text
    assert "polygon" in note_text.lower()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_modality"] == "xenium_rna_protein"
    assert manifest["pixel_size_um"] == 0.25
    assert manifest["pixel_size_source"] == "metadata"
    assert manifest["segmentation_source"] == "ranger_protein_assisted"
    assert len(manifest["cases"]) == 5

    case_dir = fixtures_dir / "scale_um_to_pixel"
    assert (case_dir / "transform.json").exists()
    assert (case_dir / "input.csv").exists()
    assert (case_dir / "expected_output.csv").exists()

    with (case_dir / "expected_output.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["id"] == "cell_centroid"
    assert rows[0]["x"] == "51.000000"
    assert rows[0]["y"] == "34.000000"
