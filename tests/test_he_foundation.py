from __future__ import annotations

import json
from pathlib import Path

from spatho.he_foundation import summarize_contours_by_structure
from spatho.schema import validate_workflow_config


def test_he_foundation_schema_defaults(tmp_path) -> None:
    base_config = tmp_path / "base.json"
    base_config.write_text(json.dumps({"dataset_root": str(tmp_path)}), encoding="utf-8")
    workflow = tmp_path / "workflow.json"
    workflow.write_text(
        json.dumps(
            {
                "case_name": "breast_case",
                "study_context": "Breast context",
                "base_pipeline_config": str(base_config),
                "output_root": str(tmp_path / "out"),
                "annotation_taxonomy": "breast",
                "openai_enabled": False,
            }
        ),
        encoding="utf-8",
    )

    cfg = validate_workflow_config(workflow)

    assert cfg.he_contour_foundation_enabled is False
    assert cfg.he_foundation_model_id == "vinid/plip"
    assert cfg.he_foundation_prompt_set == "breast_contour_v1"
    assert cfg.he_visual_override_enabled is True


def test_summarize_contours_by_structure_aggregates_visual_scores() -> None:
    summaries = summarize_contours_by_structure(
        [
            {
                "contour_id": "a",
                "structure_id": 1,
                "structure_name": "Tumor",
                "top_classes": [
                    {"label_id": "tumor", "label": "Tumor", "score": 0.8},
                    {"label_id": "stroma", "label": "Stroma", "score": 0.2},
                ],
            },
            {
                "contour_id": "b",
                "structure_id": 1,
                "structure_name": "Tumor",
                "top_classes": [
                    {"label_id": "tumor", "label": "Tumor", "score": 0.6},
                    {"label_id": "immune", "label": "Immune", "score": 0.3},
                ],
            },
        ]
    )

    assert len(summaries) == 1
    assert summaries[0]["structure_id"] == 1
    assert summaries[0]["n_contours"] == 2
    assert summaries[0]["n_classified"] == 2
    assert summaries[0]["top_label_id"] == "tumor"
    assert summaries[0]["top_mean_score"] == 0.7
