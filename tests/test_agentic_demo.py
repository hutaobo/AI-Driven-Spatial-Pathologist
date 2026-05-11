from __future__ import annotations

import json
import sys
from pathlib import Path

from spatho.agentic import build_agentic_spatial_pathologist_demo
from spatho.cli import main


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_stgpt_package(root: Path, *, qc: dict[str, object]) -> Path:
    _write_json(root / "region_qc_report.json", qc)
    _write_json(root / "evidence_manifest.json", {"schema_version": "stgpt.evidence.v1"})
    _write_text(root / "region_embeddings.parquet", "embedding")
    _write_text(root / "region_molecular_summary.parquet", "molecular")
    _write_text(root / "prototype_assignments.parquet", "prototype")
    record = {
        "evidence_id": "stgpt.region.7",
        "measured_evidence": {
            "molecular_ref": {"artifact": str((root / "region_molecular_summary.parquet").resolve()), "row_index": 7}
        },
        "model_derived_evidence": {
            "embedding_ref": {"artifact": str((root / "region_embeddings.parquet").resolve()), "row_index": 7},
            "prototype_ref": {
                "artifact": str((root / "prototype_assignments.parquet").resolve()),
                "row_index": 7,
                "prototype_id": 3,
            },
        },
        "provenance": {"checkpoint_hash": "abc123"},
        "qc_verdict": {"image_source": "contour_store"},
    }
    _write_text(root / "contour_evidence_chains.jsonl", json.dumps(record) + "\n")
    return root


def test_agentic_demo_blocks_biological_conclusions_on_fatal_qc(tmp_path: Path) -> None:
    package = _write_stgpt_package(tmp_path / "stgpt", qc={"status": "fail", "fatal_errors": ["bad registration"]})

    result = build_agentic_spatial_pathologist_demo(
        stgpt_evidence_dir=package,
        output_dir=tmp_path / "demo",
        case_name="fatal_case",
    )

    report = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
    assert report["conclusion_allowed"] is False
    assert report["claims"][0]["evidence_ids"] == []
    guardrail = Path(result["failure_guardrail_section_md"]).read_text(encoding="utf-8")
    assert "bad registration" in guardrail
    assert "Conclusion allowed: `False`" in guardrail


def test_agentic_demo_marks_warning_evidence_as_cautionary(tmp_path: Path) -> None:
    package = _write_stgpt_package(tmp_path / "stgpt", qc={"status": "pass", "warnings": ["low coverage"]})

    result = build_agentic_spatial_pathologist_demo(
        stgpt_evidence_dir=package,
        output_dir=tmp_path / "demo",
        case_name="warning_case",
    )

    report = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
    assert report["conclusion_allowed"] is True
    assert report["claims"][0]["cautionary"] is True
    assert report["claims"][0]["qc_status"] == "warning"
    report_md = Path(result["report_md"]).read_text(encoding="utf-8")
    assert "Cautionary Warnings" in report_md


def test_agentic_demo_claims_link_evidence_and_artifacts(tmp_path: Path) -> None:
    package = _write_stgpt_package(tmp_path / "stgpt", qc={"status": "pass", "warnings": []})

    result = build_agentic_spatial_pathologist_demo(
        stgpt_evidence_dir=package,
        output_dir=tmp_path / "demo",
        case_name="linked_case",
    )

    report = json.loads(Path(result["report_json"]).read_text(encoding="utf-8"))
    claim = report["claims"][0]
    assert claim["evidence_ids"] == ["stgpt.region.7"]
    assert claim["artifact_ids"]
    assert claim["checkpoint_hash"] == "abc123"
    assert claim["model_derived"] is True
    assert claim["measured_expression"] is False
    manifest = json.loads(Path(result["artifact_manifest_json"]).read_text(encoding="utf-8"))
    manifest_ids = {artifact["id"] for artifact in manifest["artifacts"]}
    assert set(claim["artifact_ids"]).issubset(manifest_ids)


def test_agentic_demo_cli_writes_bundle(tmp_path: Path, monkeypatch, capsys) -> None:
    package = _write_stgpt_package(tmp_path / "stgpt", qc={"status": "pass", "warnings": []})

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "spatho",
            "agentic-demo",
            "--stgpt-evidence-dir",
            str(package),
            "--output-dir",
            str(tmp_path / "demo"),
            "--case-name",
            "cli_case",
            "--max-records",
            "1",
        ],
    )
    main()

    payload = json.loads(capsys.readouterr().out)
    assert Path(payload["report_md"]).exists()
