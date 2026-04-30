"""Contract tests for the stgpt.runtime API consumed by spatho.

These tests verify that:

1.  ``spatho`` correctly calls ``stgpt.runtime.export_spatho_artifacts`` and
    maps its return dict to the paths expected by ``_run_local_stgpt``.
2.  The contract is resilient to both return-key variants that have appeared
    in the wild (``structure_embedding_summary`` and ``structure_summary``).
3.  When ``stgpt.runtime`` returns a valid result the resulting artifact paths
    satisfy the ``inspect_stgpt_evidence_for_paths`` guardrail.
4.  When ``stgpt.runtime`` is absent (package not installed) the error message
    is clear and actionable.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from spatho.schema import validate_workflow_config
from spatho.stgpt import _run_local_stgpt, inspect_stgpt_evidence_for_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _touch_json(path: Path, payload: object = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({} if payload is None else payload), encoding="utf-8")


def _touch_text(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _local_stgpt_config(tmp_path: Path) -> "spatho.schema.WorkflowConfig":
    base = tmp_path / "base.json"
    _touch_json(base, {"dataset_root": str(tmp_path)})
    model_path = tmp_path / "checkpoint.pt"
    model_path.write_bytes(b"fake_weights")
    config_path = tmp_path / "stgpt_config.json"
    _touch_json(config_path, {"model": "scgpt_spatial_v1"})
    cfg_path = tmp_path / "workflow.json"
    _touch_json(
        cfg_path,
        {
            "case_name": "contract_case",
            "study_context": "Contract test",
            "base_pipeline_config": str(base),
            "output_root": str(tmp_path / "out"),
            "annotation_taxonomy": "breast",
            "openai_enabled": False,
            "stgpt_enabled": True,
            "stgpt_backend": "local_stgpt",
            "stgpt_model_path": str(model_path),
            "stgpt_config_path": str(config_path),
        },
    )
    return validate_workflow_config(cfg_path)


def _make_mock_stgpt_runtime(tmp_path: Path, *, use_alt_key: bool = False) -> tuple[MagicMock, dict[str, Path]]:
    """Build a mock stgpt.runtime module and its expected artifact paths."""
    artifact_dir = tmp_path / "stgpt_runtime"
    cell_emb = artifact_dir / "cell_embeddings.parquet"
    struct_sum = artifact_dir / "structure_embedding_summary.csv"
    qc = artifact_dir / "qc_report.json"

    _touch_text(cell_emb, "placeholder parquet")
    _touch_text(struct_sum, "structure_label,n_cells\ntumor,12\n")
    _touch_json(qc, {"status": "pass"})

    # The real stgpt.runtime may return either key variant
    result_key = "structure_summary" if use_alt_key else "structure_embedding_summary"
    export_return = {
        "cell_embeddings": str(cell_emb),
        result_key: str(struct_sum),
        "qc_report": str(qc),
    }

    mock_runtime = MagicMock()
    mock_runtime.export_spatho_artifacts.return_value = export_return

    mock_stgpt = types.ModuleType("stgpt")
    mock_stgpt.runtime = mock_runtime  # type: ignore[attr-defined]

    return mock_runtime, {"cell_embeddings": cell_emb, "structure_summary": struct_sum, "qc_report": qc}


# ---------------------------------------------------------------------------
# Contract: export_spatho_artifacts return dict mapping
# ---------------------------------------------------------------------------


def test_stgpt_runtime_primary_key_contract(tmp_path: Path) -> None:
    """spatho maps 'structure_embedding_summary' to the structure_summary path."""
    cfg = _local_stgpt_config(tmp_path)
    mock_runtime, expected = _make_mock_stgpt_runtime(tmp_path, use_alt_key=False)

    with patch.dict(sys.modules, {"stgpt": MagicMock(runtime=mock_runtime), "stgpt.runtime": mock_runtime}):
        paths = _run_local_stgpt(cfg, output_root=tmp_path / "out")

    mock_runtime.export_spatho_artifacts.assert_called_once()
    call_kwargs = mock_runtime.export_spatho_artifacts.call_args

    # Verify the three canonical paths are resolved correctly
    assert paths["cell_embeddings"] == expected["cell_embeddings"].resolve()
    assert paths["structure_summary"] == expected["structure_summary"].resolve()
    assert paths["qc_report"] == expected["qc_report"].resolve()


def test_stgpt_runtime_alt_structure_key_contract(tmp_path: Path) -> None:
    """spatho falls back to 'structure_summary' key when the primary is absent."""
    cfg = _local_stgpt_config(tmp_path)
    mock_runtime, expected = _make_mock_stgpt_runtime(tmp_path, use_alt_key=True)

    with patch.dict(sys.modules, {"stgpt": MagicMock(runtime=mock_runtime), "stgpt.runtime": mock_runtime}):
        paths = _run_local_stgpt(cfg, output_root=tmp_path / "out")

    assert paths["structure_summary"] == expected["structure_summary"].resolve()


def test_stgpt_runtime_contract_passes_guardrail(tmp_path: Path) -> None:
    """Artifacts returned by the mock pass inspect_stgpt_evidence_for_paths."""
    cfg = _local_stgpt_config(tmp_path)
    mock_runtime, _ = _make_mock_stgpt_runtime(tmp_path, use_alt_key=False)

    with patch.dict(sys.modules, {"stgpt": MagicMock(runtime=mock_runtime), "stgpt.runtime": mock_runtime}):
        paths = _run_local_stgpt(cfg, output_root=tmp_path / "out")

    inspection = inspect_stgpt_evidence_for_paths(cfg, paths)
    assert inspection["ready"] is True
    assert inspection["errors"] == []


# ---------------------------------------------------------------------------
# Contract: stgpt package not installed
# ---------------------------------------------------------------------------


def test_stgpt_runtime_import_error_message(tmp_path: Path) -> None:
    """When stgpt is not installed, the ImportError message is actionable."""
    cfg = _local_stgpt_config(tmp_path)

    # Remove stgpt from sys.modules to simulate package absence
    stgpt_modules = {k: v for k, v in sys.modules.items() if k.startswith("stgpt")}
    for key in stgpt_modules:
        sys.modules.pop(key, None)

    original_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__

    import builtins

    real_import = builtins.__import__

    def _mock_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "stgpt" or name.startswith("stgpt."):
            raise ImportError("No module named 'stgpt'")
        return real_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_mock_import):
        with pytest.raises(ImportError, match="local_stgpt backend requires the stgpt package"):
            _run_local_stgpt(cfg, output_root=tmp_path / "out")

    # Restore stgpt modules
    sys.modules.update(stgpt_modules)


# ---------------------------------------------------------------------------
# Contract: missing config / model path raises ValueError
# ---------------------------------------------------------------------------


def test_stgpt_runtime_missing_model_path_raises(tmp_path: Path) -> None:
    """_run_local_stgpt raises ValueError when model_path is None."""
    base = tmp_path / "base.json"
    _touch_json(base, {"dataset_root": str(tmp_path)})
    cfg_path = tmp_path / "workflow.json"
    _touch_json(
        cfg_path,
        {
            "case_name": "missing_model",
            "study_context": "test",
            "base_pipeline_config": str(base),
            "output_root": str(tmp_path / "out"),
            "annotation_taxonomy": "breast",
            "openai_enabled": False,
            "stgpt_enabled": True,
            "stgpt_backend": "local_stgpt",
        },
    )
    cfg = validate_workflow_config(cfg_path)
    with pytest.raises(ValueError, match="stgpt_model_path"):
        _run_local_stgpt(cfg, output_root=tmp_path / "out")


# ---------------------------------------------------------------------------
# Contract: export_spatho_artifacts is called with expected keyword args
# ---------------------------------------------------------------------------


def test_stgpt_runtime_called_with_config_and_checkpoint(tmp_path: Path) -> None:
    """export_spatho_artifacts must receive config= and checkpoint= kwargs."""
    cfg = _local_stgpt_config(tmp_path)
    mock_runtime, _ = _make_mock_stgpt_runtime(tmp_path, use_alt_key=False)

    with patch.dict(sys.modules, {"stgpt": MagicMock(runtime=mock_runtime), "stgpt.runtime": mock_runtime}):
        _run_local_stgpt(cfg, output_root=tmp_path / "out")

    _, kwargs = mock_runtime.export_spatho_artifacts.call_args
    assert "config" in kwargs, "export_spatho_artifacts must receive a 'config' keyword argument"
    assert "checkpoint" in kwargs, "export_spatho_artifacts must receive a 'checkpoint' keyword argument"
    assert "output_dir" in kwargs, "export_spatho_artifacts must receive an 'output_dir' keyword argument"
