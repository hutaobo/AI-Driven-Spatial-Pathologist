from __future__ import annotations

from html import escape
from typing import Any


def build_evidence_report_section(
    *,
    summary_rows: list[dict[str, Any]],
    warnings: list[str] | None = None,
    outputs: dict[str, str] | None = None,
) -> str:
    """Build an auditable stGPT evidence HTML section."""
    warning_items = "".join(f"<li>{escape(str(item))}</li>" for item in (warnings or []))
    warning_html = (
        "<p><strong>Cautionary evidence:</strong> QC warnings were present. Interpret stGPT evidence as "
        "model-derived support, not measured expression.</p><ul>" + warning_items + "</ul>"
        if warning_items
        else "<p>QC did not report fatal stGPT evidence errors.</p>"
    )
    rows_html = []
    for row in summary_rows:
        artifact_ids = row.get("artifact_ids", "")
        if isinstance(artifact_ids, list):
            artifact_ids = "; ".join(str(item) for item in artifact_ids)
        rows_html.append(
            "<tr>"
            f"<td>{escape(str(row.get('evidence_id', '')))}</td>"
            f"<td>{escape(str(row.get('structure_label', row.get('evidence_type', 'case'))))}</td>"
            f"<td>{escape(str(row.get('n_cells', row.get('value', ''))))}</td>"
            f"<td>{escape(str(row.get('qc_flag', row.get('qc_status', 'model-derived'))))}</td>"
            f"<td>{escape(str(row.get('human_review_state', 'pending')))}</td>"
            f"<td>{escape(str(artifact_ids))}</td>"
            f"<td>{escape(str(row.get('interpretation', 'stGPT morpho-molecular evidence')))}</td>"
            "</tr>"
        )
    if not rows_html:
        rows_html.append("<tr><td colspan=\"7\">No structure-level stGPT evidence was available.</td></tr>")
    link_items = []
    for label, key in (
        ("Cell embeddings", "stgpt_cell_embeddings_parquet"),
        ("Structure summary", "stgpt_structure_embedding_summary_csv"),
        ("QC report", "stgpt_qc_report_json"),
        ("Evidence summary", "stgpt_evidence_summary_csv"),
    ):
        value = (outputs or {}).get(key)
        if value:
            link_items.append(f"<li>{escape(label)}: <code>{escape(str(value))}</code></li>")
    link_html = "<ul>" + "".join(link_items) + "</ul>" if link_items else ""
    return (
        "<!-- spatho-stgpt-evidence:start -->\n"
        "<section>\n"
        "<h2>stGPT Evidence</h2>\n"
        "<p>stGPT evidence is a morpho-molecular model output for review and triage. "
        "It must not be reported as measured expression or diagnosis without human review.</p>\n"
        f"{warning_html}\n"
        f"{link_html}\n"
        "<table><thead><tr><th>Evidence ID</th><th>Structure</th><th>Cells</th><th>QC</th>"
        "<th>Human review</th><th>Artifact IDs</th><th>Interpretation</th></tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>\n"
        "</section>\n"
        "<!-- spatho-stgpt-evidence:end -->"
    )


__all__ = ["build_evidence_report_section"]
