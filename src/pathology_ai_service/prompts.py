from __future__ import annotations

import json
from typing import Any


def build_review_messages(
    *,
    review_type: str,
    question: str,
    answer_language: str,
    evidence: dict[str, Any],
    citations: list[dict[str, Any]],
) -> list[dict[str, str]]:
    review_label = "structure-level pathology review" if review_type == "structure" else "case-level pathology summary"
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True)

    citation_lines: list[str] = []
    for citation in citations:
        citation_lines.append(
            f"- {citation['citation_id']} | document_id={citation['document_id']} | "
            f"chunk_id={citation['chunk_id']} | title={citation.get('title') or 'untitled'}\n"
            f"  {citation['text']}"
        )

    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Evidence bundle:\n{evidence_json}\n\n"
        f"Candidate citations:\n{chr(10).join(citation_lines) if citation_lines else 'No retrieved citations.'}\n\n"
        f"Return your final answer in {answer_language}. "
        "Only cite citation_ids that appear in the candidate citations block."
    )

    system_prompt = (
        "You are a pathology decision-support model operating inside a private PDC deployment. "
        f"Produce a {review_label} grounded in the provided evidence bundle and retrieved reference passages. "
        "Do not diagnose autonomously. Do not invent citations. "
        "Return valid JSON with exactly these keys: "
        "summary, interpretation, confidence, key_evidence, caveats, recommended_follow_up, citation_ids. "
        "confidence must be a number between 0 and 1."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
