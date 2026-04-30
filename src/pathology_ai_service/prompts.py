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


def build_cluster_annotation_messages(
    *,
    case_name: str,
    study_context: str,
    annotation_taxonomy: str,
    controlled_vocabulary: list[dict[str, Any]],
    cluster_evidence: dict[str, Any],
    heuristic_annotation: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "case_name": case_name,
        "study_context": study_context,
        "annotation_taxonomy": annotation_taxonomy,
        "controlled_vocabulary": controlled_vocabulary,
        "heuristic_starting_point": heuristic_annotation,
        "cluster_evidence": cluster_evidence,
    }
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    label_ids = [str(item.get("id")) for item in controlled_vocabulary if item.get("id")]
    system_prompt = (
        "You are a spatial transcriptomics cell-type annotation model running inside a private PDC deployment. "
        "Choose the single best label_id for the cluster from the provided controlled vocabulary only. "
        "Use the heuristic annotation as a starting point, but correct it when marker evidence supports another label. "
        "Use only evidence supplied in the request, including marker genes, optional scGPT-like reference mapping, "
        "pathway activity, and spatial context. Do not invent unsupported markers or labels. "
        "Return valid JSON with exactly these keys: "
        "label_id, confidence, review_priority, supporting_markers, conflicting_markers, "
        "alternative_label_ids, reasoning_summary, tumor_evidence, recommended_follow_up. "
        "confidence must be between 0 and 1. review_priority must be low, medium, or high. "
        f"Allowed label_id values are: {', '.join(label_ids)}."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_structure_multimodal_naming_messages(
    *,
    case_name: str,
    study_context: str,
    annotation_taxonomy: str,
    structure: dict[str, Any],
    current_review: dict[str, Any],
    he_visual_summary: dict[str, Any],
    multimodal_evidence: dict[str, Any],
    override_policy: dict[str, Any],
) -> list[dict[str, str]]:
    payload = {
        "case_name": case_name,
        "study_context": study_context,
        "annotation_taxonomy": annotation_taxonomy,
        "structure": structure,
        "current_review": current_review,
        "he_visual_summary": he_visual_summary,
        "multimodal_evidence": multimodal_evidence,
        "override_policy": override_policy,
    }
    user_prompt = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    system_prompt = (
        "You are a spatial pathology multimodal naming model running inside a private PDC deployment. "
        "Name the spatial structure by integrating H&E contour foundation-model evidence with RNA, "
        "cell-type composition, scGPT-like reference mapping, pathway activity, structure assignment, "
        "lightweight niche-fusion evidence, and pyXenium contour evidence. "
        "You may set visual_override=true only when the H&E evidence strongly contradicts the current "
        "name and is biologically more plausible than the molecular-only name. "
        "Do not invent image findings that are not present in he_visual_summary. "
        "Return valid JSON with exactly these keys: "
        "structure_id, pre_visual_name, final_name, visual_override, confidence, review_priority, "
        "reasoning_summary, visual_evidence, molecular_evidence, conflicts, recommended_checks. "
        "confidence must be between 0 and 1. review_priority must be low, medium, or high."
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
