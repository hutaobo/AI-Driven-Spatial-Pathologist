from __future__ import annotations

import argparse
import json

from .agentic import build_agentic_spatial_pathologist_demo
from .api import (
    build_manifest,
    init_workflow,
    list_available_organ_packs,
    run_workflow,
    workflow_doctor_report,
    write_schema,
    write_xenium_alignment_fixtures,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spatho",
        description="Public-facing CLI for the agentic spatial pathologist workflow.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Run a full spatial pathologist workflow from a JSON config.",
    )
    run_parser.add_argument("--config", required=True, help="Path to a spatho/histoseg workflow JSON.")
    run_parser.add_argument(
        "--heuristic-only",
        action="store_true",
        help="Disable OpenAI calls and force heuristic annotation/review.",
    )

    init_parser = subparsers.add_parser(
        "init-workflow",
        help="Generate a starter workflow JSON for a Xenium case.",
    )
    init_parser.add_argument("--organ", required=True, choices=["lung", "breast"], help="Organ pack to target.")
    init_parser.add_argument("--case-name", required=True, help="Case identifier used in reports and outputs.")
    init_parser.add_argument("--dataset-root", required=True, help="Root folder of the Xenium outs directory.")
    init_parser.add_argument(
        "--base-pipeline-config",
        required=True,
        help="Path to the base pipeline config consumed by histoseg.",
    )
    init_parser.add_argument("--output", required=True, help="Where to write the generated workflow JSON.")
    init_parser.add_argument("--output-root", help="Optional output directory override.")
    init_parser.add_argument("--study-context", help="Optional study context override.")
    init_parser.add_argument("--openai-model", default="gpt-5.4", help="OpenAI model to place in the template.")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Check whether the environment is ready for a workflow run.",
    )
    doctor_parser.add_argument("--config", help="Optional workflow JSON to validate.")

    subparsers.add_parser(
        "list-organ-packs",
        help="List built-in organ packs shipped with spatho.",
    )

    schema_parser = subparsers.add_parser(
        "config-schema",
        help="Export the formal workflow JSON schema.",
    )
    schema_parser.add_argument("--output", required=True, help="Where to write the workflow schema JSON.")

    manifest_parser = subparsers.add_parser(
        "build-manifest",
        help="Create or refresh an artifact manifest for an existing workflow output directory.",
    )
    manifest_parser.add_argument("--config", required=True, help="Workflow JSON used for the run.")
    manifest_parser.add_argument("--output", help="Optional explicit manifest output path.")

    xenium_parser = subparsers.add_parser(
        "write-xenium-alignment-fixtures",
        help="Write Xenium RNA+protein + H&E alignment fixtures and a method note.",
    )
    xenium_parser.add_argument("--output-dir", required=True, help="Where to write the fixture bundle.")
    xenium_parser.add_argument(
        "--metadata-pixel-size-um",
        type=float,
        help="Optional pixel size from dataset metadata. Takes precedence over the fallback value.",
    )
    xenium_parser.add_argument(
        "--fallback-pixel-size-um",
        type=float,
        help="Optional fallback pixel size when metadata is absent. Defaults to 0.2125.",
    )
    xenium_parser.add_argument(
        "--segmentation-source",
        default="ranger_default",
        choices=["ranger_protein_assisted", "ranger_default", "third_party_import"],
        help="Segmentation provenance to record in the bundle.",
    )
    demo_parser = subparsers.add_parser(
        "agentic-demo",
        help="Build an artifact-first Agentic Spatial Pathologist v0.1 report from stGPT evidence.",
    )
    demo_parser.add_argument("--stgpt-evidence-dir", required=True, help="Directory containing stGPT spatho_export artifacts.")
    demo_parser.add_argument("--output-dir", required=True, help="Where to write the demo report bundle.")
    demo_parser.add_argument("--case-name", required=True, help="Case identifier used in report outputs.")
    demo_parser.add_argument("--metrics", help="Optional stGPT evaluation_metrics.json path.")
    demo_parser.add_argument("--checkpoint-card", help="Optional stGPT model/checkpoint card JSON path.")
    demo_parser.add_argument("--pyxenium-summary", help="Optional pyXenium morphomolecular summary artifact.")
    demo_parser.add_argument("--max-records", type=int, default=100, help="Maximum evidence-chain records to sample.")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "run":
        result = run_workflow(args.config, heuristic_only=bool(args.heuristic_only))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "init-workflow":
        result = init_workflow(
            args.output,
            organ=args.organ,
            case_name=args.case_name,
            dataset_root=args.dataset_root,
            base_pipeline_config=args.base_pipeline_config,
            output_root=getattr(args, "output_root", None),
            study_context=getattr(args, "study_context", None),
            openai_model=args.openai_model,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "doctor":
        result = workflow_doctor_report(getattr(args, "config", None))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "list-organ-packs":
        print(json.dumps({"organ_packs": list_available_organ_packs()}, indent=2, ensure_ascii=False))
        return

    if args.command == "config-schema":
        result = write_schema(args.output)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "build-manifest":
        result = build_manifest(args.config, output_path=getattr(args, "output", None))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "write-xenium-alignment-fixtures":
        result = write_xenium_alignment_fixtures(
            args.output_dir,
            metadata_pixel_size_um=getattr(args, "metadata_pixel_size_um", None),
            fallback_pixel_size_um=getattr(args, "fallback_pixel_size_um", None),
            segmentation_source=args.segmentation_source,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if args.command == "agentic-demo":
        result = build_agentic_spatial_pathologist_demo(
            stgpt_evidence_dir=args.stgpt_evidence_dir,
            output_dir=args.output_dir,
            case_name=args.case_name,
            metrics_path=getattr(args, "metrics", None),
            checkpoint_card_path=getattr(args, "checkpoint_card", None),
            pyxenium_summary_path=getattr(args, "pyxenium_summary", None),
            max_records=int(args.max_records),
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
