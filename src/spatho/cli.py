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

    bench_parser = subparsers.add_parser(
        "he-benchmark",
        help="Benchmark pixel-level H&E tumor-region models on public sets and private slides.",
    )
    bench_sub = bench_parser.add_subparsers(dest="he_benchmark_command", required=True)

    bench_sub.add_parser("catalog", help="Print the models and datasets that belong in the comparison.")

    bench_init = bench_sub.add_parser("init", help="Create a benchmark directory, protocol, and dataset folders.")
    bench_init.add_argument("--output-dir", required=True, help="Where to write the benchmark workspace.")
    bench_init.add_argument("--with-synthetic-fixture", action="store_true", help="Add a tiny synthetic private case for a smoke run.")
    bench_init.add_argument("--private-images", help="Optional directory of private H&E tiles to ingest now.")
    bench_init.add_argument("--private-masks", help="Optional directory of private tumor masks paired by filename stem.")

    bench_ingest = bench_sub.add_parser("ingest", help="Register a public or private image/mask (or GeoJSON) dataset.")
    bench_ingest.add_argument("--dataset-id", required=True, help="Dataset id, e.g. private_he or public_camelyon16.")
    bench_ingest.add_argument("--output-dir", required=True, help="Dataset output directory that will contain cases.jsonl.")
    bench_ingest.add_argument("--images", help="Directory of H&E tiles.")
    bench_ingest.add_argument("--masks", help="Directory of binary tumor masks with the same stems.")
    bench_ingest.add_argument("--image", help="Single WSI/tile path used with --geojson.")
    bench_ingest.add_argument("--geojson", help="Tumor-region GeoJSON to rasterize onto --image.")
    bench_ingest.add_argument("--kind", choices=["private", "public"], default="private")
    bench_ingest.add_argument("--organ")
    bench_ingest.add_argument("--pixel-size-um", type=float)

    bench_doctor = bench_sub.add_parser("doctor", help="Check protocol paths and which models can actually run.")
    bench_doctor.add_argument("--protocol", required=True, help="Path to protocol.json.")

    bench_run = bench_sub.add_parser("run", help="Score available models and write a private/public leaderboard.")
    bench_run.add_argument("--protocol", required=True, help="Path to protocol.json.")
    bench_run.add_argument("--output-dir", help="Where to write leaderboard artifacts.")
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

    if args.command == "he-benchmark":
        from .he_benchmark import catalog_payload, doctor_benchmark, ingest_dataset, init_benchmark, run_benchmark

        if args.he_benchmark_command == "catalog":
            print(json.dumps(catalog_payload(), indent=2, ensure_ascii=False))
            return
        if args.he_benchmark_command == "init":
            result = init_benchmark(
                args.output_dir,
                with_synthetic_fixture=bool(args.with_synthetic_fixture),
                private_images_dir=getattr(args, "private_images", None),
                private_masks_dir=getattr(args, "private_masks", None),
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        if args.he_benchmark_command == "ingest":
            result = ingest_dataset(
                dataset_id=args.dataset_id,
                output_dir=args.output_dir,
                images_dir=getattr(args, "images", None),
                masks_dir=getattr(args, "masks", None),
                image_path=getattr(args, "image", None),
                geojson_path=getattr(args, "geojson", None),
                kind=args.kind,
                organ=getattr(args, "organ", None),
                pixel_size_um=getattr(args, "pixel_size_um", None),
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        if args.he_benchmark_command == "doctor":
            result = doctor_benchmark(args.protocol)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        if args.he_benchmark_command == "run":
            result = run_benchmark(args.protocol, output_dir=getattr(args, "output_dir", None))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return
        parser.error(f"Unsupported he-benchmark command: {args.he_benchmark_command}")

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
