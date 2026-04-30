# AI-Driven Spatial Pathologist Public Product Roadmap

This repo is the public-facing product layer for the AI-driven spatial pathologist workflow.

## Phase 1: Wrapper Product

Current strategy:

- keep `histoseg` as the execution engine
- keep this repo lightweight and public-facing
- expose a stable package name: `spatho`
- provide a simple CLI and README-first onboarding

What is already wrapped:

- OpenAI-driven cluster annotation
- structure discovery
- H&E overlay generation
- pathology review report generation
- formal organ-pack metadata
- workflow config schema export
- artifact manifest generation

## Why this repo exists

`histoseg` began as a segmentation and contour-generation library.
The public product experience now needs a clearer identity:

- disease-focused workflows
- case-level reporting
- reproducible workflow bundles
- user-facing documentation

This repo becomes that layer.

## Phase 2: Product Stabilization

Near-term work:

1. Expand packaged organ packs beyond `lung` and `breast`
2. Add small regression tests with tiny fixtures
3. Add GitHub Actions for `pytest`, package build, and CLI smoke tests
4. Stabilize config schema versioning and compatibility rules
5. Stabilize artifact manifest and report schema versions

## Phase 3: Dependency Inversion

The long-term goal is to reduce direct runtime coupling to sibling repos.

Planned moves:

1. move public-safe workflow code from `histoseg` into `spatho`
2. keep only geometry/segmentation primitives in `histoseg`
3. define organ packs under `spatho.organ_packs`
4. support multiple providers: OpenAI, Anthropic, local models

## Phase 4: Community Release

Before broad public release:

1. rewrite README around `spatho`, not just legacy wrappers
2. publish example datasets and example reports
3. document license boundaries clearly
4. add issue templates and contribution guide

## Phase 5: stGPT Foundation Model -> Evidence Workbench

The next AI upgrade should add an optional spatial transcriptomics foundation-model layer rather than replacing the existing workflow. The narrative is:

> stGPT learns the morpho-molecular tissue representation; spatho turns it into auditable spatial pathology evidence.

The product should be described as a closed loop:

```text
Model -> Evidence -> Agent -> Human Review -> Better Model
```

Planned moves:

1. define `stGPT Foundation`: training, model architecture, checkpoint loading, embedding, and model packaging
2. define `stGPT Evidence Suite`: QC, deterministic splits, benchmark tables, ablations, domain-shift checks, and failure analysis
3. define `stGPT Runtime / Tool API`: `embed_cells`, `evaluate_checkpoint`, `package_model`, and `export_spatho_artifacts` first; retrieval, imputation, niche scoring, region comparison, and structure explanation only after tested outputs exist
4. define `spatho Agentic Workbench`: guardrailed workflow orchestration that checks QC before biological conclusions
5. define `spatho Reports`: report sections that separate measured expression, model-derived evidence, warnings, and human review

Implementation guardrails:

- precomputed stGPT artifacts must work without importing `stgpt`
- `local_stgpt` is optional and should fail clearly when the package or model paths are missing
- fatal stGPT QC blocks a run when `stgpt_require_qc_pass=true`
- warning-only stGPT QC becomes cautionary report language, not a hard failure
- imputation, reconstruction, or embeddings must never be described as measured expression

See [stGPT Upgrade Plan](STGPT_UPGRADE_PLAN.md) for the detailed implementation route.

## Repository Roles

- `spatho`: public product and user experience layer
- `histoseg`: geometry and segmentation engine
- example web apps: optional deployment surfaces, not the core product
