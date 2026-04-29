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

## Repository Roles

- `spatho`: public product and user experience layer
- `histoseg`: geometry and segmentation engine
- example web apps: optional deployment surfaces, not the core product
