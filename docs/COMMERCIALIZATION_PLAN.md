# AI-Driven Spatial Pathologist Commercialization Plan

This document outlines a practical way to commercialize `spatho` while keeping a strong academic and non-commercial entry point.

## Reference Pattern: CyteType

CyteType is a useful benchmark because it separates free academic access from paid commercial access instead of forcing a single licensing mode.

Public documentation currently states:

- academic and non-commercial use is free
- free use is capped at three annotation runs per day
- users can also bring their own LLM API keys or run local models
- commercial use is licensed with an annual fee plus a usage-based fee
- enterprise options include isolated or on-prem deployments

Reference:

- [CyteType documentation](https://www.nygen.io/docs/cytetype)

## Recommended Spatial Pathologist Edition Split

### 1. Academic Community Edition

Goal:

- maximize adoption in academic labs
- make methods reproducible
- let users run locally with their own OpenAI key or local models

Recommended scope:

- local CLI and Python package
- built-in organ packs such as `lung` and `breast`
- workflow JSON templates
- HTML reports
- artifact manifests
- community support only

Recommended license approach:

- keep the public code under a non-commercial research license
- require commercial users to obtain a separate commercial license

Operational model:

- software itself is free for academic and non-commercial use
- API spend is paid by the user through their own OpenAI account
- no service-level guarantee

### 2. Commercial Team Edition

Goal:

- sell productivity and governance, not just source code

Recommended scope:

- commercial-use license
- managed updates and validated builds
- private organ packs and customer reference packs
- shared reports and review dashboards
- organization settings, role-based access, and audit logs
- support and onboarding

Pricing model:

- annual platform fee per organization or team
- usage-based fee per analyzed case or per annotated cluster bundle
- optionally allow bring-your-own OpenAI key for customers who want direct model billing

### 3. Enterprise Edition

Goal:

- support clinical-adjacent and regulated environments

Recommended scope:

- private cloud or on-prem deployment
- customer-managed OpenAI or local LLM credentials
- zero-retention data paths
- longer report retention controls
- SSO, audit export, and procurement support

Pricing model:

- larger annual contract
- implementation / deployment fee
- optional premium support SLA

## Why This Split Fits Spatial Pathologist

Unlike a purely algorithmic library, `spatho` has a real marginal cost when users rely on managed LLM inference.
That means a good product split is:

- free software for academic self-service users
- paid convenience, governance, support, and managed inference for commercial users

This avoids blocking academic adoption while still creating a business model.

## Recommended Technical Architecture for Editions

To support both free and paid editions cleanly, the codebase should separate four concerns:

1. Core workflow engine
   Local workflow orchestration, organ packs, schema validation, and artifact manifests.

2. Provider layer
   OpenAI, Anthropic, and local-model adapters behind a stable interface.

3. Entitlement layer
   Edition flags such as community vs commercial vs enterprise.

4. Service layer
   Optional hosted APIs, team workspaces, report storage, billing, and admin UI.

## Concrete Product Recommendation

The cleanest product packaging is:

- `spatho` Community Edition
  Local package and CLI, non-commercial, BYO API key or local models.

- `spatho` Commercial
  Licensed commercial build and support, optional managed OpenAI billing, optional hosted collaboration layer.

- `spatho Enterprise`
  Private deployment, customer-managed credentials, governance and security controls.

## Suggested Next Business Steps

1. Keep the local CLI and organ-pack workflow public and non-commercial.
2. Add provider abstraction so commercial customers can choose managed inference or BYO key.
3. Add token and job metering at the case, cluster, and structure-review levels.
4. Add authenticated report storage and organization-level access control.
5. Define a commercial license and order form separate from the public repo license.
