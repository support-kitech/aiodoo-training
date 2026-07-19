# ADR-0006: Repository Boundaries

## Status

Accepted (Phase 0)  
**Clarified:** 2026-07-19 — [ADR-0022](0022-package-surfaces-lifecycle-alignment.md), [ownership.md](../ownership.md)

## Context

AIODOO is a multi-repository ecosystem. Blurring responsibilities between
datasets, training, models, and core creates circular dependencies and unclear
ownership.

## Decision

`aiodoo-training` has hard boundaries:

- It **consumes** datasets from `aiodoo-datasets`.
- It **produces** Capability Packages (and ArtifactBundles) for handoff to
  `aiodoo-validation` and `aiodoo-model` (registry publish). It does **not**
  compose Development / Reasoning **products**.
- It does **not** generate datasets.
- It does **not** perform inference or host serving APIs.
- Heavy ML libraries are optional infrastructure, never domain dependencies.
- It is an **internal** repository executed from source (`python3 <script>.py`),
  not packaged or published to PyPI — consistent with `aiodoo-datasets`.

### Clarification (naming)

The historical name `aiodoo-models` refers to the frozen repository
**`aiodoo-model`**. Prefer the singular form in all new writing.

## Consequences

- Positive: independent release cadence and dependency isolation; same clone-and-run
  workflow as sibling AIODOO repos.
- Negative: cross-repo contracts must be versioned explicitly (protocol versions,
  Capability Package `artifact.json`, export manifests).
