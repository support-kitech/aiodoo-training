# ADR-0006: Repository Boundaries

## Status

Accepted (Phase 0)

## Context

AIODOO is a multi-repository ecosystem. Blurring responsibilities between
datasets, training, models, and core creates circular dependencies and unclear
ownership.

## Decision

`aiodoo-training` has hard boundaries:

- It **consumes** datasets from `aiodoo-datasets`.
- It **produces** trained adapters/models for `aiodoo-models`.
- It does **not** generate datasets.
- It does **not** perform inference or host serving APIs.
- Heavy ML libraries are optional infrastructure, never domain dependencies.
- It is an **internal** repository executed from source (`python3 <script>.py`),
  not packaged or published to PyPI — consistent with `aiodoo-datasets`.

## Consequences

- Positive: independent release cadence and dependency isolation; same clone-and-run
  workflow as sibling AIODOO repos.
- Negative: cross-repo contracts must be versioned explicitly (protocol versions,
  export manifests).
