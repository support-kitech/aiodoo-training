# ADR-0003: Pipeline Architecture

## Status

Accepted (Phase 0)

## Context

Training involves many ordered stages (config → data → tokenize → model →
adapt → train → eval → export). Putting this order in CLI commands or ad-hoc
scripts causes divergent execution paths and broken resume semantics.

## Decision

A single `Pipeline` orchestrator executes an ordered list of
`PipelineStageHandler` instances against an immutable `PipelineContext`.
Stages return `StageResult` values aggregated into `PipelineResult`.

Phase 0 ships the framework only — no concrete training stages.

## Consequences

- Positive: deterministic stage order, clear resume boundaries, testable orchestration.
- Negative: new capabilities must be modeled as stages or stage collaborators.
