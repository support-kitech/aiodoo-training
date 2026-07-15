# ADR-0015: Phase 4 Evaluation & Export Architecture

## Status

**Accepted** — Phase 4 architecture frozen. Implementation complete.
Phase 4 is a **permanently frozen** public contract (see [Artifact Contract](../artifact_contract.md)
and [Frozen Public Contracts](../frozen_public_contracts.md)).

## Context

Phases 0–3 are permanently frozen (ADRs 0007, 0012, 0014). Phase 4 delivers
offline evaluation and portable export without redesigning frozen
`Evaluator` / `Exporter` ports, `EvaluationReport` / `ExportArtifact` /
`ExperimentManifest`, DatasetSession, ResourcePlanner, or CheckpointManager.

## Decision

Adopt the architecture specification in
[`docs/phase4-evaluation-export-architecture.md`](../phase4-evaluation-export-architecture.md)
including hardening amendments (Artifact Contract decoupling,
ArtifactDescriptor / ArtifactIndex discovery, ArtifactValidationPolicy +
ArtifactCompatibilityPolicy).

## Consequences

- Positive: Clear handoff package; independent Models cadence; CPU golden
  eval/export.
- Negative: Frozen Evaluator/Exporter signatures constrain mid-call APIs to binders.

## Freeze

Phase 4 is **permanently frozen**. Future phases may extend evaluation/export
only through additive registrations, configuration, or new ADRs. Frozen
Evaluator/Exporter signatures and the Artifact Contract must not be redesigned
without the Section 9 change process.