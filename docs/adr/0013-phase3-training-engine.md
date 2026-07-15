# ADR-0013: Phase 3 Training Engine Architecture

## Status

**Accepted** — Phase 3 architecture frozen. Implementation authorized.

## Context

Phases 0–2 are permanently frozen. Phase 3 introduces training, checkpointing,
and resume while consuming existing ports (`TrainerBackend`, `CheckpointStore`,
`RngController`) and domain types without redesigning them.

## Decision

Adopt the architecture specification in
[`docs/phase3-training-engine-architecture.md`](../phase3-training-engine-architecture.md)
including hardening amendments:

1. TrainingSession + TrainingLifecycle; frozen TrainingProgress as emission snapshot.
2. CheckpointManager owns manifests/sidecars; CheckpointStore owns weight packages.
3. Resume via ResumePolicy + TrainerBackend.resume (frozen signature unchanged).
4. Additive OptimizerBackend / SchedulerBackend / TrainingCallback ports.
5. Frozen PipelineStage handlers only — orchestrator unchanged.
6. Framework code in infrastructure only.
7. Distributed deferred to Phase 7.
8. `training_protocol_version` on CheckpointManifest.
9. ResumePolicy STRICT | WARN | RELAXED (default STRICT).
10. No FailureRecoveryStrategy in Phase 3.
11. Resume-equivalence golden invariant (§12.1).

## Consequences

- Positive: Correct resume and determinism testable on CPU stubs.
- Negative: HF opaque state requires backend_key compatibility.

## Freeze

Phase 3 **architecture** was frozen with ADR-0013 acceptance.

Phase 3 is **permanently frozen** as an implementation + contract surface by
[ADR-0014](0014-phase3-freeze.md). Architectural redesign of Phase 3 is not
permitted without a new ADR and the Section 9 change process in
`frozen_public_contracts.md`.
