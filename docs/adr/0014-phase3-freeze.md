# ADR-0014: Phase 3 Permanent Freeze

## Status

**Accepted** — Phase 3 permanently frozen.

## Context

Phase 3 (Training Engine) completed architecture review (ADR-0013),
implementation, hardening (`docs/trainer_backend_contract.md`,
timezone-aware timestamps), and engineering verification
(ruff, mypy, pytest, coverage). Phases 0–2 are already permanently frozen
(ADRs 0007 / 0012 and governance docs).

## Summary

Phase 3 delivers production-grade **train / checkpoint / resume** on top of
frozen Phase 0–2 contracts without redesigning ports, the pipeline orchestrator,
or framework quarantine rules.

## Scope

**In scope (frozen):**

- TrainingSession / TrainingState / TrainingLifecycle
- TrainingContext binder pattern (frozen `TrainerBackend` signatures unchanged)
- CheckpointManager + CheckpointManifest + `training_protocol_version`
- ResumePolicy (STRICT | WARN | RELAXED) + ResumeCoordinator
- Additive OptimizerBackend / SchedulerBackend / TrainingCallback ports
- Event bus, MetricCollector / Aggregator / TrainingHistory
- Gradient / optimizer / scheduler / checkpoint policies
- Phase 3 config fragments and pipeline stage handlers
- Stub + thin HF trainer / checkpoint / RNG infrastructure (CPU CI via stub)
- Trainer backend compliance contract

**Out of scope (later phases):**

- Evaluation engine, export engine (Phase 4)
- Packing / curriculum sophistication (Phase 5)
- Experiment tracking backends (Phase 6)
- Distributed training / DeepSpeed / FSDP / Accelerate launch (Phase 7)
- FailureRecoveryStrategy port (explicitly deferred in ADR-0013)

## Architecture implemented

Canonical specification:
[`docs/phase3-training-engine-architecture.md`](../phase3-training-engine-architecture.md)

Mandatory trainer checklist:
[`docs/trainer_backend_contract.md`](../trainer_backend_contract.md)

Key decisions preserved from ADR-0013:

1. TrainingSession + lifecycle COW; TrainingProgress as emission snapshot
2. CheckpointManager owns manifests/sidecars; CheckpointStore owns weight packages
3. Resume via ResumePolicy + frozen `TrainerBackend.resume`
4. Additive optimizer / scheduler / callback ports
5. Frozen Pipeline handlers only — orchestrator unchanged
6. Framework code in infrastructure only
7. Resume-equivalence golden invariant

## Responsibilities introduced

| Owner | Responsibility |
|-------|----------------|
| TrainingLifecycle | Allowed TrainingStatus transitions (COW) |
| CheckpointManager | Atomic publish, manifests, validation, retention orchestration |
| CheckpointStore (infra) | Opaque weight / adapter package I/O |
| ResumeCoordinator | Bundle assembly + RNG apply after validation |
| TrainerBackend (infra) | train/resume loop; never bypasses manager or planner |
| TrainingEventBus | Ordered synchronous callback / metric dispatch |
| MetricCollector / TrainingHistory | Metric snapshots and optional JSONL history |

## Components added

Domain: `TrainingSession`, policies (`ResumePolicy`, optimizer/scheduler/gradient/checkpoint),
`CheckpointManifest`, training events, opaque optimizer/scheduler handles.

Application: `aiodoo_training/training/` (lifecycle, context, events, metrics,
checkpoint manager, resume, stub engine harness).

Ports (additive): optimizer, scheduler, callback.

Infrastructure: stub trainer + store; torch RNG/optimizer/scheduler stubs;
HF trainer/store (optional); logging/null callbacks.

Wire-up: `bootstrap_phase3`, registries, factories, training config fragments,
`pipeline/handlers.py`.

## Testing summary

Verified at freeze:

- Unit: lifecycle, policies, events, metrics, factories, builders, config, pipeline
- Integration: CPU stub train → checkpoint → resume
- Golden: resume-equivalence (`tests/golden/test_golden_resume_equivalence.py`)
- Corruption / ResumePolicy / determinism / failure / boundary suites
- **220** pytest tests passed (CPU-only CI)

## Coverage summary

Overall coverage **≥ 85%** of measured `aiodoo_training` (infra omit by design;
`fail_under=60`).

## Determinism guarantees

- Same portable config, seed, model/adapter identity, trainer key, and resolved
  `ExecutionEnvironment` → comparable progression on CPU stub.
- Experiment fingerprints remain portable (environment noise must not alter
  ExperimentId by default).
- RNG seeded / restored via `RngController` before first / resumed step.

## Resume guarantees

- Resume-equivalence invariant: interrupted@K then resume → N matches uninterrupted N
  for step/epoch/loss progression under `ResumePolicy.STRICT`.
- DatasetSession cursor, TrainingSession step, checkpoint fingerprints, and RNG
  seed continuity are restored through manager sidecars + store.

## Checkpoint guarantees

- Atomic tmp → rename publish
- Manifest + `training_protocol_version` + fingerprint validation
- Required artifact presence for FULL_STATE
- ResumePolicy-scaled compatibility checks
- Corrupt / incomplete packages rejected; `.tmp-*` cleaned

## Framework isolation guarantees

- Torch / Transformers / PEFT / bitsandbytes / Accelerate imports confined to
  `aiodoo_training/infrastructure/`
- Public surfaces use opaque AIODOO handles only
- AST boundary tests enforce quarantine

## Frozen contracts

Phase 3 joins Phases 0–2 as a **permanently frozen** public contract.
Canonical governance: [`docs/frozen_public_contracts.md`](../frozen_public_contracts.md).

Implementation phases (4+) **must preserve** these contracts. They may:

- register new backends behind existing ports
- add pipeline handlers for existing stages
- extend domain additively without mutating frozen types

They must **not**:

- widen frozen port signatures for convenience
- bypass CheckpointManager / ResourcePlanner / registries
- leak framework types outside infrastructure
- redesign Phase 3 ownership splits without a new ADR and Section 9 process

## Future extension points

- Additional trainer registry keys (custom loop, DeepSpeed, FSDP, Accelerate)
- Sharded / distributed checkpoint layouts under the same manager API
- EvaluationCompleted hooks (Phase 4)
- Tracking sinks behind existing `ExperimentTracker` port (Phase 6)

## Decision

Phase 3 is **permanently frozen**. Together with Phases 0–2 it forms the stable
train-path public surface of `aiodoo-training`.

Rules for Phase 4+:

- Never bypass an existing Port or CheckpointManager.
- Never expose third-party framework types outside infrastructure.
- Never move responsibilities between frozen layers.
- Never introduce shortcuts that violate ADRs or `trainer_backend_contract.md`.
- If a feature appears to require changing a frozen phase, **stop and explain why**.

## Consequences

- Positive: Phase 4 evaluation/export has a fixed train/resume/checkpoint boundary.
- Negative: HF-opaque optimizer state remains backend-key constrained; expanding
  cross-backend resume needs protocol bumps, not silent bypasses.
