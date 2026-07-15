# ADR-0021: Phase 7 Permanent Freeze

## Status

**Accepted** — Phase 7 permanently frozen.

## Context

Phase 7 (Distributed Readiness) completed architecture design
([`docs/phase7-distributed-readiness-architecture.md`](../phase7-distributed-readiness-architecture.md)),
architecture acceptance ([ADR-0019](0019-phase7-distributed-readiness.md)),
implementation, engineering hardening, and validation (ruff, pytest Phase 7
unit/golden suites, related regression coverage).

Phases 0–6 are already permanently frozen (ADRs 0007, 0012, 0014, 0015, 0017,
0020 and related governance). Phase 6 remains permanently frozen under
[ADR-0020](0020-phase6-freeze.md).

## Phase summary

Phase 7 delivers a **placement / synchronization / recovery façade** for
multi-process and multi-node readiness **without** owning training,
checkpoint durability, resume compatibility, packing, evaluation, export, or
tracking authority, and without redesigning frozen ResourcePlanner,
CheckpointManager, ResumePolicy, SchedulePlanner, EvaluationEngine,
ExportManager, Artifact Contract, or TrackingHealth.

## Scope

**In scope (frozen):**

- `DistributedRuntime` — ephemeral process-group lifecycle + topology / sync façade
- `DistributedSession` — immutable COW distributed lifecycle cursor
- `DistributedContext` — binder bag for distributed ports / handles
- `DeviceMesh` / `PlacementPlan` — portable mesh and placement DTOs
- `DistributedPlacementResolver` (PlacementResolver companion) — consumes
  `ExecutionEnvironment` + `DistributedSpec`; does not widen ResourcePlanner
- `DistributedBackend` port + `distributed_backend_registry`
- `DistributedCheckpointCoordinator` (+ related dist eval/export/write /
  fault-tolerance coordinators) — rank roles / barriers around frozen authorities
- `RestartPolicy` — relaunch decisions beside (never replacing) ResumePolicy
- `DistributedHealth` — runtime mesh observations (≠ TrackingHealth)
- `FakeDistributedBackend` — CI/reference infrastructure implementation
- Portable `mesh_digest` rules (no hostname / launcher-only ranks in identity)
- Distributed registries (backend / placement / sampler)
- Distributed builders / factories
- Distributed configuration (`DistributedSpec` / config fragments)
- Distributed pipeline handlers (handlers-only; no stage enum redesign)
- Distributed deterministic behaviour (fake path goldens; portable digests)

**Out of scope (extension only — not architectural redesign):**

- Real DDP / FSDP / DeepSpeed / Accelerate / XLA / UCX adapters
- Elastic world-size / torchelastic membership
- Cloud-native job schedulers inside this repository
- Bit-exact cross-SKU GPU numerics
- Mandatory new pipeline stage `INIT_DISTRIBUTED` (Section 9 only if required)
- Other AIODOO repositories (`aiodoo-models`, `aiodoo-cli`, `aiodoo-vscode`,
  master repository)

## Responsibilities introduced

| Owner | Responsibility |
|-------|----------------|
| DistributedRuntime | Ephemeral PG / topology / sync façade lifecycle |
| DistributedSession / DistributedLifecycle | Allowed DistributedStatus transitions (COW) |
| DistributedPlacementResolver | Companion placement from ExecutionEnvironment + DistributedSpec |
| DistributedBackend (infra adapters) | Collectives via frozen port + `bind()` |
| DistributedCheckpointCoordinator | Who may call CheckpointManager; never writes packages |
| DistributedEvaluation / ExportWrite coordinators | Shard/merge / single-writer around frozen engines |
| FaultToleranceCoordinator | Restart decisions that *invoke* ResumePolicy |
| RestartPolicy / DistributedHealth | Restart policy + runtime health snapshots |
| ShardPlanner / EpochCoordinator / SyncFacade | Placement helpers and sync orchestration |

## Architecture implemented

Canonical specification:
[`docs/phase7-distributed-readiness-architecture.md`](../phase7-distributed-readiness-architecture.md)

Key decisions preserved from ADR-0019:

1. Distributed façade — never a second training engine
2. Companion placement — do not widen ResourcePlanner
3. Registry-driven backends — Accelerate / DeepSpeed / FSDP / DDP / XLA via registration
4. Orthogonal checkpoint strategy — who calls CheckpointManager; Phase 3 cadence remains
5. Reuse DatasetSession placement fields only
6. Sync without framework leaks — collectives only in `infrastructure/`
7. RestartPolicy beside ResumePolicy — never weakens STRICT resume
8. DistributedHealth ≠ TrackingHealth
9. Single-writer eval/export merge; Artifact Contract unchanged
10. Handlers-only pipeline — no stage enum changes for v1
11. FakeDistributedBackend for CI / goldens

## Authority matrix (frozen)

| Concern | Authority | Phase 7 role |
|---------|-----------|--------------|
| Train cursor / status | TrainingSession + TrainerBackend | Placement / sync around train |
| Hardware plan | ResourcePlanner / ExecutionEnvironment | Companion DistributedPlacementResolver |
| Checkpoints | CheckpointManager | Coordinators invoke; never write |
| Resume compatibility | ResumePolicy | RestartPolicy may invoke resume paths |
| Evaluation | EvaluationEngine | Shard / merge around engine |
| Export / Artifact Contract | ExportManager | Single-writer / barrier around export |
| Packing / curriculum | SchedulePlanner | Rank-local plans; reserved metadata only |
| Tracking sink health | TrackingHealth | Dist health may be *mirrored* via tracker |
| Process-group lifecycle | DistributedRuntime | Owns ephemeral mesh only |

## Determinism guarantees

Same portable config + seed + mesh identity material yields identical
`mesh_digest` / placement digests under the fake backend path. Hostnames,
absolute paths, and launcher-only ranks must not enter ExperimentId or mesh
identity material.

## Framework isolation guarantees

- `torch.distributed` / Accelerate / DeepSpeed / FSDP / XLA / NCCL / UCX clients
  remain in `aiodoo_training/infrastructure/` only
- Application distributed package stays framework-free
- AST boundary tests continue to enforce quarantine

## Frozen public contracts

Phase 7 joins Phases 0–6 as a **permanently frozen** public contract.
Canonical governance: [`docs/frozen_public_contracts.md`](../frozen_public_contracts.md).

**AIODOO Training v1 architecture is complete.**

Future work **must preserve** these contracts. It may:

- register real DDP / FSDP / DeepSpeed / Accelerate / XLA / UCX (etc.) backends
- add placement / sampler strategies additively
- record DistributedHealth mirrors via ExperimentTracker

It must **not**:

- redesign DistributedRuntime into a training / checkpoint / resume authority
- widen ResourcePlanner, CheckpointManager, ResumePolicy, or TrainerBackend
  for distributed convenience
- replace TrackingHealth with DistributedHealth
- fold non-portable fields into ExperimentId / mesh_digest / fingerprints
- leak framework distributed types outside infrastructure
- redesign Phase 7 ownership splits without a new ADR and Section 9 process

## Extension points

| Extension | Mechanism |
|-----------|-----------|
| DDP / FSDP / DeepSpeed / Accelerate / XLA | `distributed_backend_registry` + infrastructure adapter |
| Placement / sampler strategies | Placement / sampler registries |
| Communication transports (UCX, etc.) | Infrastructure only; port remains AIODOO types |
| Richer fault policies | Additive RestartPolicy modes via config / domain |
| Optional INIT_DISTRIBUTED stage | Section 9 + ADR only |

**Future distributed implementations (DDP / FSDP / DeepSpeed / XLA / etc.)
MUST register through the existing extension points. They MUST NOT redesign
the architecture.**

## Testing / verification summary

At freeze acceptance (implementation validation completed successfully):

| Gate | Result |
|------|--------|
| Architecture | ADR-0019 Accepted |
| Implementation | Complete |
| Validation | Successful (lint + Phase 7 unit/golden suites and related regression coverage) |
| Bug fixes | Allowed without architectural redesign |
| Redesign | Not permitted |

## Decision

Phase 7 is **permanently frozen**.

Together with Phases 0–6 it forms the complete **AIODOO Training v1** public
surface: foundation, datasets/tokenization/resources, models/adaptation,
training/checkpoint/resume, evaluation/export, packing/curriculum/sampling,
tracking/CLI, and distributed readiness.

Rules for maintenance / extension:

- Never bypass an existing Port or Phase 0–7 Authority.
- Never expose third-party framework types outside infrastructure.
- Never move responsibilities between frozen layers.
- Never treat DistributedRuntime / coordinators as a second training engine.
- If a feature appears to require changing a frozen phase, **stop and explain why**.
- Later extensions must be **additive** (registry / infrastructure) or require
  a **new ADR**.

## Conclusion

**Phase 7 is permanently frozen.**

**AIODOO Training v1 architecture is complete.**

Future development consists of additive infrastructure adapters and work in
other AIODOO repositories. No further architectural work remains for
`aiodoo-training`.

Frozen contracts must not be modified.

## Consequences

- Positive: Real multi-GPU / multi-node backends can land via registration
  without thawing Phases 0–7.
- Negative: Full framework feature parity remains registration-driven extension
  work, not a freeze carve-out or architectural reopen.
