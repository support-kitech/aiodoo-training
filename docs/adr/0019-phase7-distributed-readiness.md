# ADR-0019: Phase 7 Distributed Readiness Architecture

## Status

**Accepted** — Phase 7 architecture. Permanently frozen under
[ADR-0021](0021-phase7-freeze.md). Canonical specification:
`docs/phase7-distributed-readiness-architecture.md` (no redesign).

## Context

Phases 0–6 are permanently frozen public contracts (ADRs 0007–0018, 0020 and related
freezes; Phase 6 permanently frozen under [ADR-0020](0020-phase6-freeze.md)). Phase 7 must prepare
AIODOO Training for multi-GPU, multi-process, multi-node, cloud launchers, and
future TPU/XLA paths **without** redesigning:

- `TrainerBackend` / trainer backend contract
- `CheckpointManager` / Phase 3 `CheckpointPolicy` / `ResumePolicy`
- `DatasetSession`
- `ResourcePlanner` / `ExecutionEnvironment` (consume + companion plan only)
- `SchedulePlanner` / packing / curriculum ports
- EvaluationEngine / ExportManager / Artifact Contract
- Phase 6 tracking (`TrackingHealth` remains sink health only)
- Pipeline stage enum order / orchestrator
- Repository root execution model

Frozen foresight already reserved:

- `DistributedSpec` on `ExperimentConfig`
- `DatasetSession` placement fields (`world_size`, ranks, shards)
- `AcceleratorKind` (`DDP` | `FSDP` | `DEEPSPEED` | `ACCELERATE`)
- `ExecutionEnvironment.accelerator`

## Decision

Adopt the architecture specification in
[`docs/phase7-distributed-readiness-architecture.md`](../phase7-distributed-readiness-architecture.md).

### Core decisions

1. **Distributed façade, not new training authority** — `DistributedSession` /
   `DistributedContext` / `DistributedRuntime` own process-group lifecycle,
   topology, and sync façades. They never replace TrainingSession,
   CheckpointManager, or ResumePolicy.
2. **Companion resource plan** — `DistributedPlacementResolver` consumes frozen
   `ExecutionEnvironment` + `DistributedSpec` to produce `DeviceMesh` /
   `PlacementPlan` / communication policies. **Do not** widen
   `ResourcePlanner` signatures.
3. **Registry-driven runtimes** — Accelerate, DeepSpeed, FSDP, DDP, XLA, and
   custom backends via `distributed_backend_registry` + trainer registry keys;
   context via `bind()` only.
4. **Orthogonal checkpoint strategy** — Phase 7
   `DistributedCheckpointPolicy` + Shard/Merge/Replica policies coordinate
   *who* calls frozen `CheckpointManager`. Phase 3 `CheckpointPolicy` remains
   cadence/retention only.
5. **Reuse DatasetSession** — `ShardPlanner` / `DistributedSampler` /
   `EpochCoordinator` populate existing placement fields only.
6. **Sync without framework leaks** — Barrier/Broadcast/Reduction/Aggregation
   policies + `SyncFacade`; collectives only in `infrastructure/`.
7. **Fault tolerance beside resume** — `RestartPolicy` decides relaunch;
   `ResumePolicy` still gates checkpoint compatibility.
8. **DistributedHealth ≠ TrackingHealth** — runtime mesh observations only.
9. **Single-writer eval/export merge** — coordinators outside EvaluationEngine /
   ExportManager; Artifact Contract unchanged.
10. **Handlers-only pipeline** — no stage enum changes for v1.
11. **CI-first fake backend** — `FakeDistributedBackend` enables CPU goldens
    without GPU.

### Explicit non-decisions (out of v1)

- Elastic world-size / torchelastic membership
- Cloud-native job schedulers inside this repository
- Bit-exact cross-SKU GPU numerics
- Mandatory new pipeline stage `INIT_DISTRIBUTED` (future Section 9 only)

## Consequences

### Positive

- Multi-process readiness without thawing Phases 0–6.
- Framework churn quarantined behind ports/registries.
- Deterministic fake path for CI and goldens.
- Clear ownership for sharded checkpoints vs Artifact Contract packages.

### Negative

- Two checkpoint-related policy names (`CheckpointPolicy` vs
  `DistributedCheckpointPolicy`) require careful docs/UX.
- Dual health surfaces (Tracking vs Distributed) need explicit CLI separation.

### Constraints

- Never bypass CheckpointManager / ExportManager / EvaluationEngine /
  SchedulePlanner / ResourcePlanner.
- Never put torch.distributed / Accelerate / DeepSpeed / XLA outside
  `infrastructure/`.
- Never fold hostnames or launcher-only ranks into ExperimentId material.
- Never use RestartPolicy to weaken STRICT resume.

## Completeness

After design review of the Phase 7 architecture document: **Phase 7 architecture
is complete and Accepted.** Permanent freeze: [ADR-0021](0021-phase7-freeze.md).

## Freeze

This ADR remains **Accepted** as the Phase 7 architecture decision.
Implementation under `docs/phase7-distributed-readiness-architecture.md` is
permanently frozen by [ADR-0021](0021-phase7-freeze.md). No Phase 8 architectural
work in this repository; extend via registration only.
