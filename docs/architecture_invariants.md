# Architecture Invariants

Binding summary of permanent rules. The full governance contract is
[Frozen Public Contracts](frozen_public_contracts.md).

These invariants are permanent. Later phases implement against them; they do not
relax them. Architecture preservation takes priority over implementation speed.

## Freeze contract

Phases **0**, **1**, **2**, **3**, **4**, **5**, **6**, and **7** are
**permanently frozen** stable public contracts
([ADR-0021](adr/0021-phase7-freeze.md) for Phase 7).

**AIODOO Training v1 architecture is complete.**

- Do not modify frozen-phase architecture.
- Bug fixes in frozen code are allowed; architectural changes are not.
- Never move responsibilities between frozen layers.
- Never introduce shortcuts that violate ADRs.
- If a future feature appears to require changing a frozen phase, **stop and
  explain why** instead of changing it automatically.

Ledger: [Phase Completion Matrix](phase_completion_matrix.md).

## Phase workflow

```text
Problem
        ↓
Architecture Review
        ↓
       ADR
        ↓
   Implementation
        ↓
    Validation
        ↓
     Freeze ADR
        ↓
   Maintenance
        ↓
Future Extension (if needed — through ADR)
        ↓
     Next Phase
```

A completed phase should normally be **permanently frozen** before the next
phase’s implementation starts. See [Frozen Public Contracts §9.1](frozen_public_contracts.md).

Orchestration vocabulary (Planner / Coordinator / Manager / Runtime / Authority):
[coordinator_conventions.md](coordinator_conventions.md).

Philosophy (informational): [engineering_principles.md](engineering_principles.md).

## Permanent rules (maintenance / extension)

1. **Never bypass an existing Port.**
2. **Never expose third-party framework types** (torch, transformers, peft,
   bitsandbytes, accelerate, …) outside `aiodoo_training/infrastructure/`.
3. **Never move responsibilities** between frozen layers.
4. **Never violate ADRs** for implementation convenience.
5. **Only implementation evolves** from this point; architecture does not.
6. **Never bypass CheckpointManager** for durable checkpoint I/O.
7. **Never bypass ResourcePlanner** for device / precision / accelerator decisions.
8. **Trainer backends** must satisfy [Trainer Backend Contract](trainer_backend_contract.md).
9. **Never bypass SchedulePlanner** for packing / curriculum / sampling
   orchestration; do not introduce competing Managers.
10. **Never convert** PackingStatistics / CurriculumStatistics into runtime
    tracking services inside Phase 5 surfaces.
11. **Never bypass DistributedRuntime** coordinators into a second training
    engine; real DDP/FSDP/DeepSpeed/XLA/etc. register via existing extension points.

## Invariants

### Invariant 1

Domain never imports torch.

### Invariant 2

Infrastructure never imports CLI.

### Invariant 3

Ports never expose third-party types.

### Invariant 4

Experiment fingerprint is deterministic.

### Invariant 5

DatasetSession is immutable.

### Invariant 6

Pipeline orchestrates only.

### Invariant 7

Trainer owns no checkpoint implementation (CheckpointManager orchestrates;
CheckpointStore is the weight port).

### Invariant 8

Evaluation owns no export logic.

### Invariant 9

Inference never belongs to aiodoo-training.

### Invariant 10

Every new backend registers through registries.

### Invariant 11

Model loading and adaptation remain separate (`ModelBackend` vs
`AdaptationStrategy`).

### Invariant 12

Hardware decisions go through `ResourcePlanner` / `ExecutionEnvironment` — never
ad-hoc CUDA / device checks in application or pipeline code.

### Invariant 13

Adapter metadata (`adapter_registry` / `AdapterProfile`) stays independent of
adaptation behavior (`adaptation_registry` / `AdaptationStrategy`).

### Invariant 14

Resumed training under matching STRICT inputs produces the same deterministic
progression as an uninterrupted run (Phase 3 golden invariant).

### Invariant 15

`training_protocol_version` gates semantic resume compatibility independently of
manifest `schema_version`.

### Invariant 16

`SchedulePlanner` is the sole Phase 5 orchestration owner (curriculum →
sampling → packing → statistics).

### Invariant 17

`PackingStatistics` and `CurriculumStatistics` are immutable completed-plan
summaries only — never runtime trackers and never Phase 5 monitoring owners.

### Invariant 18

Frozen `PackingStrategy.pack` and `CurriculumStrategy.plan` signatures are not
widened; rich context uses `bind()`.
