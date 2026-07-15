# ADR-0017: Phase 5 Permanent Freeze

## Status

**Accepted** — Phase 5 permanently frozen.

## Context

Phase 5 (Packing, Curriculum & Sampling) completed architecture design and
hardening ([`docs/phase5-packing-curriculum-architecture.md`](../phase5-packing-curriculum-architecture.md)),
architecture acceptance ([ADR-0016](0016-phase5-packing-curriculum.md)),
implementation, engineering review, and verification (ruff, mypy, pytest,
coverage). Phases 0–4 are already permanently frozen (ADRs 0007, 0012, 0014,
0015 and related governance).

## Phase summary

Phase 5 delivers deterministic **sequence packing**, **curriculum planning**,
and **sampling** so training can raise token utilization and shape example order
**without** redesigning the frozen training engine, `DatasetSession`, pipeline
orchestrator, or Phase 0–4 port signatures.

## Scope

**In scope (frozen):**

- PackingSession / PackingState / PackingProgress / PackedSpan
- PackingLifecycle + PackingContext (binder pattern)
- PackingPolicy / MemoryPackingPolicy
- Packing strategies: `none`, `concat`, `best_fit`, `length_aware`
- CurriculumSession / CurriculumState / CurriculumProgress
- CurriculumLifecycle + CurriculumContext
- Curriculum strategies: `none`, `sequential`, `weighted` (`weighted_mix`),
  `difficulty`, `random`, `mixed`
- Additive `SamplingStrategy` port + strategies: `identity`, `weighted`,
  `temperature`, `balanced`
- `SchedulePlanner` (sole Phase 5 orchestration owner) + `SchedulePlan`
- Immutable `PackingStatistics` / `CurriculumStatistics` (completed-plan only)
- Phase 5 config fragments, builders, factories, registry keys, bootstrap
- Pipeline handlers `PlanPackingStage` / `PlanCurriculumStage` (idempotent planner)
- CPU golden packing / curriculum / sampling determinism suites

**Out of scope (later phases):**

- Experiment tracking sinks / dashboards (Phase 6) — may *consume* statistics
- CLI polish beyond existing thin scripts (Phase 6)
- FlashAttention-aware / varlen packing kernels (registration later)
- Adaptive / online curriculum feedback loops
- RLHF / preference sampling training loops
- Distributed packing coordination (Phase 7)

## Responsibilities introduced

| Owner | Responsibility |
|-------|----------------|
| PackingLifecycle | Allowed PackingStatus transitions (COW) |
| CurriculumLifecycle | Allowed CurriculumStatus transitions (COW) |
| SchedulePlanner | Sole owner of curriculum → sampling → packing → statistics |
| PackingStrategy (infra/app adapters) | Emit packed `TokenBatch` via frozen `pack`; rich context via `bind()` |
| CurriculumStrategy | Emit stage sequences via frozen `plan`; context via `bind()` |
| SamplingStrategy | Deterministic reorder / strata via additive `sample` |
| Pipeline handlers | Delegate to SchedulePlanner only; no parallel Managers |
| PackingStatistics / CurriculumStatistics | Immutable completed-plan summaries (not runtime trackers) |

## Packing architecture

- Frozen port: `PackingStrategy.pack(examples, PackingSpec) -> Iterator[TokenBatch]`
- Complexity targets: none/concat **O(n)**; length_aware / best_fit **O(n log n)**
- Overflow policy: `defer` | `truncate` | `reject` via `PackingPolicy`
- Packed boundaries: separator token and/or `TokenBatch.metadata["packed_spans"]`
- Label pads / separators use `IGNORE_INDEX`

## Curriculum architecture

- Frozen port: `CurriculumStrategy.plan(examples, CurriculumSpec) -> Sequence[Sequence[TrainingExample]]`
- Stage membership / order determined by mode; seed-bound where stochastic
- Stage cursors live on `CurriculumSession` — **DatasetSession unchanged**

## Sampling architecture

- Additive port: `SamplingStrategy.sample(examples, SamplingSpec) -> Sequence[TrainingExample]`
- Registry-driven; applied per curriculum stage inside SchedulePlanner
- Deterministic under identical seed + spec

## SchedulePlanner ownership

`SchedulePlanner` is the **only** Phase 5 application orchestrator that owns:

1. curriculum planning  
2. per-stage sampling  
3. packing  
4. emission of CurriculumStatistics + PackingStatistics  

**Forbidden:** PackingManager, CurriculumManager, SamplingManager, or other
competing Managers. Pipeline handlers and builders **delegate** to the planner.

Idempotency: frozen pipeline enum order keeps `PLAN_PACKING` before
`PLAN_CURRICULUM`; planner materializes the full schedule on first need and
reuses on subsequent handlers.

## PackingStatistics

Immutable domain DTO summarizing a **completed** packing plan:

- Fingerprint + backend + example/sequence/token occupancy metrics
- Overflow counters
- **No** timestamps required for golden equality
- **Not** a runtime tracker; Phase 6 may later emit copies to sinks

## CurriculumStatistics

Immutable domain DTO summarizing a **completed** curriculum plan:

- Fingerprint + backend + stage_count / examples_per_stage / stage_names
- **Not** a runtime tracker; same non-ownership rules as packing statistics

## Determinism guarantees

Same dataset + seed + config + packing/curriculum/sampling backends +
`ExecutionEnvironment` identity material must yield identical:

- curriculum stage example_id lists  
- sampling order  
- packing / `TokenBatch` tensors and packed spans  
- packing / curriculum / sampling fingerprints  
- `PackingStatistics` and `CurriculumStatistics` field values  

## Golden test guarantees

CPU-only golden suites cover packing, curriculum, sampling, planner
determinism, statistics equality, and resume fingerprint stability of plan
outputs. CI does not require GPU or large model downloads.

## Framework isolation guarantees

- Torch / Transformers / PEFT / bitsandbytes / Accelerate imports remain confined
  to `aiodoo_training/infrastructure/`
- Packing / curriculum / sampling application code is framework-free
- AST boundary tests continue to enforce quarantine

## Frozen public contracts

Phase 5 joins Phases 0–4 as a **permanently frozen** public contract.
Canonical governance: [`docs/frozen_public_contracts.md`](../frozen_public_contracts.md).

Future phases (6+) **must preserve** these contracts. They may:

- register new packing / curriculum / sampling backends
- add Phase 6 trackers that *consume* statistics DTOs
- extend domain additively without mutating frozen types

They must **not**:

- widen frozen `pack` / `plan` signatures (or other frozen ports) for convenience
- introduce competing Managers alongside SchedulePlanner
- mutate DatasetSession for packing/curriculum cursors
- turn PackingStatistics / CurriculumStatistics into runtime trackers
- leak framework types outside infrastructure
- redesign Phase 5 ownership splits without a new ADR and Section 9 process

## Extension points

| Extension | Mechanism |
|-----------|-----------|
| FlashAttention / varlen packing | New `packing_registry` key + infrastructure adapter |
| Adaptive curriculum | New curriculum backend; may read prior metrics via binder later |
| RL / preference sampling | New `sampling_registry` key |
| Tracking / dashboards | Phase 6 consumes PackingStatistics / CurriculumStatistics |
| Distributed packing metadata | Phase 7 — rank-local plans + reserved session metadata |

## Testing summary

Verified at freeze:

- Unit: strategies, lifecycles, sessions, factories, builders, config, planner,
  pipeline handlers
- Golden: packing / curriculum / sampling determinism
- Determinism + statistics equality + resume fingerprint stability
- Complexity sanity (best_fit)
- Registry + boundary suites
- Full repo: **302** pytest tests passed (CPU-only CI)

## Coverage summary

Overall coverage **~83%** of measured `aiodoo_training` (infrastructure omit by
design; `fail_under=60`). Phase 5 packing/curriculum/sampling modules **~86%**.

## Verification summary

At freeze acceptance:

| Gate | Result |
|------|--------|
| ruff | Clean |
| mypy (strict) | Clean |
| pytest | 302 passed |
| coverage | Maintained above policy floor |
| Boundary / framework isolation | Pass |

## Decision

Phase 5 is **permanently frozen**.

Together with Phases 0–4 it forms the stable public surface covering foundation,
datasets/tokenization/resources, models/adaptation, training/checkpoint/resume,
evaluation/export, and packing/curriculum/sampling.

Rules for Phase 6+:

- Never bypass an existing Port or SchedulePlanner ownership rules.
- Never expose third-party framework types outside infrastructure.
- Never move responsibilities between frozen layers.
- Never convert completed-plan statistics into Phase 5 tracking services.
- If a feature appears to require changing a frozen phase, **stop and explain why**.

## Conclusion

**Phase 5 is permanently frozen.**

Future phases may extend it only through additive registrations,
configuration, or new ADRs.

Frozen contracts must not be modified.

## Consequences

- Positive: Phase 6 tracking has stable plan/statistics surfaces; training
  efficiency extensions stay registration-driven.
- Negative: Pipeline enum order still requires idempotent planning; packed span
  correctness remains a permanent golden invariant.
