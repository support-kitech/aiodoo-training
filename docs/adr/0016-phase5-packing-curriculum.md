# ADR-0016: Phase 5 Packing & Curriculum Architecture

## Status

**Accepted** — Phase 5 architecture frozen. Implementation complete and
permanently frozen under [ADR-0017](0017-phase5-freeze.md).

## Context

Phases 0–4 are permanently frozen public contracts (ADRs 0007, 0012, 0014,
0015 and related freezes). Phase 5 must improve training efficiency via sequence
packing, curriculum learning, and sampling **without** redesigning the training
engine, `DatasetSession`, frozen packing/curriculum **port signatures**, the
pipeline orchestrator, or checkpoint resume protocol.

Frozen ports already exist:

- `PackingStrategy.pack(examples, spec) -> Iterator[TokenBatch]`
- `CurriculumStrategy.plan(examples, spec) -> Sequence[Sequence[TrainingExample]]`

Pipeline stages `PLAN_PACKING` and `PLAN_CURRICULUM` already exist as no-ops /
stubs.

## Decision

Adopt the architecture specification in
[`docs/phase5-packing-curriculum-architecture.md`](../phase5-packing-curriculum-architecture.md),
including the post-review hardening amendments below.

### Core decisions

1. Add PackingSession / CurriculumSession lifecycles (application binders;
   domain COW DTOs).
2. Implement packing strategies (`none`, `concat`, `best_fit`, `length_aware`)
   behind the frozen `PackingStrategy` port via `bind()`.
3. Implement curriculum strategies (`none`, `sequential`, `weighted`,
   `difficulty`, `random`, `mixed`) behind the frozen `CurriculumStrategy` port.
4. Introduce additive `SamplingStrategy` port + registry (not frozen historically).
5. Use an idempotent `SchedulePlanner` so curriculum/sampling precede packing
   **without reordering frozen pipeline stage enums**.
6. Preserve `DatasetSession` unchanged; store stage/pack cursors on Phase 5
   sessions; fold fingerprints into config fingerprint material.
7. Keep FlashAttention-aware packing, adaptive curriculum, RLHF/DPO sampling as
   registration-based extension points.

### Hardening amendments (no redesign)

8. **`PackingStatistics`** — immutable domain DTO summarizing a **completed**
   packing plan only. Does not redesign `PackingSession`. Does not introduce
   runtime tracking. Produced when the plan reaches `READY` (or skipped
   identity). Consumable later by Phase 6 tracking/visualization.
9. **`CurriculumStatistics`** — immutable domain DTO summarizing a **completed**
   curriculum plan only. Does not redesign `CurriculumSession`. Same
   non-tracking rules as packing statistics.
10. **`SchedulePlanner` is the sole Phase 5 orchestration owner.** Pipeline
    handlers and lifecycles defer to it. Do **not** introduce PackingManager,
    CurriculumManager, SamplingManager, or other competing Managers.
11. **Determinism:** `PackingStatistics` and `CurriculumStatistics` are pure
    projections of completed plans + portable inputs and must be bit-equal
    under golden replay (no timestamps in the golden surface).

## Consequences

### Positive

- Higher token utilization and lower padding without TrainerBackend changes.
- Deterministic, golden-testable ordering, packing, and plan statistics on CPU.
- Clear Phase 6 integration surface (statistics DTOs) without Phase 5 owning
  tracking sinks.
- Clear extension path for FA2, adaptive curricula, and preference sampling.

### Negative

- Pipeline enum order (`PLAN_PACKING` before `PLAN_CURRICULUM`) requires
  careful idempotent planning to preserve semantic dependencies.
- Packed span metadata must be rigorously tested to avoid label corruption.

### Neutral / constraints

- No widening of frozen `pack` / `plan` signatures.
- No DatasetSession field additions.
- No TrainingEngine redesign.
- No additional Managers beyond `SchedulePlanner`.

## Completeness

After hardening review: **Phase 5 architecture is complete.** No further
architectural improvements are justified before acceptance.

## Implementation gate

This ADR is **Accepted**. Implementation is complete. Permanent freeze is
recorded in [ADR-0017](0017-phase5-freeze.md). No Phase 6 work without
explicit authorization.
