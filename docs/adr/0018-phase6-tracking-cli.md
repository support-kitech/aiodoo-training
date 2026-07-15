# ADR-0018: Phase 6 Tracking, Experiment Management & CLI Architecture

## Status

**Accepted** — Implementation authorized. Architecture is frozen; implement exactly
as specified in `docs/phase6-tracking-cli-architecture.md` (no redesign).

## Context

Phases 0–5 are permanently frozen public contracts (ADRs 0007, 0012, 0014,
0015, 0017 and related freezes). Phase 6 must deliver experiment/run tracking,
local history, provenance, reporting, logging polish, and CLI UX **without**
owning training logic and without redesigning:

- TrainingSession / CheckpointManager / ResumePolicy
- EvaluationEngine / ExportManager / Artifact Contract
- SchedulePlanner / PackingStatistics / CurriculumStatistics
- Frozen `ExperimentTracker` port signatures
- Pipeline orchestrator / stage enum order
- Repository root execution model

A frozen `ExperimentTracker` port and `TrackingSpec` / `tracker_registry`
already exist as extension points. Phase 3–5 emit observational surfaces
(events, MetricSnapshots, statistics DTOs, artifact descriptors) that Phase 6
may record.

## Decision

Adopt the architecture specification in
[`docs/phase6-tracking-cli-architecture.md`](../phase6-tracking-cli-architecture.md),
including the post-review hardening amendments below.

### Core decisions

1. Tracking is **observational only** — never authoritative for train/eval/
   export/packing control.
2. Implement tracker backends behind the frozen `ExperimentTracker` port via
   `bind(TrackingContext)`; do not widen port signatures.
3. Introduce additive ExperimentSession / RunRecord / provenance / history /
   catalog types and local TrackingStore / MetadataStore / HistoryStore.
4. Consume PackingStatistics, CurriculumStatistics, EvaluationReport, and
   export descriptors as immutable recordings — do not convert them into
   runtime trackers.
5. Polish CLI via Command Registry / Command Context while keeping root scripts
   and “not packaged” execution model.
6. Separate Logger/LogSink (diagnostics) from ExperimentTracker (science
   records).
7. Default CI tracker remains `null`; filesystem JSONL is the local UX default.
8. Tracker sink failures are nonfatal by default so goldens and train outcomes
   remain unchanged when tracking is enabled or disabled.

### Hardening amendments (no redesign)

9. **`TrackingCapability`** — immutable domain DTO declaring backend feature
   flags (`supports_metrics`, `supports_artifacts`, `supports_lineage`,
   `supports_live_stream`, `supports_resume`, `supports_remote`, …). Improves
   MLflow / W&B / TensorBoard / OpenTelemetry / custom compatibility.
   Coordinators skip unsupported emissions. **Does not** redesign
   `ExperimentTracker`.
10. **`TrackingHealth`** — immutable backend health snapshot
    (`HEALTHY` | `DEGRADED` | `READ_ONLY` | `OFFLINE` | `FAILED`) for
    diagnostics and `doctor.py`. **Backend sink health only** — never training
    health, never ResumePolicy input.
11. **`CLIProfile`** — immutable UX presets (`default` | `minimal` | `verbose`
    | `json` | `ci`). Improves developer UX. **Does not** redesign Command
    Registry.
12. **`tracking_protocol_version`** — immutable observational metadata
    (`TRACKING_PROTOCOL_VERSION = "1"`) on ExperimentSession / RunRecord /
    catalog indexes for future tracking-history migration only. **Must not**
    affect ExperimentId, fingerprints, ResumePolicy, training, evaluation,
    export, Artifact Contract, or provenance digests.
13. **`TrackingCapability.supports(feature)`** — convenience helper over
    existing boolean fields. **Does not** redesign the DTO or public behaviour.

## Consequences

### Positive

- Reproducible provenance and local experiment/run UX.
- Clear registration path for MLflow / W&B / TensorBoard / OTEL with explicit
  capability negotiation.
- Doctor can report tracker health without coupling to train status.
- No coupling leak into aiodoo-models (Artifact Contract unchanged).

### Negative

- Dual stores (checkpoints/exports vs tracking indexes) require careful UX so
  users do not confuse tracking paths with Artifact Contract packages.
- Nonfatal sink errors need explicit warnings and tests.

### Constraints

- Never bypass CheckpointManager / ExportManager / EvaluationEngine /
  SchedulePlanner.
- Never put tracker SDKs outside `infrastructure/`.
- Never alter ExperimentId derivation with non-portable tracking timestamps.
- Never widen frozen `ExperimentTracker` for capability/health (binders + DTOs).

## Completeness

After hardening review: **Phase 6 architecture is complete.** No further
architectural improvements are justified before acceptance.

## Implementation gate

This ADR is **Accepted**. Implementation proceeded under the architecture in
`docs/phase6-tracking-cli-architecture.md`.

Permanent freeze of Phase 6 public contracts is recorded in
[ADR-0020](0020-phase6-freeze.md).
