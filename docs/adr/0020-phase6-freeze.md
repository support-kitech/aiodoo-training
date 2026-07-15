# ADR-0020: Phase 6 Permanent Freeze

## Status

**Accepted** — Phase 6 permanently frozen.

## Context

Phase 6 (Tracking, Experiment Management & CLI) completed architecture design
and hardening
([`docs/phase6-tracking-cli-architecture.md`](../phase6-tracking-cli-architecture.md)),
architecture acceptance ([ADR-0018](0018-phase6-tracking-cli.md)),
implementation, engineering hardening (including immutable
`tracking_protocol_version` metadata and `TrackingCapability.supports`), and
validation (ruff, pytest Phase 6 suites, related golden/determinism coverage).

Phases 0–5 are already permanently frozen (ADRs 0007, 0012, 0014, 0015, 0017
and related governance).

## Phase summary

Phase 6 delivers **observational** experiment/run tracking, local history,
provenance, reporting, logging polish, and CLI UX **without** owning training,
checkpoint, evaluation, export, packing, or curriculum authority, and without
redesigning frozen `ExperimentTracker` port signatures or the pipeline
orchestrator.

## Scope

**In scope (frozen):**

- ExperimentSession / ExperimentHistory / ExperimentCatalog / ExperimentRegistry
- RunRecord / RunMetadata / RunIndex / run lifecycle
- TrackingPolicy / LoggingPolicy / ReportPolicy / RetentionPolicy
- TrackingCapability / TrackingHealth (sink diagnostics; not training health)
- TrackingCoordinator + TrackingContext + TrackingStore / MetadataStore /
  HistoryStore / MetricStore / ArtifactHistoryStore
- Provenance builders + digests (portable; no host noise)
- Training/evaluation/export report projections and Experiment/Run summaries
- `TRACKING_PROTOCOL_VERSION` / `tracking_protocol_version` immutable metadata
  (history migration only — not ExperimentId / fingerprints / ResumePolicy /
  Artifact Contract)
- CLI CommandRegistry / CommandContext / CommandBuilder / CLIProfile
- Config fragments, builders, factories; Null / Filesystem / MLflow trackers
  under `infrastructure/` (soft-import where applicable)
- Pipeline handler hooks only (no stage enum reorder)
- Unit / golden tracking determinism suites

**Out of scope (later phases):**

- Distributed process groups / multi-GPU readiness (Phase 7)
- Elastic membership / cloud job schedulers
- Redesign of ExperimentTracker port signatures
- Owning train / checkpoint / eval / export / packing authority

## Responsibilities introduced

| Owner | Responsibility |
|-------|----------------|
| TrackingCoordinator | Coordinates observational record around frozen authorities |
| ExperimentTracker (infra adapters) | Sink I/O via frozen port + `bind()` |
| TrackingCapability / TrackingHealth | Declared features + sink health snapshots |
| CLI CommandRegistry / CLIProfile | Command dispatch + UX presets |
| MetricStore / ArtifactHistoryStore | Persist observational copies only |
| Provenance builders | Portable digests for explanation / replay |

## Architecture implemented

Canonical specification:
[`docs/phase6-tracking-cli-architecture.md`](../phase6-tracking-cli-architecture.md)

Key decisions preserved from ADR-0018:

1. Tracking is observational only — never authoritative for train/eval/export/packing
2. Backends behind frozen `ExperimentTracker` via binder + registry
3. Additive ExperimentSession / RunRecord / provenance / history / catalog types
4. Consume PackingStatistics / CurriculumStatistics / EvaluationReport / export
   descriptors as immutable recordings
5. Logger/LogSink separate from ExperimentTracker
6. Default CI tracker `null`; filesystem JSONL for local UX
7. Nonfatal sink errors by default
8. TrackingCapability / TrackingHealth / CLIProfile hardening DTOs
9. `tracking_protocol_version` metadata only; `TrackingCapability.supports` helper

## Authority matrix (frozen)

| Concern | Authority | Phase 6 role |
|---------|-----------|--------------|
| Train cursor / status | TrainingSession + TrainerBackend | Observe / mirror |
| Checkpoints | CheckpointManager | Index references only |
| Evaluation | EvaluationEngine | Record report summaries |
| Export / Artifact Contract | ExportManager | Record lineage pointers |
| Packing / curriculum | SchedulePlanner | Record statistics DTOs |
| Tracking sink health | TrackingHealth | Sink diagnostics only |

## Determinism guarantees

Same portable config + seed + environment digests yield identical provenance
digest material (excluding wall clocks). Enabling/disabling trackers must not
change TrainingProgress / eval / export outcomes under nonfatal sink policy.

## Framework isolation guarantees

- MLflow / W&B / TensorBoard / OpenTelemetry clients remain in
  `aiodoo_training/infrastructure/` only
- Application tracking and CLI stay framework-free
- AST boundary tests continue to enforce quarantine

## Frozen public contracts

Phase 6 joins Phases 0–5 as a **permanently frozen** public contract.
Canonical governance: [`docs/frozen_public_contracts.md`](../frozen_public_contracts.md).

Future phases (7+) **must preserve** these contracts. They may:

- register new tracker backends / capability flags
- add CLI commands / profiles additively
- consume tracking DTOs and mirrors for distributed observations

They must **not**:

- redesign TrackingCoordinator into a train/checkpoint/eval/export authority
- replace TrackingHealth with training or cluster authority
- widen frozen `ExperimentTracker` signatures for convenience
- fold non-portable fields into ExperimentId / fingerprint material
- use `tracking_protocol_version` as a ResumePolicy or Artifact Contract gate
- leak tracker SDKs outside infrastructure
- redesign Phase 6 ownership splits without a new ADR and Section 9 process

## Extension points

| Extension | Mechanism |
|-----------|-----------|
| W&B / TensorBoard / OTEL sinks | Registry + infrastructure adapter + capability table |
| Richer reports | Additive report DTOs / writers |
| Distributed health mirrors | Phase 7 records DistributedHealth *via* ExperimentTracker — does not replace TrackingHealth |
| CLI subcommands | CommandRegistry registration |

## Testing / verification summary

At freeze acceptance (implementation validation completed successfully):

| Gate | Result |
|------|--------|
| Architecture | ADR-0018 Accepted |
| Implementation | Complete |
| Validation | Successful (lint + Phase 6 unit/golden suites and related regression coverage) |
| Bug fixes | Allowed without architectural redesign |
| Redesign | Not permitted |

## Decision

Phase 6 is **permanently frozen**.

Together with Phases 0–5 it forms the stable public surface covering foundation,
datasets/tokenization/resources, models/adaptation, training/checkpoint/resume,
evaluation/export, packing/curriculum/sampling, and tracking/CLI.

Rules for Phase 7+:

- Never bypass an existing Port or Phase 0–6 Authority.
- Never expose third-party framework types outside infrastructure.
- Never move responsibilities between frozen layers.
- Never treat tracking as authoritative run control.
- If a feature appears to require changing a frozen phase, **stop and explain why**.
- Later extensions must be **additive** or require a **new ADR**.

## Conclusion

**Phase 6 is permanently frozen.**

Future phases may extend it only through additive registrations,
configuration, or new ADRs.

Frozen contracts must not be modified.

## Consequences

- Positive: Phase 7 distributed readiness can consume stable tracking mirrors
  and CLI surfaces without thawing Phase 6.
- Negative: Full third-party tracker feature parity and rich train CLI remain
  registration-driven extension work, not freeze carve-outs.
