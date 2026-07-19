# AIODOO Training Architecture

Status: **Phase 0 frozen** · **Phase 1 frozen** · **Phase 2 frozen** · **Phase 3 frozen** · **Phase 4 frozen** · **Phase 5 frozen** · **Phase 6 frozen** · **Phase 7 frozen**

**AIODOO Training v1 architecture complete.**

This document records the approved architecture. It is not a redesign surface —
later work extends this structure. See also:

- [Frozen Public Contracts](frozen_public_contracts.md) — binding engineering governance
- [Architecture Invariants](architecture_invariants.md) — permanent engineering rules
- [Phase Completion Matrix](phase_completion_matrix.md) — freeze ledger
- [Coordinator conventions](coordinator_conventions.md) — Planner / Coordinator / Manager / Runtime / Authority naming
- [Engineering Principles](engineering_principles.md) — engineering philosophy (informational)
- [Phase 3 Training Engine Architecture](phase3-training-engine-architecture.md) — **permanently frozen**
- [Phase 4 Evaluation & Export Architecture](phase4-evaluation-export-architecture.md) — **permanently frozen**
- [Artifact Contract](artifact_contract.md) — Training → Models handoff
- [Phase 5 Packing & Curriculum Architecture](phase5-packing-curriculum-architecture.md) — **permanently frozen**
- [Phase 6 Tracking & CLI Architecture](phase6-tracking-cli-architecture.md) — **permanently frozen** · [ADR-0018](adr/0018-phase6-tracking-cli.md) · [ADR-0020](adr/0020-phase6-freeze.md)
- [Phase 7 Distributed Readiness Architecture](phase7-distributed-readiness-architecture.md) — **permanently frozen** · [ADR-0019](adr/0019-phase7-distributed-readiness.md) · [ADR-0021](adr/0021-phase7-freeze.md)
- [Trainer Backend Contract](trainer_backend_contract.md) — mandatory trainer checklist
- [ADR-0014](adr/0014-phase3-freeze.md) — Phase 3 permanent freeze
- [ADR-0015](adr/0015-phase4-evaluation-export.md) — Phase 4 architecture + freeze governance
- [ADR-0016](adr/0016-phase5-packing-curriculum.md) — Phase 5 architecture
- [ADR-0017](adr/0017-phase5-freeze.md) — Phase 5 permanent freeze
- [ADR-0018](adr/0018-phase6-tracking-cli.md) — Phase 6 architecture (Accepted)
- [ADR-0019](adr/0019-phase7-distributed-readiness.md) — Phase 7 architecture (Accepted)
- [ADR-0020](adr/0020-phase6-freeze.md) — Phase 6 permanent freeze
- [ADR-0021](adr/0021-phase7-freeze.md) — Phase 7 permanent freeze

## Status terminology

| Term | Meaning |
|------|---------|
| **Permanently frozen** | Architecture Accepted, Implementation Complete, Freeze ADR Accepted |
| **Accepted** | Design ADR Accepted (architecture may already be permanently frozen) |
| **Proposed** | Architecture under review; no implementation authorization |

## Responsibility

`aiodoo-training` consumes protocol datasets from `aiodoo-datasets` and produces
**Capability Packages** (and ArtifactBundles) for handoff to `aiodoo-validation`
and `aiodoo-model`. It does **not** compose Development / Reasoning products.

It does **not** generate datasets and does **not** perform inference.

Lifecycle documentation (authoritative for package/capability/product language):

- [Terminology](terminology.md)
- [Ownership](ownership.md)
- [Capability Model](capability_model.md)
- [Product Model](product_model.md)
- [Lifecycle](lifecycle.md)
- [Metadata Ownership](metadata_ownership.md)
- [Freeze Readiness](freeze_readiness.md)
- [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md)

## Execution model

Internal repository — not packaged for PyPI. Run `python3 <script>.py` from the
repository root (same model as `aiodoo-datasets`).

## Layers

1. **Domain** — immutable DTOs and enums (`DatasetSession`, `TokenBatch`,
   `ExecutionEnvironment`, `TrainingSession`, packing/curriculum sessions &
   statistics, opaque model handles, …)
2. **Ports** — abstract interfaces (`TokenizerPort`, `ChatTemplate`,
   `ResourcePlanner`, `ModelBackend`, `TrainerBackend`, `CheckpointStore`,
   `RngController`, `PackingStrategy`, `CurriculumStrategy`, additive
   `SamplingStrategy` / optimizer / scheduler / callback / evaluator / exporter, …)
3. **Application** — use-cases (`training/`, `evaluation/`, `export/`,
   `packing/` SchedulePlanner, `curriculum/`, `sampling/`, `tracking/`
   TrackingCoordinator, `distributed/` DistributedRuntime)
4. **Pipeline** — ordered stage orchestration (includes `RESOLVE_EXECUTION`,
   train/resume, evaluate/export, packing/curriculum planning handlers)
5. **Config** — YAML composition, validation, resolution, hashing
   (execution, distributed, training, evaluation/export, packing/curriculum/sampling)
6. **Registries / Builders / Factories** — wiring without framework lock-in  
   includes ChatTemplateRegistry, resource_planner_registry, trainer /
   checkpoint / rng / optimizer / scheduler / callback / packing / curriculum /
   sampling / evaluator / exporter / distributed_backend / placement registries
7. **Datasets** — JSONL loading, validation, mixing, fingerprinting, cache
8. **Tokenization** — templates, masking, TokenBatch pipeline
9. **Determinism** — fingerprints and seed management
10. **Infrastructure** — HF tokenizer + chat templates + static CPU resource
    planner + HF Causal LM + PEFT + stub/HF trainer & checkpoint + RNG/optimizer
    stubs + stub/HF eval/export + tracking + FakeDistributedBackend (quarantined).
11. **CLI** — thin root scripts + shared command helpers
12. **Models / Adaptation** — Phase 2 orchestration (`ModelLoader`,
    `AdaptationApplier`) using frozen ports only

## Frozen decisions (do not reopen)

- Immutable domain
- Ports over frameworks
- Pipeline orchestration
- Experiment fingerprint from composed (portable) config
- Adaptation separated from model loading
- Repository boundaries / source execution
- **DatasetSession** for dataset consumption state (resume / shard fields included)
- **ChatTemplateRegistry** for family-specific prompting
- **Resource management** via policies + `ResourcePlanner` + `ExecutionEnvironment`
  (ADR-0009)
- **Model loading + adaptation** (ADR-0010 / ADR-0011) — opaque handles,
  `QuantizationPolicy`, `adapter_registry` profiles, no ModelSession
- **Training engine** (ADR-0013 / ADR-0014) — TrainingSession, CheckpointManager,
  ResumePolicy, trainer backend contract, resume-equivalence invariant
- **Evaluation + export** (ADR-0015) — EvaluationSession / ExportSession,
  Artifact Contract, quality gates; Evaluator/Exporter signatures unchanged
- **Packing + curriculum + sampling** (ADR-0016 / ADR-0017) — SchedulePlanner
  sole owner; PackingStatistics / CurriculumStatistics completed-plan only;
  frozen `pack` / `plan` signatures; additive SamplingStrategy
- **Tracking + experiment management + CLI** (ADR-0018 / ADR-0020) —
  observational TrackingCoordinator; TrackingCapability / TrackingHealth;
  frozen ExperimentTracker signatures unchanged
- **Distributed readiness** (ADR-0019 / ADR-0021) — companion DistributedRuntime /
  PlacementResolver; FakeDistributedBackend for CI; no ResourcePlanner /
  CheckpointManager / ResumePolicy redesign — **permanently frozen**

## Dependency rule

Outer layers depend inward. Domain never imports torch, transformers, or PEFT.
Infrastructure is the only package allowed to import third-party ML libraries.

Public port signatures use domain types and opaque handles — never framework
classes.

## Phase map

| Phase | Scope | Status |
|-------|--------|--------|
| 0 | Foundation | **Permanently frozen** |
| 1 | Dataset loading + tokenization + resource layer | **Permanently frozen** |
| 2 | Model + adaptation backends | **Permanently frozen** |
| 3 | Trainer + checkpoint + resume | **Permanently frozen** |
| 4 | Evaluation + export | **Permanently frozen** — [artifact_contract.md](artifact_contract.md) |
| 5 | Packing + curriculum + sampling | **Permanently frozen** — [phase5-packing-curriculum-architecture.md](phase5-packing-curriculum-architecture.md) · [ADR-0017](adr/0017-phase5-freeze.md) |
| 6 | Tracking + experiment management + CLI polish | **Permanently frozen** — [phase6-tracking-cli-architecture.md](phase6-tracking-cli-architecture.md) · [ADR-0018](adr/0018-phase6-tracking-cli.md) · [ADR-0020](adr/0020-phase6-freeze.md) |
| 7 | Distributed readiness | **Permanently frozen** — [phase7-distributed-readiness-architecture.md](phase7-distributed-readiness-architecture.md) · [ADR-0019](adr/0019-phase7-distributed-readiness.md) · [ADR-0021](adr/0021-phase7-freeze.md) |

See `docs/adr/` for architectural decisions (0001–0021). Note: [ADR-0020](adr/0020-phase6-freeze.md)
is the Phase 6 Freeze ADR; [ADR-0019](adr/0019-phase7-distributed-readiness.md) is the
Phase 7 architecture ADR; [ADR-0021](adr/0021-phase7-freeze.md) is the Phase 7 Freeze ADR.
