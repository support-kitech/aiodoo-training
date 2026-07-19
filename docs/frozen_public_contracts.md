# AIODOO Training — Frozen Public Contracts

**Status:** Binding engineering governance document  
**Applies to:** All contributors (human and AI)  
**Effective:** Phases 0–7 permanently frozen — **v1.0.0 production freeze** ([repository_freeze.md](repository_freeze.md))

This is **not** end-user documentation. It is the constitutional contract of the
`aiodoo-training` repository. Future work **extends** these contracts. It does
**not** redefine them.

Related documents:

- [Architecture](architecture.md) — layer map and phase status
- [Architecture Invariants](architecture_invariants.md) — permanent engineering rules
- [Phase Completion Matrix](phase_completion_matrix.md) — phase freeze ledger
- [Terminology](terminology.md) — authoritative vocabulary
- [Ownership](ownership.md) — cross-repository ownership
- [Capability Model](capability_model.md) / [Product Model](product_model.md)
- [Lifecycle](lifecycle.md) — training and package lifecycle
- [Metadata Ownership](metadata_ownership.md)
- [Freeze Readiness](freeze_readiness.md) — lifecycle freeze checklist
- [Coordinator conventions](coordinator_conventions.md) — Planner / Coordinator / Manager / Runtime / Authority vocabulary
- [Engineering Principles](engineering_principles.md) — engineering philosophy (informational)
- [Trainer Backend Contract](trainer_backend_contract.md) — Phase 3+ trainer checklist
- [Artifact Contract](artifact_contract.md) — Phase 4 ArtifactBundle export inventory (clarified by ADR-0022)
- [Artifact Output Pipeline](artifact_output_pipeline.md) — Capability Package Drive layout
- [ADRs](adr/) — decision history (0001–0023)
- [Repository Freeze](repository_freeze.md) — v1.0.0 freeze statement
- [MAINTENANCE](MAINTENANCE.md) · [Release Checklist](release_checklist.md)
- [Phase 7 Distributed Readiness Architecture](phase7-distributed-readiness-architecture.md) — **permanently frozen** ([ADR-0019](adr/0019-phase7-distributed-readiness.md) · [ADR-0021](adr/0021-phase7-freeze.md))
- **Ecosystem ADR-0001 — AIODOO Model Lifecycle** — cross-repo lifecycle (not this repo’s ADR-0001)
---

## 1. Purpose

Long-lived training systems fail when implementation convenience quietly erodes
boundaries. Torch, Transformers, PEFT, Accelerate, and related libraries evolve
quickly; once their types leak into domain or application code, isolation becomes
expensive or impossible.

Architectural contracts exist so that:

1. **Core shapes remain stable** — domain types, ports, registries, fingerprints,
   and sessions are reliable APIs for every later phase.
2. **Framework churn is quarantined** — third-party ML libraries live only in
   infrastructure adapters and can be upgraded or replaced without redesigning
   the system.
3. **Phases compose rather than rewrite** — additive infrastructure adapters and
   other AIODOO repositories build on frozen contracts instead of inventing
   parallel ones.

Maintainability over decades requires **stable contracts**. Speed of a single
feature delivery never justifies breaking them.

If a future feature appears to require changing a frozen contract, **stop and
explain why**. Do not change frozen architecture automatically.

---

## 2. Repository Status

| Phase | Scope | Status |
|-------|--------|--------|
| **0** | Foundation (domain, ports, registries, builders, factories, config, pipeline, determinism) | **Permanently frozen** |
| **1** | Dataset loading, tokenization, resource layer, DatasetSession, ChatTemplateRegistry | **Permanently frozen** |
| **2** | Model loading, adaptation, quantization policy, model/adapter registries, hardening | **Permanently frozen** |
| **3** | Trainer, checkpoint, resume, metrics, callbacks, training config/handlers | **Permanently frozen** |
| **4** | Evaluation, export, Artifact Contract | **Permanently frozen** |
| **5** | Packing, curriculum, sampling, SchedulePlanner, plan statistics | **Permanently frozen** |
| **6** | Tracking, experiment management, CLI polish | **Permanently frozen** |
| **7** | Distributed readiness | **Permanently frozen** ([ADR-0021](adr/0021-phase7-freeze.md)) |

Frozen means: **stable public contract**. Bug fixes are allowed. Architectural
redesign is not. Only implementation behind existing ports and registries evolves
from this point onward unless the change process in Section 9 is completed.

---

## 3. Frozen Public Contracts

Each row below is a **stable API surface**. Implementations may be added behind
it; the contract itself must not be redesigned for convenience.

**Phases 0–7** are **permanently frozen** ([ADR-0021](adr/0021-phase7-freeze.md)
for Phase 7).

| Contract | Package / primary types | Responsibility |
|----------|-------------------------|----------------|
| **Domain** | `aiodoo_training.domain` | Immutable, framework-independent DTOs and enums. Source of truth for experiment identity inputs and runtime metadata. Never imports Torch/Transformers/PEFT. |
| **Ports** | `aiodoo_training.ports` | Minimal abstract interfaces (`ModelBackend`, `AdaptationStrategy`, `ResourcePlanner`, `TokenizerPort`, `ChatTemplate`, trainer/eval/export/packing/curriculum ports, …). Signatures expose only AIODOO types and opaque handles. |
| **Registries** | `aiodoo_training.registries` | Explicit, freezable name → implementation (or profile) catalogs. Every backend and every declarative profile registers here. |
| **Builders** | `aiodoo_training.builders` | Assemble immutable domain graphs without I/O and without importing infrastructure. |
| **Factories** | `aiodoo_training.factories` | Construct port implementations from registries only — never instantiate infrastructure classes directly from application code. |
| **Pipeline** | `aiodoo_training.pipeline` | Ordered stage **orchestration** only. No training logic, no ML logic, no hardware probing. |
| **Configuration** | `aiodoo_training.config` | Load, compose, validate, resolve paths, and hash experiment YAML. Experiment identity uses portable composed config (not machine-absolute paths). |
| **Determinism** | `aiodoo_training.determinism` | Seed management and fingerprint services. Extension points for Torch RNG exist without Domain importing Torch. |
| **Fingerprint system** | Config + dataset + model + adapter + experiment fingerprints | Deterministic digests for reproducibility and ExperimentId derivation. Must remain stable for identical portable inputs. |
| **Dataset framework** | `aiodoo_training.datasets` | Protocol JSONL reading, validation, mixing, caching, formatters — consumes `aiodoo-datasets` artifacts; does not generate datasets. |
| **DatasetSession** | `domain.session.DatasetSession` | Immutable consumption cursor (epoch, index, shard/rank, resume token, fingerprints). Copy-on-write updates only. **Unchanged by Phase 5** — packing/curriculum cursors live on Phase 5 sessions. |
| **Tokenization framework** | `aiodoo_training.tokenization` | Masking, TokenBatch pipeline, tokenization fingerprints — separated from chat templates and trainer code. |
| **ChatTemplateRegistry** | `chat_template_registry` + `ChatTemplate` port | Family-specific prompting, independent of tokenizer backends. |
| **Resource layer** | `DevicePolicy`, `PrecisionPolicy`, `MemoryPolicy`, `HardwareCapabilities` | Declared and discovered hardware preferences — no ad-hoc CUDA checks in domain/application/pipeline. |
| **ExecutionEnvironment** | `domain.resources.ExecutionEnvironment` | Resolved execution plan for a run. Later phases consult this object instead of probing devices directly. |
| **ResourcePlanner** | `ports.resources.ResourcePlanner` | Probe + resolve API. Static CPU planner ships; GPU planners remain infrastructure. |
| **Model backend** | `ModelBackend`, `BaseModelHandle`, `ModelLoader`, model catalogs | Load base models under an `ExecutionEnvironment`. Opaque handles only. |
| **Adaptation framework** | `AdaptationStrategy`, `TrainableModelHandle`, `AdaptationApplier` | Apply LoRA / QLoRA / full (etc.) **after** load. Behavior separated from model loading (ADR-0005). |
| **Model registry** | `model_backend_registry`, `model_family_registry`, `model_profile_registry`, `model_capability_registry` | Registration-driven model families and backends (Qwen, Llama, Mistral, …) without hardcoding families into pipeline code. |
| **Adapter registry** | `adapter_registry` (`AdapterProfile`) vs `adaptation_registry` (strategies) | Declarative adapter metadata independent of adaptation behavior classes. |
| **Quantization policy** | `QuantizationPolicy` (`QuantizationSpec` alias) | Framework-agnostic 4-bit / 8-bit / float precision policy; infrastructure maps to concrete libraries. |
| **TrainerBackend** | Frozen `train` / `resume` port + [trainer backend contract](trainer_backend_contract.md) | Training loop behind stable signatures; context via binder only. |
| **TrainingSession / Lifecycle** | `TrainingSession`, `TrainingLifecycle` | Immutable run cursor + allowed status transitions (COW). |
| **Checkpoint system** | `CheckpointStore` port + application `CheckpointManager` + `CheckpointManifest` | Store owns weight packages; manager owns manifests, sidecars, validation, atomic publish. |
| **ResumePolicy** | `STRICT` \| `WARN` \| `RELAXED` (+ `training_protocol_version`) | Compatibility severity for checkpoint resume; protocol bumps for semantic resume breakage. |
| **RngController** | Frozen seed / snapshot / restore port | Deterministic seeding and resume RNG continuity. |
| **Optimizer / Scheduler / Callback ports** | Additive `OptimizerBackend`, `SchedulerBackend`, `TrainingCallback` | Registry-driven extension without widening `TrainerBackend`. |
| **Training events / metrics** | Event bus, `MetricCollector`, `TrainingHistory` | Ordered observation; frozen `MetricSnapshot` / `TrainingProgress` as emissions. |
| **Evaluator / Exporter** | Frozen eval/export ports + EvaluationSession / ExportSession | Offline metrics and portable artifacts; Artifact Contract is the Models handoff. |
| **PackingStrategy** | Frozen `pack(examples, PackingSpec) -> Iterator[TokenBatch]` | Sequence packing; rich context via `bind()` only — **do not widen**. |
| **CurriculumStrategy** | Frozen `plan(examples, CurriculumSpec) -> Sequence[Sequence[TrainingExample]]` | Curriculum stages; context via `bind()` only — **do not widen**. |
| **SamplingStrategy** | Additive `sample(examples, SamplingSpec) -> Sequence[TrainingExample]` | Deterministic reorder / strata; registry-driven. |
| **SchedulePlanner** | `aiodoo_training.packing.planner.SchedulePlanner` | **Sole** Phase 5 orchestration owner (curriculum → sampling → packing → statistics). No competing Managers. |
| **PackingStatistics / CurriculumStatistics** | Immutable completed-plan domain DTOs | Pure projections of completed plans; never runtime trackers; never own monitoring sinks. |
| **ExperimentTracker** | Frozen `log_params` / `log_metrics` / `log_artifact` / `close` (+ binder) | Observational sinks only — **do not widen**; never authoritative for train/eval/export. |
| **TrackingCoordinator** | Application observational coordinator | Coordinates recording around frozen Authorities; never owns persistence of weight/export packages. |
| **TrackingCapability / TrackingHealth** | Immutable feature flags + sink health | Compatibility / doctor diagnostics; never training health; never ResumePolicy input. |
| **CLIProfile / CommandRegistry** | UX presets + command dispatch | Polish only; not PyPI packaging; not business logic ownership. |
| **DistributedBackend** | Port + `distributed_backend_registry` | Process-group collectives via `bind` / factory; fake CI reference; no framework types in port. |
| **DistributedRuntime / DistributedContext** | Application runtime + binder bag | Owns ephemeral PG lifecycle + health snapshots; never TrainingSession / CheckpointManager. |
| **DistributedSession / DistributedTopology** | Domain lifecycle cursor + topology | Immutable COW session; portable `mesh_digest` only. |
| **DistributedPlacementResolver** | Application companion to ResourcePlanner | Consumes `ExecutionEnvironment` + `DistributedSpec` → DeviceMesh / PlacementPlan. |
| **DistributedCheckpointCoordinator** | Application coordinator | Rank roles / barriers around CheckpointManager; never writes packages. |
| **RestartPolicy / DistributedHealth** | Domain policies + runtime health | Restart beside ResumePolicy; health ≠ TrackingHealth. |

Supporting frozen orchestration types (Phase 2): `LoadedModelContext`,
`AdaptedModelContext`, `ModelMetadata`, `AdapterMetadata`, model/adapter
fingerprints — AIODOO abstractions only; never Torch / PEFT types.

Supporting frozen training types (Phase 3): `TrainerResult`, training event DTOs,
optimizer/scheduler policies, gradient policies, CheckpointPolicy — AIODOO only.

Supporting frozen Phase 4 types: EvaluationReport, quality gates,
ArtifactBundle / ArtifactDescriptor / ArtifactIndex, validation/compatibility
policies — AIODOO only.

Supporting frozen Phase 5 types: PackingSession, CurriculumSession,
PackingPolicy, MemoryPackingPolicy, SamplingSpec, PackedSpan, SchedulePlan —
AIODOO only.

Supporting frozen Phase 6 types: ExperimentSession, RunRecord, TrackingPolicy,
provenance digests, TrackingCapability / TrackingHealth, CLIProfile,
`TRACKING_PROTOCOL_VERSION` (history metadata only) — AIODOO only; tracker SDKs
in infrastructure.

Supporting frozen Phase 7 types: DeviceMesh, PlacementPlan, SyncFacade policies,
ShardPlanner, FakeDistributedBackend, placement/sampler registries — AIODOO
abstractions; framework runtimes only in `infrastructure/distributed/`.
Permanently frozen under [ADR-0021](adr/0021-phase7-freeze.md).

---

## 4. Dependency Rules

Outer layers depend inward. Dependencies never point from domain toward
infrastructure or CLI.

| Layer | May depend on | Must not |
|-------|---------------|----------|
| **Domain** | Standard library + own domain modules | Any third-party ML library; ports; infrastructure; CLI |
| **Ports** | Domain types only | Framework types; infrastructure imports |
| **Application / models / adaptation / packing / curriculum / sampling orchestration** | Domain, ports, registries, factories, config, determinism | Direct Torch/Transformers/PEFT imports |
| **Pipeline** | Domain + pipeline framework | Business/ML/hardware logic; framework imports |
| **Infrastructure** | Domain + ports (+ third-party ML libs here only) | Being imported by domain/ports/pipeline for framework types |
| **CLI** | Application wiring, factories, config | Owning business logic or framework calls except via factories |

Additional rules:

- **Domain imports nothing external** for ML (no torch, transformers, peft,
  bitsandbytes, accelerate).
- **Ports expose only AIODOO abstractions** (including opaque handles).
- **Infrastructure owns Torch**, Transformers, PEFT, bitsandbytes, and Accelerate
  imports.
- **Pipeline orchestrates only.**
- **Application owns use-cases** (progressively filled).
- **CLI owns wiring only** (thin root scripts).

AST boundary tests enforce forbidden imports outside `infrastructure/`.

---

## 5. Breaking Change Policy

A change is **breaking** if it alters a frozen public contract in a way that
forces later phases or call sites to redesign rather than extend.

Breaking changes include (non-exhaustive):

| Category | Examples |
|----------|----------|
| Port surface | Changing method names or signatures on `ModelBackend`, `AdaptationStrategy`, `ResourcePlanner`, `TokenizerPort`, `TrainerBackend`, `PackingStrategy.pack`, `CurriculumStrategy.plan`, etc. |
| Layer moves | Moving adaptation into model load; putting checkpoint logic in the trainer port; putting export in evaluation; introducing PackingManager beside SchedulePlanner |
| Domain identity | Mutating previously immutable fields; changing `DatasetSession` semantics; altering fingerprint inputs so identical portable configs yield different ExperimentIds |
| Resource contracts | Bypassing `ResourcePlanner` / `ExecutionEnvironment`; requiring CUDA checks in application code |
| Adaptation contracts | Merging `AdaptationStrategy` with `ModelBackend`; exposing `PeftModel` / `PreTrainedModel` from ports |
| Configuration schema | Redesigning composed config shapes that existing experiments and fingerprints depend on without an ADR and migration |
| Registry contracts | Removing or renaming required registries; registering frameworks as domain types |
| Phase 5 ownership | Turning PackingStatistics / CurriculumStatistics into runtime trackers; bypassing SchedulePlanner |

If a proposed Phase 7+ feature needs any of the above, **stop**, document the
conflict against this contract and the relevant ADR, and follow Section 9.

---

## 6. Allowed Changes

Without reopening frozen architecture, the following are **allowed**:

| Allowed | Notes |
|---------|--------|
| Bug fixes | Correct incorrect behavior inside frozen modules without changing contracts |
| Performance improvements | Same public API; same fingerprints for identical portable inputs; packing asymptotics must not regress without ADR |
| Documentation | ADRs, architecture docs, this contract, README |
| New infrastructure backends | e.g. additional HF loaders, planners — confined to `infrastructure/` |
| New model families | Via `model_*` registries and chat templates — not via hardcoding in pipeline |
| New adapter strategies / profiles | Via `adaptation_registry` and `adapter_registry` |
| New packing / curriculum / sampling backends | Via existing registries; do not widen frozen port signatures |
| New registries | Additive catalogs for later phases, without removing frozen ones |
| New implementations behind existing ports | Factories resolve by key; callers stay port-oriented |
| Tests | Unit, contract, golden, integration — CPU-first; no GPU requirement in CI |
| Additive domain types for later phases | New modules that do not alter existing frozen type contracts |

Optional ML dependencies remain in `requirements/train.txt`. CI continues to use
stubs and `requirements/dev.txt` unless a dedicated GPU job is explicitly added.

---

## 7. Forbidden Changes

| Forbidden | Why |
|-----------|-----|
| Torch / Transformers / PEFT / bitsandbytes / Accelerate imports outside `infrastructure/` | Breaks quarantine; couples the ecosystem to vendor APIs |
| Changing frozen architecture for convenience | Temporary speed creates permanent debt |
| Moving logic between frozen layers | Violates hexagonal boundaries and ADRs |
| Bypassing ports | Call sites become coupled to concrete backends |
| Returning third-party framework types from public APIs | Contaminates domain/application forever |
| Hardcoding model families into pipeline or application | Registries exist for extension |
| Removing or weakening determinism / fingerprint stability | Breaks experiment identity and reproducibility |
| Ad-hoc CUDA / device / dtype checks outside ResourcePlanner resolution | Redevelops resource management in every call site |
| Introducing `ModelSession` or similar without ADR | Explicitly rejected in ADR-0011 |
| Competing Phase 5 Managers / runtime statistics trackers in packing sessions | Violates ADR-0016 / ADR-0017 |

Architecture preservation takes priority over implementation speed.

---

## 8. Architecture Invariants

These rules are permanent. See also [architecture_invariants.md](architecture_invariants.md).

1. Every framework dependency stays in **infrastructure**.
2. Every public API returns **AIODOO abstractions** (or opaque handles wrapping infrastructure privately).
3. Every fingerprint is **deterministic** for identical portable inputs.
4. Every pipeline stage is **orchestration-only** and does not hide ML or hardware logic.
5. Every registry is **explicit** and registration-driven.
6. Every domain object is **immutable** (frozen dataclasses; copy-on-write sessions).
7. Every configuration path that defines an experiment is **validated** and composed canonically.
8. Every new backend **registers** itself.
9. Every new implementation **preserves** existing contracts.
10. Model loading and adaptation remain **separate**.
11. Hardware decisions go through **ResourcePlanner** / **ExecutionEnvironment**.
12. Adapter **metadata** stays independent of adaptation **behavior**.
13. Inference never belongs to `aiodoo-training`.
14. Trainer does not own checkpoint store implementation (port separation;
    CheckpointManager may orchestrate; store remains the weight port).
15. Evaluation does not own export logic (port separation).
16. Trainer backends obey [Trainer Backend Contract](trainer_backend_contract.md).
17. Resumed and uninterrupted runs share deterministic progression under the
    Phase 3 golden invariant when STRICT resume inputs match.
18. `SchedulePlanner` is the sole Phase 5 orchestration owner.
19. `PackingStatistics` / `CurriculumStatistics` remain completed-plan summaries only.

---

## 9. Change Process

No architecture change to a frozen contract is allowed outside this process:

```text
Problem
        ↓
Architecture Review
        ↓
       ADR
        ↓
   Discussion / approval
        ↓
   Implementation
        ↓
    Validation
        ↓
      Freeze
        ↓
   Maintenance
        ↓
Future Extension (if needed — through ADR)
```

### 9.1 Normal phase lifecycle (governance)

Independent of breaking-change proposals, each **new phase** should normally
complete as:

```text
Problem / scope
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

A completed phase should normally be **permanently frozen** before
implementation of the next phase begins. Architecture design for the next phase
may be drafted while freeze paperwork for the current phase is finishing, but
shipping next-phase code against an unfrozen prior phase is discouraged unless
architecture review explicitly authorizes it.

After freeze, the phase enters **Maintenance** (bug fixes and compatible
improvements without redesign). Any architectural expansion returns through
**Problem → Architecture Review → ADR**.

1. **Architecture Review** — State the problem, why existing contracts are insufficient, and alternatives that avoid breaking changes.
2. **ADR** — Record context, decision, and consequences under `docs/adr/`.
3. **Discussion** — Explicit human approval; convenience alone is not sufficient cause.
4. **Implementation** — Minimal change; migrate callers; do not silently widen framework leakage.
5. **Validation** — Unit, contract, and boundary tests must pass; fingerprint stability tests where relevant.
6. **Freeze** — Update this document, architecture status, and ADRs; treat the new shape as the new contract.
7. **Maintenance** — Preserve frozen contracts; fix bugs without redesign.
8. **Future Extension** — Only via a new Architecture Review + ADR when contracts must grow.

Until an ADR is accepted, implementers must **build behind existing ports**.

Naming conventions for Planner / Coordinator / Manager / Runtime / Authority are
documented in [coordinator_conventions.md](coordinator_conventions.md)
(vocabulary only; not a contract redesign). Philosophy:
[engineering_principles.md](engineering_principles.md).

---

## 10. Conclusion

This document is the **constitutional contract** of AIODOO Training.

Phases 0, 1, 2, 3, 4, 5, 6, and 7 define the **permanently frozen** public
surface. **AIODOO Training v1 architecture is complete.**

- Extend by registration and infrastructure adapters.
- Orchestrate through the pipeline.
- Resolve hardware through the resource layer.
- Checkpoint and resume through CheckpointManager + ResumePolicy.
- Evaluate and export through frozen Evaluator/Exporter + ArtifactBundle contract;
  Drive-publish Capability Packages for external handoff (ADR-0022).
- Plan packing/curriculum/sampling through SchedulePlanner.
- Record observations through TrackingCoordinator / ExperimentTracker (never as run control).
- Coordinate distribution through DistributedRuntime / coordinators (never as a second training engine).
- Keep third-party ML libraries behind infrastructure walls.

Future work **extends** this contract.  
It does **not** redefine it.

Real DDP / FSDP / DeepSpeed / Accelerate / XLA / UCX (etc.) **MUST** register
through existing extension points. They **MUST NOT** redesign the architecture.
