# AIODOO Training

Production-grade AI model training framework for the AIODOO ecosystem.

## Status

**Phase 0 — Permanently frozen** (foundation)  
**Phase 1 — Permanently frozen** (dataset loading, tokenization, resource layer)  
**Phase 2 — Permanently frozen** (model loading + adaptation)  
**Phase 3 — Permanently frozen** (training engine / checkpoint / resume)  
**Phase 4 — Permanently frozen** (evaluation + export / Artifact Contract)  
**Phase 5 — Permanently frozen** (packing + curriculum + sampling)  
**Phase 6 — Permanently frozen** (tracking, experiment management, CLI polish)  
**Phase 7 — Permanently frozen** (distributed readiness; ADR-0019 / ADR-0021)

**AIODOO Training v1 architecture complete.**

Phases 0–7 are **permanently frozen** public contracts. Later work **extends**
them; it does not redesign them. See [Frozen Public Contracts](docs/frozen_public_contracts.md),
[Phase Completion Matrix](docs/phase_completion_matrix.md),
[ADR-0014](docs/adr/0014-phase3-freeze.md),
[ADR-0017](docs/adr/0017-phase5-freeze.md),
[ADR-0019](docs/adr/0019-phase7-distributed-readiness.md),
[ADR-0020](docs/adr/0020-phase6-freeze.md), and
[ADR-0021](docs/adr/0021-phase7-freeze.md).

Phase 0 provides architecture (domain, ports, registries, config, pipeline,
determinism). Phase 1 adds deterministic dataset loading, formatters,
tokenization, label masking, caching, fingerprints, and a CPU-only
`ResourcePlanner` / `ExecutionEnvironment` surface.

Phase 2 adds `ModelBackend` / `AdaptationStrategy` implementations (stub + HF /
PEFT under infrastructure), model profiles, quantization abstractions, and
fingerprints.

Phase 3 implements `TrainerBackend` / `CheckpointStore` / `RngController`
behind frozen ports, with CheckpointManager, ResumePolicy, TrainingSession,
deterministic CPU stub training, and resume-equivalence golden tests.

Phase 4 implements evaluation and export behind frozen `Evaluator` / `Exporter`
ports, quality gates, and the Artifact Contract handoff to `aiodoo-models`.

Phase 5 implements packing, curriculum, and sampling behind frozen
`PackingStrategy` / `CurriculumStrategy` ports (additive `SamplingStrategy`),
with `SchedulePlanner` as sole orchestrator and immutable completed-plan
statistics.

Phase 6 implements observational tracking, experiment/run management,
provenance, and CLI polish behind the frozen `ExperimentTracker` port
(TrackingCoordinator; TrackingCapability / TrackingHealth; CLIProfile).

Phase 7 implements distributed readiness (DistributedRuntime, FakeDistributedBackend,
placement, shard planning, sync façade, checkpoint/eval/export coordination) behind
registries — **permanently frozen**.

Optional real backends:

```bash
python3 -m pip install -r requirements/train.txt
```

CI remains CPU-only via stub backends (no GPU, no large downloads).

Frozen abstractions include:

- `DatasetSession` — dataset consumption runtime state (shard / rank / resume fields)
- `ChatTemplateRegistry` — model-family chat templates (decoupled from tokenizers)
- `ResourcePlanner` + `ExecutionEnvironment` — centralized hardware decisions (ADR-0009)
- Phase 2: `ModelLoader` / `AdaptationApplier` + registries for backends / strategies / profiles
- Phase 3: `TrainingSession`, CheckpointManager, ResumePolicy, trainer backend contract
- Phase 4: EvaluationSession / ExportSession, Artifact Contract, quality gates
- Phase 5: PackingSession / CurriculumSession, SchedulePlanner, PackingStatistics /
  CurriculumStatistics
- Phase 6: TrackingCoordinator, ExperimentSession / RunRecord, TrackingCapability /
  TrackingHealth, CLIProfile / CommandRegistry
- Phase 7: DistributedRuntime, DeviceMesh / PlacementPlan, FakeDistributedBackend,
  ShardPlanner, SyncFacade, DistributedCheckpointCoordinator

## Scope

| Owns | Does not own |
|------|----------------|
| Consume datasets from `aiodoo-datasets` | Dataset generation |
| Train and export adapters/models | Inference (`aiodoo-models`) |
| Deterministic experiments | Agent runtime (`aiodoo-core`) |

## Execution model

```bash
python3 -m pip install -r requirements/base.txt
# development / CI:
python3 -m pip install -r requirements/dev.txt
```

Do **not** `pip install -e .`. This repository is not packaged.

Optional HuggingFace tokenizer / train extras:

```bash
python3 -m pip install -r requirements/train.txt
```

## Entry scripts

```bash
python3 doctor.py
python3 validate_config.py --config configs/experiments/example.yaml
python3 fingerprint.py --config configs/experiments/example.yaml
python3 prepare_dataset.py \
  --dataset tests/fixtures/datasets/coding.jsonl \
  --dataset-type coding \
  --limit 2

python3 train.py --config configs/experiments/example.yaml
python3 evaluate.py --config configs/experiments/example.yaml
python3 export.py --config configs/experiments/example.yaml
```

## Tests

```bash
python3 tests/run_tests.py
```

## Documentation

- [Architecture](docs/architecture.md)
- [Frozen Public Contracts](docs/frozen_public_contracts.md) — binding engineering governance
- [Architecture Invariants](docs/architecture_invariants.md)
- [Phase Completion Matrix](docs/phase_completion_matrix.md)
- [Phase 3 Training Engine Architecture](docs/phase3-training-engine-architecture.md) — **permanently frozen**
- [Phase 4 Evaluation & Export Architecture](docs/phase4-evaluation-export-architecture.md) — **permanently frozen**
- [Artifact Contract](docs/artifact_contract.md) — official Training → Models handoff specification
- [Artifact Output Pipeline](docs/artifact_output_pipeline.md) — canonical Drive workspace layout
- [Production Smoke Test](docs/SMOKE.md) — end-to-end smoke procedure (`coding`)
- [Phase 5 Packing & Curriculum Architecture](docs/phase5-packing-curriculum-architecture.md) — **permanently frozen**
- [Phase 6 Tracking & CLI Architecture](docs/phase6-tracking-cli-architecture.md) — **permanently frozen**
- [Phase 7 Distributed Readiness Architecture](docs/phase7-distributed-readiness-architecture.md) — **permanently frozen**
- [Trainer Backend Contract](docs/trainer_backend_contract.md)
- [ADRs](docs/adr/) — including [0017 Phase 5 freeze](docs/adr/0017-phase5-freeze.md),
  [0018 Phase 6 architecture](docs/adr/0018-phase6-tracking-cli.md),
  [0019 Phase 7 architecture](docs/adr/0019-phase7-distributed-readiness.md),
  [0020 Phase 6 freeze](docs/adr/0020-phase6-freeze.md), and
  [0021 Phase 7 freeze](docs/adr/0021-phase7-freeze.md)
