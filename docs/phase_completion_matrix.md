# AIODOO Training — Phase Completion Matrix

**Status:** Binding phase ledger  
**Related:** [Frozen Public Contracts](frozen_public_contracts.md), [Architecture](architecture.md)

| Phase | Scope | Architecture | Implementation | Freeze ADR | Status |
|-------|--------|--------------|----------------|------------|--------|
| **0** | Foundation (domain, ports, registries, builders, factories, config, pipeline, determinism) | Accepted | Complete | [0007](adr/0007-phase0-freeze.md) | **Permanently frozen** |
| **1** | Dataset loading, tokenization, resource layer, DatasetSession, ChatTemplateRegistry | Accepted | Complete | [0008](adr/0008-dataset-session-and-chat-templates.md) / [0009](adr/0009-resource-management.md) | **Permanently frozen** |
| **2** | Model loading, adaptation, quantization, model/adapter registries, hardening | Accepted | Complete | [0012](adr/0012-phase2-freeze.md) | **Permanently frozen** |
| **3** | Trainer, checkpoint, resume, metrics, callbacks, training config/pipeline handlers | Accepted ([0013](adr/0013-phase3-training-engine.md)) | Complete | [0014](adr/0014-phase3-freeze.md) | **Permanently frozen** |
| **4** | Evaluation, export, quality gates, ArtifactBundle + Capability Package handoff | Accepted ([0015](adr/0015-phase4-evaluation-export.md); clarified [0022](adr/0022-package-surfaces-lifecycle-alignment.md)) | Complete | [0015](adr/0015-phase4-evaluation-export.md) + [artifact_contract.md](artifact_contract.md) | **Permanently frozen** |
| **5** | Packing, curriculum, sampling, SchedulePlanner, plan statistics | Accepted ([0016](adr/0016-phase5-packing-curriculum.md)) | Complete | [0017](adr/0017-phase5-freeze.md) | **Permanently frozen** |
| **6** | Tracking, experiment/run management, provenance, CLI polish | Accepted ([0018](adr/0018-phase6-tracking-cli.md)) | Complete | [0020](adr/0020-phase6-freeze.md) | **Permanently frozen** |
| **7** | Distributed readiness | Accepted ([0019](adr/0019-phase7-distributed-readiness.md)) | Complete | [0021](adr/0021-phase7-freeze.md) | **Permanently frozen** |

**AIODOO Training v1 architecture is complete.** All seven phases (0–7) are
**permanently frozen**.

Future work **consumes** frozen contracts. It does not modify them except via the
Section 9 change process in `frozen_public_contracts.md`.

**ADR note:** [ADR-0019](adr/0019-phase7-distributed-readiness.md) = Phase 7
architecture. [ADR-0020](adr/0020-phase6-freeze.md) = Phase 6 permanent freeze.
[ADR-0021](adr/0021-phase7-freeze.md) = Phase 7 permanent freeze.
[ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md) = package surfaces
and lifecycle documentation alignment.
[ADR-0023](adr/0023-repository-freeze-v1.md) = **v1.0.0 repository freeze**.

Lifecycle docs: [terminology.md](terminology.md), [ownership.md](ownership.md),
[lifecycle.md](lifecycle.md), [freeze_readiness.md](freeze_readiness.md),
[repository_freeze.md](repository_freeze.md).

## Intended phase workflow

Each phase should normally follow this sequence:

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

**Governance norm:** a completed phase should normally be **permanently frozen**
before implementation begins for the following phase. Design of phase *N+1* may
proceed while phase *N* awaits freeze, but implementing *N+1* against an
unfrozen *N* is discouraged except under explicit architecture review.

After freeze, the phase is under **Maintenance**. Architectural expansion later
returns through Architecture Review + ADR.

See also [Frozen Public Contracts §9](frozen_public_contracts.md) (change process),
[Coordinator conventions](coordinator_conventions.md), and
[Engineering Principles](engineering_principles.md).
