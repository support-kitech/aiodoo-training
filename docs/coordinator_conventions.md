# AIODOO Training — Coordinator & Orchestration Naming Conventions

**Status:** Governance / engineering conventions (documentation only)  
**Related:** [Architecture](architecture.md), [Frozen Public Contracts](frozen_public_contracts.md),
[Architecture Invariants](architecture_invariants.md),
[Engineering Principles](engineering_principles.md)

This page records the **intended engineering meaning** of major orchestration
terms used across AIODOO Training. The naming is intentional. Future phases
should keep these roles distinct rather than inventing overlapping synonyms.

These conventions do **not** change public contracts, ports, or APIs. They
clarify vocabulary so Planner / Coordinator / Manager / Runtime / Authority
responsibilities are not confused.

---

## Authority

An **Authority** is the single component that owns the **source of truth** for
a particular responsibility.

Authorities make final lifecycle decisions and own durable state (or the
canonical plan / session truth for that responsibility). Other components —
especially Coordinators — may **invoke** Authorities but never replace or
compete with them.

**Typical Authorities include:**

- `TrainingSession` (train cursor / status truth)
- `CheckpointManager` (durable checkpoint packages)
- `ExportManager` (Artifact Contract packages)
- `EvaluationEngine` (evaluation outcomes)
- `SchedulePlanner` (packing / curriculum / sampling plan ownership)

**Rules (vocabulary):**

- Every responsibility should have only one Authority.
- Coordinators coordinate Authorities.
- Runtimes own temporary execution resources.
- Managers own durable resources (when they are the Authority for that resource).
- Planners create plans (a Planner may itself be the Authority for planning).
- Authorities remain the source of truth.

This section is vocabulary only. It changes nothing architecturally.

---

## Planner

Creates **deterministic plans**.

| Rule | Detail |
|------|--------|
| Owns | Planning only (ordered stages, schedules, placements of work in abstract form) |
| Does not own | Training loops, durable persistence, framework process groups, tracking sinks |
| Output | Immutable plan / statistics DTOs when applicable |

**Example:** `SchedulePlanner` (Phase 5) — sole owner of curriculum → sampling →
packing plan orchestration.

---

## Coordinator

**Coordinates** existing frozen authorities. It never becomes an authority itself.

| Rule | Detail |
|------|--------|
| Owns | Ordering of calls, rank roles, barriers around someone else’s save/eval/export |
| Does not own | Persistence of weight packages, Artifact Contract packages, or domain session truth |
| Does not own | Domain state as source of truth (mirrors or placement helpers only) |
| Relation to Managers | Calls Managers / Engines; does not replace them |

**Examples:**

| Type | Coordinates |
|------|-------------|
| `TrackingCoordinator` | Observational recording around train/eval/export — never owns those outcomes |
| `DistributedCoordinator` | Rank roles and distributed wiring around frozen train/eval/export paths |
| `DistributedCheckpointCoordinator` | Who may call `CheckpointManager`; never writes checkpoints itself |
| `DistributedEvaluationCoordinator` | Shard eval + merge *around* EvaluationEngine |
| `FaultToleranceCoordinator` | Restart decisions that *invoke* ResumePolicy / resume paths |

If a new type would need to own durable packages or redefine session status,
it is almost certainly a **Manager** (or an existing frozen authority), not a
Coordinator.

---

## Manager

Owns **lifecycle or authoritative resources**.

| Rule | Detail |
|------|--------|
| Owns | Authoritative durability, validation, atomic publish, or export packages |
| May orchestrate | Store ports / sidecars under a single owner |
| Must not be duplicated | Do not add competing Managers for the same authority |

**Examples:**

- `CheckpointManager` — durable checkpoints, manifests, resume validation wiring
- `ExportManager` — Artifact Contract packages and export lifecycle

---

## Runtime

Owns **temporary runtime execution resources** for the duration of a job or
process group lifetime.

| Rule | Detail |
|------|--------|
| Owns | Init/teardown of ephemeral runtime handles (e.g. process-group façades) |
| Does not own | Experiment identity, checkpoint authority, or Artifact Contract |
| Lifetime | Open → use → close (or equivalent) |

**Example:** `DistributedRuntime` (Phase 7) — topology / sync façade
lifecycle without owning CheckpointManager or TrainingSession.

---

## Registry

**Name → implementation** (or profile) lookup.

| Rule | Detail |
|------|--------|
| Owns | Explicit, freezable catalogs |
| Does not own | Business logic of the registered types |

Used for backends, strategies, planners (as implementations), trackers, etc.

---

## Factory

**Creates** implementations using registries.

| Rule | Detail |
|------|--------|
| Owns | Construction from registry keys |
| Does not | Instantiate infrastructure types from application call sites ad hoc |

---

## Builder

**Builds** immutable domain (or context) objects without I/O and without
importing infrastructure.

| Rule | Detail |
|------|--------|
| Owns | Assembly of validated immutable graphs / binder bags |
| Does not | Probe hardware, open process groups, or write artifacts |

---

## Consistency rule

These roles are complementary:

```text
Builder / Factory / Registry  →  construct & select
Planner                       →  plan
Coordinator                   →  coordinate Authorities
Authority                     →  source of truth for a responsibility
Manager                       →  own durable / lifecycle resources (often an Authority)
Runtime                       →  own ephemeral execution resources
```

Prefer the existing term over inventing a new “Orchestrator”, “Controller”, or
second Manager for the same concern. When in doubt: **extend by registration**;
do not thicken Coordinators into Authorities.

---

## See also

- [Phase 5 architecture](phase5-packing-curriculum-architecture.md) — SchedulePlanner
- [Phase 6 architecture](phase6-tracking-cli-architecture.md) — TrackingCoordinator
- [Phase 7 architecture](phase7-distributed-readiness-architecture.md) — Distributed* coordinators / runtime
- [Phase Completion Matrix](phase_completion_matrix.md) — phase freeze ledger
- [Engineering Principles](engineering_principles.md) — philosophy (informational)
