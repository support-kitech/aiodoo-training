# Engineering Principles

**Status:** Engineering Philosophy (documentation only)  
**Purpose:** Capture the philosophy behind AIODOO Training without creating any
new rules, contracts, or APIs.

This document is **informational**. Binding rules remain in
[Frozen Public Contracts](frozen_public_contracts.md) and
[Architecture Invariants](architecture_invariants.md). Nothing here redesigns
architecture or widens public surfaces.

---

## Stable architecture over implementation convenience

Architecture is expected to live significantly longer than any one
implementation. Short-term convenience must not erode long-lived boundaries.

## Ports over frameworks

Frameworks evolve. Ports remain stable. Call sites depend on AIODOO
abstractions; adapters absorb vendor churn.

## Composition over redesign

Prefer additive composition. Avoid replacing frozen responsibilities. Extend
by new modules, registries, and binders — not by thawing prior phases.

## Registration over hardcoding

New capabilities arrive through registries. Pipeline and application code
resolve by key; they do not hardcode backend families or vendor class names.

## Immutable domain

Domain objects remain immutable. Session and related state changes occur
through copy-on-write helpers, not in-place mutation.

## Determinism first

Identical portable inputs should produce identical outputs (fingerprints,
cursors, golden surfaces). Host-specific noise stays out of identity material.

## Portable fingerprints

Identity material stays portable across hosts and launchers. Do not fold
hostname, absolute paths, or launcher-only ranks into experiment / mesh
fingerprints.

## Framework quarantine (infrastructure isolation)

Third-party ML libraries (Torch, Transformers, PEFT, Accelerate, DeepSpeed,
XLA, and peers) belong only inside `infrastructure/`.

## Single authority (no duplicated authority)

Each responsibility has exactly one Authority — the component that owns the
source of truth. Coordinators may invoke Authorities; they never compete with
them and never own durability that belongs to a Manager. See
[Coordinator conventions](coordinator_conventions.md).

## Documentation is architecture

Architecture documentation is considered part of the product, not an
afterthought. Specs, ADRs, and governance docs are first-class artifacts.

## Freeze before expansion

Complete **Implementation → Validation → Freeze** before implementing the next
phase whenever practical. Long-term maintenance and future extension (via ADR)
follow freeze. See [Phase Completion Matrix](phase_completion_matrix.md) and
[Frozen Public Contracts §9](frozen_public_contracts.md).

---

## Related

- [Architecture](architecture.md)
- [Frozen Public Contracts](frozen_public_contracts.md)
- [Architecture Invariants](architecture_invariants.md)
- [Coordinator Conventions](coordinator_conventions.md)
- [Phase Completion Matrix](phase_completion_matrix.md)
