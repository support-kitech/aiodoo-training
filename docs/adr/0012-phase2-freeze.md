# ADR-0012: Phase 2 Freeze

## Status

Accepted (Phase 2 permanently frozen)

## Context

Phase 2 (model loading, adaptation, resource integration, hardening) passed
architectural review. Phases 0 and 1 are already frozen.

## Decision

Phase 2 is permanently frozen. Together with Phases 0 and 1 it forms a stable
public contract. Later phases must implement against existing ports, registries,
factories, builders, config, resource management, DatasetSession,
ChatTemplateRegistry, ModelLoader, AdaptationApplier, and adapter profiles —
without redesigning frozen layers.

Rules for Phase 3+:

- Never bypass an existing Port.
- Never expose third-party framework types outside infrastructure.
- Never move responsibilities between frozen layers.
- Never introduce shortcuts that violate ADRs.
- If a feature appears to require changing a frozen phase, stop and explain why.

Canonical governance text: [`docs/frozen_public_contracts.md`](../frozen_public_contracts.md).

## Consequences

- Positive: Phase 3 trainer/checkpoint work had a fixed Phase 0–2 boundary to
  target (now also frozen under ADR-0014).
- Negative: convenience shortcuts that leak Torch/PEFT or bypass ports are rejected.
