# ADR-0022: Package Surfaces & Lifecycle Alignment

## Status

**Accepted** — Clarification only (documentation / governance).  
**Date:** 2026-07-19  
**Does not redesign** Phases 0–7 frozen contracts or change runtime behavior.

## Context

Phase A Architecture Closure established that `aiodoo-training` is
implementation-complete for Phases 0–7, but documentation drifted:

1. [Artifact Contract](../artifact_contract.md) stated that `ArtifactBundle` is
   the **only** Training → Models handoff.
2. Production Drive publish (`ArtifactOutputManager` /
   `publish_contract.py`) emits **Capability Packages** (adapter/merged
   directories with root `artifact.json`) consumed by `aiodoo-validation` and by
   frozen `aiodoo-model` `PublishingService`.
3. Configs and docs mixed **Capability** language with **Product** language
   (“product adapter”), while Development / Reasoning products correctly belong
   in `aiodoo-model`.
4. Sibling repository naming drifted (`aiodoo-models` vs frozen `aiodoo-model`).
5. Ecosystem **Model Lifecycle** (referenced here as **Ecosystem ADR-0001 —
   AIODOO Model Lifecycle**) must not be confused with this repository’s
   [ADR-0001 Immutable Domain](0001-immutable-domain.md).

This ADR records the binding clarifications. It does **not** authorize Python
API changes, metadata implementation, or capability-mapping code (those are
later implementation phases).

## Decision

### 1. Dual package surfaces (both retained)

| Surface | Role | Authority |
|---------|------|-----------|
| **Checkpoint** | Training-internal resume weights + sidecars | **Internal only** |
| **Capability Package** | Inference adapter or merged directory with root `artifact.json` (+ weights; optional checksums; local `manifest.json` for training diagnostics) | **Authoritative external handoff** to `aiodoo-validation` and to `aiodoo-model` registry publish |
| **ArtifactBundle** | Portable export inventory (`export_manifest.json`, roles, fingerprints, optional evaluation sidecars, optional merged role tree) | **First-class export package**; source for optional merged Drive publish; **not** what `PublishingService` opens today |

Multiple formats continue because they serve different lifecycle stages.
Neither surface is deleted. Phase 4 exporters and `ExportManager` remain frozen;
Drive publish remains the production external layout.

### 2. Actual pipeline relationship (not a strict chain)

Capability Package and ArtifactBundle are **siblings** produced from a successful
run, not a strict Capability → Bundle sequence:

```text
Checkpoint
   ├──→ Capability Package (adapter Drive publish from checkpoint)
   └──→ ArtifactBundle (ExportManager)
            └──→ optional Merged Capability Package (from bundle merged role)
```

Idealized single-chain diagrams in planning docs are superseded by this shape.

### 3. Capability ≠ Product

- **Capability** = skill identity (`coding`, `repair`, `execution`, `planner`,
  `context`, `conversation`, `approval`, `evaluation`) trained as one independent
  adapter from the base model.
- **Product** = composed offering (**Development**, **Reasoning**) owned by
  `aiodoo-model` (Release binding / composition policy).
- Training **never** creates products.

### 4. Context capability

**Context is an independent capability** (catalog id `context`, adapter
`aiodoo-context`). It is not folded into planner or conversation, and it is not
“infrastructure.” Validation may add a context pack later without redesigning
training ownership. See [Capability Model](../capability_model.md).

### 5. Publish terminology

| Term | Meaning |
|------|---------|
| **Drive publish** | Training writes packages into the workspace (`ArtifactOutputManager`) |
| **Registry publish** | `aiodoo-model` `PublishingService` ingests a Capability Package into the registry |

### 6. Ecosystem ADR citation

When referring to the canonical AIODOO Model Lifecycle, write:

> **Ecosystem ADR-0001 — AIODOO Model Lifecycle**

Do not confuse it with training [ADR-0001 Immutable Domain](0001-immutable-domain.md).

### 7. Related documentation (authoritative set)

| Topic | Document |
|-------|----------|
| Terminology | [terminology.md](../terminology.md) |
| Ownership | [ownership.md](../ownership.md) |
| Capability | [capability_model.md](../capability_model.md) |
| Product | [product_model.md](../product_model.md) |
| Lifecycle | [lifecycle.md](../lifecycle.md) |
| Metadata | [metadata_ownership.md](../metadata_ownership.md) |
| Freeze prep | [freeze_readiness.md](../freeze_readiness.md) |
| Drive layout | [artifact_output_pipeline.md](../artifact_output_pipeline.md) |
| Bundle contract | [artifact_contract.md](../artifact_contract.md) (clarified by this ADR) |

## Consequences

- Positive: Ends package-authority ambiguity; aligns docs with frozen
  `aiodoo-model` ingest and production Drive publish; preserves Phase 4 Bundle.
- Positive: Clear Capability vs Product ownership across the ecosystem.
- Negative: Operators must learn two package layouts; docs must keep both
  accurate.
- Non-consequence: No Python behavior change in the ADR acceptance itself.

## Freeze note

This ADR is **governance clarification**. Implementation phases that enrich
`artifact.json` fields or capability→type maps must follow existing ports and
must not reopen Phases 0–7 architecture.
