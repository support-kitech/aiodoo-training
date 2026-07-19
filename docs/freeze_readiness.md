# Freeze Readiness Documentation

**Status:** Governance checklist for freezing `aiodoo-training` under Ecosystem ADR-0001 — AIODOO Model Lifecycle  
**Related:** [Frozen Public Contracts](frozen_public_contracts.md), [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md), [Ownership](ownership.md)

This document prepares freeze criteria. It does **not** declare the repository
frozen under the Model Lifecycle yet. Phases 0–7 remain frozen as training
architecture; lifecycle alignment freeze follows implementation phases B1+.

---

## Freeze criteria (lifecycle alignment)

The repository may be declared **lifecycle-frozen** when all of the following
are true:

1. **Documentation set current** — terminology, ownership, capability, product,
   lifecycle, metadata, artifact authority (B0).
2. **Capability Package metadata complete** for the accepted capability catalog
   (implementation B1).
3. **Contract tests** prove Capability Package shape is ingestible by frozen
   model expectations / fixtures (B2).
4. **No Python imports** of `aiodoo_model` / `aiodoo_validation` / sibling
   runtimes.
5. **Product composition** remains out of scope (documented + tested).
6. **Version / coverage governance** decided and recorded.
7. **Section 9** process followed for any frozen-contract wording amendments
   (Artifact Contract clarification via ADR-0022).

---

## Public contracts (stable)

See [frozen_public_contracts.md](frozen_public_contracts.md) for Phases 0–7.

Lifecycle clarifications that are also stable after B0:

- Capability Package is the **authoritative external handoff**
- ArtifactBundle remains the **export inventory**
- Capability ≠ Product
- Context is an independent capability
- Drive publish ≠ Registry publish

---

## Compatibility guarantees

After lifecycle freeze:

- Existing Phase 0–7 ports, domain types, and registries remain stable
- ArtifactBundle protocol `"1"` layout remains producible
- Capability Package Drive layout remains the external handoff
- Additive optional metadata fields may appear without bumping training
  resume protocol
- Breaking package layout changes require ADR + protocol bump

---

## Non-goals (will never belong in aiodoo-training)

- Dataset generation
- Validation oracles / certification authority
- Model registry, promotion, compatibility policy authority
- Product composition (Development / Reasoning packages)
- Inference / serving stacks
- Agent runtime / workflow ownership (`aiodoo-core`)
- Runtime profile definitions

---

## Future extension points (additive only)

- New capability ids via config packs + naming catalog + metadata maps
- New export roles (GGUF, ONNX, …) via exporter registry + optional Bundle roles
- Richer `artifact.json` fields without importing model code
- Additional trackers behind `ExperimentTracker`
- Distributed backends behind existing Phase 7 ports

---

## Repository boundaries (summary)

```text
aiodoo-datasets → aiodoo-training → Capability Package
                                      ├→ aiodoo-validation
                                      └→ aiodoo-model (registry publish)
                                           └→ runtime / core consume
```

---

## Freeze checklist (operators)

- [ ] B0 documentation accepted
- [ ] ADR-0022 accepted
- [ ] Ecosystem ADR-0001 referenced consistently
- [ ] B1 metadata enrichment complete
- [ ] B2 contract tests green
- [ ] Coverage / version policy recorded
- [ ] `frozen_public_contracts.md` related-docs list includes lifecycle docs
- [ ] README scope table uses `aiodoo-model` and Capability language
- [ ] Explicit non-goals section unchanged
