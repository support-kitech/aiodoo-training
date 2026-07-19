# Freeze Readiness Documentation

**Status:** Lifecycle alignment freeze criteria **met** for **v1.0.1**  
**Binding freeze statement:** [repository_freeze.md](repository_freeze.md) · [ADR-0023](adr/0023-repository-freeze-v1.md)  
**Related:** [Frozen Public Contracts](frozen_public_contracts.md), [Ownership](ownership.md), [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md)

Phases 0–7 remain frozen as training architecture. Capability Package lifecycle
alignment (B0–B2) is included in the v1.0.1 production freeze. Historical git
tag **v1.0.0** (2026-07-15) is preserved and not rewritten.

---

## Freeze criteria (lifecycle alignment)

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Documentation set current (B0) | **COMPLETE** |
| 2 | Capability Package metadata complete (B1) | **COMPLETE** |
| 3 | Contract / golden tests (B2) | **COMPLETE** |
| 4 | No sibling imports in production code | **COMPLETE** |
| 5 | Product composition out of scope | **COMPLETE** |
| 6 | Coverage / version policy recorded | **COMPLETE** (`fail_under=80`, `__version__=1.0.1`) |
| 7 | Section 9 / ADR-0022 / ADR-0023 | **COMPLETE** |

### Known consumer gaps (not training defects)

| Gap | Owner | Notes |
|-----|-------|-------|
| No frozen validation profile for `context` | `aiodoo-validation` | Packages resolve; certification pack deferred |
| Merged registry deps (`base_artifact_id`, `adapter_artifact_ids`) | Caller / `aiodoo-model` `PublishingRequest` | Training cannot know registry ids before publish |
| Default `supported_odoo_versions` when config omits them | Training config | Override via config; defaults are `(17, 18, 19)` |

---

## Public contracts (stable)

See [frozen_public_contracts.md](frozen_public_contracts.md) for Phases 0–7.

Lifecycle clarifications that are also stable:

- Capability Package is the **authoritative external handoff**
- ArtifactBundle remains the **export inventory**
- Capability ≠ Product
- Context is an independent capability
- Drive publish ≠ Registry publish

---

## Compatibility guarantees

After v1.0.1 freeze:

- Existing Phase 0–7 ports, domain types, and registries remain stable
- ArtifactBundle protocol `"1"` layout remains producible
- Capability Package Drive layout remains the external handoff
- Additive optional metadata fields may appear without bumping training resume protocol
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
- PyPI packaging (clone-and-run only)

---

## Future extension points (additive only)

- New capability ids via config packs + naming catalog + metadata maps
- New export roles (GGUF, ONNX, …) via exporter registry + optional Bundle roles
- Richer `artifact.json` fields without importing model code
- Additional trackers behind `ExperimentTracker`
- Distributed backends behind existing Phase 7 ports
- Optional coverage raises beyond 80% without redesign

---

## Repository boundaries (summary)

```text
aiodoo-datasets → aiodoo-training → Capability Package
                                      ├→ aiodoo-validation
                                      └→ aiodoo-model (registry publish)
                                           └→ runtime / core consume
```

---

## Operator checklist (v1.0.1)

- [x] B0 documentation accepted
- [x] ADR-0022 accepted
- [x] B1 metadata enrichment complete
- [x] B2 contract tests green
- [x] Coverage / version policy recorded
- [x] ADR-0023 repository freeze accepted
- [x] README / CONTRIBUTING navigation updated
- [x] Release identity set to **v1.0.1** (historical **v1.0.0** tag preserved)
