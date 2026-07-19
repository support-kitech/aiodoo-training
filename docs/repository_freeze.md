# Repository Freeze (v2.0.0)

**Status:** Phases 0–7 and Capability Package lifecycle alignment are **COMPLETE**.  
Repository is **FROZEN**; ecosystem tooling release identity is **v2.0.0**.

Published historical tags **v1.0.0** and **v1.0.1** are preserved and are **not**
moved. **v1.0.0** predates Capability Package lifecycle alignment (ADR-0022).
**v1.0.1** is the first Capability Package production freeze identity.

Binding references:

- [Frozen Public Contracts](frozen_public_contracts.md)
- [Freeze Readiness](freeze_readiness.md)
- [MAINTENANCE.md](MAINTENANCE.md)
- [release_checklist.md](release_checklist.md)
- [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md)
- [ADR-0023](adr/0023-repository-freeze-v1.md)
- [AUDIT_RESOLUTION.md](../AUDIT_RESOLUTION.md)
- [CHANGELOG.md](../CHANGELOG.md)

## What is frozen

| Surface | Freeze meaning |
|---------|----------------|
| **Architecture (Phases 0–7)** | Ports, domain, pipeline, registries — extend only via Section 9 |
| **Capability Package protocol** | `artifact_type` kinds + `capability_id` / `adapter_type` / `peft_type` roles (Option A) |
| **ArtifactBundle protocol `"1"`** | Export inventory layout remains producible |
| **Repository boundaries** | No dataset gen, certification, registry, products, runtime |
| **Distribution model** | Internal clone-and-run; **not** PyPI-published |

## Semantic versioning

Follows [SemVer 2.0.0](https://semver.org/).

| Change | Bump |
|--------|------|
| Breaking public port/domain or on-disk Capability Package / Bundle break | **MAJOR** |
| Backward-compatible additive feature within frozen architecture | **MINOR** |
| Bug fix, docs, tests, hardening with identical public behavior | **PATCH** |

**v2.0.0** is an ecosystem tooling alignment major (release identity + honesty).
Protocol Option A and Phases 0–7 public contracts are unchanged from v1.0.1.

## Explicit non-claims

- Merge CLI / full PEFT merge path: deferred
- `evaluate.py` / `export.py` root wrappers: deferred (pipeline APIs exist)
- Ecosystem E2E (core / vscode dual-model runtime): out of this repository
- Training-scale approval/conversation/evaluation corpora: owned by datasets
- `context` validation profile: owned by validation
