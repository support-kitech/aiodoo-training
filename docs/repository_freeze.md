# Repository Freeze (v1.0.0)

**Status:** Phases 0–7 and Capability Package lifecycle alignment are **COMPLETE**.  
Repository is **FROZEN for v1.0.0**.

Binding references:

- [Frozen Public Contracts](frozen_public_contracts.md)
- [Freeze Readiness](freeze_readiness.md)
- [MAINTENANCE.md](MAINTENANCE.md)
- [release_checklist.md](release_checklist.md)
- [ADR-0022](adr/0022-package-surfaces-lifecycle-alignment.md)
- [ADR-0023](adr/0023-repository-freeze-v1.md)

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

Prefer **PATCH** after v1.0.0 unless an ADR justifies MINOR/MAJOR.

## Quality gates (v1.0)

```bash
python3 -m ruff check .
python3 -m ruff format --check .
python3 -m mypy aiodoo_training
python3 -m coverage run -m pytest
python3 -m coverage report -m --fail-under=80
```

Infrastructure under `aiodoo_training/infrastructure/` remains omitted from coverage
(quarantined adapters). Measured non-infrastructure coverage at freeze ≈81%.

## Explicitly not in v1.0.0

- PyPI packaging / wheels
- Product composition (Development / Reasoning)
- Validation certification engine
- Model registry / promotion
- Inference / serving
- Raising coverage to sibling 95% bars (deferred; floor is 80%)

## See also

- [Ownership](ownership.md)
- [Lifecycle](lifecycle.md)
- [Capability Model](capability_model.md)
