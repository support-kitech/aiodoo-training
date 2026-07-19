# aiodoo-training — RELEASE_REPORT (v2.0.0)

**Release identity:** annotated tag `v2.0.0` (ecosystem tooling freeze)  
**Capability Package:** Option A (unchanged from v1.0.1)  
**Date:** 2026-07-19

---

## Production Ready

| Question | Answer |
| --- | --- |
| Capability Package producer ready (in-boundary)? | **YES** |
| Merge / full PEFT HFExporter ready? | **NO** (deferred / intentional stub) |
| Dual-model / runtime / composition ready? | **NO** (out of scope) |
| Production score (in-boundary) | **8 / 10** |

---

## Quality gates (local)

| Gate | Result |
| --- | --- |
| `ruff check .` | Pass |
| `ruff format --check .` | Pass |
| `mypy aiodoo_training` | Pass (211 files) |
| `coverage run -m pytest` | **465 passed** |
| `coverage report --fail-under=80` | **81%** |

---

## Fixtures / gitignore

- `/adapters/` rooted ignore in force
- Capability Package goldens tracked:
  - `tests/fixtures/capability_packages/protocol/v1/adapters/coding/`
  - `tests/fixtures/capability_packages/protocol/v1/adapters/repair/`
  - base model + merged coding representatives

---

## Explicit non-claims

- `evaluate.py` / `export.py` / `merge` root wrappers: deferred (`NotImplementedError`)
- `HFExporter` (`hf_peft`): layout/stub path; not a full PEFT weight writer
- Product composition (Development / Reasoning): owned by `aiodoo-model`
- Sparse approval/conversation/evaluation corpora: owned by `aiodoo-datasets`
- `context` validation profile: owned by `aiodoo-validation`

---

## Architecture impact

None. Phases 0–7 and Capability Package Option A unchanged.

---

## Remaining blockers

None for in-boundary Capability Package producer freeze.

---

## Remaining future work

- Merge CLI / full PEFT merge path
- Real HFExporter PEFT write path
- evaluate/export root wrapper wiring to pipeline APIs

---

## Architectural debt

- Deferred root CLI wrappers alongside working pipeline APIs
- Coverage omits infrastructure (intentional)

---

## Repository health

**Strong** — gates green, fixtures trackable, docs honest about deferred paths.

---

## Release recommendation

**Ship annotated tag `v2.0.0`**. Preserve `v1.0.0` / `v1.0.1`. Do not claim merge
or dual-model runtime readiness.
