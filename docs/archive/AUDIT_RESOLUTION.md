> **Historical document.** Written when Git tags / release identity existed.
> Git tags and GitHub Releases were later removed ecosystem-wide.
> **Current source of truth:** branch `main` only. See `docs/STATUS.md`.
> Do not treat tag or release recommendations in this file as current instructions.

# aiodoo-training — Audit Resolution (v2.0.0)

## Batch A — tooling freeze (completed in `b6508d1`)

| Audit Finding | Category | Status |
| :--- | :--- | :--- |
| Still `__version__=1.0.1` while siblings tagged v2.0.0 | **Production Blocker** | **DONE** (`2.0.0`) |
| README listed evaluate/export as fully working | **Documentation** | **DONE** |
| `.gitignore` bare `adapters/` hid fixtures | **Bug** | **DONE** (`/adapters/`) |

## Batch B — completion residuals (this pass)

| Audit Finding | Category | Decision | Action | Implementation Required? |
| :--- | :--- | :--- | :--- | :---: |
| Missing `RELEASE_REPORT.md` | **Missing Implementation** | Fix | Write release report + verdict | **YES** |
| IMPLEMENTATION_REPORT missing completion delta | **Documentation** | Fix | Refresh for Batch B | **YES** |
| `freeze_readiness.md` / `frozen_public_contracts.md` still center v1.0.1 as current | **Documentation** | Fix | Align current identity to v2.0.0 | **YES** |
| Confirm capability-package fixtures still tracked | **Bug** (verify) | Verify | Fix gitignore only if broken | **YES** |
| Re-run ruff/mypy/pytest/coverage ≥80 | **Production Blocker** if red | Verify | Fix only if failing | **YES** |
| `cmd_merge` / merge NotImplementedError | **Future Work** | Leave | Document only | **NO** |
| HFExporter always stub | **Intentional** | Leave | Document in RELEASE_REPORT | **NO** |
| evaluate/export root wrappers deferred | **Intentional** | Leave | Already honest in README | **NO** |
| No context validation profile | **Out Of Scope** | Leave | Owned by validation | **NO** |
| Sparse approval/conversation/evaluation data | **Out Of Scope** | Leave | Owned by datasets | **NO** |
| Adapter composition / runtime inference | **Out Of Scope** | Leave | Owned by model/core | **NO** |

## Implementation batch B (YES only)

1. Refresh this file.
2. Verify gitignore/fixtures.
3. Re-run quality gates.
4. Write `RELEASE_REPORT.md`; refresh IMPLEMENTATION_REPORT + residual freeze docs.
5. Logical commits; recreate local annotated `v2.0.0`.
