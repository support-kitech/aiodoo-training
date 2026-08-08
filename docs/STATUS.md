# Status — aiodoo-training

**Living document.** Code on `main` is the only implementation Source of Truth
for this **Training** repository.  
**System documentation SoT (separate ecosystem):** `aiodoo-core/docs/SYSTEM.md`  
**Permanent branch:** `main`  
**Historical evidence:** `docs/archive/`

## Purpose

Train optional capability adapters; publish Capability Packages. **Training
never defines the System.** The System must operate correctly with foundation
models only; adapters are optional.

## Current implementation (on main)

| Item | Status |
|------|--------|
| Capability training plane | Shipped / frozen surface on `main` |
| Emits Development / Reasoning product packages | **Never** (ownership: `aiodoo-model`) |
| Product composition | Out of scope |
| Required for System correctness | **No** |

## Living docs

- `docs/architecture.md`, `docs/capability_model.md`, `docs/product_model.md`, `docs/lifecycle.md`, `docs/ownership.md`
- `docs/CONTRACT_ADOPTION.md`, `docs/adr/`, `PRODUCTION_TRAINING.md`
- `README.md`, `CHANGELOG.md`
