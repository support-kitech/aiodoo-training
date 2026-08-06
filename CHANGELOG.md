# Changelog

## Unreleased

- Documentation sync: `main` is source of truth; living posture in `docs/STATUS.md`; historical reports under `docs/archive/`; cross-references updated after archive moves and Git tag metadata reset.


All notable changes to `aiodoo-training` are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/).

## [Unreleased]

### Added

- `aiodoo_contract` adopted as the canonical Capability Contract dependency;
  `aiodoo-training` is now its second canonical consumer, after
  `aiodoo-datasets` (see `docs/CONTRACT_ADOPTION.md`).
- `aiodoo_training/contract/`: `adapters.py` (`project_<capability>`,
  `ContractAdapterError`, mirroring aiodoo-datasets' own projection layer
  against the same record shape), `prompt_bridge.py` (renders capability
  prompts exclusively via `aiodoo_contract.prompts.CapabilityPromptBuilder`
  and serializes training labels to the canonical `CapabilityResponse`
  JSON), `version_check.py` (`ensure_contract_compatible`,
  `TRAINING_CONTRACT_VERSION`).
- `datasets/formatters/formatters.py`: the six contract-mapped dataset
  types (`planner`, `coding`, `repair`, `execution`, `conversation`,
  `approval`) now build their `TrainingExample` exclusively through the new
  contract bridge instead of ad-hoc instruction/context string
  concatenation. `context`/`evaluation` are unchanged (no contract
  projection — see `docs/CONTRACT_ADOPTION.md`).
- `infrastructure/huggingface/templates.py`: `ChatTemplate` implementations
  (`QwenChatTemplate`, new `DeepSeekChatTemplate`, `GenericChatTemplate`,
  `LlamaChatTemplate`, `MistralChatTemplate`) now delegate rendering to
  `aiodoo_contract.templates` instead of the local `SimpleRoleChatTemplate`.
  The frozen `ChatTemplate` port is unchanged.
- `application/train_orchestrator.py`: calls
  `ensure_contract_compatible()` before bootstrapping, failing early on an
  incompatible installed `aiodoo_contract` version.
- `datasets/validation.py`: `DatasetValidator` additionally projects sampled
  records through `aiodoo_contract.validators.ContractValidator` for every
  capability with a canonical contract shape.
- `artifacts/publish_contract.py`: Capability Package `artifact.json` now
  carries `contract_version` (always) and `capability_package_metadata` —
  the canonical `aiodoo_contract.schemas.CapabilityPackageMetadata` — when
  derivable. Additive; frozen protocol fields unchanged.
- `docs/CONTRACT_ADOPTION.md`.

### Fixed

- **ACT-101**: `pipeline/artifact_hooks.py::maybe_publish_artifacts` no
  longer reports a training run as successful when the required adapter
  Capability Package could not be published. It now returns `bool`, and
  `pipeline/handlers.py::FinalizeStage` fails the pipeline
  (`StageStatus.FAILED`) when publishing was configured, training
  completed, and the adapter publish still failed.
- **ACT-118**: `tracking/core.py`'s `_degrade` and every hook in
  `pipeline/tracking_hooks.py` now log swallowed tracking-sink exceptions
  (`logger.warning(..., exc_info=True)`) instead of discarding them
  silently. Tracking remains best-effort; only observability changed.

### Changed

- Contract-mapped `TrainingExample.messages` now include a leading `system`
  turn (the capability's default system prompt from
  `CapabilityPromptBuilder`) — training and runtime inference now render
  capability prompts identically. Golden tokenization digests regenerated.

## [2.0.0] — 2026-07-19

### Overview

AIODOO ecosystem **tooling freeze v2.0.0**. Architecture (Phases 0–7) and
Capability Package Option A remain frozen. This major aligns release identity
with sibling repositories (`aiodoo-datasets`, `aiodoo-validation`) and corrects
documentation honesty for deferred CLI wrappers.

### Changed

- Package version **2.0.0**
- README entry-script honesty (`evaluate` / `export` / `merge` deferred)
- Includes post-v1.0.1 `.gitignore` fixture fix (`/adapters/` rooted ignore)

### Frozen (unchanged)

- Phases 0–7 architecture
- Capability Package protocol (Option A)
- ArtifactBundle protocol `"1"`
- Repository boundaries (no PyPI packaging)

### Not in this release

- Merge implementation (`cmd_merge` remains deferred)
- Real `HFExporter` PEFT write path (stub layout export remains)
- Dataset richness / validation `context` profile (other repositories)

## [1.0.1] — 2026-07-19

### Added

- Capability Package producer contract (Option A): `capability_id`, retained
  `adapter_type`, `peft_type`, family/architecture/odoo/`created_at` metadata
- Lifecycle alignment documentation (terminology, ownership, capability/product,
  lifecycle, metadata ownership, ADR-0022)
- Ecosystem contract tests and representative `protocol/v1` goldens
- Repository freeze / maintenance / release checklist governance (ADR-0023)

### Changed

- Coverage gate raised to **80%** (infrastructure omitted; measured ≈81%)
- Package version **1.0.1** — production freeze release identity
- Originally preserved historical git tag **v1.0.0** (later removed in ecosystem metadata reset; `main` is SoT)

### Frozen

- Phases 0–7 architecture
- Capability Package protocol (Option A)
- ArtifactBundle protocol `"1"`
- Repository boundaries (no PyPI packaging)

## [1.0.0] — 2026-07-15

Published git tag `v1.0.0` at the time (later removed; `main` is SoT). Phases 0–7 architecture freeze and
release hardening on that tree. Package `__version__` on that commit remained
`0.1.0`. Capability Package lifecycle (ADR-0022), repository freeze governance
(ADR-0023), and the production freeze package identity are released as **1.0.1**.

## [0.1.0] — prior

Phases 0–7 implementation under pre-freeze development versioning.
