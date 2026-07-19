# Changelog

All notable changes to `aiodoo-training` are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/).

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
- Preserves published historical git tag **v1.0.0** (immutable; see below)

### Frozen

- Phases 0–7 architecture
- Capability Package protocol (Option A)
- ArtifactBundle protocol `"1"`
- Repository boundaries (no PyPI packaging)

## [1.0.0] — 2026-07-15

Published git tag `v1.0.0` (immutable). Phases 0–7 architecture freeze and
release hardening on that tree. Package `__version__` on that commit remained
`0.1.0`. Capability Package lifecycle (ADR-0022), repository freeze governance
(ADR-0023), and the production freeze package identity are released as **1.0.1**.

## [0.1.0] — prior

Phases 0–7 implementation under pre-freeze development versioning.
