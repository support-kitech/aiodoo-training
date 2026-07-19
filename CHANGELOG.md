# Changelog

All notable changes to `aiodoo-training` are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [SemVer](https://semver.org/).

## [1.0.0] — 2026-07-19

### Added

- Capability Package producer contract (Option A): `capability_id`, retained
  `adapter_type`, `peft_type`, family/architecture/odoo/`created_at` metadata
- Lifecycle alignment documentation (terminology, ownership, capability/product,
  lifecycle, metadata ownership, ADR-0022)
- Ecosystem contract tests and representative `protocol/v1` goldens
- Repository freeze / maintenance / release checklist governance

### Changed

- Coverage gate raised to **80%** (infrastructure omitted; measured ≈81%)
- Package version **1.0.0** — production freeze

### Frozen

- Phases 0–7 architecture
- Capability Package protocol (Option A)
- ArtifactBundle protocol `"1"`
- Repository boundaries (no PyPI packaging)

## [0.1.0] — prior

Phases 0–7 implementation under pre-freeze development versioning.
