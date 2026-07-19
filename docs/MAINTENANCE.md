# Maintenance Policy (v1.x)

Policy after the v1.0.0 freeze. Freeze statement: [repository_freeze.md](repository_freeze.md).

## When ADRs are required

Write an ADR under `docs/adr/` when a change:

- Alters layer ownership or repository boundaries
- Changes Capability Package or ArtifactBundle protocol in a breaking way
- Widens frozen port signatures
- Changes resume / training protocol semantics
- Changes distribution policy (e.g. publishing wheels)

Bugfixes and docs-only edits do **not** require ADRs.

## Compatibility expectations

- Capability Package `artifact_type` values remain the frozen validation/model kinds
  (`coding_adapter`, `base_model`, `merged_model`)
- Business identity remains in `capability_id`; validation skill label in `adapter_type`;
  PEFT kind in `peft_type`
- ArtifactBundle `artifact_protocol_version = "1"` remains the produce version unless bumped via ADR
- No Python imports of `aiodoo_model` / `aiodoo_validation` in production code

## Deprecation

- Prefer additive optional metadata fields over renames
- Deprecate with docs + tests; remove only on MAJOR
- Historical phase docs may retain legacy names; ADR-0022 banners supersede for external handoff

## Quality gates

Keep CI green: ruff, mypy (strict on non-infrastructure), pytest, coverage ≥80%.

## See also

- [release_checklist.md](release_checklist.md)
- [frozen_public_contracts.md](frozen_public_contracts.md) Section 9
