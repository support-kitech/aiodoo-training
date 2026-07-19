# ADR-0023: Repository Freeze v1.0.0

## Status

**Accepted** — 2026-07-19

## Context

Phases 0–7 are permanently frozen. ADR-0022 clarified Capability Package vs
ArtifactBundle authority. B1 implemented Capability Package metadata; B2 proved
compatibility with frozen `aiodoo-validation` and `aiodoo-model`.

Remaining work was governance: version, coverage floor, maintenance/release
docs — not architecture redesign.

## Decision

1. Tag the repository line as **v1.0.0** production freeze.
2. Keep distribution as **internal clone-and-run** (not PyPI).
3. Set coverage `fail_under` to **80** (infrastructure omitted).
4. Bind maintenance to [MAINTENANCE.md](../MAINTENANCE.md) and
   [repository_freeze.md](../repository_freeze.md).

## Consequences

- Positive: clear SemVer and release process; CI matches measured quality.
- Positive: contributors have freeze/maintenance/checklist docs.
- Negative: coverage is not at sibling 95% bars; raising further is future work,
  not a freeze blocker.
- Non-consequence: no protocol or ownership redesign.

## See also

- [ADR-0022](0022-package-surfaces-lifecycle-alignment.md)
- [release_checklist.md](../release_checklist.md)
