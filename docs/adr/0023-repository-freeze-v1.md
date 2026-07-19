# ADR-0023: Repository Freeze (v1 line)

## Status

**Accepted** — 2026-07-19  
**Release identity:** production freeze published as **v1.0.1** (historical
**v1.0.0** tag preserved).

## Context

Phases 0–7 are permanently frozen. ADR-0022 clarified Capability Package vs
ArtifactBundle authority. B1 implemented Capability Package metadata; B2 proved
compatibility with frozen `aiodoo-validation` and `aiodoo-model`.

Remaining work was governance: version, coverage floor, maintenance/release
docs — not architecture redesign.

A git tag `v1.0.0` was already published on 2026-07-15 for an earlier
architecture cut (that tree still reported package `__version__ = 0.1.0` and
lacked ADR-0022/0023). Rewriting that tag is not acceptable.

## Decision

1. Publish the production freeze as **v1.0.1** (package + annotated git tag).
2. Leave historical tag **v1.0.0** immutable.
3. Keep distribution as **internal clone-and-run** (not PyPI).
4. Set coverage `fail_under` to **80** (infrastructure omitted).
5. Bind maintenance to [MAINTENANCE.md](../MAINTENANCE.md) and
   [repository_freeze.md](../repository_freeze.md).

## Consequences

- Positive: clear SemVer and release process; CI matches measured quality.
- Positive: contributors have freeze/maintenance/checklist docs.
- Positive: published release history remains immutable.
- Negative: coverage is not at sibling 95% bars; raising further is future work,
  not a freeze blocker.
- Non-consequence: no protocol or ownership redesign.

## See also

- [ADR-0022](0022-package-surfaces-lifecycle-alignment.md)
- [release_checklist.md](../release_checklist.md)
- [CHANGELOG.md](../../CHANGELOG.md)
