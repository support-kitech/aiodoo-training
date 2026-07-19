# ADR-0001: Immutable Domain

## Status

Accepted (Phase 0)

## Citation note

This is the **training-repository** ADR-0001 (immutable domain types).

The cross-repository Model Lifecycle decision is cited as **Ecosystem ADR-0001 —
AIODOO Model Lifecycle**. Do not conflate the two. See
[terminology.md](../terminology.md) and
[ADR-0022](0022-package-surfaces-lifecycle-alignment.md).

## Context

Training pipelines share configuration, progress, and artifact metadata across
stages, processes, and future distributed workers. Mutable domain objects create
silent coupling and non-deterministic bugs.

## Decision

All domain entities are implemented as frozen dataclasses
(`@dataclass(frozen=True, slots=True)`). Once constructed, their state cannot
be modified in place.

## Consequences

- Positive: safe sharing across pipeline stages and future workers; clearer
  invariants; better testability.
- Negative: updates require `dataclasses.replace()` or new instances (copy-on-write).
