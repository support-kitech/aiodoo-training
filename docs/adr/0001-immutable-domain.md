# ADR-0001: Immutable Domain

## Status

Accepted (Phase 0)

## Context

Training pipelines share configuration, progress, and artifact metadata across
stages, processes, and future distributed workers. Mutable domain objects create
silent coupling and non-deterministic bugs.

## Decision

All domain entities are implemented as frozen dataclasses
(`@dataclass(frozen=True, slots=True)`). Once constructed, their state cannot
be modified in place.

## Consequences

- Positive: safe sharing, clearer invariants, better testability.
- Negative: updates require `dataclasses.replace()` or new instances.
