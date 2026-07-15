# ADR-0004: Experiment Fingerprint

## Status

Accepted (Phase 0)

## Context

Reproducibility is a product requirement. Two runs are comparable only if their
configuration, datasets, and relevant versions can be shown to match.

## Decision

Every experiment derives an `ExperimentId` from a deterministic fingerprint
composed of:

- canonical hash of the **composed** (portable) configuration
- dataset fingerprint (path placeholders in Phase 0; content digests later)
- version fingerprint
- package fingerprint

Environment fingerprint is recorded for diagnostics but excluded from the
default identity digest so the same experiment is portable across machines.

Path resolution to absolute paths is a **runtime** concern and must not feed
experiment identity hashing.

## Consequences

- Positive: configs and fingerprints can be validated without GPU access; IDs
  remain stable across hosts.
- Negative: fingerprint inputs must be carefully versioned as phases add digests.
