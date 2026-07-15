# ADR-0005: Adaptation Separation

## Status

Accepted (Phase 0)

## Context

LoRA, QLoRA, and full fine-tuning are strategies applied to a loaded base model.
Coupling adaptation into model loading recreates monolithic training scripts and
complicates future adapter types.

## Decision

`ModelBackend` loads the base model only. `AdaptationStrategy` applies
LoRA / QLoRA / full fine-tuning afterward. Factories and registries treat them
as independent collaborators.

## Consequences

- Positive: clear extension point for new adaptation methods.
- Negative: callers must wire two ports instead of one convenience function.
