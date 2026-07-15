# ADR-0002: Ports over Frameworks

## Status

Accepted (Phase 0)

## Context

HuggingFace Transformers, PEFT, and related libraries evolve quickly. Embedding
their concrete types into domain and application code would permanently couple
AIODOO training to one stack.

## Decision

All ML capabilities are expressed as ports (abstract interfaces) under
`aiodoo_training.ports`. Concrete backends live only under
`aiodoo_training.infrastructure` and are selected via registries and factories.

## Consequences

- Positive: backends can be swapped (HF Trainer today, custom loop / DeepSpeed later).
- Negative: slight indirection and more files than a script-centric approach.
