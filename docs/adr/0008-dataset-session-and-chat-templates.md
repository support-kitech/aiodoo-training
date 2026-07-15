# ADR-0008: DatasetSession and ChatTemplateRegistry

## Status

Accepted (Phase 0 freeze amendment)

## Context

Phase 1 introduces dataset streaming/tokenization. Without a frozen session
abstraction, resume and distributed work would force domain churn. Without a
chat template registry, tokenizer backends would hardcode model-family prompting.

## Decision

1. **DatasetSession** — immutable domain object representing dataset consumption
   runtime state (epoch, iterator position, fingerprint, worker assignment,
   resume metadata). No tokenization or training logic.

2. **ChatTemplateRegistry** — dedicated registry for `ChatTemplate` ports, keyed
   by family/template name. Tokenizers resolve templates via this registry only.

## Consequences

- Positive: resume/curriculum/distributed training can evolve without redesign;
  model families stay decoupled from tokenizer implementations.
- Negative: one additional registry and session type to maintain.
