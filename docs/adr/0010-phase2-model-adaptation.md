# ADR-0010: Phase 2 Model Loading & Adaptation

## Status

Accepted (Phase 2 permanently frozen — ADR-0012)

## Context

Phase 2 introduces base-model loading and adaptation without training loops.
Frozen Phase 0/1 ports already define `ModelBackend`, `AdaptationStrategy`,
`ResourcePlanner`, and opaque handles. Implementation must not enlarge those
contracts with framework types.

## Decision

1. Reuse frozen ports and registries; factories instantiate registered backends.
2. HuggingFace Causal LM and PEFT LoRA/QLoRA live only under `infrastructure/`.
3. CI uses `StubModelBackend` (no Torch required).
4. `QuantizationPolicy` abstracts 4-bit / 8-bit / float precisions (alias:
   `QuantizationSpec`); infrastructure maps to bitsandbytes / dtypes.
5. Model / adapter fingerprints compose on top of `FingerprintService` without
   changing its frozen signature.
6. New catalogs (`model_family_registry`, `model_profile_registry`,
   `model_capability_registry`, `adapter_registry`) are registration-driven.
7. Hardening decisions: see ADR-0011 (no ModelSession; adapter profiles;
   strengthened boundary tests).

## Consequences

- Positive: Llama / Mistral / DeepSeek / Gemma / Phi profiles register without
  architecture changes.
- Positive: framework quarantine remains enforceable by AST boundary tests.
- Negative: real HF weight loading requires optional `requirements/train.txt`.
