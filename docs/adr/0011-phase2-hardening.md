# ADR-0011: Phase 2 Hardening (pre-freeze)

## Status

Accepted — **Phase 2 permanently frozen**

## Context

Before permanently freezing Phase 2, several potential abstractions were reviewed:
ModelSession, AdapterRegistry vs AdaptationStrategy, QuantizationPolicy naming,
ModelMetadata serializability, and framework boundary enforcement.

## Decisions

### 1. No ModelSession

**Not introduced.**

`DatasetSession` tracks a consumption cursor (epoch / index / shard). Loaded
models have no analogous continuous cursor. Phase 2 already exposes:

- `LoadedModelContext` — handle + metadata + fingerprint + execution
- `AdaptedModelContext` — trainable handle + adapter metadata + fingerprint

Resume / adapter stacking / training lifecycle belong to Phase 3
`CheckpointHandle` and trainer state. A ModelSession today would duplicate
those contexts without a consumer and increase surface area.

### 2. AdapterRegistry (`adapter_registry`) for profiles

**Introduced.**

Mirrors `model_profile_registry`: declarative `AdapterProfile` metadata
(rank, targets, capabilities, `strategy_key`) is independent of
`AdaptationStrategy` behavior classes in `adaptation_registry`.

### 3. QuantizationPolicy

**Canonical name** aligned with `DevicePolicy` / `PrecisionPolicy` /
`MemoryPolicy`. `QuantizationSpec` remains a backward-compatible alias.
This is naming alignment, not a new architectural layer — load decisions still
flow through `ExecutionEnvironment.precision_policy` and fingerprints.

### 4. ModelMetadata

Hardened: frozen mappings, `Mapping[str, str]` extras, `to_dict` / `from_dict`
for JSON round-trips. Remains framework-independent.

### 5. Boundary tests

AST scanner now also flags `__import__("…")` and `importlib.import_module("…")`
for torch / transformers / peft / bitsandbytes / accelerate outside
`infrastructure/`.

## Consequences

- Positive: clearer metadata vs behavior split for adapters.
- Positive: stronger leak detection before Phase 3.
- Positive: no speculative ModelSession complexity.
