# ADR-0009: Resource Management Layer

## Status

Accepted (Phase 1 freeze amendment)

## Context

Phase 2 will introduce Torch, Transformers weight loading, PEFT, and eventually
CUDA / MPS / multi-GPU planners. Ad-hoc device and dtype checks in application
or pipeline code would permanently couple later phases to one accelerator stack
and force redesign when DeepSpeed, FSDP, Apple Silicon, or CPU-only training appear.

## Decision

Hardware decisions are centralized behind a **Resource Management** surface:

| Artifact | Layer | Role |
|----------|-------|------|
| `DevicePolicy`, `PrecisionPolicy`, `MemoryPolicy` | Domain | Declared preferences |
| `HardwareCapabilities` | Domain | Probe result |
| `ExecutionEnvironment` | Domain | Resolved run plan |
| `ExecutionSpec`, `DistributedSpec` | Domain / Config | Experiment-declared policies |
| `ResourcePlanner` | Port | Probe + resolve API |
| `StaticResourcePlanner` | Infrastructure | CPU-only planner (no Torch) |

Opaque model handles (`BaseModelHandle`, `TrainableModelHandle`) keep Torch /
PEFT types out of ports. Model loading and adaptation consume `ExecutionEnvironment`
instead of inspecting CUDA directly.

A dedicated pipeline stage name `RESOLVE_EXECUTION` exists for future wiring;
Phase 1 does not execute training stages.

## Consequences

- Positive: Llama / Mistral / DeepSeek / Gemma / Phi families and accelerators
  can land without schema or domain redesign.
- Positive: CI and Phase 1 remain CPU-only via `StaticResourcePlanner`.
- Negative: callers must resolve an environment before loading models (intentional).
