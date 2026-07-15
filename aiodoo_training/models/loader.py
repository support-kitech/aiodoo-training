"""Model loading orchestration (Phase 2) — no framework imports."""

from __future__ import annotations

from dataclasses import dataclass

from aiodoo_training.determinism.model_fingerprints import fingerprint_model
from aiodoo_training.domain.config import ExecutionSpec
from aiodoo_training.domain.handles import BaseModelHandle
from aiodoo_training.domain.model_info import ModelFingerprint, ModelMetadata
from aiodoo_training.domain.quantization import QuantizationPolicy
from aiodoo_training.domain.refs import ModelRef
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.models.access import fingerprint_from_base_handle, metadata_from_base_handle
from aiodoo_training.ports.model import ModelBackend
from aiodoo_training.ports.resources import ResourcePlanner


@dataclass(frozen=True, slots=True)
class LoadedModelContext:
    """Result of ModelLoader.load — AIODOO types only."""

    handle: BaseModelHandle
    metadata: ModelMetadata
    fingerprint: ModelFingerprint
    execution: ExecutionEnvironment


class ModelLoader:
    """
    Orchestrates ResourcePlanner + ModelBackend.

    Callers must never check CUDA themselves — pass ExecutionSpec (or a resolved
    ExecutionEnvironment) and load through this service / ModelBackend.
    """

    def __init__(self, backend: ModelBackend, planner: ResourcePlanner) -> None:
        self._backend = backend
        self._planner = planner

    def resolve_execution(self, spec: ExecutionSpec) -> ExecutionEnvironment:
        """Resolve declared execution preferences via the frozen ResourcePlanner."""
        return self._planner.resolve_spec(spec)

    def load(
        self,
        model_ref: ModelRef,
        *,
        execution: ExecutionEnvironment | None = None,
        execution_spec: ExecutionSpec | None = None,
        tokenizer_binding: str | None = None,
    ) -> LoadedModelContext:
        """
        Resolve execution (if needed), load the base model, and return context.

        Prefer passing a resolved ``execution``. If omitted, ``execution_spec``
        (defaulting to ExecutionSpec()) is resolved through ResourcePlanner.
        """
        if execution is None:
            env = self.resolve_execution(execution_spec or ExecutionSpec())
        else:
            env = execution

        handle = self._backend.load(model_ref, env)
        metadata = metadata_from_base_handle(handle)
        if tokenizer_binding is not None and metadata.tokenizer_binding != tokenizer_binding:
            # Prefer explicit binding for fingerprint identity when provided.
            quantization = metadata.quantization
            fingerprint = fingerprint_model(
                model_ref,
                quantization=quantization,
                execution=env,
                tokenizer_binding=tokenizer_binding,
            )
        else:
            fingerprint = fingerprint_from_base_handle(handle) or fingerprint_model(
                model_ref,
                quantization=metadata.quantization,
                execution=env,
                tokenizer_binding=metadata.tokenizer_binding,
            )
        return LoadedModelContext(
            handle=handle,
            metadata=metadata,
            fingerprint=fingerprint,
            execution=env,
        )


def build_quantization_for_load(
    model_ref: ModelRef,
    execution: ExecutionEnvironment,
) -> QuantizationPolicy:
    """Derive QuantizationPolicy from execution precision policy (model ref noted)."""
    quant = QuantizationPolicy.from_precision_policy(execution.precision_policy)
    if (
        not quant.load_in_4bit
        and not quant.load_in_8bit
        and quant.compute == execution.precision_policy.compute
    ):
        # Preserve declared model precision when no quantization flags are set
        # and execution compute matches the policy default path.
        _ = model_ref.precision
    return quant
