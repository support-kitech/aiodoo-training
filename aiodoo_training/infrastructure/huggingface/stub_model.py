"""CPU stub ModelBackend — no Torch / Transformers downloads."""

from __future__ import annotations

from types import MappingProxyType

from aiodoo_training.determinism.model_fingerprints import fingerprint_model
from aiodoo_training.domain.enums import DeviceKind
from aiodoo_training.domain.handles import BaseModelHandle
from aiodoo_training.domain.model_info import ModelCapabilities, ModelMetadata
from aiodoo_training.domain.quantization import QuantizationPolicy
from aiodoo_training.domain.refs import ModelRef
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.exceptions import DomainError
from aiodoo_training.infrastructure.model_handles import OpaqueBaseModel, as_base_handle
from aiodoo_training.ports.model import ModelBackend


class StubModelBackend(ModelBackend):
    """
    Deterministic in-process ModelBackend for CI and architecture tests.

    Does not import Torch. Simulates a tiny parameter set for countable adapters.
    """

    BACKEND_KEY = "stub"

    def load(
        self,
        model_ref: ModelRef,
        execution: ExecutionEnvironment,
    ) -> BaseModelHandle:
        if execution.selected_device != DeviceKind.CPU:
            raise DomainError(
                f"StubModelBackend only supports CPU execution; "
                f"got {execution.selected_device.value}."
            )

        quantization = QuantizationPolicy.from_precision_policy(execution.precision_policy)
        num_parameters = 1_000 + (len(model_ref.identifier) * 17)
        metadata = ModelMetadata(
            identifier=model_ref.identifier,
            family=model_ref.family,
            revision=model_ref.revision,
            precision=model_ref.precision,
            quantization=quantization,
            tokenizer_binding=model_ref.family.value,
            capabilities=ModelCapabilities(
                num_parameters=num_parameters,
                vocab_size=32_000,
                hidden_size=64,
                max_position_embeddings=2048,
                extra=MappingProxyType({"stub": "true"}),
            ),
            backend_key=self.BACKEND_KEY,
        )
        fingerprint = fingerprint_model(
            model_ref,
            quantization=quantization,
            execution=execution,
            tokenizer_binding=metadata.tokenizer_binding,
        )
        carrier = OpaqueBaseModel(
            framework_model={
                "kind": "stub",
                "identifier": model_ref.identifier,
                "num_parameters": num_parameters,
                "weights": tuple(range(16)),
            },
            aiodoo_metadata=metadata,
            aiodoo_fingerprint=fingerprint,
            backend_key=self.BACKEND_KEY,
        )
        return as_base_handle(carrier)
