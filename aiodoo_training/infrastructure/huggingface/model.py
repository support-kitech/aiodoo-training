"""
HuggingFace Causal LM ModelBackend.

Torch / Transformers imports are confined to this module. CI uses StubModelBackend.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

from aiodoo_training.determinism.model_fingerprints import fingerprint_model
from aiodoo_training.domain.enums import DeviceKind, Precision
from aiodoo_training.domain.handles import BaseModelHandle
from aiodoo_training.domain.model_info import ModelCapabilities, ModelMetadata
from aiodoo_training.domain.quantization import QuantizationPolicy
from aiodoo_training.domain.refs import ModelRef
from aiodoo_training.domain.resources import ExecutionEnvironment
from aiodoo_training.exceptions import DomainError
from aiodoo_training.infrastructure.model_handles import OpaqueBaseModel, as_base_handle
from aiodoo_training.ports.model import ModelBackend


def _torch_dtype_for(precision: Precision) -> Any:
    """Map Precision to a torch dtype — imports torch only when called."""
    import torch

    mapping = {
        Precision.FP32: torch.float32,
        Precision.FP16: torch.float16,
        Precision.BF16: torch.bfloat16,
        Precision.INT8: torch.float16,  # compute dtype when int8 weights requested
        Precision.INT4: torch.float16,
    }
    return mapping.get(precision, torch.float32)


def _device_map_for(execution: ExecutionEnvironment) -> str | dict[str, Any]:
    """Translate ExecutionEnvironment into HF device_map without CUDA probes."""
    selected = execution.selected_device
    if selected == DeviceKind.CPU:
        return "cpu"
    if selected == DeviceKind.CUDA:
        ids = execution.device_policy.device_ids
        if ids:
            return {"": ids[0]}
        return "cuda"
    if selected == DeviceKind.MPS:
        return "mps"
    if selected == DeviceKind.XPU:
        return "xpu"
    return "cpu"


class HuggingFaceCausalLMBackend(ModelBackend):
    """
    Loads a HuggingFace causal LM using Transformers + Torch.

    Honors ExecutionEnvironment for device and PrecisionPolicy / QuantizationPolicy
    for dtype and optional 4/8-bit load flags. Does not call ``torch.cuda.is_available``
    directly for decisions — the planner already selected the device.
    """

    BACKEND_KEY = "hf_causal"

    def load(
        self,
        model_ref: ModelRef,
        execution: ExecutionEnvironment,
    ) -> BaseModelHandle:
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
        except ImportError as exc:  # pragma: no cover - exercised when deps missing
            raise DomainError(
                "transformers and torch are required for HuggingFaceCausalLMBackend. "
                "Install requirements/train.txt or use backend key 'stub'."
            ) from exc

        quantization = QuantizationPolicy.from_precision_policy(execution.precision_policy)
        source = (
            str(model_ref.local_path) if model_ref.local_path is not None else model_ref.identifier
        )
        if model_ref.local_path is not None and not Path(model_ref.local_path).exists():
            raise DomainError(f"Model local_path does not exist: {model_ref.local_path}")

        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
        }
        if model_ref.revision:
            load_kwargs["revision"] = model_ref.revision

        if quantization.load_in_4bit or quantization.load_in_8bit:
            load_kwargs.update(self._bnb_kwargs(quantization, execution))
        else:
            load_kwargs["torch_dtype"] = _torch_dtype_for(quantization.compute)
            load_kwargs["device_map"] = _device_map_for(execution)

        try:
            model = AutoModelForCausalLM.from_pretrained(source, **load_kwargs)
            config = AutoConfig.from_pretrained(
                source,
                **{k: v for k, v in load_kwargs.items() if k in {"revision", "trust_remote_code"}},
            )
        except Exception as exc:  # noqa: BLE001 — wrap library failures
            raise DomainError(f"Failed to load HuggingFace model '{source}': {exc}") from exc

        num_parameters = int(sum(int(p.numel()) for p in model.parameters()))
        metadata = ModelMetadata(
            identifier=model_ref.identifier,
            family=model_ref.family,
            revision=model_ref.revision,
            precision=model_ref.precision,
            quantization=quantization,
            tokenizer_binding=model_ref.family.value,
            capabilities=ModelCapabilities(
                num_parameters=num_parameters,
                vocab_size=getattr(config, "vocab_size", None),
                hidden_size=getattr(config, "hidden_size", None),
                max_position_embeddings=getattr(config, "max_position_embeddings", None),
                extra=MappingProxyType({"backend": self.BACKEND_KEY}),
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
            framework_model=model,
            aiodoo_metadata=metadata,
            aiodoo_fingerprint=fingerprint,
            backend_key=self.BACKEND_KEY,
        )
        return as_base_handle(carrier)

    @staticmethod
    def _bnb_kwargs(
        quantization: QuantizationPolicy,
        execution: ExecutionEnvironment,
    ) -> dict[str, Any]:
        """Build bitsandbytes load kwargs from QuantizationPolicy (infra only)."""
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise DomainError(
                "bitsandbytes / BitsAndBytesConfig required for 4-bit/8-bit loading."
            ) from exc

        compute = _torch_dtype_for(quantization.compute)
        bnb = BitsAndBytesConfig(
            load_in_4bit=quantization.load_in_4bit,
            load_in_8bit=quantization.load_in_8bit,
            bnb_4bit_compute_dtype=compute if quantization.load_in_4bit else None,
        )
        return {
            "quantization_config": bnb,
            "device_map": _device_map_for(execution),
        }


def register_default_model_backends(*, overwrite: bool = False) -> None:
    """Register stub and HF Causal LM backends."""
    from aiodoo_training.infrastructure.huggingface.stub_model import StubModelBackend
    from aiodoo_training.registries import model_backend_registry

    mappings: dict[str, type[ModelBackend]] = {
        "stub": StubModelBackend,
        "hf_causal": HuggingFaceCausalLMBackend,
        "huggingface": HuggingFaceCausalLMBackend,
    }
    for key, cls in mappings.items():
        if not model_backend_registry.exists(key) or overwrite:
            model_backend_registry.register(key, cls, overwrite=overwrite)
