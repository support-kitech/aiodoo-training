"""Phase 2 model / adaptation / execution builders (immutable assembly)."""

from __future__ import annotations

from pathlib import Path

from aiodoo_training.adaptation.applier import AdaptedModelContext
from aiodoo_training.domain.config import AdaptationSpec, ExecutionSpec
from aiodoo_training.domain.enums import (
    AcceleratorKind,
    AdapterType,
    DeviceKind,
    ModelFamily,
    Precision,
)
from aiodoo_training.domain.handles import BaseModelHandle
from aiodoo_training.domain.model_info import ModelFingerprint, ModelMetadata
from aiodoo_training.domain.quantization import QuantizationPolicy
from aiodoo_training.domain.refs import ModelRef
from aiodoo_training.domain.resources import (
    DevicePolicy,
    ExecutionEnvironment,
    MemoryPolicy,
    PrecisionPolicy,
)
from aiodoo_training.exceptions import BuilderError
from aiodoo_training.models.loader import LoadedModelContext


class ModelBuilder:
    """Builds an immutable :class:`ModelRef` from typed fields."""

    def __init__(self) -> None:
        self._identifier: str | None = None
        self._family: ModelFamily = ModelFamily.UNKNOWN
        self._revision: str | None = None
        self._local_path: Path | None = None
        self._precision: Precision = Precision.BF16

    def with_identifier(self, identifier: str) -> ModelBuilder:
        self._identifier = identifier
        return self

    def with_family(self, family: ModelFamily | str) -> ModelBuilder:
        self._family = family if isinstance(family, ModelFamily) else ModelFamily(family)
        return self

    def with_revision(self, revision: str | None) -> ModelBuilder:
        self._revision = revision
        return self

    def with_local_path(self, path: Path | str | None) -> ModelBuilder:
        self._local_path = None if path is None else Path(path)
        return self

    def with_precision(self, precision: Precision | str) -> ModelBuilder:
        self._precision = precision if isinstance(precision, Precision) else Precision(precision)
        return self

    def build(self) -> ModelRef:
        if not self._identifier or not self._identifier.strip():
            raise BuilderError("ModelBuilder requires a non-empty identifier.")
        return ModelRef(
            identifier=self._identifier.strip(),
            family=self._family,
            revision=self._revision,
            local_path=self._local_path,
            precision=self._precision,
        )


class AdaptationBuilder:
    """Builds an immutable :class:`AdaptationSpec`."""

    def __init__(self) -> None:
        self._adapter_type: AdapterType = AdapterType.LORA
        self._rank: int | None = 8
        self._alpha: int | None = 16
        self._dropout: float | None = 0.05
        self._target_modules: tuple[str, ...] = ()

    def with_adapter_type(self, adapter_type: AdapterType | str) -> AdaptationBuilder:
        self._adapter_type = (
            adapter_type if isinstance(adapter_type, AdapterType) else AdapterType(adapter_type)
        )
        return self

    def with_rank(self, rank: int | None) -> AdaptationBuilder:
        self._rank = rank
        return self

    def with_alpha(self, alpha: int | None) -> AdaptationBuilder:
        self._alpha = alpha
        return self

    def with_dropout(self, dropout: float | None) -> AdaptationBuilder:
        self._dropout = dropout
        return self

    def with_target_modules(self, modules: tuple[str, ...] | list[str]) -> AdaptationBuilder:
        self._target_modules = tuple(modules)
        return self

    def build(self) -> AdaptationSpec:
        if self._adapter_type in {AdapterType.LORA, AdapterType.QLORA}:
            if self._rank is not None and self._rank < 1:
                raise BuilderError("AdaptationBuilder.rank must be >= 1 for LoRA/QLoRA.")
        return AdaptationSpec(
            adapter_type=self._adapter_type,
            rank=self._rank,
            alpha=self._alpha,
            dropout=self._dropout,
            target_modules=self._target_modules,
        )


class ExecutionContextBuilder:
    """Builds Device/Precision/Memory policies and an ExecutionSpec."""

    def __init__(self) -> None:
        self._preferred: DeviceKind = DeviceKind.AUTO
        self._allow_cpu_fallback: bool = True
        self._device_ids: tuple[int, ...] = ()
        self._compute: Precision = Precision.BF16
        self._load_in_4bit: bool = False
        self._load_in_8bit: bool = False
        self._activation_checkpointing: bool = False
        self._allow_cpu_offload: bool = False
        self._max_memory_gb: float | None = None
        self._accelerator: AcceleratorKind = AcceleratorKind.NONE

    def with_device(
        self,
        preferred: DeviceKind | str = DeviceKind.AUTO,
        *,
        allow_cpu_fallback: bool = True,
        device_ids: tuple[int, ...] = (),
    ) -> ExecutionContextBuilder:
        self._preferred = preferred if isinstance(preferred, DeviceKind) else DeviceKind(preferred)
        self._allow_cpu_fallback = allow_cpu_fallback
        self._device_ids = device_ids
        return self

    def with_precision(
        self,
        compute: Precision | str = Precision.BF16,
        *,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ) -> ExecutionContextBuilder:
        self._compute = compute if isinstance(compute, Precision) else Precision(compute)
        self._load_in_4bit = load_in_4bit
        self._load_in_8bit = load_in_8bit
        return self

    def with_memory(
        self,
        *,
        activation_checkpointing: bool = False,
        allow_cpu_offload: bool = False,
        max_memory_gb: float | None = None,
    ) -> ExecutionContextBuilder:
        self._activation_checkpointing = activation_checkpointing
        self._allow_cpu_offload = allow_cpu_offload
        self._max_memory_gb = max_memory_gb
        return self

    def with_accelerator(self, accelerator: AcceleratorKind | str) -> ExecutionContextBuilder:
        self._accelerator = (
            accelerator
            if isinstance(accelerator, AcceleratorKind)
            else AcceleratorKind(accelerator)
        )
        return self

    def build_spec(self) -> ExecutionSpec:
        if self._load_in_4bit and self._load_in_8bit:
            raise BuilderError("Cannot enable both load_in_4bit and load_in_8bit.")
        return ExecutionSpec(
            device=DevicePolicy(
                preferred=self._preferred,
                allow_cpu_fallback=self._allow_cpu_fallback,
                device_ids=self._device_ids,
            ),
            precision=PrecisionPolicy(
                compute=self._compute,
                load_in_4bit=self._load_in_4bit,
                load_in_8bit=self._load_in_8bit,
            ),
            memory=MemoryPolicy(
                max_memory_gb=self._max_memory_gb,
                activation_checkpointing=self._activation_checkpointing,
                allow_cpu_offload=self._allow_cpu_offload,
            ),
            accelerator=self._accelerator,
        )

    def build_quantization(self) -> QuantizationPolicy:
        return QuantizationPolicy(
            compute=self._compute,
            load_in_4bit=self._load_in_4bit,
            load_in_8bit=self._load_in_8bit,
        )


class ModelContextBuilder:
    """Assembles a LoadedModelContext from already-resolved pieces (no I/O)."""

    def __init__(self) -> None:
        self._handle: BaseModelHandle | None = None
        self._metadata: ModelMetadata | None = None
        self._fingerprint: ModelFingerprint | None = None
        self._execution: ExecutionEnvironment | None = None

    def with_handle(self, handle: BaseModelHandle) -> ModelContextBuilder:
        self._handle = handle
        return self

    def with_metadata(self, metadata: ModelMetadata) -> ModelContextBuilder:
        self._metadata = metadata
        return self

    def with_fingerprint(self, fingerprint: ModelFingerprint) -> ModelContextBuilder:
        self._fingerprint = fingerprint
        return self

    def with_execution(self, execution: ExecutionEnvironment) -> ModelContextBuilder:
        self._execution = execution
        return self

    def from_loaded(self, loaded: LoadedModelContext) -> ModelContextBuilder:
        self._handle = loaded.handle
        self._metadata = loaded.metadata
        self._fingerprint = loaded.fingerprint
        self._execution = loaded.execution
        return self

    def build(self) -> LoadedModelContext:
        if self._handle is None or self._metadata is None or self._fingerprint is None:
            raise BuilderError("ModelContextBuilder requires handle, metadata, and fingerprint.")
        if self._execution is None:
            raise BuilderError("ModelContextBuilder requires execution environment.")
        return LoadedModelContext(
            handle=self._handle,
            metadata=self._metadata,
            fingerprint=self._fingerprint,
            execution=self._execution,
        )


class AdaptedModelContextBuilder:
    """Assembles AdaptedModelContext without I/O (test / wiring helper)."""

    def __init__(self) -> None:
        self._ctx: AdaptedModelContext | None = None

    def from_adapted(self, adapted: AdaptedModelContext) -> AdaptedModelContextBuilder:
        self._ctx = adapted
        return self

    def build(self) -> AdaptedModelContext:
        if self._ctx is None:
            raise BuilderError("AdaptedModelContextBuilder requires an AdaptedModelContext.")
        return self._ctx
