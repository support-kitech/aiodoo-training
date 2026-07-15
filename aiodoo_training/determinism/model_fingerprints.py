"""Deterministic model / adaptation fingerprints (Phase 2)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from aiodoo_training.config.system import ConfigHasher
from aiodoo_training.determinism.fingerprints import ExperimentFingerprint, FingerprintService
from aiodoo_training.domain.adapter_info import AdapterFingerprint
from aiodoo_training.domain.config import AdaptationSpec
from aiodoo_training.domain.identifiers import ExperimentId
from aiodoo_training.domain.model_info import ModelFingerprint
from aiodoo_training.domain.quantization import QuantizationPolicy
from aiodoo_training.domain.refs import ModelRef
from aiodoo_training.domain.resources import ExecutionEnvironment


def _digest(parts: tuple[str, ...]) -> str:
    material = "\n".join(parts).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def fingerprint_quantization(policy: QuantizationPolicy) -> str:
    """Return a digest for a QuantizationPolicy."""
    return _digest(policy.canonical_parts())


def fingerprint_execution(execution: ExecutionEnvironment) -> str:
    """Return a portable digest for a resolved ExecutionEnvironment."""
    parts = (
        f"selected={execution.selected_device.value}",
        f"preferred={execution.device_policy.preferred.value}",
        f"fallback={execution.device_policy.allow_cpu_fallback}",
        f"device_ids={','.join(str(i) for i in execution.device_policy.device_ids)}",
        f"compute={execution.precision_policy.compute.value}",
        f"4bit={execution.precision_policy.load_in_4bit}",
        f"8bit={execution.precision_policy.load_in_8bit}",
        f"max_mem={execution.memory_policy.max_memory_gb}",
        f"act_ckpt={execution.memory_policy.activation_checkpointing}",
        f"cpu_offload={execution.memory_policy.allow_cpu_offload}",
        f"accelerator={execution.accelerator.value}",
    )
    return _digest(parts)


def fingerprint_model(
    model_ref: ModelRef,
    *,
    quantization: QuantizationPolicy,
    execution: ExecutionEnvironment,
    tokenizer_binding: str | None = None,
) -> ModelFingerprint:
    """Fingerprint base model identity + quantization + execution plan."""
    quant_digest = fingerprint_quantization(quantization)
    exec_digest = fingerprint_execution(execution)
    revision = model_ref.revision or ""
    path = str(model_ref.local_path) if model_ref.local_path is not None else ""
    parts = (
        f"id={model_ref.identifier}",
        f"family={model_ref.family.value}",
        f"revision={revision}",
        f"local={path}",
        f"precision={model_ref.precision.value}",
        f"tokenizer={tokenizer_binding or ''}",
        f"quant={quant_digest}",
        f"exec={exec_digest}",
    )
    return ModelFingerprint(
        digest=_digest(parts),
        identifier=model_ref.identifier,
        revision=model_ref.revision,
        family=model_ref.family.value,
        quantization_digest=quant_digest,
        execution_digest=exec_digest,
    )


def fingerprint_adapter(
    spec: AdaptationSpec,
    *,
    quantization: QuantizationPolicy,
) -> AdapterFingerprint:
    """Fingerprint adaptation configuration (strategy-agnostic)."""
    quant_digest = fingerprint_quantization(quantization)
    modules = tuple(sorted(spec.target_modules))
    hasher = ConfigHasher()
    extra = dict(spec.extra)
    extra_json = hasher.canonical_json(extra) if extra else ""
    parts = (
        f"type={spec.adapter_type.value}",
        f"rank={spec.rank}",
        f"alpha={spec.alpha}",
        f"dropout={spec.dropout}",
        f"modules={','.join(modules)}",
        f"extra={extra_json}",
        f"quant={quant_digest}",
    )
    return AdapterFingerprint(
        digest=_digest(parts),
        adapter_type=spec.adapter_type.value,
        rank=spec.rank,
        alpha=spec.alpha,
        target_modules=modules,
        quantization_digest=quant_digest,
    )


def combine_model_adaptation_digests(
    model: ModelFingerprint,
    adapter: AdapterFingerprint,
) -> str:
    """Combine model + adapter digests for experiment identity inputs."""
    return _digest((f"model={model.digest}", f"adapter={adapter.digest}"))


def experiment_fingerprint_with_model_adaptation(
    config_data: dict[str, Any],
    *,
    package_version: str,
    model: ModelFingerprint,
    adapter: AdapterFingerprint,
    dataset_paths: tuple[Path, ...] = (),
    include_environment: bool = False,
    package_extra: tuple[tuple[str, str], ...] = (),
) -> ExperimentFingerprint:
    """
    Extend :class:`FingerprintService` experiment identity with model/adapter digests.

    Does not modify the frozen FingerprintService API — composes on top.
    """
    base = FingerprintService().experiment_fingerprint(
        config_data,
        package_version=package_version,
        dataset_paths=dataset_paths,
        include_environment=include_environment,
        package_extra=package_extra,
    )
    model_adapter = combine_model_adaptation_digests(model, adapter)
    digest = _digest((f"base={base.digest}", f"model_adapter={model_adapter}"))
    return ExperimentFingerprint(
        config=base.config,
        datasets=base.datasets,
        versions=base.versions,
        packages=base.packages,
        environment=base.environment,
        digest=digest,
        experiment_id=ExperimentId(value=f"exp_{digest[:16]}"),
    )
