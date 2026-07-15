"""Phase 2 hardening: ModelSession decision, adapter profiles, metadata, quantization."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from aiodoo_training.bootstrap import bootstrap_phase2
from aiodoo_training.domain.adapter_info import AdapterProfile
from aiodoo_training.domain.enums import AdapterType, ModelFamily, Precision
from aiodoo_training.domain.model_info import ModelCapabilities, ModelMetadata
from aiodoo_training.domain.quantization import QuantizationPolicy, QuantizationSpec
from aiodoo_training.factories import (
    AdaptationStrategyFactory,
    ModelBackendFactory,
    ResourcePlannerFactory,
)
from aiodoo_training.models import ModelLoader
from aiodoo_training.registries import adaptation_registry, adapter_registry


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase2(overwrite=True)


def test_quantization_policy_is_canonical_with_spec_alias() -> None:
    policy = QuantizationPolicy(compute=Precision.FP16, load_in_4bit=True)
    assert QuantizationSpec is QuantizationPolicy
    assert policy.to_dict()["compute"] == "fp16"
    assert QuantizationPolicy.from_dict(policy.to_dict()) == policy
    with pytest.raises(ValueError):
        QuantizationPolicy(load_in_4bit=True, load_in_8bit=True)


def test_model_metadata_immutable_serializable_roundtrip() -> None:
    meta = ModelMetadata(
        identifier="fixture/model",
        family=ModelFamily.QWEN,
        precision=Precision.BF16,
        quantization=QuantizationPolicy(compute=Precision.BF16),
        tokenizer_binding="qwen",
        capabilities=ModelCapabilities(num_parameters=1234, extra={"a": "1"}),
        backend_key="stub",
        extra={"source": "test"},
    )
    with pytest.raises(FrozenInstanceError):
        meta.identifier = "x"  # type: ignore[misc]
    payload = meta.to_dict()
    # Must be JSON-serializable without framework types
    encoded = json.dumps(payload)
    restored = ModelMetadata.from_dict(json.loads(encoded))
    assert restored.identifier == meta.identifier
    assert restored.family == ModelFamily.QWEN
    assert restored.capabilities.num_parameters == 1234
    assert restored.extra["source"] == "test"


def test_adapter_registry_independent_of_adaptation_strategy() -> None:
    assert adapter_registry.exists("lora-r8")
    assert adaptation_registry.exists("lora")
    profile = adapter_registry.get("lora-r8")
    assert isinstance(profile, AdapterProfile)
    assert profile.adapter_type == AdapterType.LORA
    assert profile.strategy_key == "lora"
    spec = profile.to_adaptation_spec()
    assert spec.rank == 8
    # Strategy resolved independently from profile metadata
    strategy = AdaptationStrategyFactory().create(profile.strategy_key)
    assert strategy.__class__.__name__ == "LoraAdaptationStrategy"


def test_loaded_model_context_sufficient_without_model_session() -> None:
    """
    ModelSession is intentionally not introduced.

    LoadedModelContext already binds handle + metadata + fingerprint + execution.
    DatasetSession-style cursors do not apply to loaded models; Phase 3 resume
    belongs to CheckpointHandle / trainer state, not a parallel ModelSession.
    """
    from aiodoo_training.builders import ModelBuilder

    ref = ModelBuilder().with_identifier("s").with_family("qwen").build()
    planner = ResourcePlannerFactory().create()
    loaded = ModelLoader(ModelBackendFactory().create("stub"), planner).load(ref)
    assert loaded.metadata.backend_key == "stub"
    assert loaded.execution.selected_device.value == "cpu"
    assert loaded.fingerprint.digest
    # No ModelSession type is part of the public phase-2 surface
    import aiodoo_training.domain as domain

    assert not hasattr(domain, "ModelSession")
