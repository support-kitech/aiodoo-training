"""Phase 4 configuration parsing and policy mapping tests."""

from __future__ import annotations

import pytest

from aiodoo_training.bootstrap import bootstrap_phase4
from aiodoo_training.config.evaluation_config import (
    parse_evaluation_config,
    to_acceptance_policy,
    to_evaluation_policy,
)
from aiodoo_training.config.export_config import (
    parse_export_config,
    to_compatibility_policy,
    to_validation_policy,
)
from aiodoo_training.domain.enums import ComparisonOp
from aiodoo_training.domain.export_manifest import ArtifactValidationPolicy
from aiodoo_training.exceptions import ConfigError


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase4(overwrite=True)


def test_parse_evaluation_config_defaults() -> None:
    frag = parse_evaluation_config({})
    assert frag.backend == "stub"
    assert frag.enabled is False
    assert frag.metrics == ["loss", "perplexity", "token_accuracy"]


def test_parse_export_config_defaults() -> None:
    frag = parse_export_config({})
    assert frag.backend == "stub"
    assert frag.enabled is False
    assert "model_card" in frag.export_types


def test_invalid_evaluation_backend_raises() -> None:
    with pytest.raises(ConfigError, match="Invalid evaluation config"):
        parse_evaluation_config({"backend": ""})


def test_invalid_export_backend_raises() -> None:
    with pytest.raises(ConfigError, match="Invalid export config"):
        parse_export_config({"backend": ""})


def test_to_evaluation_policy_maps_fields() -> None:
    frag = parse_evaluation_config(
        {"backend": "stub", "profile": "default", "metrics": ["loss"], "seed": 99}
    )
    policy = to_evaluation_policy(frag)
    assert policy.backend_key == "stub"
    assert policy.metrics == ("loss",)
    assert policy.seed == 99


def test_op_alias_ge_maps_to_comparison_op_ge() -> None:
    frag = parse_evaluation_config(
        {
            "acceptance": {
                "thresholds": [
                    {"metric_key": "token_accuracy", "op": "ge", "value": 0.5},
                ],
            },
        }
    )
    acceptance = to_acceptance_policy(frag)
    assert len(acceptance.thresholds) == 1
    threshold = acceptance.thresholds[0]
    assert threshold.op is ComparisonOp.GE
    assert threshold.op.value == ">="


def test_to_export_policies() -> None:
    frag = parse_export_config({"validation_policy": "warn"})
    assert to_validation_policy(frag) is ArtifactValidationPolicy.WARN
    compat = to_compatibility_policy(frag)
    assert "1" in compat.accepted_artifact_protocols
    assert "peft_adapter" in compat.required_roles
