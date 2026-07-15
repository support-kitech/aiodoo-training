"""Phase 4 evaluation / export profile and metric catalog registration."""

from __future__ import annotations

from aiodoo_training.domain.evaluation_policies import EvaluationProfile, ExportProfile
from aiodoo_training.domain.metric_definition import MetricAggregation, MetricDefinition
from aiodoo_training.registries.defaults import (
    evaluation_profile_registry,
    export_profile_registry,
    metric_registry,
)

__all__ = [
    "EvaluationProfile",
    "ExportProfile",
    "register_default_evaluation_profiles",
    "register_default_export_profiles",
    "register_default_metrics",
    "register_phase4_catalogs",
]


def register_default_metrics(*, overwrite: bool = False) -> None:
    defaults = (
        MetricDefinition(
            name="loss",
            aggregation=MetricAggregation.MEAN,
            higher_is_better=False,
            unit="nats",
        ),
        MetricDefinition(
            name="perplexity",
            aggregation=MetricAggregation.MEAN,
            higher_is_better=False,
            unit="",
        ),
        MetricDefinition(
            name="token_accuracy",
            aggregation=MetricAggregation.MEAN,
            higher_is_better=True,
            unit="ratio",
        ),
    )
    for definition in defaults:
        if not metric_registry.exists(definition.name) or overwrite:
            metric_registry.register(definition.name, definition, overwrite=overwrite)


def register_default_evaluation_profiles(*, overwrite: bool = False) -> None:
    profile = EvaluationProfile(key="default")
    if not evaluation_profile_registry.exists("default") or overwrite:
        evaluation_profile_registry.register("default", profile, overwrite=overwrite)


def register_default_export_profiles(*, overwrite: bool = False) -> None:
    profile = ExportProfile(key="peft_default")
    if not export_profile_registry.exists("peft_default") or overwrite:
        export_profile_registry.register("peft_default", profile, overwrite=overwrite)


def register_phase4_catalogs(*, overwrite: bool = False) -> None:
    register_default_metrics(overwrite=overwrite)
    register_default_evaluation_profiles(overwrite=overwrite)
    register_default_export_profiles(overwrite=overwrite)
