"""Bootstrap default registries for Phase 1–3."""

from __future__ import annotations

from aiodoo_training.adaptation.profiles import register_default_adapter_profiles
from aiodoo_training.curriculum import register_default_curriculum
from aiodoo_training.datasets.formatters import register_default_formatters
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.evaluation.profiles import register_phase4_catalogs
from aiodoo_training.infrastructure.callbacks import register_default_callbacks
from aiodoo_training.infrastructure.huggingface import (
    register_default_chat_templates,
    register_default_tokenizers,
)
from aiodoo_training.infrastructure.huggingface.model import register_default_model_backends
from aiodoo_training.infrastructure.peft import register_default_adaptation_strategies
from aiodoo_training.infrastructure.resources import register_default_resource_planners
from aiodoo_training.infrastructure.stub.checkpoint_store import register_default_checkpoint_stores
from aiodoo_training.infrastructure.stub.evaluator import register_default_evaluators
from aiodoo_training.infrastructure.stub.exporter import register_default_exporters
from aiodoo_training.infrastructure.stub.trainer import register_default_trainers
from aiodoo_training.infrastructure.torch.optimizer import register_default_optimizers
from aiodoo_training.infrastructure.torch.rng import register_default_rng
from aiodoo_training.infrastructure.torch.scheduler import register_default_schedulers
from aiodoo_training.infrastructure.tracking import register_default_trackers
from aiodoo_training.models.profiles import register_default_model_profiles
from aiodoo_training.packing import register_default_packing
from aiodoo_training.registries import dataset_source_registry
from aiodoo_training.sampling import register_default_sampling


def bootstrap_phase1(*, overwrite: bool = False) -> None:
    """Register Phase 1 formatters, templates, tokenizers, packing, planners, and sources."""
    register_default_formatters(overwrite=overwrite)
    register_default_chat_templates(overwrite=overwrite)
    register_default_tokenizers(overwrite=overwrite)
    register_default_packing(overwrite=overwrite)
    register_default_resource_planners(overwrite=overwrite)
    if not dataset_source_registry.exists("jsonl") or overwrite:
        dataset_source_registry.register("jsonl", JsonlDatasetSource, overwrite=overwrite)


def bootstrap_phase2(*, overwrite: bool = False) -> None:
    """
    Register Phase 2 model backends, adaptation strategies, profiles, and catalogs.

    Extends Phase 1 registrations; does not alter Phase 1 contracts.
    """
    bootstrap_phase1(overwrite=overwrite)
    register_default_model_backends(overwrite=overwrite)
    register_default_adaptation_strategies(overwrite=overwrite)
    register_default_adapter_profiles(overwrite=overwrite)
    register_default_model_profiles(overwrite=overwrite)


def bootstrap_phase3(*, overwrite: bool = False) -> None:
    """
    Register Phase 3 training infrastructure (stub + HF trainers, stores, RNG, etc.).

    Extends Phase 2; does not alter Phase 0–2 contracts.
    """
    bootstrap_phase2(overwrite=overwrite)
    register_default_trainers(overwrite=overwrite)
    register_default_checkpoint_stores(overwrite=overwrite)
    register_default_optimizers(overwrite=overwrite)
    register_default_schedulers(overwrite=overwrite)
    register_default_rng(overwrite=overwrite)
    register_default_callbacks(overwrite=overwrite)


def bootstrap_phase4(*, overwrite: bool = False) -> None:
    """
    Register Phase 4 evaluation and export infrastructure (stub + HF adapters).

    Extends Phase 3; does not alter Phase 0–3 contracts.
    """
    bootstrap_phase3(overwrite=overwrite)
    register_default_evaluators(overwrite=overwrite)
    register_default_exporters(overwrite=overwrite)
    register_phase4_catalogs(overwrite=overwrite)


def bootstrap_phase5(*, overwrite: bool = False) -> None:
    """
    Register Phase 5 packing, curriculum, and sampling backends.

    Extends Phase 4; does not alter Phase 0–4 contracts.
    """
    bootstrap_phase4(overwrite=overwrite)
    register_default_packing(overwrite=overwrite)
    register_default_curriculum(overwrite=overwrite)
    register_default_sampling(overwrite=overwrite)


def bootstrap_phase6(*, overwrite: bool = False) -> None:
    """
    Register Phase 6 tracking backends.

    Extends Phase 5; does not alter Phase 0–5 contracts.
    """
    bootstrap_phase5(overwrite=overwrite)
    register_default_trackers(overwrite=overwrite)


def bootstrap_phase7(*, overwrite: bool = False) -> None:
    """
    Register Phase 7 distributed backends / placement / samplers.

    Extends Phase 6; does not alter Phase 0–6 contracts.
    """
    from aiodoo_training.infrastructure.distributed import register_default_distributed_backends
    from aiodoo_training.infrastructure.distributed.placement import (
        register_default_placement_strategies,
    )
    from aiodoo_training.infrastructure.distributed.samplers import (
        register_default_distributed_samplers,
    )

    bootstrap_phase6(overwrite=overwrite)
    register_default_distributed_backends(overwrite=overwrite)
    register_default_placement_strategies(overwrite=overwrite)
    register_default_distributed_samplers(overwrite=overwrite)
