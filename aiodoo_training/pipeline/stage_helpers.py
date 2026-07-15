"""Thin helpers for pipeline stage orchestration (no architecture ownership)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aiodoo_training.domain.config import ExperimentConfig
from aiodoo_training.pipeline.pipeline import PipelineContext


def raw_config(context: PipelineContext) -> dict[str, Any]:
    raw = context.get("raw_config")
    return dict(raw) if isinstance(raw, dict) else {}


def resource_planner_key(context: PipelineContext) -> str:
    """Resolve ResourcePlanner registry key from raw execution config."""
    execution = raw_config(context).get("execution")
    if isinstance(execution, dict):
        key = execution.get("resource_planner")
        if isinstance(key, str) and key.strip():
            return key.strip()
    return "static"


def model_backend_key(context: PipelineContext, config: ExperimentConfig) -> str:
    """Resolve ModelBackend registry key; prefer raw model.backend, else stub."""
    model = raw_config(context).get("model")
    if isinstance(model, dict):
        key = model.get("backend")
        if isinstance(key, str) and key.strip():
            return key.strip()
    meta_model = config.metadata.get("model") if config.metadata else None
    if isinstance(meta_model, dict):
        key = meta_model.get("backend")
        if isinstance(key, str) and key.strip():
            return key.strip()
    return "stub"


def tokenizer_registry_key(model_backend: str) -> str:
    """Map model backend key to TokenizerPort registry key."""
    if model_backend in {"hf_causal", "huggingface"}:
        return "huggingface"
    return "stub"


def trainer_backend_key(context: PipelineContext, config: ExperimentConfig) -> str:
    fragments = context.get("phase3_fragments") or {}
    key = fragments.get("trainer_backend_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    training = raw_config(context).get("training")
    if isinstance(training, dict):
        backend = training.get("backend")
        if isinstance(backend, str) and backend.strip():
            return backend.strip()
    meta_training = config.metadata.get("training") if config.metadata else None
    if isinstance(meta_training, dict):
        backend = meta_training.get("backend")
        if isinstance(backend, str) and backend.strip():
            return backend.strip()
    return "stub"


def missing_dataset_paths(config: ExperimentConfig) -> tuple[Path, ...]:
    return tuple(Path(ref.path) for ref in config.datasets.datasets if not Path(ref.path).exists())
