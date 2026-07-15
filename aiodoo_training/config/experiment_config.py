"""Map composed/resolved YAML into immutable :class:`ExperimentConfig`."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Any

from aiodoo_training.config.curriculum_config import parse_curriculum_config, to_curriculum_spec
from aiodoo_training.config.distributed_config import (
    parse_phase7_distributed_config,
    to_distributed_spec,
)
from aiodoo_training.config.evaluation_config import parse_evaluation_config
from aiodoo_training.config.export_config import parse_export_config
from aiodoo_training.config.model_config import (
    parse_adaptation_config,
    parse_execution_config,
    parse_model_config,
    to_adaptation_spec,
    to_execution_spec,
    to_model_ref,
)
from aiodoo_training.config.packing_config import parse_packing_config
from aiodoo_training.domain.config import (
    CheckpointingSpec,
    DatasetMixSpec,
    DeterminismSpec,
    EvaluationSpec,
    ExperimentConfig,
    ExportSpec,
    OptimizationSpec,
    PackingSpec,
    PrecisionSpec,
    TrackingSpec,
)
from aiodoo_training.domain.enums import (
    DatasetType,
    PackingMode,
    Precision,
    TrackerType,
    TrainingBackend,
)
from aiodoo_training.domain.identifiers import ExperimentId
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.exceptions import ConfigError

_OPT_KEYS = (
    "learning_rate",
    "weight_decay",
    "warmup_ratio",
    "num_epochs",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "max_steps",
)


def _as_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    raise ConfigError(f"{label} must be a mapping when provided.")


def _precision_section(value: Any) -> dict[str, Any]:
    """Accept mapping or scalar precision name (from flat model includes)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        return {"precision": value}
    raise ConfigError("precision must be a mapping or precision name when provided.")


def _merge_training_hyperparams(resolved: dict[str, Any]) -> dict[str, Any]:
    """
    Merge nested ``training:`` with flat include keys (e.g. ``default.yaml``).

    Nested ``training`` wins on conflicts.
    """
    training = _as_mapping(resolved.get("training"), label="training")
    merged = dict(training)
    for key in _OPT_KEYS:
        if key not in merged and key in resolved:
            merged[key] = resolved[key]
    return merged


def parse_dataset_mix(raw: dict[str, Any] | list[Any] | None, *, seed: int = 42) -> DatasetMixSpec:
    """Parse dataset mix from list or ``{datasets, shuffle, seed}`` mapping."""
    if raw is None:
        return DatasetMixSpec(seed=seed)
    if isinstance(raw, list):
        entries = raw
        shuffle = True
        mix_seed = seed
    elif isinstance(raw, dict):
        entries = raw.get("datasets") or raw.get("refs") or []
        if not isinstance(entries, list):
            raise ConfigError("datasets.datasets must be a list when provided.")
        shuffle = bool(raw.get("shuffle", True))
        mix_seed = int(raw.get("seed", seed))
    else:
        raise ConfigError("datasets must be a list or mapping.")

    refs: list[DatasetRef] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(f"datasets[{index}] must be a mapping.")
        path_raw = entry.get("path")
        if not isinstance(path_raw, str) or not path_raw.strip():
            raise ConfigError(f"datasets[{index}].path is required.")
        type_raw = entry.get("dataset_type") or entry.get("type") or "mixed"
        try:
            dataset_type = DatasetType(str(type_raw))
        except ValueError as exc:
            raise ConfigError(f"datasets[{index}].dataset_type invalid: {type_raw!r}") from exc
        protocol = str(entry.get("protocol_version") or "1.0")
        refs.append(
            DatasetRef(
                path=Path(path_raw),
                dataset_type=dataset_type,
                protocol_version=protocol,
                checksum=entry.get("checksum") if isinstance(entry.get("checksum"), str) else None,
                weight=float(entry.get("weight", 1.0)),
                name=entry.get("name") if isinstance(entry.get("name"), str) else None,
            )
        )
    return DatasetMixSpec(datasets=tuple(refs), shuffle=shuffle, seed=mix_seed)


def to_optimization_spec(training_raw: dict[str, Any]) -> OptimizationSpec:
    """Map training hyperparameter mapping to :class:`OptimizationSpec`."""
    try:
        return OptimizationSpec(
            learning_rate=float(training_raw.get("learning_rate", 2e-4)),
            weight_decay=float(training_raw.get("weight_decay", 0.0)),
            warmup_ratio=float(training_raw.get("warmup_ratio", 0.03)),
            num_epochs=float(training_raw.get("num_epochs", 1.0)),
            per_device_batch_size=int(training_raw.get("per_device_batch_size", 1)),
            gradient_accumulation_steps=int(training_raw.get("gradient_accumulation_steps", 1)),
            max_steps=(
                int(training_raw["max_steps"])
                if training_raw.get("max_steps") is not None
                else None
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid optimization hyperparameters: {exc}") from exc


def to_checkpointing_spec(raw: dict[str, Any] | None) -> CheckpointingSpec:
    data = _as_mapping(raw, label="checkpointing")
    output = data.get("output_dir", "artifacts/checkpoints")
    resume = data.get("resume_from")
    return CheckpointingSpec(
        output_dir=Path(output) if output is not None else Path("artifacts/checkpoints"),
        save_steps=int(data.get("save_steps", 500)),
        save_total_limit=int(data.get("save_total_limit", 3)),
        resume_from=Path(resume) if resume else None,
    )


def to_evaluation_spec(raw: dict[str, Any] | None, *, seed: int) -> EvaluationSpec:
    fragment = parse_evaluation_config(raw if isinstance(raw, dict) else {})
    dataset_raw = raw.get("datasets") if isinstance(raw, dict) else None
    refs = parse_dataset_mix(dataset_raw, seed=seed).datasets if dataset_raw else ()
    return EvaluationSpec(
        enabled=fragment.enabled,
        dataset_refs=refs,
        eval_steps=None,
    )


def to_export_spec(raw: dict[str, Any] | None) -> ExportSpec:
    fragment = parse_export_config(raw if isinstance(raw, dict) else {})
    output = fragment.output_dir or Path("artifacts/export")
    return ExportSpec(
        output_dir=Path(output),
        export_types=tuple(fragment.export_types),
    )


def to_tracking_spec(raw: dict[str, Any] | None) -> TrackingSpec:
    data = _as_mapping(raw, label="tracking")
    tracker_raw = data.get("tracker_type") or data.get("backend") or "null"
    try:
        tracker = TrackerType(str(tracker_raw))
    except ValueError as exc:
        raise ConfigError(f"Invalid tracking.tracker_type: {tracker_raw!r}") from exc
    uri = data.get("tracking_uri") or data.get("root_dir")
    return TrackingSpec(
        tracker_type=tracker,
        experiment_name=data.get("experiment_name")
        if isinstance(data.get("experiment_name"), str)
        else None,
        tracking_uri=str(uri) if uri is not None else None,
    )


def to_determinism_spec(raw: dict[str, Any] | None, *, seed: int) -> DeterminismSpec:
    data = _as_mapping(raw, label="determinism")
    return DeterminismSpec(
        seed=int(data.get("seed", seed)),
        cudnn_deterministic=bool(data.get("cudnn_deterministic", True)),
        cudnn_benchmark=bool(data.get("cudnn_benchmark", False)),
    )


def to_packing_spec(raw: dict[str, Any] | None) -> PackingSpec:
    fragment = parse_packing_config(raw if isinstance(raw, dict) else {})
    return PackingSpec(
        mode=PackingMode(fragment.mode),
        max_sequence_length=fragment.max_sequence_length,
    )


def _training_backend(training_raw: dict[str, Any]) -> TrainingBackend:
    backend = str(training_raw.get("backend") or "stub")
    if backend in {"hf_trainer", "huggingface"}:
        return TrainingBackend.HF_TRAINER
    if backend in {"custom_loop", "custom"}:
        return TrainingBackend.CUSTOM_LOOP
    # stub and unknown backends: keep domain enum at HF_TRAINER default used by
    # make_stub_experiment_config; runtime key lives in metadata["training"]["backend"].
    return TrainingBackend.HF_TRAINER


def to_experiment_config(
    resolved: dict[str, Any],
    *,
    experiment_id: ExperimentId,
    composed: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """
    Build :class:`ExperimentConfig` from a path-resolved (or composed) YAML mapping.

    Raises:
        ConfigError: when required sections are missing or invalid.
    """
    if not isinstance(resolved, dict):
        raise ConfigError("Experiment config root must be a mapping.")

    name = resolved.get("name")
    schema_version = resolved.get("schema_version")
    if not isinstance(name, str) or not name.strip():
        raise ConfigError("Experiment config requires non-empty 'name'.")
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise ConfigError("Experiment config requires non-empty 'schema_version'.")

    seed = int(resolved.get("seed", 42))
    training_raw = _merge_training_hyperparams(resolved)

    model_raw = _as_mapping(resolved.get("model"), label="model")
    # Flat include fragments (e.g. models/*.yaml) may place model fields at root.
    for key in (
        "identifier",
        "family",
        "revision",
        "local_path",
        "precision",
        "backend",
        "tokenizer_binding",
    ):
        if key not in model_raw and key in resolved:
            model_raw[key] = resolved[key]
    if not model_raw:
        raise ConfigError("Experiment config requires a 'model' section.")
    model_cfg = parse_model_config(model_raw)
    model_ref = to_model_ref(model_cfg)

    adaptation_raw = _as_mapping(resolved.get("adaptation"), label="adaptation")
    for key in ("adapter_type", "rank", "alpha", "dropout", "target_modules", "strategy"):
        if key not in adaptation_raw and key in resolved:
            adaptation_raw[key] = resolved[key]
    adaptation = to_adaptation_spec(
        parse_adaptation_config(adaptation_raw or {"adapter_type": "lora"})
    )

    execution_raw = _as_mapping(resolved.get("execution"), label="execution")
    execution = to_execution_spec(parse_execution_config(execution_raw))

    datasets_raw = resolved.get("datasets")
    if isinstance(datasets_raw, (dict, list)) or datasets_raw is None:
        datasets = parse_dataset_mix(datasets_raw, seed=seed)
    else:
        raise ConfigError("datasets must be a list or mapping.")

    # Top-level shuffle/seed from dataset include fragments.
    if isinstance(datasets_raw, list):
        shuffle = bool(resolved.get("shuffle", datasets.shuffle))
        mix_seed = int(resolved.get("seed", datasets.seed)) if "seed" in resolved else datasets.seed
        datasets = DatasetMixSpec(datasets=datasets.datasets, shuffle=shuffle, seed=mix_seed)

    packing = to_packing_spec(_as_mapping(resolved.get("packing"), label="packing") or None)
    curriculum = to_curriculum_spec(
        parse_curriculum_config(_as_mapping(resolved.get("curriculum"), label="curriculum") or None)
    )

    precision_raw = _precision_section(resolved.get("precision"))
    precision_value = precision_raw.get("precision") or precision_raw.get("compute")
    if precision_value is None:
        precision_value = model_cfg.precision.value
    try:
        precision = PrecisionSpec(
            precision=Precision(str(precision_value)),
            gradient_checkpointing=bool(precision_raw.get("gradient_checkpointing", False)),
        )
    except ValueError as exc:
        raise ConfigError(f"Invalid precision: {precision_value!r}") from exc

    distributed_raw = _as_mapping(resolved.get("distributed"), label="distributed")
    distributed = to_distributed_spec(parse_phase7_distributed_config(distributed_raw or None))

    # Metadata preserves portable composed YAML + backend keys for stage fragments.
    portable = composed if isinstance(composed, dict) else resolved
    meta: dict[str, Any] = dict(portable)
    # TrainingFragment allows only backend/max_steps/logging_steps (not hyperparams).
    training_src = _as_mapping(meta.get("training"), label="training")
    training_meta: dict[str, Any] = {
        key: training_src[key]
        for key in ("backend", "max_steps", "logging_steps")
        if key in training_src
    }
    backend = str(training_raw.get("backend") or model_cfg.backend or "stub")
    training_meta.setdefault("backend", backend)
    if training_raw.get("max_steps") is not None:
        training_meta.setdefault("max_steps", training_raw["max_steps"])
    if training_raw.get("logging_steps") is not None:
        training_meta.setdefault("logging_steps", training_raw["logging_steps"])
    meta["training"] = training_meta
    meta["model"] = {
        **(_as_mapping(meta.get("model"), label="model")),
        "backend": model_cfg.backend,
    }

    return ExperimentConfig(
        name=name,
        schema_version=schema_version,
        seed=seed,
        model=model_ref,
        datasets=datasets,
        adaptation=adaptation,
        optimization=to_optimization_spec(training_raw),
        precision=precision,
        packing=packing,
        curriculum=curriculum,
        checkpointing=to_checkpointing_spec(
            _as_mapping(resolved.get("checkpointing"), label="checkpointing") or None
        ),
        evaluation=to_evaluation_spec(
            _as_mapping(resolved.get("evaluation"), label="evaluation") or None,
            seed=seed,
        ),
        export=to_export_spec(_as_mapping(resolved.get("export"), label="export") or None),
        tracking=to_tracking_spec(_as_mapping(resolved.get("tracking"), label="tracking") or None),
        determinism=to_determinism_spec(
            _as_mapping(resolved.get("determinism"), label="determinism") or None,
            seed=seed,
        ),
        execution=execution,
        distributed=distributed,
        training_backend=_training_backend(training_raw),
        experiment_id=experiment_id,
        metadata=MappingProxyType(meta),
    )


__all__ = [
    "parse_dataset_mix",
    "to_checkpointing_spec",
    "to_determinism_spec",
    "to_evaluation_spec",
    "to_experiment_config",
    "to_export_spec",
    "to_optimization_spec",
    "to_packing_spec",
    "to_tracking_spec",
]
