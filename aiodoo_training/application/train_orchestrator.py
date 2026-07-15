"""Application training orchestrator — public config → Pipeline → structured result."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.config import ConfigSystem
from aiodoo_training.config.experiment_config import to_experiment_config
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.pipeline import PipelineResult
from aiodoo_training.exceptions import AiodooTrainingError, ConfigError
from aiodoo_training.pipeline import Pipeline, PipelineContext, build_phase4_pipeline

logger = logging.getLogger("aiodoo_training.train")


@dataclass(frozen=True, slots=True)
class ExecutionResult:
    """Structured result of a public train.py execution."""

    success: bool
    adapter_path: Path | None
    checkpoint_path: Path | None
    merged_model_path: Path | None
    export_path: Path | None
    metrics_path: Path | None
    logs_path: Path | None
    duration_seconds: float
    message: str
    experiment_id: str | None = None
    run_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload


def _env_path(name: str) -> Path | None:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return None
    return Path(raw)


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def apply_colab_path_overrides(resolved: dict[str, Any]) -> dict[str, Any]:
    """
    Apply optional aiodoo-colab environment hints onto a resolved config mapping.

    Only overrides paths that are present in the environment. Does not invent
    training semantics.
    """
    data = dict(resolved)
    model_path = _env_path("AIODOO_COLAB_MODEL_PATH")
    if model_path is not None:
        model = _as_dict(data.get("model"))
        model["local_path"] = str(model_path)
        data["model"] = model

    dataset_root = _env_path("AIODOO_COLAB_DATASET_PATH")
    if dataset_root is not None:
        data["datasets"] = _rewrite_dataset_entries(data.get("datasets"), dataset_root)

    checkpoint_out = _env_path("AIODOO_COLAB_CHECKPOINTS_OUTPUT") or _env_path(
        "AIODOO_COLAB_ADAPTER_OUTPUT"
    )
    if checkpoint_out is not None:
        ckpt = _as_dict(data.get("checkpointing"))
        ckpt["output_dir"] = str(checkpoint_out)
        data["checkpointing"] = ckpt

    export_out = _env_path("AIODOO_COLAB_EXPORT_OUTPUT")
    if export_out is not None:
        export = _as_dict(data.get("export"))
        export["output_dir"] = str(export_out)
        data["export"] = export

    metrics_out = _env_path("AIODOO_COLAB_METRICS_OUTPUT")
    if metrics_out is not None:
        metrics = _as_dict(data.get("metrics"))
        metrics["history_path"] = str(metrics_out / "history.jsonl")
        data["metrics"] = metrics

    logs_out = _env_path("AIODOO_COLAB_LOGS_OUTPUT")
    if logs_out is not None:
        tracking = _as_dict(data.get("tracking"))
        tracking["root_dir"] = str(logs_out / "tracking")
        data["tracking"] = tracking

    return data


def _rewrite_dataset_entries(datasets: Any, dataset_root: Path) -> Any:
    """
    Join Colab dataset version root with each entry's filename.

    EXP configs store workspace-relative filenames (e.g. ``coding_v1_0.jsonl``).
    ``AIODOO_COLAB_DATASET_PATH`` is the version root
    (``…/datasets/v1.0.0``), not a single JSONL file.
    """

    def _join_one(entry: Any) -> dict[str, Any]:
        item = dict(entry) if isinstance(entry, dict) else {"path": str(entry)}
        raw = item.get("path")
        if not isinstance(raw, str) or not raw.strip():
            item["path"] = str(dataset_root)
            return item
        path = Path(raw)
        # Already under the Colab dataset root — keep.
        try:
            if path.is_absolute() and dataset_root in path.parents:
                item["path"] = str(path)
                return item
        except OSError:
            pass
        # Prefer basename so absolutized config-dir paths still resolve to the file.
        name = path.name if path.name else raw
        item["path"] = str(dataset_root / name)
        return item

    if isinstance(datasets, list):
        return [_join_one(entry) for entry in datasets]
    if isinstance(datasets, dict):
        mix = dict(datasets)
        entries = list(mix.get("datasets") or [])
        if entries:
            mix["datasets"] = [_join_one(entry) for entry in entries]
        else:
            mix["datasets"] = [
                {
                    "path": str(dataset_root),
                    "dataset_type": "mixed",
                    "protocol_version": "1.0",
                }
            ]
        return mix
    return [
        {
            "path": str(dataset_root),
            "dataset_type": "mixed",
            "protocol_version": "1.0",
        }
    ]


def _artifact_paths(
    resolved: dict[str, Any],
    *,
    pipeline_result: PipelineResult | None,
) -> dict[str, Path | None]:
    ckpt_raw = _as_dict(resolved.get("checkpointing"))
    export_raw = _as_dict(resolved.get("export"))
    metrics_raw = _as_dict(resolved.get("metrics"))

    checkpoint_path = Path(str(ckpt_raw["output_dir"])) if ckpt_raw.get("output_dir") else None
    adapter_path = (
        _env_path("AIODOO_COLAB_ADAPTER_OUTPUT") or checkpoint_path or Path("artifacts/adapters")
    )
    export_path = (
        Path(str(export_raw["output_dir"]))
        if export_raw.get("output_dir")
        else _env_path("AIODOO_COLAB_EXPORT_OUTPUT")
    )
    metrics_path = (
        Path(str(metrics_raw["history_path"])).parent
        if metrics_raw.get("history_path")
        else _env_path("AIODOO_COLAB_METRICS_OUTPUT")
    )
    logs_path = _env_path("AIODOO_COLAB_LOGS_OUTPUT") or (
        checkpoint_path / "logs" if checkpoint_path is not None else Path("artifacts/logs")
    )
    merged_path = _env_path("AIODOO_COLAB_MERGED_OUTPUT")

    # Prefer export stage payload path when present.
    if pipeline_result is not None:
        for stage in pipeline_result.stage_results:
            if stage.stage.value == "export" and stage.payload:
                if "output_dir" in stage.payload:
                    export_path = Path(str(stage.payload["output_dir"]))
                if "bundle_dir" in stage.payload:
                    export_path = Path(str(stage.payload["bundle_dir"]))

    return {
        "adapter_path": adapter_path,
        "checkpoint_path": checkpoint_path,
        "merged_model_path": merged_path,
        "export_path": export_path,
        "metrics_path": metrics_path,
        "logs_path": logs_path,
    }


def _validate_workspace(config_path: Path, resolved: dict[str, Any]) -> None:
    """Fail early when required filesystem assumptions are violated."""
    if not config_path.is_file():
        raise ConfigError(f"Config file not found: {config_path}")
    ckpt = _as_dict(resolved.get("checkpointing"))
    output_dir = ckpt.get("output_dir")
    if output_dir:
        Path(str(output_dir)).mkdir(parents=True, exist_ok=True)
    export = _as_dict(resolved.get("export"))
    if export.get("output_dir"):
        Path(str(export["output_dir"])).mkdir(parents=True, exist_ok=True)


def run_train_from_config(config_path: Path, *, run_id: RunId | None = None) -> ExecutionResult:
    """
    Load config, validate, run the training pipeline, return a structured result.

    Converts uncaught failures into :class:`ExecutionResult` with ``success=False``.
    """
    started = time.perf_counter()
    resolved: dict[str, Any] = {}
    experiment_id: ExperimentId | None = None
    active_run_id = run_id or RunId(value=f"run-{uuid4().hex[:12]}")

    # Ensure diagnostics reach stderr when invoked outside logging-configured hosts.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )

    try:
        logger.info("Bootstrapping registries")
        bootstrap_phase7(overwrite=True)

        system = ConfigSystem()
        _raw_model, experiment_id, resolved = system.load_experiment(config_path)
        composed = system.composer.compose(config_path)
        resolved = apply_colab_path_overrides(resolved)

        logger.info("Validating workspace paths for %s", config_path)
        _validate_workspace(config_path, resolved)

        logger.info("Mapping resolved config → ExperimentConfig")
        config = to_experiment_config(
            resolved,
            experiment_id=experiment_id,
            composed=composed,
        )

        # Ensure checkpoint / export directories exist before pipeline I/O.
        config.checkpointing.output_dir.mkdir(parents=True, exist_ok=True)
        config.export.output_dir.mkdir(parents=True, exist_ok=True)

        hasher = system.hasher
        config_fingerprint = hasher.hash(composed)

        context = PipelineContext(
            experiment_id=experiment_id,
            run_id=active_run_id,
            config=config,
        ).with_values(
            raw_config=resolved,
            config_fingerprint=config_fingerprint,
            metrics_history_path=(
                Path(str(resolved["metrics"]["history_path"]))
                if isinstance(resolved.get("metrics"), dict)
                and resolved["metrics"].get("history_path")
                else config.checkpointing.output_dir / "metrics" / "history.jsonl"
            ),
        )

        logger.info(
            "Executing pipeline experiment_id=%s run_id=%s",
            experiment_id.value,
            active_run_id.value,
        )
        pipeline_result = Pipeline(build_phase4_pipeline()).run(context)
        duration = time.perf_counter() - started
        paths = _artifact_paths(resolved, pipeline_result=pipeline_result)
        success = pipeline_result.succeeded
        message = pipeline_result.message or (
            "Training completed successfully" if success else "Training pipeline failed"
        )
        error: str | None = None
        if not success:
            failed = next(
                (s for s in pipeline_result.stage_results if s.status.value == "failed"),
                None,
            )
            error = failed.error if failed is not None else message
            logger.error("Pipeline failed: %s", error)
        else:
            logger.info("Pipeline completed in %.2fs", duration)

        return ExecutionResult(
            success=success,
            duration_seconds=duration,
            message=message,
            experiment_id=experiment_id.value,
            run_id=active_run_id.value,
            error=error,
            **paths,
        )
    except AiodooTrainingError as exc:
        duration = time.perf_counter() - started
        logger.exception("Training failed with domain error")
        paths = _artifact_paths(resolved, pipeline_result=None)
        return ExecutionResult(
            success=False,
            duration_seconds=duration,
            message=str(exc),
            experiment_id=experiment_id.value if experiment_id is not None else None,
            run_id=active_run_id.value,
            error=f"{type(exc).__name__}: {exc}",
            **paths,
        )
    except Exception as exc:  # noqa: BLE001 — public boundary converts all failures
        duration = time.perf_counter() - started
        logger.exception("Training failed with unexpected error")
        paths = _artifact_paths(resolved, pipeline_result=None)
        return ExecutionResult(
            success=False,
            duration_seconds=duration,
            message="Uncaught exception during training execution.",
            experiment_id=experiment_id.value if experiment_id is not None else None,
            run_id=active_run_id.value,
            error=f"{type(exc).__name__}: {exc}",
            **paths,
        )


def train_exit_code(result: ExecutionResult) -> int:
    """Map structured execution result to a process exit code."""
    return 0 if result.success else 1


def emit_execution_result(result: ExecutionResult, *, as_json: bool = False) -> None:
    """Print structured execution result to stdout."""
    payload = result.to_dict()
    if as_json:
        print(json.dumps(payload, sort_keys=True, default=str))
        return
    for key, value in payload.items():
        print(f"{key}: {value}")


__all__ = [
    "ExecutionResult",
    "apply_colab_path_overrides",
    "emit_execution_result",
    "run_train_from_config",
    "train_exit_code",
]
