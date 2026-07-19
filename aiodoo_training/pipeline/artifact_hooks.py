"""Pipeline hooks for canonical artifact publishing at finalize."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from aiodoo_training.artifacts.output_manager import ArtifactOutputManager
from aiodoo_training.artifacts.publish_contract import PublishError
from aiodoo_training.pipeline.pipeline import PipelineContext

logger = logging.getLogger("aiodoo_training.artifacts")

_PUBLISH_IO_ERRORS = (OSError, PublishError)


def maybe_publish_artifacts(context: PipelineContext) -> None:
    """
    Publish final adapter, merged model, and experiment summary to Drive.

    Publish failures are logged but never fail the pipeline (finalize has already
    succeeded). Experiment summary ``success`` reflects training and evaluation
    outcomes when present.
    """
    raw = context.get("raw_config")
    if not isinstance(raw, dict):
        return

    progress = context.get("training_progress")
    status = getattr(getattr(progress, "status", None), "value", "")
    if status == "failed":
        logger.info("Skipping artifact publish — training did not succeed")
        return

    manager = ArtifactOutputManager.from_resolved(raw)
    if manager is None:
        return

    layout = manager.layout
    run_id = context.run_id.value if context.run_id is not None else "unknown"

    # Ensure base model has artifact.json for validation handoff.
    model_path = _resolve_model_path(raw)
    if model_path is not None:
        try:
            manager.publish_base_model_artifact(model_path)
            logger.info("Wrote base model artifact.json at %s", model_path)
        except _PUBLISH_IO_ERRORS as exc:
            logger.warning("Base model artifact publish failed: %s", exc)

    # Publish adapter from latest checkpoint (including final train-end save).
    ckpt_dir = layout.adapter_checkpoints_dir
    latest = manager.find_latest_checkpoint(ckpt_dir)
    adapter_dest: Path | None = None
    if latest is not None:
        try:
            adapter_dest = manager.publish_adapter_from_checkpoint(latest)
            logger.info("Published adapter to %s", adapter_dest)
        except _PUBLISH_IO_ERRORS as exc:
            logger.warning("Adapter publish failed: %s", exc)
    else:
        logger.warning(
            "No checkpoint under %s — adapter was not published. "
            "Training must save a final checkpoint before finalize.",
            ckpt_dir,
        )

    # Publish merged weights from export bundle if present.
    bundle = context.get("artifact_bundle")
    merged_dest: Path | None = None
    if bundle is not None:
        bundle_root = getattr(bundle, "root", None)
        if bundle_root is not None:
            try:
                merged_dest = manager.publish_merged_from_bundle(Path(bundle_root))
                if merged_dest is not None:
                    logger.info("Published merged model to %s", merged_dest)
            except _PUBLISH_IO_ERRORS as exc:
                logger.warning("Merged model publish failed: %s", exc)

    # Config snapshot.
    config_path = context.get("config_path")
    if isinstance(config_path, Path):
        try:
            manager.snapshot_config(config_path)
        except OSError as exc:
            logger.warning("Config snapshot failed: %s", exc)

    duration = float(getattr(progress, "duration_seconds", 0.0) or 0.0)
    paths: dict[str, str | None] = {
        "adapter": str(adapter_dest) if adapter_dest else None,
        "merged": str(merged_dest) if merged_dest else None,
        "checkpoints": str(ckpt_dir),
        "exports": str(layout.export_dir),
        "metrics": str(layout.experiment_metrics_dir),
        "logs": str(layout.experiment_logs_dir),
    }
    bundle_root = getattr(bundle, "root", None) if bundle is not None else None
    if bundle_root is not None:
        paths["export_bundle"] = str(bundle_root)

    extra: dict[str, object] = {
        "capability_id": layout.training_id,
    }
    dataset_version = raw.get("dataset_version")
    if isinstance(dataset_version, str) and dataset_version.strip():
        extra["dataset_version"] = dataset_version

    experiment_success = _experiment_success(context)

    try:
        manager.write_experiment_summary(
            run_id=run_id,
            success=experiment_success,
            duration_seconds=duration,
            paths=paths,
            extra=extra,
        )
    except OSError as exc:
        logger.warning("Experiment summary write failed: %s", exc)


def _experiment_success(context: PipelineContext) -> bool:
    """Return whether the experiment summary should record success."""
    progress = context.get("training_progress")
    status = getattr(getattr(progress, "status", None), "value", "")
    if status == "failed":
        return False

    evaluation_report = context.get("evaluation_report")
    if evaluation_report is not None and not bool(getattr(evaluation_report, "passed", True)):
        return False

    quality_report = context.get("quality_report")
    if quality_report is not None and not bool(getattr(quality_report, "passed", True)):
        return False

    return True


def _resolve_model_path(raw: dict) -> Path | None:
    model = raw.get("model")
    if isinstance(model, dict):
        local_path = model.get("local_path")
        if isinstance(local_path, str) and local_path.strip():
            return Path(local_path)
    env_path = os.environ.get("AIODOO_COLAB_MODEL_PATH")
    if env_path and str(env_path).strip():
        return Path(env_path)
    return None


__all__ = ["maybe_publish_artifacts"]
