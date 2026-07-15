"""Checkpoint orchestration — manifests, sidecars, validation, atomic save."""

from __future__ import annotations

import json
import os
import shutil
import uuid
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from aiodoo_training.domain.checkpoint_manifest import (
    CheckpointManifest,
    compute_checkpoint_fingerprint,
)
from aiodoo_training.domain.enums import CheckpointType
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.session import DatasetSession
from aiodoo_training.domain.training import CheckpointHandle, MetricSnapshot, TrainingProgress
from aiodoo_training.domain.training_policies import TRAINING_PROTOCOL_VERSION, ResumePolicy
from aiodoo_training.domain.training_session import TrainingSession
from aiodoo_training.exceptions import CheckpointCorruption, IncompatibleResume, ResumeWarning
from aiodoo_training.ports.trainer import CheckpointStore, RngController

if TYPE_CHECKING:
    from aiodoo_training.training.resume import ResumeBundle

MANIFEST_FILENAME = "manifest.json"
RNG_FILENAME = "rng.json"
DATASET_SESSION_FILENAME = "dataset_session.json"
METRICS_FILENAME = "metrics.json"
INDEX_FILENAME = "checkpoints.json"
MANIFEST_SCHEMA_VERSION = "1"

DEFAULT_FULL_STATE_REQUIRED: tuple[str, ...] = (
    MANIFEST_FILENAME,
    RNG_FILENAME,
    DATASET_SESSION_FILENAME,
    METRICS_FILENAME,
)


def dataset_session_to_dict(session: DatasetSession) -> dict[str, Any]:
    """Serialize a :class:`DatasetSession` for checkpoint sidecars."""
    return {
        "session_id": session.session_id,
        "experiment_id": session.experiment_id.value if session.experiment_id else None,
        "run_id": session.run_id.value if session.run_id else None,
        "dataset_fingerprint": session.dataset_fingerprint,
        "mix_fingerprint": session.mix_fingerprint,
        "epoch": session.epoch,
        "example_index": session.example_index,
        "examples_seen": session.examples_seen,
        "examples_total": session.examples_total,
        "shuffle_seed": session.shuffle_seed,
        "worker_id": session.worker_id,
        "world_size": session.world_size,
        "global_rank": session.global_rank,
        "local_rank": session.local_rank,
        "shard_id": session.shard_id,
        "num_shards": session.num_shards,
        "resume_token": session.resume_token,
        "metadata": dict(session.metadata),
    }


def dataset_session_from_dict(data: Mapping[str, Any]) -> DatasetSession:
    """Deserialize a :class:`DatasetSession` from checkpoint sidecars."""
    experiment_raw = data.get("experiment_id")
    run_raw = data.get("run_id")
    return DatasetSession(
        session_id=str(data["session_id"]),
        experiment_id=ExperimentId(value=str(experiment_raw)) if experiment_raw else None,
        run_id=RunId(value=str(run_raw)) if run_raw else None,
        dataset_fingerprint=data.get("dataset_fingerprint"),
        mix_fingerprint=data.get("mix_fingerprint"),
        epoch=int(data.get("epoch", 0)),
        example_index=int(data.get("example_index", 0)),
        examples_seen=int(data.get("examples_seen", 0)),
        examples_total=data.get("examples_total"),
        shuffle_seed=data.get("shuffle_seed"),
        worker_id=int(data.get("worker_id", 0)),
        world_size=int(data.get("world_size", 1)),
        global_rank=int(data.get("global_rank", 0)),
        local_rank=int(data.get("local_rank", 0)),
        shard_id=int(data.get("shard_id", 0)),
        num_shards=int(data.get("num_shards", 1)),
        resume_token=data.get("resume_token"),
        metadata=dict(data.get("metadata") or {}),
    )


@dataclass(frozen=True, slots=True)
class ResumeValidationContext:
    """Expected portable identity fields for resume validation."""

    experiment_id: ExperimentId
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    config_fingerprint: str = ""
    execution_digest: str = ""
    trainer_backend_key: str = ""
    checkpoint_fingerprint: str | None = None
    software: Mapping[str, str] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class SaveCheckpointRequest:
    """Inputs required to publish a durable checkpoint."""

    model: TrainableModelHandle
    progress: TrainingProgress
    training_session: TrainingSession
    dataset_session: DatasetSession
    experiment_id: ExperimentId
    run_id: RunId
    output_dir: Path
    checkpoint_type: CheckpointType = CheckpointType.FULL_STATE
    model_fingerprint: str = ""
    adapter_fingerprint: str = ""
    config_fingerprint: str = ""
    execution_digest: str = ""
    quantization_digest: str = ""
    trainer_backend_key: str = "hf_trainer"
    adaptation_strategy_key: str = ""
    optimizer_policy_key: str = "adamw"
    scheduler_policy_key: str = "cosine"
    software: Mapping[str, str] = MappingProxyType({})
    metrics: Sequence[MetricSnapshot] = ()


class CheckpointIndex:
    """Ordered index of published checkpoints under an output directory."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._path = output_dir / INDEX_FILENAME
        self._entries: list[dict[str, Any]] = []
        self._load()

    @property
    def entries(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._entries)

    def _load(self) -> None:
        if not self._path.is_file():
            self._entries = []
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruption(f"Unreadable checkpoint index: {self._path}") from exc
        items = raw.get("checkpoints") if isinstance(raw, dict) else raw
        if not isinstance(items, list):
            raise CheckpointCorruption(f"Invalid checkpoint index shape: {self._path}")
        self._entries = [dict(item) for item in items]

    def add(self, handle: CheckpointHandle) -> None:
        entry = {
            "path": str(handle.path),
            "global_step": handle.global_step,
            "checkpoint_type": handle.checkpoint_type.value,
            "created_at": handle.created_at.isoformat() if handle.created_at else None,
        }
        self._entries = [e for e in self._entries if e.get("path") != entry["path"]]
        self._entries.append(entry)
        self._entries.sort(key=lambda item: int(item.get("global_step", 0)))
        self._persist()

    def _persist(self) -> None:
        self._output_dir.mkdir(parents=True, exist_ok=True)
        payload = {"checkpoints": self._entries}
        _write_json_atomic(self._path, payload)


class CheckpointManager:
    """
    Application owner of checkpoint policy, manifest I/O, validation, and
    orchestration of the frozen :class:`CheckpointStore` plus sidecars.
    """

    def __init__(
        self,
        *,
        checkpoint_store: CheckpointStore,
        rng: RngController,
    ) -> None:
        self._store = checkpoint_store
        self._rng = rng

    def cleanup_incomplete_tmp_dirs(self, output_dir: Path) -> list[Path]:
        """Remove stale ``.tmp-*`` directories under ``output_dir``."""
        removed: list[Path] = []
        if not output_dir.exists():
            return removed
        for child in output_dir.iterdir():
            if child.is_dir() and child.name.startswith(".tmp-"):
                shutil.rmtree(child, ignore_errors=True)
                removed.append(child)
        return removed

    def save(self, request: SaveCheckpointRequest) -> CheckpointHandle:
        """Publish a checkpoint atomically via tmp dir → rename."""
        output_dir = request.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        self.cleanup_incomplete_tmp_dirs(output_dir)

        tmp_dir = output_dir / f".tmp-{uuid.uuid4().hex}"
        tmp_dir.mkdir(parents=True, exist_ok=False)
        final_dir = output_dir / f"checkpoint-{request.progress.global_step}"

        try:
            self._store.save(
                request.model,
                request.progress,
                request.experiment_id,
                request.run_id,
                tmp_dir,
            )

            rng_state = self._rng.snapshot()
            dataset_payload = dataset_session_to_dict(request.dataset_session)
            metrics_payload = [_metric_to_dict(m) for m in request.metrics]

            _write_json(tmp_dir / RNG_FILENAME, rng_state)
            _write_json(tmp_dir / DATASET_SESSION_FILENAME, dataset_payload)
            _write_json(tmp_dir / METRICS_FILENAME, {"metrics": metrics_payload})

            artifact_paths = _discover_artifact_paths(tmp_dir, exclude=set())
            required_artifacts = _required_artifacts_for(
                request.checkpoint_type,
                artifact_paths=artifact_paths,
            )

            manifest_body = {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "training_protocol_version": TRAINING_PROTOCOL_VERSION,
                "experiment_id": request.experiment_id.value,
                "run_id": request.run_id.value,
                "global_step": request.progress.global_step,
                "epoch": request.progress.epoch,
                "checkpoint_type": request.checkpoint_type.value,
                "model_fingerprint": request.model_fingerprint,
                "adapter_fingerprint": request.adapter_fingerprint,
                "config_fingerprint": request.config_fingerprint,
                "execution_digest": request.execution_digest,
                "quantization_digest": request.quantization_digest,
                "metadata": {
                    "trainer_backend_key": request.trainer_backend_key,
                    "adaptation_strategy_key": request.adaptation_strategy_key,
                    "optimizer_policy_key": request.optimizer_policy_key,
                    "scheduler_policy_key": request.scheduler_policy_key,
                    "software": dict(request.software),
                },
                "artifact_paths": list(artifact_paths),
                "required_artifacts": list(required_artifacts),
                "dataset_session": dataset_payload,
                "rng_snapshot": rng_state,
                "created_at": datetime.now(UTC).isoformat(),
            }
            fingerprint = compute_checkpoint_fingerprint(manifest_body)
            manifest_body["checkpoint_fingerprint"] = fingerprint
            manifest = CheckpointManifest.from_dict(manifest_body)

            manifest_path = tmp_dir / MANIFEST_FILENAME
            _write_json(manifest_path, manifest.to_dict())

            for path in (
                manifest_path,
                tmp_dir / RNG_FILENAME,
                tmp_dir / DATASET_SESSION_FILENAME,
                tmp_dir / METRICS_FILENAME,
            ):
                _fsync_file(path)

            fingerprint = manifest.checkpoint_fingerprint
            published_handle = CheckpointHandle(
                path=final_dir,
                experiment_id=request.experiment_id,
                run_id=request.run_id,
                checkpoint_type=request.checkpoint_type,
                global_step=request.progress.global_step,
                created_at=datetime.now(UTC),
                metadata=(("checkpoint_fingerprint", fingerprint),),
            )

            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(tmp_dir, final_dir)

            published_handle = CheckpointHandle(
                path=final_dir,
                experiment_id=published_handle.experiment_id,
                run_id=published_handle.run_id,
                checkpoint_type=published_handle.checkpoint_type,
                global_step=published_handle.global_step,
                created_at=published_handle.created_at,
                metadata=published_handle.metadata,
            )

            index = CheckpointIndex(output_dir)
            index.add(published_handle)
            self._store.prune(output_dir, keep=_infer_keep_limit(output_dir))
            return published_handle
        except Exception:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

    def load_and_validate(
        self,
        path: Path,
        *,
        expected: ResumeValidationContext,
        policy: ResumePolicy,
        training_session: TrainingSession,
    ) -> ResumeBundle:
        """Validate checkpoint compatibility and assemble a :class:`ResumeBundle`."""
        from aiodoo_training.training.resume import ResumeBundle

        checkpoint_dir = path.resolve()
        if not checkpoint_dir.is_dir():
            raise CheckpointCorruption(f"Checkpoint directory not found: {checkpoint_dir}")

        manifest_path = checkpoint_dir / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise CheckpointCorruption(f"Missing {MANIFEST_FILENAME} in {checkpoint_dir}")

        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruption(f"Unreadable manifest: {manifest_path}") from exc

        try:
            manifest = CheckpointManifest.from_dict(manifest_data)
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruption(f"Invalid manifest schema: {manifest_path}") from exc

        collected_warnings: list[str] = []

        self._validate_protocol(manifest)
        self._validate_schema(manifest)
        self._validate_required_artifacts(checkpoint_dir, manifest)

        self._check_field(
            policy,
            field="checkpoint_fingerprint",
            expected=expected.checkpoint_fingerprint,
            actual=manifest.checkpoint_fingerprint,
            computed=self._recompute_fingerprint(manifest_data),
            warnings=collected_warnings,
        )
        self._check_field(
            policy,
            field="experiment_id",
            expected=expected.experiment_id.value,
            actual=manifest.experiment_id.value,
            warnings=collected_warnings,
        )
        self._check_field(
            policy,
            field="trainer_backend_key",
            expected=expected.trainer_backend_key,
            actual=manifest.metadata.trainer_backend_key,
            warnings=collected_warnings,
        )
        self._check_field(
            policy,
            field="model_fingerprint",
            expected=expected.model_fingerprint,
            actual=manifest.model_fingerprint,
            warnings=collected_warnings,
        )
        self._check_field(
            policy,
            field="adapter_fingerprint",
            expected=expected.adapter_fingerprint,
            actual=manifest.adapter_fingerprint,
            warnings=collected_warnings,
        )
        self._check_field(
            policy,
            field="config_fingerprint",
            expected=expected.config_fingerprint,
            actual=manifest.config_fingerprint,
            warnings=collected_warnings,
        )
        self._check_field(
            policy,
            field="execution_digest",
            expected=expected.execution_digest,
            actual=manifest.execution_digest,
            warnings=collected_warnings,
        )
        self._check_software_versions(
            policy, expected.software, manifest.metadata.software, collected_warnings
        )

        handle = CheckpointHandle(
            path=checkpoint_dir,
            experiment_id=manifest.experiment_id,
            run_id=manifest.run_id,
            checkpoint_type=manifest.checkpoint_type,
            global_step=manifest.global_step,
            created_at=manifest.created_at,
            metadata=(("checkpoint_fingerprint", manifest.checkpoint_fingerprint),),
        )

        try:
            model = self._store.restore(handle)
        except Exception as exc:
            raise CheckpointCorruption(f"Failed to restore weights from {checkpoint_dir}") from exc

        rng_state = _read_json(checkpoint_dir / RNG_FILENAME)
        dataset_session = dataset_session_from_dict(
            _read_json(checkpoint_dir / DATASET_SESSION_FILENAME)
        )

        resumed_session = training_session.with_checkpoint(
            manifest.checkpoint_fingerprint,
            path=checkpoint_dir,
        )

        meta = dict(resumed_session.metadata)
        for index, warning in enumerate(collected_warnings):
            meta[f"resume_warning_{index}"] = warning
            warnings.warn(warning, ResumeWarning, stacklevel=2)

        from dataclasses import replace

        resumed_session = replace(
            resumed_session,
            global_step=manifest.global_step,
            epoch=manifest.epoch,
            dataset_session=dataset_session,
            metadata=MappingProxyType(meta),
        )

        return ResumeBundle(
            model=model,
            checkpoint=handle,
            training_session=resumed_session,
            dataset_session=dataset_session,
            rng_state=dict(rng_state),
            manifest=manifest,
            warnings=tuple(collected_warnings),
        )

    def list_checkpoints(self, output_dir: Path) -> Sequence[CheckpointHandle]:
        return self._store.list(output_dir)

    @staticmethod
    def _validate_protocol(manifest: CheckpointManifest) -> None:
        if manifest.training_protocol_version != TRAINING_PROTOCOL_VERSION:
            raise IncompatibleResume(
                f"Unsupported training_protocol_version: {manifest.training_protocol_version!r} "
                f"(expected {TRAINING_PROTOCOL_VERSION!r})."
            )

    @staticmethod
    def _validate_schema(manifest: CheckpointManifest) -> None:
        if not manifest.schema_version:
            raise CheckpointCorruption("Manifest schema_version is empty.")

    @staticmethod
    def _validate_required_artifacts(checkpoint_dir: Path, manifest: CheckpointManifest) -> None:
        required = manifest.required_artifacts or _required_artifacts_for(manifest.checkpoint_type)
        missing = [name for name in required if not (checkpoint_dir / name).exists()]
        if missing:
            raise CheckpointCorruption(
                f"Missing required checkpoint artifacts in {checkpoint_dir}: {', '.join(missing)}"
            )

    @staticmethod
    def _recompute_fingerprint(manifest_data: Mapping[str, Any]) -> str:
        return compute_checkpoint_fingerprint(dict(manifest_data))

    def _check_field(
        self,
        policy: ResumePolicy,
        *,
        field: str,
        expected: str | None,
        actual: str,
        computed: str | None = None,
        warnings: list[str],
    ) -> None:
        if field == "checkpoint_fingerprint" and expected is None:
            if actual and computed and actual != computed:
                self._apply_policy(
                    policy,
                    field,
                    message=(
                        f"checkpoint_fingerprint mismatch (stored={actual}, computed={computed})"
                    ),
                    warnings=warnings,
                )
            return
        if not expected:
            return

        if actual == expected:
            return

        self._apply_policy(
            policy,
            field,
            message=f"{field} mismatch (expected={expected!r}, actual={actual!r})",
            warnings=warnings,
        )

    def _check_software_versions(
        self,
        policy: ResumePolicy,
        expected: Mapping[str, str],
        actual: Mapping[str, str],
        warnings: list[str],
    ) -> None:
        if policy is ResumePolicy.STRICT:
            return
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if actual_value is None:
                continue
            if actual_value != expected_value:
                message = (
                    f"software.{key} mismatch "
                    f"(expected={expected_value!r}, actual={actual_value!r})"
                )
                if policy is ResumePolicy.WARN:
                    warnings.append(message)
                elif policy is ResumePolicy.RELAXED:
                    pass

    def _apply_policy(
        self,
        policy: ResumePolicy,
        field: str,
        *,
        message: str,
        warnings: list[str],
    ) -> None:
        action = _POLICY_MATRIX.get(field, {}).get(policy, "reject")
        if action == "ignore":
            return
        if action == "warn":
            warnings.append(message)
            return
        raise IncompatibleResume(message)


_POLICY_MATRIX: dict[str, dict[ResumePolicy, str]] = {
    "checkpoint_fingerprint": {
        ResumePolicy.STRICT: "reject",
        ResumePolicy.WARN: "reject",
        ResumePolicy.RELAXED: "warn",
    },
    "experiment_id": {
        ResumePolicy.STRICT: "reject",
        ResumePolicy.WARN: "reject",
        ResumePolicy.RELAXED: "warn",
    },
    "trainer_backend_key": {
        ResumePolicy.STRICT: "reject",
        ResumePolicy.WARN: "warn",
        ResumePolicy.RELAXED: "warn",
    },
    "model_fingerprint": {
        ResumePolicy.STRICT: "reject",
        ResumePolicy.WARN: "warn",
        ResumePolicy.RELAXED: "warn",
    },
    "adapter_fingerprint": {
        ResumePolicy.STRICT: "reject",
        ResumePolicy.WARN: "warn",
        ResumePolicy.RELAXED: "warn",
    },
    "config_fingerprint": {
        ResumePolicy.STRICT: "reject",
        ResumePolicy.WARN: "warn",
        ResumePolicy.RELAXED: "warn",
    },
    "execution_digest": {
        ResumePolicy.STRICT: "reject",
        ResumePolicy.WARN: "warn",
        ResumePolicy.RELAXED: "ignore",
    },
}


def _required_artifacts_for(
    checkpoint_type: CheckpointType,
    *,
    artifact_paths: Sequence[str] = (),
) -> tuple[str, ...]:
    if checkpoint_type is CheckpointType.FULL_STATE:
        extras = tuple(path for path in artifact_paths if path not in DEFAULT_FULL_STATE_REQUIRED)
        return DEFAULT_FULL_STATE_REQUIRED + extras
    if checkpoint_type is CheckpointType.ADAPTER_ONLY:
        return (
            MANIFEST_FILENAME,
            *tuple(path for path in artifact_paths if path != MANIFEST_FILENAME),
        )
    return (MANIFEST_FILENAME, METRICS_FILENAME)


def _metric_to_dict(metric: MetricSnapshot) -> dict[str, Any]:
    return {
        "name": metric.name,
        "value": metric.value,
        "step": metric.step,
        "timestamp": metric.timestamp.isoformat() if metric.timestamp else None,
        "tags": list(metric.tags),
    }


def _write_json(path: Path, payload: Mapping[str, Any] | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, default=str), encoding="utf-8")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    _write_json(tmp, payload)
    _fsync_file(tmp)
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CheckpointCorruption(f"Missing JSON artifact: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointCorruption(f"Unreadable JSON artifact: {path}") from exc
    if not isinstance(raw, dict):
        raise CheckpointCorruption(f"Expected JSON object in {path}")
    return raw


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    if not path.exists():
        return
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _discover_artifact_paths(tmp_dir: Path, *, exclude: set[str]) -> tuple[str, ...]:
    paths = [
        str(item.relative_to(tmp_dir))
        for item in tmp_dir.rglob("*")
        if item.is_file() and item.name not in exclude and not item.name.startswith(".")
    ]
    return tuple(sorted(set(paths)))


def _infer_keep_limit(output_dir: Path) -> int:
    index_path = output_dir / INDEX_FILENAME
    if index_path.is_file():
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            keep = raw.get("save_total_limit")
            if isinstance(keep, int) and keep >= 1:
                return keep
        except (OSError, json.JSONDecodeError):
            pass
    return 3
