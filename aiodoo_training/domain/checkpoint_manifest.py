"""Checkpoint manifest domain — JSON-serializable resume inventory."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any

from aiodoo_training.domain.enums import CheckpointType
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.training_policies import TRAINING_PROTOCOL_VERSION


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    trainer_backend_key: str
    adaptation_strategy_key: str = ""
    optimizer_policy_key: str = "adamw"
    scheduler_policy_key: str = "cosine"
    software: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "software",
            MappingProxyType({str(k): str(v) for k, v in dict(self.software).items()}),
        )


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    """Inventory of a checkpoint package for resume validation."""

    schema_version: str
    training_protocol_version: str
    experiment_id: ExperimentId
    run_id: RunId
    global_step: int
    epoch: float
    checkpoint_type: CheckpointType
    model_fingerprint: str
    adapter_fingerprint: str
    config_fingerprint: str
    execution_digest: str
    quantization_digest: str
    metadata: CheckpointMetadata
    artifact_paths: tuple[str, ...] = ()
    required_artifacts: tuple[str, ...] = ()
    dataset_session: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    rng_snapshot: Mapping[str, Any] = field(default_factory=lambda: MappingProxyType({}))
    checkpoint_fingerprint: str = ""
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.training_protocol_version != TRAINING_PROTOCOL_VERSION and False:
            # Validation happens in CheckpointManager — keep field free here.
            pass
        object.__setattr__(
            self,
            "dataset_session",
            MappingProxyType(dict(self.dataset_session)),
        )
        # rng may contain non-JSON-native tuples — store as mapping for serde via manager
        object.__setattr__(self, "rng_snapshot", MappingProxyType(dict(self.rng_snapshot)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "training_protocol_version": self.training_protocol_version,
            "experiment_id": self.experiment_id.value,
            "run_id": self.run_id.value,
            "global_step": self.global_step,
            "epoch": self.epoch,
            "checkpoint_type": self.checkpoint_type.value,
            "model_fingerprint": self.model_fingerprint,
            "adapter_fingerprint": self.adapter_fingerprint,
            "config_fingerprint": self.config_fingerprint,
            "execution_digest": self.execution_digest,
            "quantization_digest": self.quantization_digest,
            "metadata": {
                "trainer_backend_key": self.metadata.trainer_backend_key,
                "adaptation_strategy_key": self.metadata.adaptation_strategy_key,
                "optimizer_policy_key": self.metadata.optimizer_policy_key,
                "scheduler_policy_key": self.metadata.scheduler_policy_key,
                "software": dict(self.metadata.software),
            },
            "artifact_paths": list(self.artifact_paths),
            "required_artifacts": list(self.required_artifacts),
            "dataset_session": dict(self.dataset_session),
            "rng_snapshot": _jsonable(dict(self.rng_snapshot)),
            "checkpoint_fingerprint": self.checkpoint_fingerprint,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CheckpointManifest:
        meta_raw = data.get("metadata") or {}
        meta = CheckpointMetadata(
            trainer_backend_key=str(meta_raw.get("trainer_backend_key", "")),
            adaptation_strategy_key=str(meta_raw.get("adaptation_strategy_key", "")),
            optimizer_policy_key=str(meta_raw.get("optimizer_policy_key", "adamw")),
            scheduler_policy_key=str(meta_raw.get("scheduler_policy_key", "cosine")),
            software=dict(meta_raw.get("software") or {}),
        )
        created = data.get("created_at")
        return cls(
            schema_version=str(data.get("schema_version", "1")),
            training_protocol_version=str(data.get("training_protocol_version", "")),
            experiment_id=ExperimentId(value=str(data["experiment_id"])),
            run_id=RunId(value=str(data["run_id"])),
            global_step=int(data.get("global_step", 0)),
            epoch=float(data.get("epoch", 0.0)),
            checkpoint_type=CheckpointType(str(data.get("checkpoint_type", "full_state"))),
            model_fingerprint=str(data.get("model_fingerprint", "")),
            adapter_fingerprint=str(data.get("adapter_fingerprint", "")),
            config_fingerprint=str(data.get("config_fingerprint", "")),
            execution_digest=str(data.get("execution_digest", "")),
            quantization_digest=str(data.get("quantization_digest", "")),
            metadata=meta,
            artifact_paths=tuple(str(x) for x in (data.get("artifact_paths") or ())),
            required_artifacts=tuple(str(x) for x in (data.get("required_artifacts") or ())),
            dataset_session=dict(data.get("dataset_session") or {}),
            rng_snapshot=dict(data.get("rng_snapshot") or {}),
            checkpoint_fingerprint=str(data.get("checkpoint_fingerprint", "")),
            created_at=datetime.fromisoformat(created) if isinstance(created, str) else None,
        )


def compute_checkpoint_fingerprint(manifest_body: Mapping[str, Any]) -> str:
    """Digest of manifest contents excluding the fingerprint field itself."""
    payload = {k: v for k, v in manifest_body.items() if k != "checkpoint_fingerprint"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
