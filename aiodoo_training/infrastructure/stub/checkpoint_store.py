"""Stub CheckpointStore — JSON weight packages for CPU CI."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aiodoo_training.domain.adapter_info import AdapterCapabilities, AdapterMetadata
from aiodoo_training.domain.enums import AdapterType, CheckpointType, ModelFamily, Precision
from aiodoo_training.domain.handles import TrainableModelHandle
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.model_info import ModelCapabilities, ModelFingerprint, ModelMetadata
from aiodoo_training.domain.quantization import QuantizationPolicy
from aiodoo_training.domain.training import CheckpointHandle, TrainingProgress
from aiodoo_training.exceptions import CheckpointCorruption, DomainError
from aiodoo_training.infrastructure.model_handles import (
    OpaqueBaseModel,
    OpaqueTrainableModel,
    as_trainable_handle,
    require_trainable_carrier,
)
from aiodoo_training.ports.trainer import CheckpointStore

WEIGHTS_FILENAME = "weights.json"
OPTIMIZER_FILENAME = "optimizer.json"


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _adapter_from_dict(data: Mapping[str, Any]) -> AdapterMetadata:
    caps_raw = data.get("capabilities") or {}
    capabilities = AdapterCapabilities(
        supports_merge=bool(caps_raw.get("supports_merge", True)),
        supports_resume=bool(caps_raw.get("supports_resume", True)),
        requires_quantization=bool(caps_raw.get("requires_quantization", False)),
        extra=dict(caps_raw.get("extra") or {}),
    )
    return AdapterMetadata(
        adapter_type=AdapterType(str(data.get("adapter_type", "none"))),
        rank=int(data["rank"]) if data.get("rank") is not None else None,
        alpha=int(data["alpha"]) if data.get("alpha") is not None else None,
        dropout=float(data["dropout"]) if data.get("dropout") is not None else None,
        target_modules=tuple(str(x) for x in (data.get("target_modules") or ())),
        trainable_parameters=(
            int(data["trainable_parameters"])
            if data.get("trainable_parameters") is not None
            else None
        ),
        total_parameters=(
            int(data["total_parameters"]) if data.get("total_parameters") is not None else None
        ),
        capabilities=capabilities,
        strategy_key=str(data.get("strategy_key", "stub")),
        profile_key=str(data["profile_key"]) if data.get("profile_key") is not None else None,
        extra=dict(data.get("extra") or {}),
    )


def _fingerprint_from_dict(data: Mapping[str, Any] | None) -> ModelFingerprint | None:
    if not data:
        return None
    digest = str(data.get("digest", ""))
    if len(digest) < 16:
        digest = (digest + "0" * 16)[:16]
    return ModelFingerprint(
        digest=digest,
        identifier=str(data.get("identifier", "stub")),
        revision=str(data["revision"]) if data.get("revision") is not None else None,
        family=str(data.get("family", "unknown")),
        quantization_digest=str(data.get("quantization_digest", "")),
        execution_digest=str(data.get("execution_digest", "")),
    )


def _base_from_dict(data: Mapping[str, Any] | None) -> OpaqueBaseModel | None:
    if not data:
        return None
    meta_raw = data.get("aiodoo_metadata") or {}
    if isinstance(meta_raw, Mapping) and meta_raw.get("identifier"):
        metadata = ModelMetadata.from_dict(meta_raw)
    else:
        metadata = ModelMetadata(
            identifier=str(data.get("identifier", "stub")),
            family=ModelFamily.UNKNOWN,
            precision=Precision.FP32,
            quantization=QuantizationPolicy(),
            capabilities=ModelCapabilities(),
            backend_key=str(data.get("backend_key", "stub")),
        )
    fingerprint = _fingerprint_from_dict(data.get("aiodoo_fingerprint"))
    if fingerprint is None:
        fingerprint = ModelFingerprint(
            digest="stub_restored_base00",
            identifier=metadata.identifier,
            revision=metadata.revision,
            family=metadata.family.value,
            quantization_digest="",
            execution_digest="",
        )
    return OpaqueBaseModel(
        framework_model=data.get("framework_model"),
        aiodoo_metadata=metadata,
        aiodoo_fingerprint=fingerprint,
        backend_key=str(data.get("backend_key", metadata.backend_key)),
    )


def _carrier_to_weights_payload(carrier: OpaqueTrainableModel) -> dict[str, Any]:
    base_payload: dict[str, Any] | None = None
    if carrier.base is not None:
        base_payload = {
            "framework_model": _jsonable(carrier.base.framework_model),
            "aiodoo_metadata": carrier.base.aiodoo_metadata.to_dict(),
            "aiodoo_fingerprint": {
                "digest": carrier.base.aiodoo_fingerprint.digest,
                "identifier": carrier.base.aiodoo_fingerprint.identifier,
                "revision": carrier.base.aiodoo_fingerprint.revision,
                "family": carrier.base.aiodoo_fingerprint.family,
                "quantization_digest": carrier.base.aiodoo_fingerprint.quantization_digest,
                "execution_digest": carrier.base.aiodoo_fingerprint.execution_digest,
            },
            "backend_key": carrier.base.backend_key,
        }
    return {
        "kind": "stub_trainable",
        "strategy_key": carrier.strategy_key,
        "framework_model": _jsonable(carrier.framework_model),
        "adapter_metadata": carrier.aiodoo_adapter_metadata.to_dict(),
        "base": base_payload,
    }


def _optimizer_payload(carrier: OpaqueTrainableModel) -> dict[str, Any] | None:
    payload = getattr(carrier, "optimizer_state", None)
    if payload is None and isinstance(carrier.framework_model, dict):
        payload = carrier.framework_model.get("optimizer")
    if payload is None:
        return None
    if not isinstance(payload, dict):
        return {"value": _jsonable(payload)}
    return _jsonable(payload)


class StubCheckpointStore(CheckpointStore):
    """
    Persist stub trainable weights as ``weights.json`` (and optional
    ``optimizer.json``). Does not own manifests / RNG / DatasetSession sidecars.
    """

    BACKEND_KEY = "stub"

    def save(
        self,
        model: TrainableModelHandle,
        progress: TrainingProgress,
        experiment_id: ExperimentId,
        run_id: RunId,
        destination: Path,
    ) -> CheckpointHandle:
        destination.mkdir(parents=True, exist_ok=True)
        carrier = require_trainable_carrier(model)
        weights_path = destination / WEIGHTS_FILENAME
        weights_path.write_text(
            json.dumps(_carrier_to_weights_payload(carrier), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        optimizer = _optimizer_payload(carrier)
        if optimizer is not None:
            (destination / OPTIMIZER_FILENAME).write_text(
                json.dumps(optimizer, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        return CheckpointHandle(
            path=destination,
            experiment_id=experiment_id,
            run_id=run_id,
            checkpoint_type=CheckpointType.FULL_STATE,
            global_step=progress.global_step,
            created_at=datetime.now(UTC),
            metadata=(("store", self.BACKEND_KEY),),
        )

    def restore(self, handle: CheckpointHandle) -> TrainableModelHandle:
        directory = handle.path
        weights_path = directory / WEIGHTS_FILENAME
        if not weights_path.is_file():
            # Allow handle.path to point at the weights file itself.
            if directory.is_file() and directory.name == WEIGHTS_FILENAME:
                weights_path = directory
                directory = directory.parent
            else:
                raise CheckpointCorruption(f"Missing stub weights at {weights_path}")
        try:
            raw = json.loads(weights_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruption(f"Unreadable stub weights: {weights_path}") from exc
        if not isinstance(raw, dict):
            raise CheckpointCorruption(f"Invalid stub weights payload: {weights_path}")

        framework_model = raw.get("framework_model")
        if framework_model is None and raw.get("kind") == "stub":
            framework_model = raw
        if not isinstance(framework_model, dict):
            raise DomainError("StubCheckpointStore.restore expects a dict framework_model.")

        # Prefer mutable weights list for continued in-place updates.
        if "weights" in framework_model and isinstance(framework_model["weights"], list):
            framework_model["weights"] = [float(w) for w in framework_model["weights"]]
        elif "weights" in framework_model and isinstance(framework_model["weights"], tuple):
            framework_model["weights"] = [float(w) for w in framework_model["weights"]]

        adapter_raw = raw.get("adapter_metadata") or {}
        if isinstance(adapter_raw, Mapping) and adapter_raw:
            adapter_metadata = _adapter_from_dict(adapter_raw)
        else:
            adapter_metadata = AdapterMetadata(
                adapter_type=AdapterType.NONE,
                strategy_key=str(raw.get("strategy_key", "stub")),
            )

        opt_path = directory / OPTIMIZER_FILENAME
        if opt_path.is_file():
            try:
                optimizer = json.loads(opt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise CheckpointCorruption(f"Unreadable optimizer.json: {opt_path}") from exc
            framework_model["optimizer"] = optimizer

        carrier = OpaqueTrainableModel(
            framework_model=framework_model,
            aiodoo_adapter_metadata=adapter_metadata,
            base=_base_from_dict(raw.get("base") if isinstance(raw.get("base"), dict) else None),
            strategy_key=str(raw.get("strategy_key", adapter_metadata.strategy_key)),
        )
        return as_trainable_handle(carrier)

    def list(self, directory: Path) -> Sequence[CheckpointHandle]:
        if not directory.is_dir():
            return ()
        handles: list[CheckpointHandle] = []
        for child in sorted(directory.iterdir(), key=lambda p: p.name):
            if not child.is_dir() or not child.name.startswith("checkpoint-"):
                continue
            if child.name.startswith(".tmp-"):
                continue
            if not (child / WEIGHTS_FILENAME).is_file():
                continue
            step = _parse_checkpoint_step(child.name)
            handles.append(
                CheckpointHandle(
                    path=child,
                    experiment_id=ExperimentId(value="unknown"),
                    run_id=RunId(value="unknown"),
                    checkpoint_type=CheckpointType.FULL_STATE,
                    global_step=step,
                    created_at=None,
                    metadata=(("store", self.BACKEND_KEY),),
                )
            )
        handles.sort(key=lambda h: h.global_step)
        return tuple(handles)

    def prune(self, directory: Path, keep: int) -> Sequence[CheckpointHandle]:
        if keep < 0:
            raise DomainError("prune keep must be >= 0.")
        handles = list(self.list(directory))
        if len(handles) <= keep:
            return ()
        removed = handles[: len(handles) - keep] if keep > 0 else handles
        for handle in removed:
            if handle.path.is_dir():
                shutil.rmtree(handle.path, ignore_errors=True)
        return tuple(removed)


def _parse_checkpoint_step(name: str) -> int:
    # checkpoint-<step>
    suffix = name.removeprefix("checkpoint-")
    try:
        return int(suffix)
    except ValueError:
        return 0


def register_default_checkpoint_stores(*, overwrite: bool = False) -> None:
    """Register ``stub`` and ``hf`` / ``huggingface`` checkpoint stores."""
    from aiodoo_training.infrastructure.huggingface.checkpoint_store import (
        register_hf_checkpoint_store,
    )
    from aiodoo_training.registries import checkpoint_store_registry

    if not checkpoint_store_registry.exists("stub") or overwrite:
        checkpoint_store_registry.register("stub", StubCheckpointStore, overwrite=overwrite)
    register_hf_checkpoint_store(overwrite=overwrite)
