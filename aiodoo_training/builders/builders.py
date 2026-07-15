"""Builder skeletons for immutable domain assembly (Phase 0).

Builders construct frozen domain graphs. They do not perform I/O and do not
resolve registries — those responsibilities belong to config loaders and
factories respectively (same separation as aiodoo-datasets).
"""

from __future__ import annotations

from aiodoo_training.domain.artifacts import ExperimentManifest, ExportArtifact
from aiodoo_training.domain.config import CurriculumSpec, DatasetMixSpec, ExperimentConfig
from aiodoo_training.domain.refs import DatasetRef
from aiodoo_training.exceptions import BuilderError


class ExperimentConfigBuilder:
    """
    Assembles an :class:`ExperimentConfig` from typed fragments.

    Phase 0 skeleton: YAML composition lives in :mod:`aiodoo_training.config`.
    This builder will map validated raw config into domain objects in a later phase.
    """

    def __init__(self) -> None:
        self._data: dict[str, object] = {}

    def with_name(self, name: str) -> ExperimentConfigBuilder:
        self._data["name"] = name
        return self

    def with_fragment(self, key: str, value: object) -> ExperimentConfigBuilder:
        self._data[key] = value
        return self

    def build(self) -> ExperimentConfig:
        raise BuilderError(
            "ExperimentConfigBuilder.build() is not implemented in Phase 0. "
            "Use ConfigSystem.load_experiment() for YAML composition and validation."
        )


class TrainingContextBuilder:
    """
    Builds a resolved runtime :class:`~aiodoo_training.training.context.TrainingContext`.

    Requires collaborator handles to be provided (factories resolve ports).
    """

    def __init__(self) -> None:
        self._pieces: dict[str, object] = {}

    def with_config(self, config: ExperimentConfig) -> TrainingContextBuilder:
        self._pieces["config"] = config
        return self

    def with_piece(self, key: str, value: object) -> TrainingContextBuilder:
        self._pieces[key] = value
        return self

    def build(self, config: ExperimentConfig | None = None) -> object:
        from aiodoo_training.domain.training_policies import (
            CheckpointPolicy,
            GradientAccumulationPolicy,
            GradientClippingPolicy,
            LossScalingPolicy,
            MixedPrecisionPolicy,
            OptimizerPolicy,
            SchedulerPolicy,
        )
        from aiodoo_training.training.context import TrainingContext

        cfg = config if config is not None else self._pieces.get("config")
        if not isinstance(cfg, ExperimentConfig):
            raise BuilderError("TrainingContextBuilder requires ExperimentConfig.")

        required = (
            "execution",
            "model",
            "dataset_session",
            "training_session",
            "trainer",
            "checkpoint_store",
            "rng",
        )
        missing = [key for key in required if key not in self._pieces]
        if missing:
            raise BuilderError(
                "TrainingContextBuilder missing required pieces: " + ", ".join(missing)
            )

        return TrainingContext(
            config=cfg,
            execution=self._pieces["execution"],  # type: ignore[arg-type]
            model=self._pieces["model"],  # type: ignore[arg-type]
            dataset_session=self._pieces["dataset_session"],  # type: ignore[arg-type]
            training_session=self._pieces["training_session"],  # type: ignore[arg-type]
            trainer=self._pieces["trainer"],  # type: ignore[arg-type]
            checkpoint_store=self._pieces["checkpoint_store"],  # type: ignore[arg-type]
            rng=self._pieces["rng"],  # type: ignore[arg-type]
            optimizer_policy=self._pieces.get("optimizer_policy") or OptimizerPolicy(),  # type: ignore[arg-type]
            scheduler_policy=self._pieces.get("scheduler_policy") or SchedulerPolicy(),  # type: ignore[arg-type]
            gradient_accumulation_policy=self._pieces.get("gradient_accumulation_policy")
            or GradientAccumulationPolicy(),  # type: ignore[arg-type]
            gradient_clipping_policy=self._pieces.get("gradient_clipping_policy")
            or GradientClippingPolicy(),  # type: ignore[arg-type]
            mixed_precision_policy=self._pieces.get("mixed_precision_policy")
            or MixedPrecisionPolicy(),  # type: ignore[arg-type]
            loss_scaling_policy=self._pieces.get("loss_scaling_policy") or LossScalingPolicy(),  # type: ignore[arg-type]
            checkpoint_policy=self._pieces.get("checkpoint_policy") or CheckpointPolicy(),  # type: ignore[arg-type]
            callbacks=tuple(self._pieces.get("callbacks") or ()),  # type: ignore[arg-type]
            tracker=self._pieces.get("tracker"),  # type: ignore[arg-type]
            event_bus=self._pieces.get("event_bus"),  # type: ignore[arg-type]
            checkpoint_manager=self._pieces.get("checkpoint_manager"),  # type: ignore[arg-type]
            metric_collector=self._pieces.get("metric_collector"),  # type: ignore[arg-type]
            training_history=self._pieces.get("training_history"),  # type: ignore[arg-type]
            trainer_backend_key=str(self._pieces.get("trainer_backend_key") or "stub"),
            model_fingerprint=str(self._pieces.get("model_fingerprint") or ""),
            adapter_fingerprint=str(self._pieces.get("adapter_fingerprint") or ""),
            config_fingerprint=str(self._pieces.get("config_fingerprint") or ""),
            execution_digest=str(self._pieces.get("execution_digest") or ""),
            quantization_digest=str(self._pieces.get("quantization_digest") or ""),
            adaptation_strategy_key=str(self._pieces.get("adaptation_strategy_key") or ""),
            bind_extra=_as_str_object_dict(self._pieces.get("bind_extra")),
        )


def _as_str_object_dict(value: object | None) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): v for k, v in value.items()}
    raise BuilderError("bind_extra must be a mapping when provided.")



class DatasetMixBuilder:
    """Builds a :class:`DatasetMixSpec` from dataset references and weights."""

    def __init__(self) -> None:
        self._refs: list[DatasetRef] = []
        self._shuffle: bool = True
        self._seed: int = 42

    def add_ref(self, ref: DatasetRef) -> DatasetMixBuilder:
        self._refs.append(ref)
        return self

    def with_shuffle(self, shuffle: bool, *, seed: int = 42) -> DatasetMixBuilder:
        self._shuffle = shuffle
        self._seed = seed
        return self

    def build(self) -> DatasetMixSpec:
        raise BuilderError(
            "DatasetMixBuilder.build() is not implemented in Phase 0 "
            f"(collected {len(self._refs)} refs)."
        )


class CurriculumBuilder:
    """Builds a :class:`CurriculumSpec` from ordered stage names (Phase 5)."""

    def __init__(self) -> None:
        self._stages: list[str] = []
        self._mode: str | None = None

    def add_stage(self, name: str) -> CurriculumBuilder:
        self._stages.append(name)
        return self

    def with_mode(self, mode: str) -> CurriculumBuilder:
        self._mode = mode
        return self

    def build(self) -> CurriculumSpec:
        from aiodoo_training.domain.enums import CurriculumMode

        if self._mode is not None:
            mode = CurriculumMode(self._mode)
        elif self._stages:
            mode = CurriculumMode.SEQUENTIAL
        else:
            mode = CurriculumMode.NONE
        return CurriculumSpec(mode=mode, stages=tuple(self._stages))


class ManifestBuilder:
    """Builds an :class:`ExperimentManifest` for aiodoo-models handoff."""

    def build(self) -> ExperimentManifest:
        raise BuilderError("ManifestBuilder.build() is not implemented in Phase 0.")


class ExportBundleBuilder:
    """Builds a tuple of :class:`ExportArtifact` entries for export packages."""

    def build(self) -> tuple[ExportArtifact, ...]:
        raise BuilderError("ExportBundleBuilder.build() is not implemented in Phase 0.")
