"""Factory and builder skeleton tests."""

import pytest

from aiodoo_training.builders import DatasetMixBuilder, ExperimentConfigBuilder
from aiodoo_training.exceptions import BuilderError, FactoryError
from aiodoo_training.factories import DatasetSourceFactory, TrainerBackendFactory
from aiodoo_training.registries import Registry


def test_experiment_config_builder_is_skeleton() -> None:
    with pytest.raises(BuilderError, match="Phase 0"):
        ExperimentConfigBuilder().with_name("x").build()


def test_dataset_mix_builder_tracks_refs_but_stays_skeleton() -> None:
    builder = DatasetMixBuilder().with_shuffle(False, seed=1)
    with pytest.raises(BuilderError, match="0 refs"):
        builder.build()


def test_factory_unknown_key_lists_known() -> None:
    with pytest.raises(FactoryError, match="Known keys:"):
        DatasetSourceFactory().create("missing")


def test_factory_accepts_injected_empty_registry() -> None:
    from aiodoo_training.ports.trainer import TrainerBackend

    registry: Registry[type[TrainerBackend]] = Registry("test-trainers")
    factory = TrainerBackendFactory(registry=registry)
    with pytest.raises(FactoryError, match=r"Known keys: \(none\)"):
        factory.create("hf_trainer")
