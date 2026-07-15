"""Phase 0 freeze: DatasetSession and ChatTemplateRegistry architecture tests."""

from dataclasses import FrozenInstanceError

import pytest

from aiodoo_training.domain import DatasetSession, ExperimentId
from aiodoo_training.exceptions import RegistryError
from aiodoo_training.ports import ChatTemplate
from aiodoo_training.registries import Registry, chat_template_registry


def test_dataset_session_is_immutable_and_copy_on_write() -> None:
    session = DatasetSession(session_id="s1", experiment_id=ExperimentId(value="exp_1"))
    with pytest.raises(FrozenInstanceError):
        session.epoch = 1  # type: ignore[misc]
    advanced = session.advance(steps=3)
    assert session.example_index == 0
    assert advanced.example_index == 3
    assert advanced.examples_seen == 3
    nxt = advanced.next_epoch()
    assert nxt.epoch == 1
    assert nxt.example_index == 0


def test_dataset_session_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        DatasetSession(session_id="")
    with pytest.raises(ValueError):
        DatasetSession(session_id="s", world_size=0)
    with pytest.raises(ValueError):
        DatasetSession(session_id="s", shard_id=1, num_shards=1)


def test_dataset_session_resume_and_placement_fields() -> None:
    session = DatasetSession(
        session_id="s1",
        mix_fingerprint="mix_abc",
        examples_total=10,
        global_rank=1,
        local_rank=0,
        shard_id=1,
        num_shards=4,
        resume_token="tok",
    )
    assert session.mix_fingerprint == "mix_abc"
    assert session.examples_total == 10
    mixed = session.with_mix_fingerprint("mix_def")
    assert mixed.mix_fingerprint == "mix_def"
    assert session.mix_fingerprint == "mix_abc"


def test_chat_template_registry_registration() -> None:
    registry: Registry[type[ChatTemplate]] = Registry("chat_templates_test")

    class DummyTemplate(ChatTemplate):
        @property
        def name(self) -> str:
            return "dummy"

        @property
        def family(self) -> str:
            return "dummy"

        def render(self, messages):  # type: ignore[no-untyped-def]
            return "x"

        def fingerprint(self) -> str:
            return "fp"

    registry.register("dummy", DummyTemplate)
    assert registry.exists("dummy")
    assert registry.get("dummy") is DummyTemplate
    registry.freeze()
    with pytest.raises(RegistryError):
        registry.register("other", DummyTemplate)


def test_global_chat_template_registry_exists() -> None:
    assert chat_template_registry.name == "chat_templates"
