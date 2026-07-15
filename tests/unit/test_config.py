"""Config hashing, composition, and identity tests."""

from pathlib import Path

import pytest

from aiodoo_training.config import ConfigComposer, ConfigHasher, ConfigSystem, deep_merge
from aiodoo_training.exceptions import ConfigError

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "configs" / "experiments" / "example.yaml"


def test_canonical_hash_is_order_independent() -> None:
    hasher = ConfigHasher()
    data = {"name": "demo", "seed": 42, "nested": {"a": 1, "b": 2}}
    reordered = {"nested": {"b": 2, "a": 1}, "seed": 42, "name": "demo"}
    assert hasher.hash(data) == hasher.hash(reordered)


def test_hash_changes_when_values_change() -> None:
    hasher = ConfigHasher()
    assert hasher.hash({"seed": 1}) != hasher.hash({"seed": 2})


def test_non_serializable_config_raises() -> None:
    hasher = ConfigHasher()
    with pytest.raises(ConfigError, match="not canonically serializable"):
        hasher.hash({"bad": {1, 2, 3}})  # type: ignore[dict-item]


def test_experiment_id_prefix() -> None:
    hasher = ConfigHasher()
    eid = hasher.experiment_id({"name": "x", "seed": 1})
    assert eid.value.startswith("exp_")
    assert len(eid.value) == len("exp_") + 16


def test_deep_merge_nested_dicts() -> None:
    merged = deep_merge({"a": 1, "nested": {"x": 1}}, {"b": 2, "nested": {"y": 2}})
    assert merged == {"a": 1, "b": 2, "nested": {"x": 1, "y": 2}}


def test_compose_example_experiment() -> None:
    composed = ConfigComposer().compose(EXAMPLE)
    assert composed["name"] == "example-phase0"
    assert composed["schema_version"] == "1.0"
    assert "model" in composed
    assert composed["execution"]["device"]["preferred"] == "auto"
    assert composed["distributed"]["enabled"] is False


def test_load_experiment_identity_uses_composed_not_absolute_paths(tmp_path: Path) -> None:
    """Absolute path resolution must not change ExperimentId."""
    fragment = tmp_path / "frag.yaml"
    fragment.write_text("seed: 7\n", encoding="utf-8")
    experiment = tmp_path / "exp.yaml"
    experiment.write_text(
        "schema_version: '1.0'\n"
        "name: portable\n"
        "include:\n"
        "  - frag.yaml\n"
        "checkpointing:\n"
        "  output_dir: relative/out\n",
        encoding="utf-8",
    )

    system = ConfigSystem()
    _model, experiment_id, resolved = system.load_experiment(experiment)
    assert experiment_id.value.startswith("exp_")
    assert Path(resolved["checkpointing"]["output_dir"]).is_absolute()

    # Re-hash composed manually — must equal experiment_id derivation.
    composed = system.composer.compose(experiment)
    assert system.hasher.experiment_id(composed) == experiment_id


def test_missing_config_raises() -> None:
    with pytest.raises(ConfigError, match="not found"):
        ConfigSystem().load_experiment(Path("/tmp/does-not-exist-aiodoo.yaml"))


def test_circular_include_detected(tmp_path: Path) -> None:
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    a.write_text(
        "include:\n  - b.yaml\nname: a\nschema_version: '1.0'\n",
        encoding="utf-8",
    )
    b.write_text("include:\n  - a.yaml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="Circular"):
        ConfigComposer().compose(a)
