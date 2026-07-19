"""B2 ecosystem contract tests for Capability Packages.

Always-on tests encode frozen consumer rules without importing siblings.

Optional sibling tests import ``aiodoo-validation`` / ``aiodoo-model`` from the
workspace when present — proving live compatibility without making them
package dependencies of ``aiodoo-training``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from aiodoo_training import __version__ as TRAINING_VERSION
from aiodoo_training.artifacts.output_layout import ArtifactOutputLayout
from aiodoo_training.artifacts.output_manager import ArtifactOutputManager
from aiodoo_training.artifacts.publish_contract import (
    ADAPTER_PROTOCOL_ARTIFACT_TYPE,
    ARTIFACT_METADATA_FILENAME,
    BASE_PROTOCOL_ARTIFACT_TYPE,
    MERGED_PROTOCOL_ARTIFACT_TYPE,
    build_adapter_artifact_json,
    build_base_model_artifact_json,
    build_merged_artifact_json,
)
from aiodoo_training.naming import TRAINING_IDS

FIXTURES = (
    Path(__file__).resolve().parents[1] / "fixtures" / "capability_packages" / "protocol" / "v1"
)
ECOSYSTEM_ROOT = Path(__file__).resolve().parents[3]
VALIDATION_ROOT = ECOSYSTEM_ROOT / "aiodoo-validation"
MODEL_ROOT = ECOSYSTEM_ROOT / "aiodoo-model"

# Volatile fields omitted from committed goldens (see fixtures README).
_GOLDEN_STRIP = frozenset({"training_version", "producer", "source_checkpoint", "source_bundle"})

# Frozen protocol kinds accepted by aiodoo-validation ArtifactType enum.
_VALIDATION_ARTIFACT_TYPES = frozenset({"base_model", "coding_adapter", "merged_model"})

# Required adapter fields for self-describing model normalize (without request overlays).
_ADAPTER_REQUIRED_FOR_MODEL = frozenset(
    {
        "artifact_type",
        "protocol_major",
        "identifier",
        "capability_id",
        "adapter_type",
        "peft_type",
        "model_family",
        "architecture",
        "supported_odoo_versions",
        "created_at",
    }
)

_BASE_REQUIRED_FOR_MODEL = frozenset(
    {
        "artifact_type",
        "protocol_major",
        "identifier",
        "model_family",
        "architecture",
        "created_at",
    }
)


def _resolved(capability: str) -> dict[str, Any]:
    return {
        "experiment": {"id": capability},
        "datasets": [{"path": f"{capability}.jsonl", "dataset_type": capability}],
        "model": {"family": "qwen", "base_model": "Qwen/Qwen3-8B", "identifier": "Qwen/Qwen3-8B"},
        "adaptation": {"adapter_type": "qlora", "strategy": "qlora"},
        "dataset_version": "v1.0.0",
    }


def _load_golden(*parts: str) -> dict[str, Any]:
    path = FIXTURES.joinpath(*parts) / "artifact.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _stable_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in payload.items() if k not in _GOLDEN_STRIP}


def _assert_created_at_utc_z(value: object) -> None:
    assert isinstance(value, str)
    assert value.endswith("Z")
    assert "T" in value


# ---------------------------------------------------------------------------
# Always-on: golden package structure
# ---------------------------------------------------------------------------


def test_golden_base_model_shape() -> None:
    golden = _load_golden("base_models", "qwen3-8b")
    assert golden["artifact_type"] == BASE_PROTOCOL_ARTIFACT_TYPE
    assert golden["protocol_major"] == 1
    assert _BASE_REQUIRED_FOR_MODEL <= set(golden)
    assert golden["artifact_type"] in _VALIDATION_ARTIFACT_TYPES
    assert "training_version" not in golden  # stripped volatile field


def test_golden_adapter_coding_shape() -> None:
    golden = _load_golden("adapters", "coding")
    assert golden["artifact_type"] == ADAPTER_PROTOCOL_ARTIFACT_TYPE
    assert golden["capability_id"] == "coding"
    assert golden["adapter_type"] == "coding"
    assert golden["peft_type"] == "qlora"
    assert _ADAPTER_REQUIRED_FOR_MODEL <= set(golden)


def test_golden_adapter_repair_not_collapsed_to_coding() -> None:
    golden = _load_golden("adapters", "repair")
    assert golden["artifact_type"] == ADAPTER_PROTOCOL_ARTIFACT_TYPE
    assert golden["capability_id"] == "repair"
    assert golden["adapter_type"] == "repair"
    assert golden["capability_id"] != "coding"


def test_golden_merged_shape() -> None:
    golden = _load_golden("merged", "coding")
    assert golden["artifact_type"] == MERGED_PROTOCOL_ARTIFACT_TYPE
    assert golden["capability_id"] == "coding"
    assert golden["artifact_type"] in _VALIDATION_ARTIFACT_TYPES
    assert "peft_type" not in golden


def test_representative_catalog_is_intentionally_small() -> None:
    """Guard against accidental exhaustive JSON catalogs."""
    adapters = sorted(p.name for p in (FIXTURES / "adapters").iterdir() if p.is_dir())
    merged = sorted(p.name for p in (FIXTURES / "merged").iterdir() if p.is_dir())
    bases = sorted(p.name for p in (FIXTURES / "base_models").iterdir() if p.is_dir())
    assert adapters == ["coding", "repair"]
    assert merged == ["coding"]
    assert bases == ["qwen3-8b"]


def test_live_builders_match_golden_stable_fields() -> None:
    """Builders emit golden protocol fields; volatile producer fields may differ."""
    base = build_base_model_artifact_json(
        model_id="Qwen/Qwen3-8B",
        model_family="qwen",
        architecture="qwen",
        created_at="2026-07-19T12:00:00Z",
    )
    assert _BASE_REQUIRED_FOR_MODEL <= set(base)
    assert base["training_version"] == TRAINING_VERSION
    assert _stable_view(base) == _load_golden("base_models", "qwen3-8b")

    adapter = build_adapter_artifact_json(
        experiment_id="aiodoo-coding",
        resolved=_resolved("coding"),
        created_at="2026-07-19T12:00:00Z",
    )
    assert _ADAPTER_REQUIRED_FOR_MODEL <= set(adapter)
    assert _stable_view(adapter) == _load_golden("adapters", "coding")

    repair = build_adapter_artifact_json(
        experiment_id="aiodoo-repair",
        resolved=_resolved("repair"),
        created_at="2026-07-19T12:00:00Z",
    )
    assert _stable_view(repair) == _load_golden("adapters", "repair")

    merged = build_merged_artifact_json(
        experiment_id="aiodoo-coding",
        resolved=_resolved("coding"),
        created_at="2026-07-19T12:00:00Z",
    )
    assert _stable_view(merged) == _load_golden("merged", "coding")


# ---------------------------------------------------------------------------
# Always-on: cross-capability protocol stability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("capability", list(TRAINING_IDS))
def test_every_capability_emits_stable_protocol(capability: str) -> None:
    payload = build_adapter_artifact_json(
        experiment_id=f"aiodoo-{capability}",
        resolved=_resolved(capability),
        created_at="2026-07-19T12:00:00Z",
    )
    assert payload["artifact_type"] == ADAPTER_PROTOCOL_ARTIFACT_TYPE
    assert payload["artifact_type"] in _VALIDATION_ARTIFACT_TYPES
    assert payload["protocol_major"] == 1
    assert isinstance(payload["protocol_major"], int)
    assert payload["capability_id"] == capability
    assert payload["adapter_type"] == capability
    assert payload["peft_type"] in {"lora", "qlora", "full"}
    assert payload["supported_odoo_versions"]
    assert all(isinstance(v, int) for v in payload["supported_odoo_versions"])
    _assert_created_at_utc_z(payload["created_at"])
    assert _ADAPTER_REQUIRED_FOR_MODEL <= set(payload)


@pytest.mark.parametrize("capability", list(TRAINING_IDS))
def test_published_layout_consistent(capability: str, tmp_path: Path) -> None:
    workspace = tmp_path / "AIODOO"
    workspace.mkdir()
    resolved = _resolved(capability)
    resolved["workspace"] = {"layout": "drive_v1", "root": str(workspace)}
    manager = ArtifactOutputManager(
        layout=ArtifactOutputLayout.for_training(workspace, capability),
        resolved=resolved,
    )
    ckpt = workspace / "training" / "cache" / capability / "checkpoints" / "checkpoint-1"
    ckpt.mkdir(parents=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (ckpt / "adapter_model.safetensors").write_text("w", encoding="utf-8")
    (ckpt / "rng.json").write_text("{}", encoding="utf-8")

    dest = manager.publish_adapter_from_checkpoint(ckpt)
    assert dest is not None
    assert dest == workspace / "models" / "adapters" / f"aiodoo-{capability}"
    assert (dest / "adapter_config.json").is_file()
    assert (dest / "adapter_model.safetensors").is_file()
    assert not (dest / "rng.json").exists()
    assert (dest / ARTIFACT_METADATA_FILENAME).is_file()
    assert (dest / "manifest.json").is_file()

    artifact = json.loads((dest / ARTIFACT_METADATA_FILENAME).read_text(encoding="utf-8"))
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert artifact["capability_id"] == capability
    assert artifact["adapter_type"] == capability
    assert artifact["artifact_type"] == ADAPTER_PROTOCOL_ARTIFACT_TYPE
    assert manifest["capability_id"] == capability
    assert manifest["training_id"] == capability


def test_backward_compat_old_minimal_artifact_still_has_protocol_fields() -> None:
    """Pre-B1 minimal coding package shape remains a valid validation protocol kind."""
    legacy = {
        "artifact_type": "coding_adapter",
        "protocol_major": 1,
        "identifier": "aiodoo-coding",
        "adapter_type": "coding",
    }
    assert legacy["artifact_type"] in _VALIDATION_ARTIFACT_TYPES
    assert isinstance(legacy["protocol_major"], int)


def test_merged_not_self_describing_for_model_deps() -> None:
    """Merged packages intentionally omit registry dependency ids (caller supplies)."""
    payload = build_merged_artifact_json(
        experiment_id="aiodoo-coding",
        resolved=_resolved("coding"),
        created_at="2026-07-19T12:00:00Z",
    )
    assert "base_artifact_id" not in payload
    assert "adapter_artifact_ids" not in payload
    assert payload["artifact_type"] == MERGED_PROTOCOL_ARTIFACT_TYPE
    assert payload["capability_id"] == "coding"


# ---------------------------------------------------------------------------
# Optional: live sibling consumers
# ---------------------------------------------------------------------------


def _maybe_sys_path(root: Path) -> None:
    if root.is_dir() and str(root) not in sys.path:
        sys.path.insert(0, str(root))


@pytest.mark.skipif(not VALIDATION_ROOT.is_dir(), reason="aiodoo-validation sibling missing")
@pytest.mark.parametrize("capability", [c for c in TRAINING_IDS if c != "context"])
def test_live_validation_resolve_and_profile(capability: str, tmp_path: Path) -> None:
    _maybe_sys_path(VALIDATION_ROOT)
    from aiodoo_validation.domain.artifacts import ArtifactBundle
    from aiodoo_validation.domain.enums import ArtifactType, FingerprintPolicy
    from aiodoo_validation.domain.request import ValidationRequest
    from aiodoo_validation.profiles.adapter_profile import (
        validate_adapter_profile_compatibility,
    )
    from aiodoo_validation.resolution.common import resolve_descriptor
    from aiodoo_validation.resolution.fingerprint import PlaceholderFingerprintProvider

    base_dir = tmp_path / "base"
    base_dir.mkdir()
    (base_dir / "weights.bin").write_bytes(b"b")
    (base_dir / "artifact.json").write_text(
        json.dumps(
            build_base_model_artifact_json(
                model_id="Qwen/Qwen3-8B",
                model_family="qwen",
                architecture="qwen",
                created_at="2026-07-19T12:00:00Z",
            )
        ),
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter_dir / "adapter_model.bin").write_bytes(b"a")
    (adapter_dir / "artifact.json").write_text(
        json.dumps(
            build_adapter_artifact_json(
                experiment_id=f"aiodoo-{capability}",
                resolved=_resolved(capability),
                created_at="2026-07-19T12:00:00Z",
            )
        ),
        encoding="utf-8",
    )

    req = ValidationRequest(
        profile_name=capability,
        base_model_ref=str(base_dir),
        adapter_ref=str(adapter_dir),
        protocol_major=1,
        protocol_minor=0,
        fingerprint_policy=FingerprintPolicy.OFF,
    )
    fp = PlaceholderFingerprintProvider()
    base_desc, base_err, _ = resolve_descriptor(
        logical_ref="base",
        path_ref=str(base_dir),
        expected_type=ArtifactType.BASE_MODEL,
        request=req,
        fingerprint_provider=fp,
        fingerprint_policy=FingerprintPolicy.OFF,
    )
    adapter_desc, adapter_err, _ = resolve_descriptor(
        logical_ref="adapter",
        path_ref=str(adapter_dir),
        expected_type=ArtifactType.CODING_ADAPTER,
        request=req,
        fingerprint_provider=fp,
        fingerprint_policy=FingerprintPolicy.OFF,
    )
    assert base_desc is not None and not base_err
    assert adapter_desc is not None and not adapter_err
    bundle = ArtifactBundle(
        base_model=base_desc,
        adapter=adapter_desc,
        merged_model=None,
        protocol_major=1,
        protocol_minor=0,
        fingerprint_policy=FingerprintPolicy.OFF,
        bundle_digest="x",
    )
    errors = validate_adapter_profile_compatibility(bundle, profile_name=capability)
    assert errors == ()


@pytest.mark.skipif(not MODEL_ROOT.is_dir(), reason="aiodoo-model sibling missing")
@pytest.mark.parametrize("capability", list(TRAINING_IDS))
def test_live_model_publish_adapter(capability: str, tmp_path: Path) -> None:
    _maybe_sys_path(MODEL_ROOT)
    from aiodoo_model.publishing import PublishingRequest, PublishingService
    from aiodoo_model.registry import FileBackedRegistry
    from aiodoo_model.storage import StorageManager

    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "adapter_config.json").write_text("{}", encoding="utf-8")
    (pkg / "adapter_model.bin").write_bytes(b"w")
    (pkg / "artifact.json").write_text(
        json.dumps(
            build_adapter_artifact_json(
                experiment_id=f"aiodoo-{capability}",
                resolved=_resolved(capability),
                created_at="2026-07-19T12:00:00Z",
            )
        ),
        encoding="utf-8",
    )
    service = PublishingService(
        FileBackedRegistry(tmp_path / "registry"),
        StorageManager.create_default(tmp_path / "store"),
    )
    result = service.publish(
        PublishingRequest(
            source_path=str(pkg),
            artifact_id=f"art-{capability}",
            family_id="qwen3-8b",
            base_artifact_id="art-base-1",
        )
    )
    assert result.artifact.adapter_spec is not None
    assert result.artifact.adapter_spec.capability_id == capability
    assert result.artifact.adapter_spec.adapter_type == "qlora"


@pytest.mark.skipif(not MODEL_ROOT.is_dir(), reason="aiodoo-model sibling missing")
def test_live_model_merged_requires_request_deps(tmp_path: Path) -> None:
    _maybe_sys_path(MODEL_ROOT)
    from aiodoo_model.exceptions import PublishNormalizeError
    from aiodoo_model.publishing import PublishingRequest, PublishingService
    from aiodoo_model.registry import FileBackedRegistry
    from aiodoo_model.storage import StorageManager

    pkg = tmp_path / "merged"
    pkg.mkdir()
    (pkg / "weights.bin").write_bytes(b"m")
    (pkg / "artifact.json").write_text(
        json.dumps(
            build_merged_artifact_json(
                experiment_id="aiodoo-coding",
                resolved=_resolved("coding"),
                created_at="2026-07-19T12:00:00Z",
            )
        ),
        encoding="utf-8",
    )
    service = PublishingService(
        FileBackedRegistry(tmp_path / "registry"),
        StorageManager.create_default(tmp_path / "store"),
    )
    with pytest.raises(PublishNormalizeError, match="base_artifact_id"):
        service.publish(
            PublishingRequest(
                source_path=str(pkg),
                artifact_id="art-merged",
                family_id="qwen3-8b",
            )
        )
    result = service.publish(
        PublishingRequest(
            source_path=str(pkg),
            artifact_id="art-merged-2",
            family_id="qwen3-8b",
            base_artifact_id="art-base-1",
            adapter_artifact_ids=("art-coding",),
        )
    )
    assert result.artifact.kind.value == "merged"


@pytest.mark.skipif(not MODEL_ROOT.is_dir(), reason="aiodoo-model sibling missing")
def test_live_model_without_capability_id_collapses_to_coding(tmp_path: Path) -> None:
    """Regression guard: capability_id is mandatory to avoid model fallback to coding."""
    _maybe_sys_path(MODEL_ROOT)
    from aiodoo_model.publishing import PublishingRequest, PublishingService
    from aiodoo_model.registry import FileBackedRegistry
    from aiodoo_model.storage import StorageManager

    meta = build_adapter_artifact_json(
        experiment_id="aiodoo-repair",
        resolved=_resolved("repair"),
        created_at="2026-07-19T12:00:00Z",
    )
    del meta["capability_id"]
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "adapter_config.json").write_text("{}", encoding="utf-8")
    (pkg / "adapter_model.bin").write_bytes(b"w")
    (pkg / "artifact.json").write_text(json.dumps(meta), encoding="utf-8")
    service = PublishingService(
        FileBackedRegistry(tmp_path / "registry"),
        StorageManager.create_default(tmp_path / "store"),
    )
    result = service.publish(
        PublishingRequest(
            source_path=str(pkg),
            artifact_id="art-strip",
            family_id="qwen3-8b",
        )
    )
    assert result.artifact.adapter_spec is not None
    assert result.artifact.adapter_spec.capability_id == "coding"
