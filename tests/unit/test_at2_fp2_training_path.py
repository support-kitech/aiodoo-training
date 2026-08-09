"""AT-2 — FP2 TrainingExample loader, PEFT export, provenance."""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3] if "tests" in Path(__file__).parts else Path(__file__).resolve().parents[2]
from unittest.mock import MagicMock

import pytest

from aiodoo_training.artifacts.publish_contract import build_adapter_artifact_json
from aiodoo_training.bootstrap import bootstrap_phase7
from aiodoo_training.datasets.source import JsonlDatasetSource
from aiodoo_training.domain.enums import DatasetType, ExportType
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.refs import (
    RECORD_FORMAT_FP2_TRAINING_EXAMPLE,
    DatasetRef,
)
from aiodoo_training.exceptions import DomainError
from aiodoo_training.infrastructure.huggingface.exporter import HFExporter
from aiodoo_training.infrastructure.model_handles import (
    OpaqueTrainableModel,
    as_trainable_handle,
)
from aiodoo_training.system_training_contract.version import SYSTEM_TRAINING_CONTRACT_VERSION

BATCH2 = WORKSPACE / "aiodoo-datasets/datasets/fp2/controlled_batch_2"
LEGACY_CODING = WORKSPACE / "aiodoo-datasets/datasets/coding_v1_0.jsonl"


@pytest.fixture(autouse=True)
def _bootstrap() -> None:
    bootstrap_phase7(overwrite=True)


def test_fp2_pack_loads_without_protocol_formatter() -> None:
    pack = BATCH2 / "pack_development.jsonl"
    assert pack.is_file()
    ref = DatasetRef(
        path=pack,
        dataset_type=DatasetType.CODING,
        protocol_version="1.0.0",
        record_format=RECORD_FORMAT_FP2_TRAINING_EXAMPLE,
        name="fp2_dev",
    )
    examples = list(JsonlDatasetSource(validate=True).load([ref]))
    assert len(examples) == 1004
    first = examples[0]
    assert first.messages[0]["role"] == "user"
    assert first.metadata.get("fp2_native") is True
    assert first.metadata.get("record_format") == "fp2_training_example"
    # Must not have been rewritten by Protocol CodingFormatter (no contract projection).
    assert "AIODOO System Training Contract" in first.messages[0]["content"]


def test_fp2_malformed_fails_clearly(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"example_id": "x", "dataset_type": "coding"}) + "\n")
    ref = DatasetRef(
        path=bad,
        dataset_type=DatasetType.CODING,
        protocol_version="1.0.0",
        record_format=RECORD_FORMAT_FP2_TRAINING_EXAMPLE,
    )
    with pytest.raises(DomainError, match="missing keys"):
        list(JsonlDatasetSource(validate=True).load([ref]))


@pytest.mark.skipif(not LEGACY_CODING.is_file(), reason="legacy coding missing")
def test_legacy_protocol_v1_still_loads() -> None:
    ref = DatasetRef(
        path=LEGACY_CODING,
        dataset_type=DatasetType.CODING,
        protocol_version="1.0",
        record_format="protocol_v1",
    )
    examples = list(JsonlDatasetSource(validate=False).load([ref]))
    assert len(examples) >= 1
    assert examples[0].dataset_type == DatasetType.CODING


def test_fp2_provenance_in_artifact_json() -> None:
    checksum = "abc123"
    payload = build_adapter_artifact_json(
        experiment_id="aiodoo-coding",
        resolved={
            "name": "at2-fp2-smoke",
            "model": {"identifier": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"},
            "fp2": {
                "system_training_contract_version": SYSTEM_TRAINING_CONTRACT_VERSION,
                "corpus_version": "fp2-controlled-2.0.0-tr7",
                "corpus_checksum": checksum,
                "source_pack": "pack_development",
                "split": "smoke",
                "smoke_id": "at2-smoke-001",
            },
        },
    )
    assert payload["provider_contract_version"]
    assert payload["contract_version"] == payload["provider_contract_version"]
    assert payload["system_training_contract_version"] == "1.0.0"
    assert payload["corpus_version"] == "fp2-controlled-2.0.0-tr7"
    assert payload["corpus_checksum"] == checksum
    assert payload["foundation_model_id"] == "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
    assert payload["foundation_hub_id"] == payload["foundation_model_id"]
    assert payload["training_config_id"] == "at2-fp2-smoke"


def test_hf_exporter_writes_real_peft_weights(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    pytest.importorskip("transformers")
    pytest.importorskip("torch")

    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import GPT2Config, GPT2LMHeadModel

    from aiodoo_training.domain.config import ExportSpec
    from aiodoo_training.domain.enums import AcceleratorKind, DeviceKind, Precision
    from aiodoo_training.domain.export_session import ExportSession
    from aiodoo_training.domain.resources import (
        DevicePolicy,
        ExecutionEnvironment,
        HardwareCapabilities,
        MemoryPolicy,
        PrecisionPolicy,
    )
    from aiodoo_training.export.context import ExportContext
    from aiodoo_training.training.engine import make_stub_experiment_config

    config = GPT2Config(
        n_layer=1,
        n_head=2,
        n_embd=32,
        n_positions=32,
        vocab_size=128,
        bos_token_id=0,
        eos_token_id=0,
    )
    base = GPT2LMHeadModel(config)
    adapted = get_peft_model(
        base,
        LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["c_attn"],
            task_type=TaskType.CAUSAL_LM,
            fan_in_fan_out=True,
        ),
    )
    model = as_trainable_handle(
        OpaqueTrainableModel(
            framework_model=adapted,
            aiodoo_adapter_metadata=MagicMock(),
            base=None,
            strategy_key="lora",
        )
    )
    tmp = tmp_path / "export_tmp"
    tmp.mkdir()
    execution = ExecutionEnvironment(
        selected_device=DeviceKind.CPU,
        capabilities=HardwareCapabilities(available_devices=(DeviceKind.CPU,), device_count=0),
        device_policy=DevicePolicy(preferred=DeviceKind.CPU, allow_cpu_fallback=True),
        precision_policy=PrecisionPolicy(compute=Precision.FP32),
        memory_policy=MemoryPolicy(),
        accelerator=AcceleratorKind.NONE,
    )
    exp = make_stub_experiment_config(output_dir=tmp_path)
    session = ExportSession(
        session_id="sess-export",
        experiment_id=ExperimentId(value="at2-export-unit"),
        run_id=RunId(value="run1"),
        model_fingerprint="m",
        adapter_fingerprint="a",
        config_fingerprint="c",
    )
    ctx = ExportContext(
        config=exp,
        export_spec=ExportSpec(
            output_dir=tmp_path / "out",
            export_types=(ExportType.PEFT_ADAPTER,),
        ),
        model=model,
        execution=execution,
        export_session=session,
        exporter=HFExporter(),
        output_dir=tmp_path / "out",
        tmp_dir=tmp,
        exporter_backend_key="hf_peft",
        export_types=("peft_adapter",),
        bind_extra={
            "fp2_provenance": {
                "system_training_contract_version": "1.0.0",
                "corpus_version": "fp2-controlled-2.0.0-tr7",
            }
        },
    )
    artifacts = HFExporter().bind(ctx).export(
        model,
        ctx.export_spec,
        ExperimentId(value="at2-export-unit"),
        RunId(value="run1"),
    )
    peft_arts = [a for a in artifacts if a.export_type == ExportType.PEFT_ADAPTER]
    assert peft_arts
    adapter_dir = Path(peft_arts[0].path)
    assert (adapter_dir / "adapter_config.json").is_file()
    weights = list(adapter_dir.glob("adapter_model.*"))
    assert weights and weights[0].stat().st_size > 0
    reloaded = PeftModel.from_pretrained(base, str(adapter_dir))
    assert reloaded is not None
    assert (tmp / "artifacts" / "fp2_provenance.json").is_file()


def test_hf_exporter_rejects_stub_model(tmp_path: Path) -> None:
    from aiodoo_training.domain.config import ExportSpec
    from aiodoo_training.domain.enums import AcceleratorKind, DeviceKind, Precision
    from aiodoo_training.domain.export_session import ExportSession
    from aiodoo_training.domain.resources import (
        DevicePolicy,
        ExecutionEnvironment,
        HardwareCapabilities,
        MemoryPolicy,
        PrecisionPolicy,
    )
    from aiodoo_training.export.context import ExportContext
    from aiodoo_training.training.engine import make_stub_experiment_config

    model = as_trainable_handle(
        OpaqueTrainableModel(
            framework_model={"kind": "stub", "num_parameters": 10},
            aiodoo_adapter_metadata=MagicMock(),
            base=None,
            strategy_key="lora",
        )
    )
    tmp = tmp_path / "t"
    tmp.mkdir()
    exp_cfg = make_stub_experiment_config(output_dir=tmp_path / "ckpts")
    execution = ExecutionEnvironment(
        selected_device=DeviceKind.CPU,
        capabilities=HardwareCapabilities(available_devices=(DeviceKind.CPU,), device_count=0),
        device_policy=DevicePolicy(preferred=DeviceKind.CPU, allow_cpu_fallback=True),
        precision_policy=PrecisionPolicy(compute=Precision.FP32),
        memory_policy=MemoryPolicy(),
        accelerator=AcceleratorKind.NONE,
    )
    ctx = ExportContext(
        config=exp_cfg,
        export_spec=ExportSpec(
            output_dir=tmp_path / "out",
            export_types=(ExportType.PEFT_ADAPTER,),
        ),
        model=model,
        execution=execution,
        export_session=ExportSession(
            session_id="sess",
            experiment_id=ExperimentId(value="x"),
            run_id=RunId(value="r"),
            model_fingerprint="m",
            adapter_fingerprint="a",
            config_fingerprint="c",
        ),
        exporter=HFExporter(),
        output_dir=tmp_path / "out",
        tmp_dir=tmp,
        export_types=("peft_adapter",),
    )
    with pytest.raises(DomainError, match="stub"):
        HFExporter().bind(ctx).export(
            model, ctx.export_spec, ExperimentId(value="x"), RunId(value="r")
        )
