"""FW0 smoke: training configs bind frozen dual foundations."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "configs" / "training"

DEVELOPMENT_HUB = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"
REASONING_HUB = "deepseek-ai/deepseek-vl2"

DEV_CAPS = ("coding", "repair", "execution", "context")
REASON_CAPS = ("planner", "conversation", "approval", "evaluation")


def _load_identifier(capability: str) -> str:
    path = CONFIG_ROOT / capability / "model.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return str(payload["model"]["identifier"])


def test_development_capabilities_use_coder_foundation() -> None:
    for cap in DEV_CAPS:
        assert _load_identifier(cap) == DEVELOPMENT_HUB


def test_reasoning_capabilities_use_vl2_foundation() -> None:
    for cap in REASON_CAPS:
        assert _load_identifier(cap) == REASONING_HUB


def test_no_r1_or_qwen3_8b_foundation_defaults() -> None:
    for cap in (*DEV_CAPS, *REASON_CAPS):
        text = (CONFIG_ROOT / cap / "model.yaml").read_text(encoding="utf-8")
        assert "Qwen/Qwen3-8B" not in text
        assert "DeepSeek-R1" not in text
        assert "future_models" not in text
