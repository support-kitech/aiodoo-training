"""Tokenization fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from aiodoo_training.domain.examples import TokenizationConfig


def fingerprint_masking_policy(*, mask_prompt: bool, ignore_index: int) -> str:
    payload = json.dumps(
        {"mask_prompt": mask_prompt, "ignore_index": ignore_index},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def fingerprint_tokenizer_identity(name: str, *, vocab_size: int, extra: str = "") -> str:
    material = f"tokenizer={name}|vocab={vocab_size}|extra={extra}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def fingerprint_tokenization_config(config: TokenizationConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
