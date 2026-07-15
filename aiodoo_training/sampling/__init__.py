"""Phase 5 sampling strategies."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from aiodoo_training.domain.examples import TrainingExample
from aiodoo_training.domain.packing_policies import SamplingSpec
from aiodoo_training.ports.packing import SamplingStrategy
from aiodoo_training.registries import sampling_registry


def _rng(seed: int | None) -> random.Random:
    return random.Random(42 if seed is None else seed)


def _weight(example: TrainingExample) -> float:
    try:
        return max(float(example.metadata.get("weight", 1.0)), 0.0)
    except (TypeError, ValueError):
        return 1.0


class IdentitySampling(SamplingStrategy):
    BACKEND_KEY = "identity"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> IdentitySampling:
        self._context = context
        return self

    def sample(
        self, examples: Sequence[TrainingExample], spec: SamplingSpec
    ) -> Sequence[TrainingExample]:
        _ = spec
        return tuple(examples)


class WeightedSampling(SamplingStrategy):
    BACKEND_KEY = "weighted"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> WeightedSampling:
        self._context = context
        return self

    def sample(
        self, examples: Sequence[TrainingExample], spec: SamplingSpec
    ) -> Sequence[TrainingExample]:
        rng = _rng(spec.seed)
        items = list(examples)
        items.sort(key=lambda e: e.example_id)
        scored: list[tuple[tuple[float, str], TrainingExample]] = []
        for example in items:
            w = _weight(example)
            u = rng.random()
            g = -math.log(max(u, 1e-12))
            scored.append(((g / max(w, 1e-12), example.example_id), example))
        scored.sort(key=lambda x: x[0])
        return tuple(e for _, e in scored)


class TemperatureSampling(SamplingStrategy):
    BACKEND_KEY = "temperature"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> TemperatureSampling:
        self._context = context
        return self

    def sample(
        self, examples: Sequence[TrainingExample], spec: SamplingSpec
    ) -> Sequence[TrainingExample]:
        rng = _rng(spec.seed)
        tau = spec.temperature
        items = list(examples)
        items.sort(key=lambda e: e.example_id)
        logits = [_weight(e) / tau for e in items]
        max_logit = max(logits) if logits else 0.0
        exps = [math.exp(x - max_logit) for x in logits]
        scored = []
        for example, mass in zip(items, exps, strict=True):
            u = rng.random()
            g = -math.log(max(u, 1e-12))
            scored.append(((g / max(mass, 1e-12), example.example_id), example))
        scored.sort(key=lambda x: x[0])
        return tuple(e for _, e in scored)


class BalancedSampling(SamplingStrategy):
    BACKEND_KEY = "balanced"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> BalancedSampling:
        self._context = context
        return self

    def sample(
        self, examples: Sequence[TrainingExample], spec: SamplingSpec
    ) -> Sequence[TrainingExample]:
        strata_key = spec.strata_key
        buckets: dict[str, list[TrainingExample]] = {}
        for example in examples:
            key = str(example.metadata.get(strata_key, example.dataset_type.value))
            buckets.setdefault(key, []).append(example)
        for key in buckets:
            buckets[key].sort(key=lambda e: e.example_id)
        keys = sorted(buckets.keys())
        pointers = {k: 0 for k in keys}
        out: list[TrainingExample] = []
        remaining = len(examples)
        while remaining > 0:
            progressed = False
            for key in keys:
                idx = pointers[key]
                bucket = buckets[key]
                if idx < len(bucket):
                    out.append(bucket[idx])
                    pointers[key] = idx + 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break
        return tuple(out)


def register_default_sampling(*, overwrite: bool = False) -> None:
    if not sampling_registry.exists("identity") or overwrite:
        sampling_registry.register("identity", IdentitySampling, overwrite=overwrite)
    if not sampling_registry.exists("weighted") or overwrite:
        sampling_registry.register("weighted", WeightedSampling, overwrite=overwrite)
    if not sampling_registry.exists("temperature") or overwrite:
        sampling_registry.register("temperature", TemperatureSampling, overwrite=overwrite)
    if not sampling_registry.exists("balanced") or overwrite:
        sampling_registry.register("balanced", BalancedSampling, overwrite=overwrite)
