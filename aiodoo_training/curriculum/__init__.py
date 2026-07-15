"""Curriculum lifecycle, context, and strategies."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from aiodoo_training.domain.config import CurriculumSpec
from aiodoo_training.domain.curriculum_session import CurriculumSession
from aiodoo_training.domain.enums import CurriculumMode, CurriculumStatus
from aiodoo_training.domain.examples import TrainingExample
from aiodoo_training.exceptions import CurriculumLifecycleError
from aiodoo_training.ports.packing import CurriculumStrategy
from aiodoo_training.registries import curriculum_registry

_VALID: dict[CurriculumStatus, frozenset[CurriculumStatus]] = {
    CurriculumStatus.PENDING: frozenset({CurriculumStatus.PLANNING, CurriculumStatus.SKIPPED}),
    CurriculumStatus.PLANNING: frozenset({CurriculumStatus.READY, CurriculumStatus.FAILED}),
    CurriculumStatus.READY: frozenset({CurriculumStatus.ACTIVE}),
    CurriculumStatus.ACTIVE: frozenset({CurriculumStatus.COMPLETED}),
    CurriculumStatus.FAILED: frozenset({CurriculumStatus.PENDING}),
    CurriculumStatus.COMPLETED: frozenset(),
    CurriculumStatus.SKIPPED: frozenset(),
}


class CurriculumLifecycle:
    def _transition(
        self,
        session: CurriculumSession,
        target: CurriculumStatus,
        *,
        message: str | None = None,
    ) -> CurriculumSession:
        allowed = _VALID.get(session.status, frozenset())
        if target not in allowed:
            raise CurriculumLifecycleError(
                f"Cannot transition from {session.status.value!r} to {target.value!r}."
            )
        return session.with_status(target, message=message)

    def begin(self, session: CurriculumSession, *, message: str | None = None) -> CurriculumSession:
        return self._transition(session, CurriculumStatus.PLANNING, message=message)

    def skip(self, session: CurriculumSession, *, message: str | None = None) -> CurriculumSession:
        return self._transition(session, CurriculumStatus.SKIPPED, message=message)

    def ready(self, session: CurriculumSession, *, message: str | None = None) -> CurriculumSession:
        return self._transition(session, CurriculumStatus.READY, message=message)

    def activate(
        self, session: CurriculumSession, *, message: str | None = None
    ) -> CurriculumSession:
        return self._transition(session, CurriculumStatus.ACTIVE, message=message)

    def complete(
        self, session: CurriculumSession, *, message: str | None = None
    ) -> CurriculumSession:
        return self._transition(session, CurriculumStatus.COMPLETED, message=message)

    def fail(self, session: CurriculumSession, *, message: str | None = None) -> CurriculumSession:
        return self._transition(session, CurriculumStatus.FAILED, message=message)


@dataclass(frozen=True, slots=True)
class CurriculumContext:
    curriculum_session: CurriculumSession
    curriculum_spec: CurriculumSpec
    seed: int = 42
    backend_key: str = "none"
    bind_extra: dict[str, Any] = field(default_factory=dict)

    def with_session(self, session: CurriculumSession) -> CurriculumContext:
        return replace(self, curriculum_session=session)


def _difficulty(example: TrainingExample) -> float:
    raw = example.metadata.get("difficulty")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            pass
    content_len = sum(len(str(m.get("content", ""))) for m in example.messages)
    return float(content_len)


def _seeded_rng(seed: int) -> random.Random:
    return random.Random(seed)


class NoneCurriculum(CurriculumStrategy):
    BACKEND_KEY = "none"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> NoneCurriculum:
        self._context = context
        return self

    def plan(
        self, examples: Sequence[TrainingExample], spec: CurriculumSpec
    ) -> Sequence[Sequence[TrainingExample]]:
        _ = spec
        return (tuple(examples),)


class SequentialCurriculum(CurriculumStrategy):
    BACKEND_KEY = "sequential"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> SequentialCurriculum:
        self._context = context
        return self

    def plan(
        self, examples: Sequence[TrainingExample], spec: CurriculumSpec
    ) -> Sequence[Sequence[TrainingExample]]:
        stages = spec.stages or ("all",)
        if stages == ("all",) or not stages:
            return (tuple(examples),)
        buckets: dict[str, list[TrainingExample]] = {s: [] for s in stages}
        other: list[TrainingExample] = []
        for example in examples:
            stage = str(example.metadata.get("stage", ""))
            if stage in buckets:
                buckets[stage].append(example)
            else:
                other.append(example)
        # Equal chunk overflow into stages by sorted example_id order
        if other:
            other_sorted = sorted(other, key=lambda e: e.example_id)
            n = len(stages)
            for index, example in enumerate(other_sorted):
                buckets[stages[index % n]].append(example)
        return tuple(tuple(buckets[s]) for s in stages)


class WeightedCurriculum(CurriculumStrategy):
    BACKEND_KEY = "weighted"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> WeightedCurriculum:
        self._context = context
        return self

    def plan(
        self, examples: Sequence[TrainingExample], spec: CurriculumSpec
    ) -> Sequence[Sequence[TrainingExample]]:
        stages = spec.stages or ("all",)
        if len(stages) <= 1:
            return (tuple(examples),)
        seed = int(getattr(self._context, "seed", 42) if self._context else 42)
        rng = _seeded_rng(seed)
        # Assign each example to a stage proportional to weight metadata or uniform.
        result: dict[str, list[TrainingExample]] = {s: [] for s in stages}
        for example in sorted(examples, key=lambda e: e.example_id):
            weights = []
            for stage in stages:
                w = example.metadata.get(f"weight_{stage}")
                if w is None:
                    w = example.metadata.get("weight", 1.0)
                try:
                    weights.append(max(float(w), 0.0))
                except (TypeError, ValueError):
                    weights.append(1.0)
            total = sum(weights) or 1.0
            pick = rng.random() * total
            acc = 0.0
            chosen = stages[-1]
            for stage, weight in zip(stages, weights, strict=True):
                acc += weight
                if pick <= acc:
                    chosen = stage
                    break
            result[chosen].append(example)
        return tuple(tuple(result[s]) for s in stages)


class DifficultyCurriculum(CurriculumStrategy):
    BACKEND_KEY = "difficulty"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> DifficultyCurriculum:
        self._context = context
        return self

    def plan(
        self, examples: Sequence[TrainingExample], spec: CurriculumSpec
    ) -> Sequence[Sequence[TrainingExample]]:
        ordered = sorted(examples, key=lambda e: (_difficulty(e), e.example_id))
        stages = spec.stages
        if not stages or len(stages) == 1:
            return (tuple(ordered),)
        # Equal contiguous bins by sorted difficulty
        n = len(ordered)
        k = len(stages)
        out: list[tuple[TrainingExample, ...]] = []
        for i, _name in enumerate(stages):
            start = (i * n) // k
            end = ((i + 1) * n) // k
            out.append(tuple(ordered[start:end]))
        return tuple(out)


class RandomCurriculum(CurriculumStrategy):
    BACKEND_KEY = "random"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context

    def bind(self, context: Any) -> RandomCurriculum:
        self._context = context
        return self

    def plan(
        self, examples: Sequence[TrainingExample], spec: CurriculumSpec
    ) -> Sequence[Sequence[TrainingExample]]:
        seed = int(getattr(self._context, "seed", 42) if self._context else 42)
        rng = _seeded_rng(seed)
        items = list(examples)
        items.sort(key=lambda e: e.example_id)
        rng.shuffle(items)
        stages = spec.stages
        if not stages or len(stages) == 1:
            return (tuple(items),)
        n = len(items)
        k = len(stages)
        return tuple(tuple(items[(i * n) // k : ((i + 1) * n) // k]) for i in range(k))


class MixedCurriculum(CurriculumStrategy):
    BACKEND_KEY = "mixed"

    def __init__(self, context: Any | None = None) -> None:
        self._context = context
        self._sequential = SequentialCurriculum(context)
        self._weighted = WeightedCurriculum(context)

    def bind(self, context: Any) -> MixedCurriculum:
        self._context = context
        self._sequential.bind(context)
        self._weighted.bind(context)
        return self

    def plan(
        self, examples: Sequence[TrainingExample], spec: CurriculumSpec
    ) -> Sequence[Sequence[TrainingExample]]:
        # Stage membership via sequential tags; order within stages via weighted RNG order.
        stages = self._sequential.plan(examples, spec)
        seed = int(getattr(self._context, "seed", 42) if self._context else 42)
        out: list[tuple[TrainingExample, ...]] = []
        for index, stage_examples in enumerate(stages):
            rng = _seeded_rng(seed + index)
            items = list(stage_examples)
            items.sort(key=lambda e: e.example_id)

            def weight_of(example: TrainingExample) -> float:
                try:
                    return max(float(example.metadata.get("weight", 1.0)), 0.0)
                except (TypeError, ValueError):
                    return 1.0

            # Deterministic weighted order without replacement (Gumbel-max style)
            scored = []
            for example in items:
                w = weight_of(example)
                u = rng.random()
                g = -math.log(max(u, 1e-12))
                key = (g / max(w, 1e-12), example.example_id)
                scored.append((key, example))
            scored.sort(key=lambda x: x[0])
            out.append(tuple(e for _, e in scored))
        return tuple(out)


def register_default_curriculum(*, overwrite: bool = False) -> None:
    if not curriculum_registry.exists("none") or overwrite:
        curriculum_registry.register("none", NoneCurriculum, overwrite=overwrite)
    if not curriculum_registry.exists("sequential") or overwrite:
        curriculum_registry.register("sequential", SequentialCurriculum, overwrite=overwrite)
    if not curriculum_registry.exists("weighted") or overwrite:
        curriculum_registry.register("weighted", WeightedCurriculum, overwrite=overwrite)
    if not curriculum_registry.exists("weighted_mix") or overwrite:
        curriculum_registry.register("weighted_mix", WeightedCurriculum, overwrite=overwrite)
    if not curriculum_registry.exists("difficulty") or overwrite:
        curriculum_registry.register("difficulty", DifficultyCurriculum, overwrite=overwrite)
    if not curriculum_registry.exists("random") or overwrite:
        curriculum_registry.register("random", RandomCurriculum, overwrite=overwrite)
    if not curriculum_registry.exists("mixed") or overwrite:
        curriculum_registry.register("mixed", MixedCurriculum, overwrite=overwrite)


def curriculum_key_for_mode(mode: CurriculumMode) -> str:
    if mode == CurriculumMode.WEIGHTED_MIX:
        return "weighted"
    return mode.value


def fingerprint_stages(stages: Sequence[Sequence[TrainingExample]]) -> str:
    material = "||".join(",".join(e.example_id for e in stage) for stage in stages)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()
