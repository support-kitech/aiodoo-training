"""Phase 5 packing builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from aiodoo_training.domain.config import PackingSpec
from aiodoo_training.domain.enums import PackingMode, PackingOverflow
from aiodoo_training.domain.examples import TrainingExample
from aiodoo_training.domain.identifiers import ExperimentId, RunId
from aiodoo_training.domain.packing_policies import MemoryPackingPolicy, PackingPolicy
from aiodoo_training.domain.packing_session import PackingSession
from aiodoo_training.exceptions import BuilderError
from aiodoo_training.packing.context import PackingContext
from aiodoo_training.packing.token_rows import TokenRow, resolve_token_rows


class PackingBuilder:
    """Assembles :class:`PackingPolicy` / :class:`PackingSpec` from fragments."""

    def __init__(self) -> None:
        self._backend_key = "none"
        self._mode = PackingMode.NONE
        self._max_sequence_length = 2048
        self._max_examples_per_sequence: int | None = None
        self._separator_token_id: int | None = None
        self._overflow: PackingOverflow = PackingOverflow.DEFER
        self._drop_last = False
        self._seed: int | None = 42
        self._pad_to_multiple_of: int | None = None
        self._memory = MemoryPackingPolicy()

    def with_backend(self, key: str) -> PackingBuilder:
        self._backend_key = key
        return self

    def with_mode(self, mode: PackingMode | str) -> PackingBuilder:
        self._mode = PackingMode(mode) if not isinstance(mode, PackingMode) else mode
        return self

    def with_max_sequence_length(self, value: int) -> PackingBuilder:
        self._max_sequence_length = value
        return self

    def with_seed(self, seed: int | None) -> PackingBuilder:
        self._seed = seed
        return self

    def with_memory_policy(self, policy: MemoryPackingPolicy) -> PackingBuilder:
        self._memory = policy
        return self

    def with_policy(self, policy: PackingPolicy) -> PackingBuilder:
        self._backend_key = policy.backend_key
        self._mode = policy.mode
        self._max_sequence_length = policy.max_sequence_length
        self._max_examples_per_sequence = policy.max_examples_per_sequence
        self._separator_token_id = policy.separator_token_id
        self._overflow = policy.overflow
        self._drop_last = policy.drop_last
        self._seed = policy.seed
        self._pad_to_multiple_of = policy.pad_to_multiple_of
        return self

    def build_policy(self) -> PackingPolicy:
        return PackingPolicy(
            backend_key=self._backend_key,
            mode=self._mode,
            max_sequence_length=self._max_sequence_length,
            max_examples_per_sequence=self._max_examples_per_sequence,
            separator_token_id=self._separator_token_id,
            overflow=self._overflow,
            drop_last=self._drop_last,
            seed=self._seed,
            pad_to_multiple_of=self._pad_to_multiple_of,
        )

    def build_spec(self) -> PackingSpec:
        return PackingSpec(mode=self._mode, max_sequence_length=self._max_sequence_length)

    def build_memory_policy(self) -> MemoryPackingPolicy:
        return self._memory


class PackingContextBuilder:
    """Assembles a resolved :class:`PackingContext`."""

    def __init__(self) -> None:
        self._pieces: dict[str, object] = {}

    def with_piece(self, key: str, value: object) -> PackingContextBuilder:
        self._pieces[key] = value
        return self

    def with_examples(self, examples: Sequence[TrainingExample]) -> PackingContextBuilder:
        self._pieces["examples"] = tuple(examples)
        return self

    def with_session(self, session: PackingSession) -> PackingContextBuilder:
        self._pieces["packing_session"] = session
        return self

    def with_policy(self, policy: PackingPolicy) -> PackingContextBuilder:
        self._pieces["packing_policy"] = policy
        return self

    def with_spec(self, spec: PackingSpec) -> PackingContextBuilder:
        self._pieces["packing_spec"] = spec
        return self

    def build(self) -> PackingContext:
        examples = self._pieces.get("examples")
        session = self._pieces.get("packing_session")
        policy = self._pieces.get("packing_policy")
        if not isinstance(examples, tuple):
            raise BuilderError("PackingContextBuilder requires examples tuple.")
        if not isinstance(session, PackingSession):
            raise BuilderError("PackingContextBuilder requires PackingSession.")
        if not isinstance(policy, PackingPolicy):
            policy = PackingBuilder().build_policy()
        spec = self._pieces.get("packing_spec")
        if not isinstance(spec, PackingSpec):
            spec = PackingSpec(mode=policy.mode, max_sequence_length=policy.max_sequence_length)
        token_rows_obj = self._pieces.get("token_rows")
        token_rows: dict[str, TokenRow]
        if isinstance(token_rows_obj, Mapping):
            token_rows = {str(k): v for k, v in token_rows_obj.items() if isinstance(v, TokenRow)}
        else:
            token_rows = resolve_token_rows(examples, max_length=policy.max_sequence_length)
        memory = self._pieces.get("memory_policy")
        if not isinstance(memory, MemoryPackingPolicy):
            memory = MemoryPackingPolicy()
        seed_raw = self._pieces.get("seed", policy.seed if policy.seed is not None else 42)
        seed = seed_raw if isinstance(seed_raw, int) else 42
        extra = self._pieces.get("bind_extra")
        bind_extra: dict[str, Any] = dict(extra) if isinstance(extra, dict) else {}
        return PackingContext(
            examples=examples,
            packing_session=session,
            packing_spec=spec,
            packing_policy=policy,
            token_rows=token_rows,
            memory_policy=memory,
            seed=seed,
            bind_extra=bind_extra,
        )


def build_packing_session(
    *,
    experiment_id: ExperimentId,
    run_id: RunId,
    session_id: str = "pack-session",
) -> PackingSession:
    from datetime import UTC, datetime

    now = datetime.now(UTC)
    return PackingSession(
        session_id=session_id,
        experiment_id=experiment_id,
        run_id=run_id,
        created_at=now,
        updated_at=now,
    )

