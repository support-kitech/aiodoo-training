"""SyncFacade — Barrier / Broadcast / Reduction without framework primitives."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from aiodoo_training.domain.distributed_policies import (
    AggregationPolicy,
    BarrierPolicy,
    BroadcastPolicy,
    ReductionPolicy,
)
from aiodoo_training.domain.enums import BarrierTimeoutAction, ReductionOp
from aiodoo_training.exceptions import DistributedError
from aiodoo_training.ports.distributed import DistributedBackend


class SyncFacade:
    """Application façade over DistributedBackend collectives."""

    def __init__(
        self,
        backend: DistributedBackend,
        *,
        default_group: str = "default",
        require_deterministic_order: bool = True,
    ) -> None:
        self._backend = backend
        self._default_group = default_group
        self._require_order = require_deterministic_order

    def barrier(
        self, name: str = "default", policy: BarrierPolicy | None = None
    ) -> None:
        pol = policy or BarrierPolicy()
        group = name or self._default_group
        try:
            self._backend.barrier(group, timeout_sec=pol.timeout_sec)
        except TimeoutError as exc:
            if pol.on_timeout is BarrierTimeoutAction.WARN_CONTINUE:
                return
            raise DistributedError(f"Barrier timeout on group {group!r}: {exc}") from exc

    def broadcast_obj(self, obj: Any, policy: BroadcastPolicy | None = None) -> Any:
        pol = policy or BroadcastPolicy()
        payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode(pol.encoding)
        if len(payload) > pol.max_bytes:
            raise DistributedError("Broadcast payload exceeds max_bytes.")
        out = self._backend.broadcast_bytes(
            self._default_group, payload, src_rank=pol.src_rank
        )
        return json.loads(out.decode(pol.encoding))

    def reduce_metrics(
        self,
        metrics: Mapping[str, float],
        policy: ReductionPolicy | AggregationPolicy | None = None,
    ) -> Mapping[str, float]:
        if isinstance(policy, AggregationPolicy):
            keys = policy.metric_keys or tuple(sorted(metrics))
            values = {k: float(metrics[k]) for k in keys if k in metrics}
            op = policy.reduction.op
            if self._require_order or policy.reduction:
                values = {k: values[k] for k in sorted(values)}
            return self._backend.all_reduce_metrics(
                self._default_group, values, op=op
            )
        pol = policy if isinstance(policy, ReductionPolicy) else ReductionPolicy()
        values = {k: float(metrics[k]) for k in sorted(metrics)}
        return self._backend.all_reduce_metrics(
            self._default_group, values, op=pol.op
        )

    def all_reduce(
        self, values: Mapping[str, float], *, op: ReductionOp = ReductionOp.MEAN
    ) -> Mapping[str, float]:
        ordered = {k: float(values[k]) for k in sorted(values)}
        return self._backend.all_reduce_metrics(self._default_group, ordered, op=op)
