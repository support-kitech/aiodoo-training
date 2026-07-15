"""FakeDistributedBackend — CPU-only deterministic collectives (mandatory CI path)."""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock

from aiodoo_training.domain.distributed_session import DistributedTopology
from aiodoo_training.domain.enums import ReductionOp
from aiodoo_training.exceptions import DistributedError
from aiodoo_training.ports.distributed import DistributedBackend


class FakeDistributedBackend(DistributedBackend):
    """
    Architectural reference implementation for distributed collectives.

    Single-process simulation of N logical ranks. Barriers / broadcasts /
    reductions are deterministic and require no GPU, NCCL, CUDA, Accelerate,
    DeepSpeed, FSDP, or MPI.

    Future DDP / FSDP / DeepSpeed / XLA adapters should preserve the same
    observable DistributedBackend behaviour while changing only infrastructure
    internals.
    """

    def __init__(self) -> None:
        self._topology: DistributedTopology | None = None
        self._open = False
        self._lock = Lock()
        # Simulated store for broadcast/all-reduce across sequential fake calls.
        self._broadcast_payload: bytes | None = None
        self._metric_scratch: dict[str, list[float]] = {}

    def initialize(self, topology: DistributedTopology) -> Mapping[str, str]:
        with self._lock:
            self._topology = topology
            self._open = True
            self._broadcast_payload = None
            self._metric_scratch.clear()
            return {gid: "ready" for gid in topology.groups}

    def barrier(self, group_id: str, *, timeout_sec: float) -> None:
        self._require_open()
        if group_id not in (self._topology.groups if self._topology else {}):
            # allow named groups used by SyncFacade that map to same PG
            if group_id not in {
                "default",
                "epoch",
                "ckpt_before",
                "ckpt_after",
                "eval_merge",
                "export_before",
                "export_after",
            }:
                raise DistributedError(f"Unknown process group {group_id!r}")
        if timeout_sec <= 0:
            raise TimeoutError("fake barrier timeout")
        # Single-process: immediate success.

    def broadcast_bytes(self, group_id: str, payload: bytes, *, src_rank: int) -> bytes:
        self._require_open()
        assert self._topology is not None
        if src_rank < 0 or src_rank >= self._topology.world_size:
            raise DistributedError(f"src_rank {src_rank} out of range")
        # Deterministic: src provides payload; all callers receive it.
        if self._topology.global_rank == src_rank or self._broadcast_payload is None:
            self._broadcast_payload = bytes(payload)
        return bytes(self._broadcast_payload)

    def all_reduce_metrics(
        self,
        group_id: str,
        values: Mapping[str, float],
        *,
        op: ReductionOp,
    ) -> Mapping[str, float]:
        del group_id  # single-process logical group
        self._require_open()
        assert self._topology is not None
        # Accumulate contributions; for world_size==1 reduce is identity.
        # For multi-logical-rank tests in one process, callers may invoke once
        # per logical rank by temporarily swapping topology; default is identity
        # over provided values when scratch empty, else combined.
        with self._lock:
            for key, value in values.items():
                self._metric_scratch.setdefault(key, []).append(float(value))
            # When only one contribution per key, treat as world aggregate for CI.
            out: dict[str, float] = {}
            for key in sorted(values):
                samples = list(self._metric_scratch.get(key, [float(values[key])]))
                out[key] = _reduce(samples, op)
            # Reset per-call for deterministic single-shot goldens.
            self._metric_scratch.clear()
            return out

    def finalize(self) -> None:
        with self._lock:
            self._open = False
            self._topology = None
            self._broadcast_payload = None
            self._metric_scratch.clear()

    def _require_open(self) -> None:
        if not self._open or self._topology is None:
            raise DistributedError("FakeDistributedBackend is not initialized.")


def _reduce(samples: list[float], op: ReductionOp) -> float:
    if not samples:
        return 0.0
    if op is ReductionOp.SUM:
        return float(sum(samples))
    if op is ReductionOp.MEAN:
        return float(sum(samples) / len(samples))
    if op is ReductionOp.MAX:
        return float(max(samples))
    if op is ReductionOp.MIN:
        return float(min(samples))
    raise DistributedError(f"Unsupported reduction op: {op}")


class RegistrationOnlyDistributedBackend(DistributedBackend):
    """Placeholder for DDP/FSDP/DeepSpeed/Accelerate/XLA registration points."""

    def __init__(self, key: str) -> None:
        self._key = key

    def initialize(self, topology: DistributedTopology) -> Mapping[str, str]:
        del topology
        raise DistributedError(
            f"Distributed backend {self._key!r} is registration-only in this build; "
            "use backend='fake' for CPU CI."
        )

    def barrier(self, group_id: str, *, timeout_sec: float) -> None:
        raise DistributedError(f"{self._key} backend not initialized")

    def broadcast_bytes(self, group_id: str, payload: bytes, *, src_rank: int) -> bytes:
        raise DistributedError(f"{self._key} backend not initialized")

    def all_reduce_metrics(
        self,
        group_id: str,
        values: Mapping[str, float],
        *,
        op: ReductionOp,
    ) -> Mapping[str, float]:
        raise DistributedError(f"{self._key} backend not initialized")

    def finalize(self) -> None:
        return None


def register_default_distributed_backends(*, overwrite: bool = False) -> None:
    from aiodoo_training.registries import distributed_backend_registry

    distributed_backend_registry.register("fake", FakeDistributedBackend, overwrite=overwrite)
    for key in ("ddp", "fsdp", "deepspeed", "accelerate", "xla"):
        # Late binding via factory key — registry stores class; factory wraps.
        distributed_backend_registry.register(
            key, _registration_only_type(key), overwrite=overwrite
        )


_REG_TYPES: dict[str, type[DistributedBackend]] = {}


def _registration_only_type(key: str) -> type[DistributedBackend]:
    if key in _REG_TYPES:
        return _REG_TYPES[key]

    class _Backend(RegistrationOnlyDistributedBackend):
        def __init__(self) -> None:
            super().__init__(key)

    _Backend.__name__ = f"{key.title()}DistributedBackendStub"
    _REG_TYPES[key] = _Backend
    return _Backend
