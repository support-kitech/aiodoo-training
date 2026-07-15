"""RNG controllers — Python-stdlib primary; Torch/NumPy optional extensions."""

from __future__ import annotations

import random
from typing import Any

from aiodoo_training.exceptions import DeterminismError
from aiodoo_training.ports.trainer import RngController


def _serialize_python_random_state(state: Any) -> dict[str, Any]:
    """Convert ``random.getstate()`` into a JSON-friendly mapping."""
    version, internal, gauss = state
    return {
        "version": int(version),
        "internal": [int(x) for x in internal],
        "gauss": None if gauss is None else float(gauss),
    }


def _deserialize_python_random_state(payload: Any) -> Any:
    """Rebuild a ``random.setstate``-compatible tuple from JSON data."""
    if isinstance(payload, tuple) and len(payload) == 3:
        version, internal, gauss = payload
        return (version, tuple(internal), gauss)
    if isinstance(payload, list) and len(payload) == 3:
        version, internal, gauss = payload
        return (version, tuple(internal), gauss)
    if isinstance(payload, dict):
        version = payload.get("version")
        internal = payload.get("internal")
        gauss = payload.get("gauss")
        if not isinstance(version, int) or not isinstance(internal, list):
            raise DeterminismError("Invalid python_random payload in RNG snapshot.")
        return (version, tuple(int(x) for x in internal), gauss)
    raise DeterminismError("Invalid python_random payload in RNG snapshot.")


class _DeterministicPrng:
    """
    Serialisable 64-bit LCG used for stub-training draws independent of
    ``random`` module call order in unrelated code.
    """

    # Numerical Recipes LCG constants
    _A = 1664525
    _C = 1013904223
    _M = 2**32

    def __init__(self, seed: int = 0) -> None:
        self._state = int(seed) % self._M

    @property
    def state(self) -> int:
        return self._state

    def seed(self, seed: int) -> None:
        self._state = int(seed) % self._M

    def random(self) -> float:
        self._state = (self._A * self._state + self._C) % self._M
        return self._state / float(self._M)

    def snapshot(self) -> dict[str, int]:
        return {"state": self._state, "modulus": self._M}

    def restore(self, payload: dict[str, Any]) -> None:
        state = payload.get("state")
        if not isinstance(state, int):
            raise DeterminismError("Invalid deterministic PRNG state in RNG snapshot.")
        self._state = int(state) % self._M


class PythonRngController(RngController):
    """
    CPU-safe RNG controller using only the Python standard library.

    Always seeds ``random``. Maintains a private LCG whose state is included in
    snapshots so stub training remains deterministic without Torch/NumPy.
    """

    BACKEND_KEY = "python"

    def __init__(self) -> None:
        self._seed = 0
        self._prng = _DeterministicPrng(0)

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def prng(self) -> _DeterministicPrng:
        """Private deterministic PRNG for infra consumers (stub trainer, etc.)."""
        return self._prng

    def seed_all(self, seed: int) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise DeterminismError("Seed must be a non-negative integer.")
        self._seed = seed
        random.seed(seed)
        self._prng.seed(seed)
        self._maybe_seed_numpy(seed)

    def snapshot(self) -> dict[str, object]:
        return {
            "backend": self.BACKEND_KEY,
            "seed": self._seed,
            "python_random": _serialize_python_random_state(random.getstate()),
            "deterministic_prng": self._prng.snapshot(),
        }

    def restore(self, state: dict[str, object]) -> None:
        seed = state.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise DeterminismError("Invalid seed in RNG snapshot.")
        self._seed = seed
        python_state = state.get("python_random")
        if python_state is not None:
            random.setstate(_deserialize_python_random_state(python_state))
        else:
            random.seed(seed)
        prng_payload = state.get("deterministic_prng")
        if isinstance(prng_payload, dict):
            self._prng.restore(prng_payload)
        else:
            self._prng.seed(seed)
        self._maybe_seed_numpy(seed)

    @staticmethod
    def _maybe_seed_numpy(seed: int) -> None:
        try:
            import numpy as np
        except ImportError:
            return
        np.random.seed(seed)


class TorchRngController(PythonRngController):
    """Extends :class:`PythonRngController` with Torch (and NumPy) seeding."""

    BACKEND_KEY = "torch"

    def seed_all(self, seed: int) -> None:
        super().seed_all(seed)
        self._maybe_seed_torch(seed)

    def snapshot(self) -> dict[str, object]:
        payload = super().snapshot()
        payload["backend"] = self.BACKEND_KEY
        try:
            import torch

            payload["torch_rng"] = torch.random.get_rng_state()
            if torch.cuda.is_available():
                payload["torch_cuda_rng"] = torch.cuda.get_rng_state_all()
        except ImportError:
            pass
        try:
            import numpy as np

            payload["numpy_rng"] = np.random.get_state()
        except ImportError:
            pass
        return payload

    def restore(self, state: dict[str, object]) -> None:
        super().restore(state)
        try:
            import torch

            torch_state = state.get("torch_rng")
            if torch_state is not None:
                torch.random.set_rng_state(torch_state)
            else:
                torch.manual_seed(self._seed)
            cuda_state = state.get("torch_cuda_rng")
            if cuda_state is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(cuda_state)
        except ImportError:
            pass
        try:
            import numpy as np

            numpy_state = state.get("numpy_rng")
            if numpy_state is not None:
                np.random.set_state(numpy_state)
        except ImportError:
            pass

    @staticmethod
    def _maybe_seed_torch(seed: int) -> None:
        try:
            import torch
        except ImportError:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


# Alias preferred by architecture drafts that named the stub controller explicitly.
StubRngController = PythonRngController


def register_default_rng(*, overwrite: bool = False) -> None:
    """Register ``python`` and optionally ``torch`` RNG controllers."""
    from aiodoo_training.registries import rng_registry

    if not rng_registry.exists("python") or overwrite:
        rng_registry.register("python", PythonRngController, overwrite=overwrite)
    # Always register torch key — TorchRngController degrades without torch installed.
    if not rng_registry.exists("torch") or overwrite:
        rng_registry.register("torch", TorchRngController, overwrite=overwrite)
