"""Portable mesh_digest — never hostnames, IPs, PIDs, UUIDs, paths, or timestamps."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

# Truncation length for portable digests. Changing this changes digest strings;
# keep stable for fingerprint / golden compatibility.
MESH_DIGEST_LENGTH = 32


def compute_mesh_digest(
    *,
    world_size: int,
    mesh_axes: Sequence[str],
    mesh_shape: Sequence[int],
    placement_key: str,
    communication_backend_key: str,
    accelerator: str,
    runtime_backend_key: str,
    rank_to_coord: Mapping[int, tuple[int, ...]] | None = None,
) -> str:
    """
    Compute a portable topology digest for fingerprints / RestartPolicy.

    Only stable topology identity contributes. Callers must not pass hostname,
    IP, MAC, PID, launcher ids, timestamps, UUIDs, filesystem paths, or
    environment-specific identifiers.
    """
    coords = rank_to_coord or {}
    coord_part = "|".join(
        f"{rank}:{'x'.join(str(c) for c in coords[rank])}"
        for rank in sorted(coords)
    )
    material = "|".join(
        [
            f"ws={world_size}",
            f"axes={','.join(mesh_axes)}",
            f"shape={','.join(str(s) for s in mesh_shape)}",
            f"place={placement_key}",
            f"comm={communication_backend_key}",
            f"accel={accelerator}",
            f"rt={runtime_backend_key}",
            f"coords={coord_part}",
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:MESH_DIGEST_LENGTH]
