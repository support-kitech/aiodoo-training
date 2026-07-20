"""Capability Contract version compatibility gate for aiodoo-training.

Training pins :data:`TRAINING_CONTRACT_VERSION` — the Capability Contract
version its prompt/schema/template integration was built and tested
against (ADR-0009 in aiodoo-contract). Every training entry point that
bootstraps the pipeline calls :func:`ensure_contract_compatible` first and
fails closed, rather than silently training against a contract version
whose schemas, prompts, or templates may have moved.
"""

from __future__ import annotations

from aiodoo_contract.version import CompatibilityResult, check_compatibility

from aiodoo_training.exceptions import ConfigError

__all__ = [
    "TRAINING_CONTRACT_VERSION",
    "ContractVersionError",
    "ensure_contract_compatible",
]

#: The Capability Contract version aiodoo-training's contract integration
#: (prompts, chat templates, schemas, publishing metadata) is pinned to.
#: Bump deliberately when adopting a new contract release — never let this
#: silently drift from what was actually validated against.
TRAINING_CONTRACT_VERSION = "1.0.0"


class ContractVersionError(ConfigError):
    """Raised when the installed `aiodoo_contract` is incompatible with training's pin."""


def ensure_contract_compatible(
    *,
    consumer_version: str = TRAINING_CONTRACT_VERSION,
) -> CompatibilityResult:
    """Verify the installed ``aiodoo_contract`` is compatible with ``consumer_version``.

    Returns:
        The :class:`~aiodoo_contract.version.CompatibilityResult` on success
        (``COMPATIBLE`` or logged ``MINOR_MISMATCH`` — see below).

    Raises:
        ContractVersionError: if the installed contract is a different
            major version, or a strictly older minor version than
            ``consumer_version`` expects (``MINOR_MISMATCH``) — training
            must fail early rather than train against prompts/schemas the
            installed contract does not actually provide.
    """
    result = check_compatibility(consumer_version)
    if not result.is_compatible:
        raise ContractVersionError(
            "aiodoo-training is incompatible with the installed aiodoo_contract: "
            f"{result.reason} (training pinned to {consumer_version!r}, "
            f"installed contract is {result.contract_version!s}). "
            "Upgrade/downgrade aiodoo_contract or update "
            "aiodoo_training.contract.version_check.TRAINING_CONTRACT_VERSION "
            "after re-validating the contract integration."
        )
    return result
