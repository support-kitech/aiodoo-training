"""Phase 6 CLI profile domain."""

from __future__ import annotations

from dataclasses import dataclass

from aiodoo_training.domain.enums import CLIProfileName


@dataclass(frozen=True, slots=True)
class CLIProfile:
    """Immutable UX preset — does not redesign Command Registry."""

    name: CLIProfileName = CLIProfileName.DEFAULT
    progress: bool = True
    color: str = "auto"
    output: str = "text"
    verbosity: int = 0
    confirm_destructive: bool = True

    def __post_init__(self) -> None:
        if self.verbosity < 0 or self.verbosity > 2:
            raise ValueError("CLIProfile.verbosity must be in 0..2.")
        if self.color not in {"auto", "always", "never"}:
            raise ValueError("CLIProfile.color must be auto|always|never.")
        if self.output not in {"text", "json"}:
            raise ValueError("CLIProfile.output must be text|json.")


PROFILE_PRESETS: dict[CLIProfileName, CLIProfile] = {
    CLIProfileName.DEFAULT: CLIProfile(name=CLIProfileName.DEFAULT),
    CLIProfileName.MINIMAL: CLIProfile(
        name=CLIProfileName.MINIMAL,
        progress=False,
        verbosity=0,
        confirm_destructive=True,
    ),
    CLIProfileName.VERBOSE: CLIProfile(
        name=CLIProfileName.VERBOSE,
        progress=True,
        verbosity=2,
    ),
    CLIProfileName.JSON: CLIProfile(
        name=CLIProfileName.JSON,
        progress=False,
        color="never",
        output="json",
        verbosity=0,
        confirm_destructive=False,
    ),
    CLIProfileName.CI: CLIProfile(
        name=CLIProfileName.CI,
        progress=False,
        color="never",
        output="json",
        verbosity=0,
        confirm_destructive=False,
    ),
}


def resolve_cli_profile(name: str | CLIProfileName) -> CLIProfile:
    key = CLIProfileName(name) if not isinstance(name, CLIProfileName) else name
    return PROFILE_PRESETS[key]
