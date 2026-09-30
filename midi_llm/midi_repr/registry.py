"""Registry of representation backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import MidiRepresentation

# Active arms for Paper 2 experiments (training, eval, token-stats).
REPR_NAMES = ("remi", "amt", "octuple")

# Reserved / disabled until a real MIDI↔ABC pipeline is implemented and tested.
DISABLED_REPR_NAMES = ("abc",)

_REPR_CACHE: dict[str, MidiRepresentation] = {}


class RepresentationDisabledError(ValueError):
    """Raised when a representation arm is registered but not yet usable."""


def get_repr(name: str) -> MidiRepresentation:
    key = name.lower().strip()
    if key in DISABLED_REPR_NAMES:
        raise RepresentationDisabledError(
            f"representation {name!r} is disabled: MIDI→ABC conversion is not yet "
            f"implemented. Active arms: {list(REPR_NAMES)}"
        )
    if key in _REPR_CACHE:
        return _REPR_CACHE[key]

    if key == "remi":
        from .remi_repr import REMIRepresentation

        backend: MidiRepresentation = REMIRepresentation()
    elif key == "amt":
        from .amt_repr import AMTRepresentation

        backend = AMTRepresentation()
    elif key == "octuple":
        from .octuple_repr import OctupleRepresentation

        backend = OctupleRepresentation()
    else:
        raise ValueError(f"unknown representation {name!r}; choose from {REPR_NAMES}")

    _REPR_CACHE[key] = backend
    return backend


def clear_repr_cache() -> None:
    """Clear cached representation backends (for tests)."""
    _REPR_CACHE.clear()


def list_reprs(*, include_disabled: bool = False) -> list[str]:
    if include_disabled:
        return [*REPR_NAMES, *DISABLED_REPR_NAMES]
    return list(REPR_NAMES)
