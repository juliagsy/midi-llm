"""Registry of representation backends."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import MidiRepresentation

REPR_NAMES = ("abc", "remi", "amt", "octuple")


def get_repr(name: str) -> MidiRepresentation:
    key = name.lower().strip()
    if key == "abc":
        from .abc_repr import ABCRepresentation

        return ABCRepresentation()
    if key == "remi":
        from .remi_repr import REMIRepresentation

        return REMIRepresentation()
    if key == "amt":
        from .amt_repr import AMTRepresentation

        return AMTRepresentation()
    if key == "octuple":
        from .octuple_repr import OctupleRepresentation

        return OctupleRepresentation()
    raise ValueError(f"unknown representation {name!r}; choose from {REPR_NAMES}")


def list_reprs() -> list[str]:
    return list(REPR_NAMES)
