"""ABC text representation (disabled until real MIDI↔ABC conversion exists)."""

from __future__ import annotations

from pathlib import Path

from .base import EncodeResult, MidiRepresentation

_DISABLED_REASON = (
    "ABC arm is disabled: music21's score.write('abc') does not produce valid ABC "
    "notation from MIDI. Re-enable after implementing and testing a real converter."
)


class ABCRepresentation(MidiRepresentation):
    """Placeholder for ChatMusician-style ABC text (not yet available)."""

    name = "abc"
    enabled = False

    def encode(self, midi_path: str | Path) -> EncodeResult:
        raise NotImplementedError(_DISABLED_REASON)

    def decode_to_midi(
        self,
        *,
        token_ids: list[int] | None = None,
        text: str | None = None,
        output_path: str | Path,
    ) -> Path:
        raise NotImplementedError(_DISABLED_REASON)
