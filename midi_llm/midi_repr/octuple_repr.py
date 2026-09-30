"""Octuple compound-token representation via MidiTok."""

from __future__ import annotations

from pathlib import Path

from .base import EncodeResult, MidiRepresentation
from ._miditok_backend import (
    build_stats,
    decode_with_tokenizer,
    default_tokenizer_config,
    encode_with_tokenizer,
)


class OctupleRepresentation(MidiRepresentation):
    name = "octuple"

    def __init__(self) -> None:
        from miditok import Octuple

        self._tokenizer = Octuple(default_tokenizer_config())

    def encode(self, midi_path: str | Path) -> EncodeResult:
        path = Path(midi_path)
        token_ids, _ = encode_with_tokenizer(self._tokenizer, path)
        stats = build_stats(path, len(token_ids))
        return EncodeResult(repr_name=self.name, token_ids=token_ids, stats=stats)

    def decode_to_midi(
        self,
        *,
        token_ids: list[int] | None = None,
        text: str | None = None,
        output_path: str | Path,
    ) -> Path:
        if token_ids is None:
            raise ValueError("Octuple decode requires token_ids")
        if text is not None:
            raise ValueError("Octuple does not decode from text")
        return decode_with_tokenizer(self._tokenizer, token_ids, output_path)
