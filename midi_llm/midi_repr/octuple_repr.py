"""Octuple compound-token representation via MidiTok."""

from __future__ import annotations

from pathlib import Path

from ._miditok_backend import (
    build_stats,
    decode_with_tokenizer,
    default_tokenizer_config,
    encode_compound_with_tokenizer,
)
from .base import EncodeResult, MidiRepresentation


class OctupleRepresentation(MidiRepresentation):
    name = "octuple"

    def __init__(self) -> None:
        from miditok import Octuple

        self._tokenizer = Octuple(default_tokenizer_config())

    def encode(self, midi_path: str | Path) -> EncodeResult:
        path = Path(midi_path)
        compound_token_ids, _ = encode_compound_with_tokenizer(self._tokenizer, path)
        stats = build_stats(path, len(compound_token_ids))
        return EncodeResult(
            repr_name=self.name,
            compound_token_ids=compound_token_ids,
            stats=stats,
        )

    def decode_to_midi(
        self,
        *,
        token_ids: list[int] | None = None,
        compound_token_ids: list[list[int]] | None = None,
        text: str | None = None,
        output_path: str | Path,
    ) -> Path:
        if compound_token_ids is None and token_ids is None:
            raise ValueError("Octuple decode requires compound_token_ids")
        if text is not None:
            raise ValueError("Octuple does not decode from text")
        return decode_with_tokenizer(
            self._tokenizer,
            token_ids,
            output_path,
            compound_token_ids=compound_token_ids,
        )
