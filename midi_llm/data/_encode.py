"""Shared MIDI encoding helpers for shard builders."""

from __future__ import annotations

import tempfile
from pathlib import Path

from midi_llm.midi_repr.payload import serialize_midi_payload
from midi_llm.midi_repr.registry import get_repr


def encode_midi_file(repr_name: str, midi_path: str | Path) -> str:
    backend = get_repr(repr_name)
    encoded = backend.encode(midi_path)
    return serialize_midi_payload(repr_name, encoded)


def encode_midi_bytes(repr_name: str, midi_bytes: bytes, *, suffix: str = ".mid") -> str:
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        tmp = Path(handle.name)
        handle.write(midi_bytes)
    try:
        return encode_midi_file(repr_name, tmp)
    finally:
        tmp.unlink(missing_ok=True)
