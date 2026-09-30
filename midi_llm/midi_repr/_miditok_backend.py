"""Shared MidiTok encode/decode helpers for REMI and Octuple."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._midi_stats import midi_note_stats


def _require_miditok():
    try:
        from miditok import TokenizerConfig
        from symusic import Score
    except ImportError as exc:
        raise ImportError(
            "MidiTok backend requires optional deps: pip install -e '.[repr]'"
        ) from exc
    return TokenizerConfig, Score


def default_tokenizer_config():
    TokenizerConfig, _ = _require_miditok()
    return TokenizerConfig(
        num_velocities=32,
        use_chords=True,
        use_programs=True,
        use_tempos=True,
        use_time_signatures=True,
        beat_res={(0, 4): 8},
    )


def load_score(midi_path: str | Path):
    _, Score = _require_miditok()
    return Score(str(midi_path))


def flatten_token_ids(encoded: Any) -> list[int]:
    if hasattr(encoded, "ids"):
        return list(encoded.ids)
    if isinstance(encoded, list):
        if not encoded:
            return []
        if isinstance(encoded[0], list):
            flat: list[int] = []
            for track_tokens in encoded:
                flat.extend(int(t) for t in track_tokens)
            return flat
        return [int(t) for t in encoded]
    raise TypeError(f"unexpected encoded token type: {type(encoded)!r}")


def encode_with_tokenizer(tokenizer, midi_path: str | Path) -> tuple[list[int], Any]:
    score = load_score(midi_path)
    encoded = tokenizer.encode(score)
    return flatten_token_ids(encoded), encoded


def decode_with_tokenizer(tokenizer, token_ids: list[int], output_path: str | Path) -> Path:
    score = tokenizer.decode(token_ids)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    score.dump_midi(str(out))
    return out


def build_stats(midi_path: str | Path, n_tokens: int):
    from .base import EncodeStats

    n_notes, n_tracks, duration = midi_note_stats(midi_path)
    return EncodeStats.from_counts(
        n_notes=n_notes,
        n_tokens=n_tokens,
        n_tracks=n_tracks,
        duration_sec=duration,
    )
