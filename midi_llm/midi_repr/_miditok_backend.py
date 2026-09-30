"""Shared MidiTok encode/decode helpers for REMI and Octuple."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ._midi_stats import midi_note_stats
from .payload import extract_compound_token_ids, extract_flat_token_ids


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


def encode_with_tokenizer(tokenizer, midi_path: str | Path) -> tuple[list[int], Any]:
    score = load_score(midi_path)
    encoded = tokenizer.encode(score)
    return extract_flat_token_ids(encoded), encoded


def encode_compound_with_tokenizer(tokenizer, midi_path: str | Path) -> tuple[list[list[int]], Any]:
    score = load_score(midi_path)
    encoded = tokenizer.encode(score)
    return extract_compound_token_ids(encoded), encoded


def decode_with_tokenizer(
    tokenizer,
    token_ids: list[int] | None,
    output_path: str | Path,
    *,
    compound_token_ids: list[list[int]] | None = None,
) -> Path:
    ids = compound_token_ids if compound_token_ids is not None else token_ids
    if ids is None:
        raise ValueError("decode requires token_ids or compound_token_ids")
    score = tokenizer.decode(ids)
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
