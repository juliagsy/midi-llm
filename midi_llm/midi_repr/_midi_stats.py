"""Helpers to summarize MIDI files for encode stats."""

from __future__ import annotations

from pathlib import Path

import pretty_midi


def midi_note_stats(midi_path: str | Path) -> tuple[int, int, float]:
    """Return (n_notes, n_tracks, duration_sec)."""
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    n_notes = sum(len(inst.notes) for inst in midi.instruments)
    n_tracks = len(midi.instruments)
    duration = midi.get_end_time()
    return n_notes, n_tracks, duration
