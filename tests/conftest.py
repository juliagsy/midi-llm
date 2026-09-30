"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pretty_midi
import pytest


@pytest.fixture
def sample_midi(tmp_path: Path) -> Path:
    """Small piano + drums seed for tokenizer tests."""
    path = tmp_path / "sample.mid"
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    piano = pretty_midi.Instrument(program=0, name="piano")
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="drums")
    beat = 0.5
    for i in range(8):
        t0 = i * beat
        piano.notes.append(pretty_midi.Note(velocity=80, pitch=60 + (i % 4), start=t0, end=t0 + 0.4))
        if i % 2 == 0:
            drums.notes.append(pretty_midi.Note(velocity=100, pitch=36, start=t0, end=t0 + 0.05))
    midi.instruments.extend([piano, drums])
    midi.write(str(path))
    return path


@pytest.fixture
def miditok_available() -> bool:
    try:
        import miditok  # noqa: F401
        import symusic  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.fixture
def music21_available() -> bool:
    try:
        import music21  # noqa: F401

        return True
    except ImportError:
        return False
