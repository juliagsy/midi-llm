"""Shared fixtures."""

from __future__ import annotations

import json
import shutil
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
        piano.notes.append(
            pretty_midi.Note(velocity=80, pitch=60 + (i % 4), start=t0, end=t0 + 0.4)
        )
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


@pytest.fixture
def instruct_mini_manifest(tmp_path: Path, sample_midi: Path) -> Path:
    """Minimal MIDI-Instruct-style manifest with local MIDI files (no external repos)."""
    root = tmp_path / "instruct_mini"
    root.mkdir()
    in_path = root / "in.mid"
    gold_path = root / "gold.mid"
    shutil.copy2(sample_midi, in_path)
    shutil.copy2(sample_midi, gold_path)

    manifest = root / "manifest.jsonl"
    rows = [
        {
            "item_id": "train_001",
            "split": "train",
            "midi_in": "in.mid",
            "gold_midi": "gold.mid",
            "instruction": "Transpose up 2 semitones.",
        },
        {
            "item_id": "test_001",
            "split": "test",
            "midi_in": "in.mid",
            "gold_midi": "gold.mid",
            "instruction": "Increase velocity on track 0.",
        },
    ]
    manifest.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    return manifest
