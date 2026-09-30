"""ABC text representation for native LLM tokenization."""

from __future__ import annotations

import tempfile
from pathlib import Path

from .base import EncodeResult, MidiRepresentation
from ._midi_stats import midi_note_stats


def _require_music21():
    try:
        from music21 import converter
    except ImportError as exc:
        raise ImportError(
            "ABC representation requires music21: pip install music21"
        ) from exc
    return converter


class ABCRepresentation(MidiRepresentation):
    """Serialize MIDI as ABC notation text (ChatMusician-style arm)."""

    name = "abc"

    def encode(self, midi_path: str | Path) -> EncodeResult:
        converter = _require_music21()
        path = Path(midi_path)
        score = converter.parse(str(path))
        with tempfile.NamedTemporaryFile(suffix=".abc", delete=False) as handle:
            abc_tmp = Path(handle.name)
        written: Path | None = None
        try:
            abc_path = score.write("abc", fp=str(abc_tmp))
            written = Path(abc_path[0] if isinstance(abc_path, list) else abc_path)
            text = written.read_text(encoding="utf-8", errors="replace").strip()
        finally:
            abc_tmp.unlink(missing_ok=True)
            if written is not None and written != abc_tmp:
                written.unlink(missing_ok=True)

        n_notes, n_tracks, duration = midi_note_stats(path)
        word_count = len(text.split())
        from .base import EncodeStats

        stats = EncodeStats.from_counts(
            n_notes=n_notes,
            n_tokens=word_count,
            n_tracks=n_tracks,
            duration_sec=duration,
        )
        return EncodeResult(repr_name=self.name, text=text, stats=stats, metadata={"format": "abc"})

    def decode_to_midi(
        self,
        *,
        token_ids: list[int] | None = None,
        text: str | None = None,
        output_path: str | Path,
    ) -> Path:
        if text is None:
            raise ValueError("ABC decode requires text")
        if token_ids is not None:
            raise ValueError("ABC does not decode from token_ids")

        converter = _require_music21()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".abc", delete=False, encoding="utf-8") as handle:
            handle.write(text)
            abc_tmp = Path(handle.name)
        try:
            score = converter.parse(str(abc_tmp))
            score.write("midi", fp=str(out))
        finally:
            abc_tmp.unlink(missing_ok=True)
        return out
