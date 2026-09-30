"""AMT (Anticipatory Music Transformer) arrival-time MIDI tokens."""

from __future__ import annotations

from pathlib import Path

from .base import EncodeResult, MidiRepresentation
from ._midi_stats import midi_note_stats


def _require_anticipation():
    try:
        from anticipation.convert import events_to_midi, midi_to_events
        from anticipation.vocab import VOCAB_SIZE
    except ImportError as exc:
        raise ImportError(
            "AMT representation requires anticipation from GitHub: "
            "pip install -e '.[amt]' (https://github.com/jthickstun/anticipation)"
        ) from exc
    return events_to_midi, midi_to_events, VOCAB_SIZE


class AMTRepresentation(MidiRepresentation):
    """Arrival-time events: onset, duration, instrument×pitch triplets."""

    name = "amt"

    def encode(self, midi_path: str | Path) -> EncodeResult:
        _, midi_to_events, vocab_size = _require_anticipation()
        path = Path(midi_path)
        token_ids = [int(t) for t in midi_to_events(str(path))]
        n_notes, n_tracks, duration = midi_note_stats(path)
        from .base import EncodeStats

        stats = EncodeStats.from_counts(
            n_notes=n_notes,
            n_tokens=len(token_ids),
            n_tracks=n_tracks,
            duration_sec=duration,
        )
        return EncodeResult(
            repr_name=self.name,
            token_ids=token_ids,
            stats=stats,
            metadata={"vocab_size": vocab_size, "tokens_per_note": 3},
        )

    def decode_to_midi(
        self,
        *,
        token_ids: list[int] | None = None,
        text: str | None = None,
        output_path: str | Path,
    ) -> Path:
        if token_ids is None:
            raise ValueError("AMT decode requires token_ids")
        if text is not None:
            raise ValueError("AMT does not decode from text")

        events_to_midi, _, _ = _require_anticipation()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        mid = events_to_midi(token_ids)
        mid.save(str(out))
        return out
