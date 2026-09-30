"""Aggregate representation + BPE token statistics for one MIDI file."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from midi_llm.config import load_config
from midi_llm.midi_repr.registry import get_repr
from midi_llm.tokenization.bpe_counts import count_llama_bpe_tokens, serialized_payload


@dataclass
class TokenStatsRow:
    repr: str
    n_repr_tokens: int
    n_compound_tokens: int | None
    n_bpe_tokens: int | None
    bpe_tokens_per_note: float | None
    n_notes: int | None
    repr_tokens_per_note: float | None
    n_tracks: int | None
    duration_sec: float | None
    payload_chars: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def token_stats_for_midi(
    midi_path: str | Path,
    repr_name: str,
    *,
    model_name: str | None = None,
    count_bpe: bool = True,
) -> TokenStatsRow:
    """Encode one MIDI file and compare representation vs Llama BPE token counts."""
    try:
        backend = get_repr(repr_name)
        encoded = backend.encode(midi_path)
    except ImportError as exc:
        return TokenStatsRow(
            repr=repr_name,
            n_repr_tokens=0,
            n_compound_tokens=None,
            n_bpe_tokens=None,
            bpe_tokens_per_note=None,
            n_notes=None,
            repr_tokens_per_note=None,
            n_tracks=None,
            duration_sec=None,
            payload_chars=0,
            error=str(exc),
        )

    payload = serialized_payload(encoded)
    stats = encoded.stats
    n_notes = stats.n_notes if stats else None
    n_repr = encoded.n_tokens
    n_compound = len(encoded.compound_token_ids) if encoded.compound_token_ids is not None else None
    repr_tpn = stats.tokens_per_note if stats else None

    n_bpe: int | None = None
    bpe_tpn: float | None = None
    bpe_error: str | None = None
    if count_bpe:
        cfg = load_config(repr_name)
        backbone = model_name or cfg["model"]["backbone"]
        try:
            n_bpe = count_llama_bpe_tokens(payload, backbone)
            bpe_tpn = n_bpe / n_notes if n_notes else 0.0
        except Exception as exc:  # noqa: BLE001 — report HF/network failures per repr
            bpe_error = str(exc)

    if bpe_error:
        return TokenStatsRow(
            repr=repr_name,
            n_repr_tokens=n_repr,
            n_compound_tokens=n_compound,
            n_bpe_tokens=None,
            bpe_tokens_per_note=None,
            n_notes=n_notes,
            repr_tokens_per_note=round(repr_tpn, 3) if repr_tpn is not None else None,
            n_tracks=stats.n_tracks if stats else None,
            duration_sec=round(stats.duration_sec, 3) if stats else None,
            payload_chars=len(payload),
            error=bpe_error,
        )

    return TokenStatsRow(
        repr=repr_name,
        n_repr_tokens=n_repr,
        n_compound_tokens=n_compound,
        n_bpe_tokens=n_bpe,
        bpe_tokens_per_note=round(bpe_tpn, 3) if bpe_tpn is not None else None,
        n_notes=n_notes,
        repr_tokens_per_note=round(repr_tpn, 3) if repr_tpn is not None else None,
        n_tracks=stats.n_tracks if stats else None,
        duration_sec=round(stats.duration_sec, 3) if stats else None,
        payload_chars=len(payload),
    )
