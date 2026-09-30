"""Shared types for MIDI representation backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EncodeStats:
    n_notes: int
    n_tokens: int
    tokens_per_note: float
    n_tracks: int
    duration_sec: float

    @classmethod
    def from_counts(
        cls,
        *,
        n_notes: int,
        n_tokens: int,
        n_tracks: int,
        duration_sec: float,
    ) -> EncodeStats:
        tpn = n_tokens / n_notes if n_notes else 0.0
        return cls(
            n_notes=n_notes,
            n_tokens=n_tokens,
            tokens_per_note=tpn,
            n_tracks=n_tracks,
            duration_sec=duration_sec,
        )


@dataclass
class EncodeResult:
    """Encoded MIDI in a representation-specific form."""

    repr_name: str
    token_ids: list[int] | None = None
    text: str | None = None
    stats: EncodeStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_tokens(self) -> int:
        if self.token_ids is not None:
            return len(self.token_ids)
        if self.text is not None:
            return len(self.text.split())
        return 0


@dataclass
class RoundTripResult:
    repr_name: str
    source: Path
    output: Path | None
    success: bool
    error: str | None = None
    stats_before: EncodeStats | None = None
    stats_after: EncodeStats | None = None
    fidelity_ok: bool | None = None
    fidelity_summary: str | None = None


class MidiRepresentation(ABC):
    """Encode MIDI files to tokens or text and decode back to MIDI."""

    name: str

    @abstractmethod
    def encode(self, midi_path: str | Path) -> EncodeResult:
        raise NotImplementedError

    @abstractmethod
    def decode_to_midi(
        self,
        *,
        token_ids: list[int] | None = None,
        text: str | None = None,
        output_path: str | Path,
    ) -> Path:
        raise NotImplementedError

    def roundtrip(self, midi_path: str | Path, output_dir: str | Path | None = None) -> RoundTripResult:
        source = Path(midi_path)
        out_dir = Path(output_dir) if output_dir else source.parent / "_roundtrip"
        out_dir.mkdir(parents=True, exist_ok=True)
        destination = out_dir / f"{source.stem}.{self.name}.mid"

        try:
            encoded = self.encode(source)
            if encoded.token_ids is not None:
                self.decode_to_midi(token_ids=encoded.token_ids, output_path=destination)
            elif encoded.text is not None:
                self.decode_to_midi(text=encoded.text, output_path=destination)
            else:
                return RoundTripResult(
                    repr_name=self.name,
                    source=source,
                    output=None,
                    success=False,
                    error="encode produced neither token_ids nor text",
                    stats_before=encoded.stats,
                )
            from ._fidelity import compare_midi_fidelity

            fidelity = compare_midi_fidelity(source, destination)
            reencoded = self.encode(destination)
            if not fidelity.ok:
                return RoundTripResult(
                    repr_name=self.name,
                    source=source,
                    output=destination,
                    success=False,
                    error=f"round-trip fidelity failed: {fidelity.summary}",
                    stats_before=encoded.stats,
                    stats_after=reencoded.stats,
                    fidelity_ok=False,
                    fidelity_summary=fidelity.summary,
                )
            return RoundTripResult(
                repr_name=self.name,
                source=source,
                output=destination,
                success=True,
                stats_before=encoded.stats,
                stats_after=reencoded.stats,
                fidelity_ok=True,
                fidelity_summary=fidelity.summary,
            )
        except Exception as exc:  # noqa: BLE001 — surface round-trip failures to CLI
            return RoundTripResult(
                repr_name=self.name,
                source=source,
                output=None,
                success=False,
                error=str(exc),
            )
