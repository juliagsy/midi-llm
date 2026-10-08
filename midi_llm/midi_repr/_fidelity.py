"""Compare MIDI files after encode/decode round-trips."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pretty_midi


@dataclass(frozen=True)
class NoteEvent:
    pitch: int
    start: float
    end: float
    is_drum: bool


@dataclass
class FidelityReport:
    ok: bool
    n_notes_source: int
    n_notes_output: int
    n_tracks_source: int
    n_tracks_output: int
    duration_source: float
    duration_output: float
    matched_notes: int
    note_recall: float
    note_precision: float
    issues: list[str]

    @property
    def summary(self) -> str:
        if self.ok:
            return "fidelity ok"
        return "; ".join(self.issues)


def _extract_notes(midi_path: Path) -> list[NoteEvent]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    events: list[NoteEvent] = []
    for inst in midi.instruments:
        for note in inst.notes:
            events.append(
                NoteEvent(
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=float(note.end),
                    is_drum=bool(inst.is_drum),
                )
            )
    events.sort(key=lambda n: (n.start, n.pitch, n.is_drum))
    return events


def _quantize_time(value: float, step: float) -> float:
    if step <= 0:
        return value
    return round(value / step) * step


def _match_notes(
    source: list[NoteEvent],
    output: list[NoteEvent],
    *,
    onset_tol_sec: float,
    duration_tol_sec: float,
) -> tuple[int, int, int]:
    """Return (matched, unmatched_source, unmatched_output)."""
    used_output: set[int] = set()
    matched = 0
    for src in source:
        src_start = _quantize_time(src.start, onset_tol_sec)
        src_duration = src.end - src.start
        found = False
        for idx, out in enumerate(output):
            if idx in used_output:
                continue
            if out.pitch != src.pitch or out.is_drum != src.is_drum:
                continue
            if abs(_quantize_time(out.start, onset_tol_sec) - src_start) > onset_tol_sec:
                continue
            out_duration = out.end - out.start
            if abs(out_duration - src_duration) > duration_tol_sec:
                continue
            used_output.add(idx)
            matched += 1
            found = True
            break
        if not found:
            pass
    unmatched_source = len(source) - matched
    unmatched_output = len(output) - len(used_output)
    return matched, unmatched_source, unmatched_output


def compare_midi_fidelity(
    source: str | Path,
    output: str | Path,
    *,
    duration_tol_sec: float = 0.25,
    onset_tol_sec: float = 0.05,
    min_note_recall: float = 0.9,
    min_note_precision: float = 0.9,
) -> FidelityReport:
    """Check whether decoded MIDI preserves structure from the source file."""
    src_path = Path(source)
    out_path = Path(output)
    src_notes = _extract_notes(src_path)
    out_notes = _extract_notes(out_path)

    src_midi = pretty_midi.PrettyMIDI(str(src_path))
    out_midi = pretty_midi.PrettyMIDI(str(out_path))
    duration_source = float(src_midi.get_end_time())
    duration_output = float(out_midi.get_end_time())
    n_tracks_source = len(src_midi.instruments)
    n_tracks_output = len(out_midi.instruments)

    matched, _, _ = _match_notes(
        src_notes,
        out_notes,
        onset_tol_sec=onset_tol_sec,
        duration_tol_sec=onset_tol_sec,
    )
    recall = matched / len(src_notes) if src_notes else 1.0
    precision = matched / len(out_notes) if out_notes else 1.0

    issues: list[str] = []
    if n_tracks_source != n_tracks_output:
        issues.append(f"track count {n_tracks_source} -> {n_tracks_output}")
    if abs(duration_source - duration_output) > duration_tol_sec:
        issues.append(
            f"duration {duration_source:.3f}s -> {duration_output:.3f}s (tol {duration_tol_sec}s)"
        )
    if recall < min_note_recall:
        issues.append(f"note recall {recall:.3f} < {min_note_recall}")
    if precision < min_note_precision:
        issues.append(f"note precision {precision:.3f} < {min_note_precision}")

    ok = not issues
    return FidelityReport(
        ok=ok,
        n_notes_source=len(src_notes),
        n_notes_output=len(out_notes),
        n_tracks_source=n_tracks_source,
        n_tracks_output=n_tracks_output,
        duration_source=duration_source,
        duration_output=duration_output,
        matched_notes=matched,
        note_recall=recall,
        note_precision=precision,
        issues=issues,
    )
