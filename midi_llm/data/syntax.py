"""Build Stage S0 syntax shards from local MIDI files."""

from __future__ import annotations

import json
from pathlib import Path

from midi_llm.midi_repr.payload import serialize_midi_payload
from midi_llm.midi_repr.registry import get_repr

from .templates import TaskKind


def _iter_midi_files(root: Path, max_files: int | None) -> list[Path]:
    files = sorted(root.rglob("*.mid")) + sorted(root.rglob("*.midi"))
    if max_files is not None:
        return files[:max_files]
    return files


def _encode_payload(repr_name: str, midi_path: Path) -> str:
    backend = get_repr(repr_name)
    encoded = backend.encode(midi_path)
    return serialize_midi_payload(repr_name, encoded)


def build_syntax_shard(
    midi_root: str | Path,
    *,
    repr_name: str,
    output_path: str | Path,
    max_files: int | None = None,
) -> int:
    """Write JSONL records for next-token MIDI syntax learning (S0).

    Each record: prompt="" and completion=<encoded MIDI>.
    """
    root = Path(midi_root)
    if not root.is_dir():
        raise NotADirectoryError(midi_root)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as writer:
        for midi_path in _iter_midi_files(root, max_files):
            try:
                payload = _encode_payload(repr_name, midi_path)
            except Exception:
                continue
            if not payload.strip():
                continue
            record = {
                "task": TaskKind.CAPTION_TO_MIDI.value,
                "repr_name": repr_name,
                "prompt": "",
                "completion": payload,
                "metadata": {"source_midi": str(midi_path), "stage": "s0_syntax"},
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        raise RuntimeError(
            f"no syntax examples written to {out}. "
            "Check midi_dir path, file extensions, and representation dependencies."
        )
    return count
