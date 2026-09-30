"""Build Stage S0 syntax shards from HuggingFace GigaMIDI (no Lakh)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .templates import TaskKind
from ._encode import encode_midi_bytes, encode_midi_file


def _extract_midi_bytes(row: dict[str, Any]) -> bytes | None:
    music = row.get("music")
    if isinstance(music, dict):
        raw = music.get("bytes")
        if isinstance(raw, bytes):
            return raw
    for key in ("midi_bytes", "bytes", "content"):
        raw = row.get(key)
        if isinstance(raw, bytes):
            return raw
    return None


def build_gigamidi_shard(
    *,
    repr_name: str,
    output_path: str | Path,
    dataset_name: str = "Metacreation/GigaMIDI",
    split: str = "train",
    limit: int | None = 100,
    streaming: bool = True,
) -> int:
    """Write S0 JSONL from GigaMIDI rows that include embedded MIDI bytes."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("GigaMIDI shard requires datasets: pip install -e '.[data]'") from exc

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if streaming:
        ds = load_dataset(dataset_name, split=split, streaming=True)
    else:
        ds = load_dataset(dataset_name, split=split)

    count = 0
    skipped = 0
    with out.open("w", encoding="utf-8") as writer:
        for row in ds:
            if limit is not None and count >= limit:
                break
            payload: str | None = None
            item_id = str(row.get("id") or row.get("file") or count)
            raw = _extract_midi_bytes(row)
            if raw:
                try:
                    payload = encode_midi_bytes(repr_name, raw)
                except Exception:
                    skipped += 1
                    continue
            else:
                path_val = row.get("path") or row.get("midi_path")
                if path_val and Path(str(path_val)).is_file():
                    try:
                        payload = encode_midi_file(repr_name, path_val)
                    except Exception:
                        skipped += 1
                        continue
                else:
                    skipped += 1
                    continue

            if not payload.strip():
                skipped += 1
                continue

            record = {
                "task": TaskKind.CAPTION_TO_MIDI.value,
                "repr_name": repr_name,
                "prompt": "",
                "completion": payload,
                "metadata": {"item_id": item_id, "stage": "s0_syntax", "source": "gigamidi"},
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        raise RuntimeError(
            f"no GigaMIDI examples written (skipped={skipped}). "
            "Check dataset schema or network access."
        )
    return count
