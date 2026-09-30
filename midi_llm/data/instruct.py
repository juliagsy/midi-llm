"""Build SFT shards from MIDI-Instruct manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from ._encode import encode_midi_file
from .templates import SFTExample, build_edit_example


def _load_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def build_instruct_shard(
    manifest_path: str | Path,
    *,
    repr_name: str,
    output_path: str | Path,
    split: str | None = "train",
) -> int:
    """Write JSONL SFT examples for MIDI editing (Stage S3). Returns example count."""
    manifest = Path(manifest_path)
    root = manifest.parent
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with out.open("w", encoding="utf-8") as writer:
        for record in _load_jsonl(manifest):
            if split is not None and record.get("split") != split:
                continue
            gold_rel = record.get("gold_midi")
            if not gold_rel:
                continue

            midi_in = root / record["midi_in"]
            gold_midi = root / gold_rel
            if not midi_in.is_file() or not gold_midi.is_file():
                continue

            example: SFTExample = build_edit_example(
                repr_name=repr_name,
                instruction=record["instruction"],
                input_midi_payload=encode_midi_file(repr_name, midi_in),
                output_midi_payload=encode_midi_file(repr_name, gold_midi),
                item_id=record["item_id"],
                split=record.get("split", split or "train"),
            )
            writer.write(json.dumps(example.model_dump(), ensure_ascii=False) + "\n")
            count += 1
    return count
