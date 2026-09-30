"""Build SFT shards from MIDI-Instruct manifests."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ._encode import encode_midi_file
from .jsonl_io import MANIFEST_FIELDS, iter_jsonl
from .templates import SFTExample, build_edit_example


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

    skipped: Counter[str] = Counter()
    count = 0
    with out.open("w", encoding="utf-8") as writer:
        for _line_no, record in iter_jsonl(
            manifest,
            required_fields=MANIFEST_FIELDS,
            label="manifest",
        ):
            if split is not None and record.get("split") != split:
                skipped["wrong_split"] += 1
                continue
            gold_rel = record.get("gold_midi")
            if not gold_rel:
                skipped["missing_gold_midi_field"] += 1
                continue

            midi_in = root / record["midi_in"]
            gold_midi = root / gold_rel
            if not midi_in.is_file():
                skipped["missing_midi_in"] += 1
                continue
            if not gold_midi.is_file():
                skipped["missing_gold_midi_file"] += 1
                continue

            try:
                example: SFTExample = build_edit_example(
                    repr_name=repr_name,
                    instruction=record["instruction"],
                    input_midi_payload=encode_midi_file(repr_name, midi_in),
                    output_midi_payload=encode_midi_file(repr_name, gold_midi),
                    item_id=record["item_id"],
                    split=record.get("split", split or "train"),
                )
            except Exception as exc:  # noqa: BLE001 — count encode failures per row
                skipped[f"encode_error:{type(exc).__name__}"] += 1
                continue

            writer.write(json.dumps(example.model_dump(), ensure_ascii=False) + "\n")
            count += 1

    if count == 0:
        skip_summary = ", ".join(f"{key}={value}" for key, value in sorted(skipped.items()))
        raise RuntimeError(
            f"no instruct examples written to {out} (split={split!r}, skipped: {skip_summary}). "
            "Check manifest paths and representation dependencies."
        )

    if skipped:
        skip_summary = ", ".join(f"{key}={value}" for key, value in sorted(skipped.items()))
        print(f"instruct shard: skipped {sum(skipped.values())} manifest rows ({skip_summary})")

    return count
