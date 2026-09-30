"""Build Stage S2 caption↔MIDI shards from MidiCaps (no Lakh required)."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ._encode import encode_midi_bytes, encode_midi_file
from .templates import build_caption_to_midi, build_midi_to_caption


def _resolve_midi_path(midi_root: Path, midi_ref: str) -> Path | None:
    """Resolve a MidiCaps path against a local MIDI directory tree."""
    ref = Path(midi_ref)
    candidates = [
        midi_root / ref,
        midi_root / ref.name,
        midi_root / "lmd_full" / ref,
        midi_root / "lmd_matched" / ref,
    ]
    if len(ref.parts) >= 2:
        candidates.append(midi_root / ref.parts[-2] / ref.name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _extract_midi_bytes(row: dict) -> bytes | None:
    for key in ("midi_bytes", "bytes", "midi"):
        raw = row.get(key)
        if isinstance(raw, bytes):
            return raw
    music = row.get("music")
    if isinstance(music, dict) and isinstance(music.get("bytes"), bytes):
        return music["bytes"]
    return None


def build_midicaps_shard(
    *,
    repr_name: str,
    output_path: str | Path,
    midi_root: str | Path | None = None,
    lakh_root: str | Path | None = None,
    dataset_name: str = "amaai-lab/MidiCaps",
    split: str = "train",
    limit: int | None = None,
    bidirectional: bool = True,
) -> int:
    """Write S2 SFT JSONL from HuggingFace MidiCaps.

    MIDI resolution order:
    1. Embedded bytes on the HF row (if present)
    2. Local ``midi_root`` (MidiCaps tarball extract or any flat MIDI tree)
    3. Legacy ``lakh_root`` alias for ``midi_root``
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("MidiCaps shard requires datasets: pip install -e '.[data]'") from exc

    root = Path(midi_root or lakh_root) if (midi_root or lakh_root) else None
    ds = load_dataset(dataset_name, split=split)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0
    with out.open("w", encoding="utf-8") as writer:
        for row in ds:
            if limit is not None and count >= limit:
                break

            caption = row.get("caption") or row.get("text") or row.get("description")
            midi_ref = row.get("location") or row.get("midi") or row.get("midi_path")
            if not caption:
                skipped += 1
                continue

            payload: str | None = None
            raw = _extract_midi_bytes(row)
            if raw:
                try:
                    payload = encode_midi_bytes(repr_name, raw)
                except Exception:
                    skipped += 1
                    continue
            elif root is not None and midi_ref:
                midi_path = _resolve_midi_path(root, str(midi_ref))
                if midi_path is None:
                    skipped += 1
                    continue
                try:
                    payload = encode_midi_file(repr_name, midi_path)
                except Exception:
                    skipped += 1
                    continue
            else:
                skipped += 1
                continue

            item_id = str(row.get("id") or row.get("file") or midi_ref or count)
            examples = [
                build_caption_to_midi(
                    repr_name=repr_name,
                    caption=str(caption),
                    midi_payload=payload,
                    item_id=item_id,
                )
            ]
            if bidirectional:
                examples.append(
                    build_midi_to_caption(
                        repr_name=repr_name,
                        caption=str(caption),
                        midi_payload=payload,
                        item_id=item_id,
                    )
                )

            for example in examples:
                writer.write(json.dumps(example.model_dump(), ensure_ascii=False) + "\n")
                count += 1
                if limit is not None and count >= limit:
                    break
    if count == 0:
        raise RuntimeError(
            f"no MidiCaps examples written (skipped={skipped}). "
            "Provide --midi-root with extracted MidiCaps MIDIs, or use a HF split with bytes."
        )
    return count


def lakh_id_from_path(midi_path: str | Path) -> str | None:
    """Extract Lakh-style hex id from a path if present."""
    match = re.search(r"([0-9a-fA-F]{24,})", str(midi_path))
    return match.group(1).lower() if match else None
