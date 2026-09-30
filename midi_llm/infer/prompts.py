"""Build edit prompts from MIDI-Instruct manifest items."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from midi_llm.data._encode import encode_midi_file
from midi_llm.data.jsonl_io import MANIFEST_FIELDS, iter_jsonl
from midi_llm.data.templates import build_edit_example
from midi_llm.train.dataset import format_llama_instruct


def iter_manifest_records(manifest_path: str | Path, *, split: str | None = None) -> Iterator[dict]:
    for _line_no, record in iter_jsonl(
        manifest_path,
        required_fields=MANIFEST_FIELDS,
        label="manifest",
    ):
        if split is None or record.get("split") == split:
            yield record


def build_edit_prompt(
    manifest_path: str | Path,
    record: dict,
    *,
    repr_name: str,
    chat_template: bool = True,
) -> tuple[str, str, str]:
    """Return (item_id, model_input, plain_prompt) for one benchmark item."""
    root = Path(manifest_path).parent
    midi_in = root / record["midi_in"]
    payload = encode_midi_file(repr_name, midi_in)
    example = build_edit_example(
        repr_name=repr_name,
        instruction=record["instruction"],
        input_midi_payload=payload,
        output_midi_payload="",
        item_id=record["item_id"],
        split=record.get("split", "test"),
    )
    if chat_template:
        parts = format_llama_instruct(example.prompt, "")
        model_input = parts["prompt"]
    else:
        model_input = example.prompt
    return record["item_id"], model_input, example.prompt
